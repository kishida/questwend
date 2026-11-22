// Inkling 2 vision tower. The graph and the preprocessing follow llama.cpp's
// inkling2v (tools/mtmd, MIT), including the Pillow-compatible resize, so the
// embeddings can be compared with it directly.

#include "vision_inkling.h"
#include "gguf_util.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <thread>

namespace questwend {

namespace {

constexpr int   GRAPH_NODES      = 4096;
constexpr int   MAX_IMAGE_TOKENS = 4096;
constexpr float ROPE_BASE        = 10000.0f;

// ---- Pillow's bicubic resize: antialiased, 22-bit fixed point ----

constexpr int PRECISION = 32 - 8 - 2;

struct Kernel {
    int size = 0;
    std::vector<int>     first, count;
    std::vector<int32_t> weight;
};

double bicubic(double x) {
    const double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

Kernel make_kernel(int in, int out) {
    const double scale   = (double) in / out;
    const double widen   = std::max(1.0, scale);
    const double support = 2.0 * widen;
    Kernel k;
    k.size = (int) std::ceil(support) * 2 + 1;
    k.first.resize((size_t) out);
    k.count.resize((size_t) out);
    k.weight.assign((size_t) out * k.size, 0);
    std::vector<double> w((size_t) k.size);
    for (int i = 0; i < out; ++i) {
        const double center = (i + 0.5) * scale;
        const int lo = std::max(0, (int) (center - support + 0.5));
        const int n  = std::min(in, (int) (center + support + 0.5)) - lo;
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            w[(size_t) j] = bicubic((j + lo - center + 0.5) * (1.0 / widen));
            sum += w[(size_t) j];
        }
        for (int j = 0; j < n; ++j) {
            const double v = sum != 0.0 ? w[(size_t) j] / sum : w[(size_t) j];
            k.weight[(size_t) i * k.size + j] = (int32_t) (v * (1 << PRECISION) + (v < 0 ? -0.5 : 0.5));
        }
        k.first[(size_t) i] = lo;
        k.count[(size_t) i] = n;
    }
    return k;
}

// One axis of the separable resize: `lines` independent lines of `in` pixels.
// step = between pixels of a line, stride = between lines (both in bytes).
void resample(const uint8_t * src, size_t src_step, size_t src_stride,
              uint8_t * dst, size_t dst_step, size_t dst_stride,
              int lines, int out, const Kernel & k) {
    for (int l = 0; l < lines; ++l) {
        const uint8_t * s = src + (size_t) l * src_stride;
        uint8_t *       d = dst + (size_t) l * dst_stride;
        for (int i = 0; i < out; ++i) {
            const int32_t * w = &k.weight[(size_t) i * k.size];
            const uint8_t * p = s + (size_t) k.first[(size_t) i] * src_step;
            int32_t acc[3] = { 1 << (PRECISION - 1), 1 << (PRECISION - 1), 1 << (PRECISION - 1) };
            for (int j = 0; j < k.count[(size_t) i]; ++j, p += src_step)
                for (int c = 0; c < 3; ++c) acc[c] += p[c] * w[j];
            for (int c = 0; c < 3; ++c)
                d[(size_t) i * dst_step + c] = (uint8_t) std::min(255, std::max(0, acc[c] >> PRECISION));
        }
    }
}

std::vector<uint8_t> resize_bicubic(const uint8_t * rgb, int w, int h, int tw, int th) {
    std::vector<uint8_t> across((size_t) tw * h * 3);
    if (tw == w) std::copy_n(rgb, across.size(), across.begin());
    else resample(rgb, 3, (size_t) w * 3, across.data(), 3, (size_t) tw * 3, h, tw, make_kernel(w, tw));
    if (th == h) return across;
    std::vector<uint8_t> out((size_t) tw * th * 3);
    resample(across.data(), (size_t) tw * 3, 3, out.data(), (size_t) tw * 3, 3, tw, th, make_kernel(h, th));
    return out;
}

} // namespace

struct InklingVision::Impl {
    int   n_layer = 0, dim = 0, n_head = 0, d_head = 0, proj_dim = 0;
    int   patch = 0, merge = 0;
    int   pos_side = 0;          // the learned position grid (32 x 32)
    float eps = 1e-5f;
    float mean[3] = { 0.5f, 0.5f, 0.5f }, sd[3] = { 0.5f, 0.5f, 0.5f };
    ggml_backend_t gpu  = nullptr;   // the caller's; may itself be the CPU
    ggml_backend_t cpu  = nullptr;
    ggml_context * host = nullptr;   // every weight, in RAM
    using Weights = std::map<std::string, ggml_tensor *>;
    Weights host_w;

    ~Impl() {
        if (cpu)  ggml_backend_free(cpu);
        if (host) ggml_free(host);
    }

    // Weights and graph of one encode, on one backend.
    struct Graph {
        ggml_backend_t        backend = nullptr;
        ggml_context *        wctx = nullptr;
        ggml_backend_buffer_t wbuf = nullptr;
        Weights               w;
        ggml_context * ctx   = nullptr;
        ggml_cgraph *  gf    = nullptr;
        ggml_gallocr_t alloc = nullptr;
        ggml_tensor * pixels = nullptr, * pos_w = nullptr, * pos_h = nullptr, * out = nullptr;
        Graph() = default;
        Graph(const Graph &) = delete;
        Graph & operator=(const Graph &) = delete;
        ~Graph() {
            if (alloc) ggml_gallocr_free(alloc);
            if (ctx)   ggml_free(ctx);
            if (wbuf)  ggml_backend_buffer_free(wbuf);
            if (wctx)  ggml_free(wctx);
        }
    };

    void load(const std::string & path);
    bool prepare(Graph & g, ggml_backend_t be, int width, int height) const;
    bool build(Graph & g, int width, int height) const;
};

void InklingVision::Impl::load(const std::string & path) {
    gguf_init_params ip{};
    ip.no_alloc = false;
    ip.ctx      = &host;
    gguf_context * gg = gguf_init_from_file(path.c_str(), ip);
    if (!gg) throw std::runtime_error("failed to open mmproj: " + path);
    struct Free {
        gguf_context * g;
        ~Free() { gguf_free(g); }
    } guard{ gg };

    n_layer  = (int) gguf_u32(gg, "clip.vision.block_count");
    dim      = (int) gguf_u32(gg, "clip.vision.embedding_length");
    n_head   = (int) gguf_u32(gg, "clip.vision.attention.head_count");
    proj_dim = (int) gguf_u32(gg, "clip.vision.projection_dim");
    patch    = (int) gguf_u32(gg, "clip.vision.patch_size");
    merge    = (int) gguf_u32(gg, "clip.vision.spatial_merge_size", 2);
    eps      = gguf_f32(gg, "clip.vision.attention.layer_norm_epsilon", 1e-5f);
    if (!n_layer || !dim || !n_head || !proj_dim || !patch || !merge)
        throw std::runtime_error("mmproj: missing clip.vision.* metadata");
    d_head = dim / n_head;
    auto triple = [&](const char * key, float * v) {
        const int64_t k = gguf_find_key(gg, key);
        if (k < 0 || gguf_get_arr_type(gg, k) != GGUF_TYPE_FLOAT32 || gguf_get_arr_n(gg, k) != 3) return;
        std::copy_n((const float *) gguf_get_arr_data(gg, k), 3, v);
    };
    triple("clip.vision.image_mean", mean);
    triple("clip.vision.image_std", sd);

    const int64_t n_tensors = gguf_get_n_tensors(gg);
    size_t bytes = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gg, i);
        ggml_tensor * t = ggml_get_tensor(host, name);
        host_w[name] = t;
        bytes += ggml_nbytes(t);
    }
    const auto pos = host_w.find("v.position_embd.weight");
    if (pos == host_w.end()) throw std::runtime_error("mmproj: missing tensor v.position_embd.weight");
    pos_side = (int) std::lround(std::sqrt((double) pos->second->ne[1]));

    cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, std::max(4, (int) std::thread::hardware_concurrency() / 2));
    fprintf(stderr, "mmproj: %s | ViT %d layers dim=%d heads=%d, %dpx cells, up to %d tokens x %d"
                    " | %zu MB, in VRAM only while encoding\n",
            InklingVision::PROJECTOR, n_layer, dim, n_head, patch * merge, MAX_IMAGE_TOKENS, proj_dim, bytes >> 20);
}

// false = `be` has no room for the weights or the compute buffer
bool InklingVision::Impl::prepare(Graph & g, ggml_backend_t be, int width, int height) const {
    g.backend = be;
    if (ggml_backend_is_cpu(be)) {
        g.w = host_w;
    } else {
        ggml_init_params wp{};
        wp.mem_size = ggml_tensor_overhead() * (host_w.size() + 1);
        wp.no_alloc = true;
        g.wctx = ggml_init(wp);
        for (const auto & kv : host_w) {
            ggml_tensor * t = ggml_dup_tensor(g.wctx, kv.second);
            ggml_set_name(t, kv.first.c_str());
            g.w[kv.first] = t;
        }
        g.wbuf = ggml_backend_alloc_ctx_tensors(g.wctx, be);
        if (!g.wbuf) return false;
        for (const auto & kv : host_w)
            ggml_backend_tensor_set(g.w[kv.first], kv.second->data, 0, ggml_nbytes(kv.second));
    }
    return build(g, width, height);
}

bool InklingVision::Impl::build(Graph & g, int width, int height) const {
    const Weights & w = g.w;
    auto W = [&w](const std::string & name) -> ggml_tensor * {
        const auto it = w.find(name);
        if (it == w.end()) throw std::runtime_error("mmproj: missing tensor " + name);
        return it->second;
    };
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false);
    p.no_alloc = true;
    g.ctx = ggml_init(p);
    g.gf  = ggml_new_graph_custom(g.ctx, GRAPH_NODES, false);
    ggml_context * c = g.ctx;

    const int pw = width / patch, ph = height / patch, np = pw * ph, n_pos = np + 1;
    g.pixels = ggml_new_tensor_4d(c, GGML_TYPE_F32, width, height, 3, 1);
    g.pos_w  = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_pos);
    g.pos_h  = ggml_new_tensor_1d(c, GGML_TYPE_I32, n_pos);
    for (ggml_tensor * t : { g.pixels, g.pos_w, g.pos_h }) ggml_set_input(t);

    auto norm = [&](ggml_tensor * x, const std::string & name) {
        x = ggml_mul(c, ggml_norm(c, x, eps), W(name + ".weight"));
        return ggml_add(c, x, W(name + ".bias"));
    };
    auto linear = [&](ggml_tensor * x, const std::string & name) {
        x = ggml_mul_mat(c, W(name + ".weight"), x);
        const auto b = w.find(name + ".bias");
        return b == w.end() ? x : ggml_add(c, x, b->second);
    };

    // patches + learned positions (interpolated to this grid), CLS in front
    ggml_tensor * cur = ggml_conv_2d(c, W("v.patch_embd.weight"), g.pixels, patch, patch, 0, 0, 1, 1);
    cur = ggml_cont(c, ggml_transpose(c, ggml_reshape_2d(c, cur, np, dim)));
    ggml_tensor * pos = W("v.position_embd.weight");
    if (pw != pos_side || ph != pos_side) {
        pos = ggml_permute(c, ggml_reshape_3d(c, pos, dim, pos_side, pos_side), 2, 0, 1, 3);
        pos = ggml_interpolate(c, pos, pw, ph, dim, 1, GGML_SCALE_MODE_BILINEAR);
        pos = ggml_cont_2d(c, ggml_permute(c, pos, 1, 2, 0, 3), dim, np);
    }
    cur = ggml_add(c, cur, pos);
    ggml_tensor * cls = ggml_add(c, W("v.class_embd"), W("v.class_position_embd.weight"));
    cur = ggml_concat(c, ggml_reshape_2d(c, cls, dim, 1), cur, 1);
    cur = norm(cur, "v.pre_ln");

    // half of each head rotates by the column, the other half by the row
    auto rope = [&](ggml_tensor * x) {
        x = ggml_rope_ext(c, x, g.pos_w, nullptr, d_head / 2, 0, 0, ROPE_BASE, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        x = ggml_rope_ext(c, x, g.pos_h, nullptr, d_head / 2, 0, 0, ROPE_BASE, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        return ggml_rope_set_offset(x, d_head / 2);
    };
    const float kq_scale = 1.0f / std::sqrt((float) d_head);
    for (int il = 0; il < n_layer; ++il) {
        const std::string b = "v.blk." + std::to_string(il) + ".";
        ggml_tensor * x = norm(cur, b + "ln1");
        auto heads = [&](const char * name) {
            return ggml_reshape_3d(c, linear(x, b + name), d_head, n_head, n_pos);
        };
        ggml_tensor * q = ggml_permute(c, rope(heads("attn_q")), 0, 2, 1, 3);
        ggml_tensor * k = ggml_permute(c, rope(heads("attn_k")), 0, 2, 1, 3);
        ggml_tensor * v = ggml_permute(c, heads("attn_v"), 0, 2, 1, 3);
        ggml_tensor * att = ggml_flash_attn_ext(c, q, ggml_cast(c, k, GGML_TYPE_F16), ggml_cast(c, v, GGML_TYPE_F16),
                                                nullptr, kq_scale, 0.0f, 0.0f);
        att = linear(ggml_reshape_2d(c, att, dim, n_pos), b + "attn_out");
        cur = ggml_add(c, ggml_mul(c, att, W(b + "ls1.weight")), cur);

        x = norm(cur, b + "ln2");
        x = linear(ggml_gelu_erf(c, linear(x, b + "ffn_up")), b + "ffn_down");
        cur = ggml_add(c, ggml_mul(c, x, W(b + "ls2.weight")), cur);
    }

    // drop CLS, merge 2x2 cells, project
    cur = ggml_view_2d(c, cur, dim, np, cur->nb[1], cur->nb[1]);
    cur = ggml_reshape_3d(c, cur, dim, pw, ph);
    cur = ggml_reshape_3d(c, cur, dim * merge, pw / merge, ph);
    cur = ggml_permute(c, cur, 0, 2, 1, 3);
    cur = ggml_cont_3d(c, cur, dim * merge * merge, ph / merge, pw / merge);
    cur = ggml_permute(c, cur, 0, 2, 1, 3);
    cur = ggml_cont_2d(c, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
    cur = linear(ggml_gelu_erf(c, linear(cur, "mm.1")), "mm.2");
    ggml_set_output(cur);
    g.out = cur;
    ggml_build_forward_expand(g.gf, cur);

    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g.backend));
    return ggml_gallocr_alloc_graph(g.alloc, g.gf);
}

InklingVision::InklingVision(const std::string & path, ggml_backend_t backend) : impl_(new Impl()) {
    impl_->gpu = backend;
    impl_->load(path);
}
InklingVision::~InklingVision() = default;
int InklingVision::n_embd() const { return impl_->proj_dim; }
std::vector<float> InklingVision::encode_rgb(const uint8_t * rgb, int w, int h) {
    const Impl & im = *impl_;
    if (w <= 0 || h <= 0) throw std::runtime_error("vision: empty image");
    const int cell = im.patch * im.merge;
    const int64_t max_px = (int64_t) MAX_IMAGE_TOKENS * cell * cell;
    auto align = [cell](int64_t v) { return (v + cell - 1) / cell * cell; };

    // never resized, except to bound the token count
    std::vector<uint8_t> shrunk;
    if (align(w) * align(h) > max_px) {
        const float beta = std::sqrt((float) h * (float) w / (float) max_px);
        auto fit = [&](int v) { return std::max(cell, (int) std::floor((float) v / beta / (float) cell) * cell); };
        const int tw = fit(w), th = fit(h);
        if (tw > 65536 || th > 65536) throw std::runtime_error("vision: image aspect ratio out of range");
        shrunk = resize_bicubic(rgb, w, h, tw, th);
        rgb = shrunk.data();
        w = tw;
        h = th;
    }

    // pad right / bottom with -1 (before normalisation, as the reference does)
    const int W = (int) align(w), H = (int) align(h);
    std::vector<float> px((size_t) W * H * 3);
    for (int c = 0; c < 3; ++c) {
        const float m = im.mean[c] * 255.0f, s = im.sd[c] * 255.0f;
        float * plane = &px[(size_t) c * W * H];
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                const float v = (x < w && y < h) ? (float) rgb[((size_t) y * w + x) * 3 + c] : -1.0f;
                plane[(size_t) y * W + x] = (v - m) / s;
            }
    }
    // CLS at 0, patches from 1
    std::vector<int32_t> pos_w{ 0 }, pos_h{ 0 };
    for (int y = 0; y < H / im.patch; ++y)
        for (int x = 0; x < W / im.patch; ++x) {
            pos_w.push_back(x + 1);
            pos_h.push_back(y + 1);
        }

    auto gp = std::make_unique<Impl::Graph>();
    if (!im.prepare(*gp, im.gpu, W, H)) {
        gp.reset();   // give back what did fit before the CPU takes over
        gp = std::make_unique<Impl::Graph>();
        fprintf(stderr, "vision: no VRAM for a %dx%d image, encoding it on the CPU\n", W, H);
        if (!im.prepare(*gp, im.cpu, W, H))
            throw std::runtime_error("vision: not enough memory to encode the image");
    }
    Impl::Graph & g = *gp;
    ggml_backend_tensor_set(g.pixels, px.data(), 0, px.size() * sizeof(float));
    ggml_backend_tensor_set(g.pos_w, pos_w.data(), 0, pos_w.size() * sizeof(int32_t));
    ggml_backend_tensor_set(g.pos_h, pos_h.data(), 0, pos_h.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(g.backend, g.gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("mmproj: compute failed");
    std::vector<float> out((size_t) ggml_nelements(g.out));
    ggml_backend_tensor_get(g.out, out.data(), 0, out.size() * sizeof(float));
    return out;
}

} // namespace questwend
