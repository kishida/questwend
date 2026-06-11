#include "vision.h"
#include "gguf_util.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_NO_THREAD_LOCALS
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

namespace qwencpp {

static const int VIS_GRAPH_SIZE = 4096;

struct VisionEncoder::Impl {
    // hyper-parameters (clip.* metadata)
    int   n_layer  = 0;     // ViT blocks
    int   dim      = 0;     // ViT embedding dim
    int   ffn      = 0;     // ViT FFN hidden dim
    int   n_head   = 0;
    int   d_head   = 0;
    int   proj_dim = 0;     // LLM embedding dim
    int   patch    = 0;     // 16
    int   img_size = 0;     // 768
    int   merge    = 0;     // 2
    float ln_eps   = 1e-6f;
    int   n_side   = 0;     // img_size / patch (48)
    int   n_patch  = 0;     // n_side^2 (2304)
    int   n_tokens = 0;     // (n_side/merge)^2 (576)

    ggml_backend_t        backend = nullptr;
    ggml_context *        wctx    = nullptr;   // weight tensors (backend buffer)
    ggml_backend_buffer_t wbuf    = nullptr;
    std::map<std::string, ggml_tensor *> w;

    // persistent forward graph
    ggml_context * gctx   = nullptr;
    ggml_cgraph  * gf     = nullptr;
    ggml_gallocr_t galloc = nullptr;

    ~Impl() {
        if (galloc) ggml_gallocr_free(galloc);
        if (gctx)   ggml_free(gctx);
        if (wbuf)   ggml_backend_buffer_free(wbuf);
        if (wctx)   ggml_free(wctx);
    }

    ggml_tensor * W(const std::string & name) {
        auto it = w.find(name);
        if (it == w.end()) throw std::runtime_error("mmproj: missing tensor " + name);
        return it->second;
    }

    // 2x2-block-contiguous reorder of patch index p (row-major py*n_side+px).
    int reorder_index(int py, int px) const {
        const int B = n_side / merge;
        const int by = py / merge, yi = py % merge;
        const int bx = px / merge, xi = px % merge;
        return ((by * B + bx) * merge + yi) * merge + xi;
    }

    void load_weights(const std::string & path);
    void build_graph();
    std::vector<float> forward(const std::vector<float> & patches);
};

// ---- weight loading -------------------------------------------------------

void VisionEncoder::Impl::load_weights(const std::string & path) {
    // read the whole mmproj into a CPU ggml context (mmproj files are small)
    ggml_context * cpu_ctx = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = false;
    ip.ctx      = &cpu_ctx;
    gguf_context * g = gguf_init_from_file(path.c_str(), ip);
    if (!g) throw std::runtime_error("failed to open mmproj: " + path);

    const std::string arch = gguf_str(g, "general.architecture");
    if (arch != "clip") {
        gguf_free(g); ggml_free(cpu_ctx);
        throw std::runtime_error("mmproj: unexpected architecture '" + arch + "' (want clip)");
    }
    const std::string ptype = gguf_str(g, "clip.projector_type");
    if (ptype != "qwen3vl_merger")
        fprintf(stderr, "mmproj: warning: projector_type '%s' (expected qwen3vl_merger)\n", ptype.c_str());

    n_layer  = (int) gguf_u32(g, "clip.vision.block_count");
    dim      = (int) gguf_u32(g, "clip.vision.embedding_length");
    ffn      = (int) gguf_u32(g, "clip.vision.feed_forward_length");
    n_head   = (int) gguf_u32(g, "clip.vision.attention.head_count");
    proj_dim = (int) gguf_u32(g, "clip.vision.projection_dim");
    patch    = (int) gguf_u32(g, "clip.vision.patch_size");
    img_size = (int) gguf_u32(g, "clip.vision.image_size");
    merge    = (int) gguf_u32(g, "clip.vision.spatial_merge_size", 2);
    ln_eps   = gguf_f32(g, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    if (!n_layer || !dim || !n_head || !proj_dim || !patch || !img_size)
        throw std::runtime_error("mmproj: missing clip.vision.* metadata");
    d_head   = dim / n_head;
    n_side   = img_size / patch;
    n_patch  = n_side * n_side;
    n_tokens = (n_side / merge) * (n_side / merge);

    // helper: fetch a CPU-side tensor as f32
    auto fetch_f32 = [&](const std::string & name, bool required = true) -> std::vector<float> {
        ggml_tensor * t = ggml_get_tensor(cpu_ctx, name.c_str());
        if (!t) {
            if (required) throw std::runtime_error("mmproj: missing tensor " + name);
            return {};
        }
        const int64_t ne = ggml_nelements(t);
        std::vector<float> out(ne);
        if (t->type == GGML_TYPE_F32) memcpy(out.data(), t->data, ne * sizeof(float));
        else ggml_get_type_traits(t->type)->to_float(t->data, out.data(), ne);
        return out;
    };

    // ---- precompute host-side transforms ----
    // patch embedding: sum the still-image and temporal conv kernels (both are
    // applied and added, per HF/llama.cpp), flattened to [K, dim] for matmul
    const int K = patch * patch * 3;
    std::vector<float> w0 = fetch_f32("v.patch_embd.weight");
    std::vector<float> w1 = fetch_f32("v.patch_embd.weight.1", false);
    if (!w1.empty())
        for (size_t i = 0; i < w0.size(); ++i) w0[i] += w1[i];

    // position embedding reordered into 2x2-block-contiguous token order
    std::vector<float> pos = fetch_f32("v.position_embd.weight");   // [dim, n_patch]
    std::vector<float> posR(pos.size());
    for (int py = 0; py < n_side; ++py)
        for (int px = 0; px < n_side; ++px) {
            const int src = py * n_side + px;
            const int dst = reorder_index(py, px);
            memcpy(&posR[(size_t) dst * dim], &pos[(size_t) src * dim], dim * sizeof(float));
        }

    // ---- create backend weight tensors ----
    const int n_named = 8 + n_layer * 12;
    ggml_init_params wp{};
    wp.mem_size = ggml_tensor_overhead() * (n_named + 8) + 1024;
    wp.no_alloc = true;
    wctx = ggml_init(wp);

    std::vector<std::pair<ggml_tensor *, const float *>> uploads;
    std::vector<std::vector<float>> keep;   // keep host data alive until upload

    auto mk = [&](const std::string & name, std::vector<float> && data,
                  int64_t ne0, int64_t ne1 = 1) {
        ggml_tensor * t = ne1 > 1 ? ggml_new_tensor_2d(wctx, GGML_TYPE_F32, ne0, ne1)
                                  : ggml_new_tensor_1d(wctx, GGML_TYPE_F32, ne0);
        ggml_set_name(t, name.c_str());
        if ((int64_t) data.size() != ne0 * ne1)
            throw std::runtime_error("mmproj: size mismatch for " + name);
        keep.emplace_back(std::move(data));
        uploads.push_back({ t, keep.back().data() });
        w[name] = t;
        return t;
    };

    mk("patch_w", std::move(w0), K, dim);
    mk("patch_b", fetch_f32("v.patch_embd.bias"), dim);
    mk("pos_r",   std::move(posR), dim, n_patch);
    for (int il = 0; il < n_layer; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        mk(p + "ln1.w",      fetch_f32(p + "ln1.weight"),      dim);
        mk(p + "ln1.b",      fetch_f32(p + "ln1.bias"),        dim);
        mk(p + "qkv.w",      fetch_f32(p + "attn_qkv.weight"), dim, 3 * dim);
        mk(p + "qkv.b",      fetch_f32(p + "attn_qkv.bias"),   3 * dim);
        mk(p + "attn_out.w", fetch_f32(p + "attn_out.weight"), dim, dim);
        mk(p + "attn_out.b", fetch_f32(p + "attn_out.bias"),   dim);
        mk(p + "ln2.w",      fetch_f32(p + "ln2.weight"),      dim);
        mk(p + "ln2.b",      fetch_f32(p + "ln2.bias"),        dim);
        mk(p + "up.w",       fetch_f32(p + "ffn_up.weight"),   dim, ffn);
        mk(p + "up.b",       fetch_f32(p + "ffn_up.bias"),     ffn);
        mk(p + "down.w",     fetch_f32(p + "ffn_down.weight"), ffn, dim);
        mk(p + "down.b",     fetch_f32(p + "ffn_down.bias"),   dim);
    }
    mk("post_ln.w", fetch_f32("v.post_ln.weight"), dim);
    mk("post_ln.b", fetch_f32("v.post_ln.bias"),   dim);
    const int mm_in = merge * merge * dim;
    {
        std::vector<float> mm0 = fetch_f32("mm.0.weight");
        const int mm_hidden = (int) (mm0.size() / mm_in);
        mk("mm0.w", std::move(mm0), mm_in, mm_hidden);
        mk("mm0.b", fetch_f32("mm.0.bias"), mm_hidden);
        mk("mm2.w", fetch_f32("mm.2.weight"), mm_hidden, proj_dim);
        mk("mm2.b", fetch_f32("mm.2.bias"), proj_dim);
    }

    // vision M-RoPE cos/sin tables: pairs (d, d + d_head/2) for d in [0, d_head/2);
    // section 0 (d < d_head/4) uses py, section 1 uses px;
    // theta_scale = 10000^(-4/d_head), theta resets per section.
    {
        const float theta_scale = std::pow(10000.0f, -4.0f / (float) d_head);
        std::vector<float> C((size_t) d_head * n_patch), S((size_t) d_head * n_patch);
        const int quarter = d_head / 4, half = d_head / 2;
        for (int py = 0; py < n_side; ++py)
            for (int px = 0; px < n_side; ++px) {
                const int t = reorder_index(py, px);
                float theta = (float) py;
                for (int d = 0; d < half; ++d) {
                    if (d == quarter) theta = (float) px;
                    const float c = std::cos(theta), s = std::sin(theta);
                    C[(size_t) t * d_head + d]        = c;
                    C[(size_t) t * d_head + d + half] = c;
                    S[(size_t) t * d_head + d]        = s;
                    S[(size_t) t * d_head + d + half] = s;
                    theta *= theta_scale;
                }
            }
        mk("rope_cos", std::move(C), d_head, n_patch);
        mk("rope_sin", std::move(S), d_head, n_patch);
    }

    wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuf) throw std::runtime_error("mmproj: failed to alloc weight buffer");
    for (auto & u : uploads)
        ggml_backend_tensor_set(u.first, u.second, 0, ggml_nbytes(u.first));

    gguf_free(g);
    ggml_free(cpu_ctx);

    fprintf(stderr, "mmproj: %s | ViT %d layers dim=%d heads=%d -> %d tokens x %d (LLM dim)\n",
            ptype.c_str(), n_layer, dim, n_head, n_tokens, proj_dim);
}

// ---- forward graph --------------------------------------------------------

void VisionEncoder::Impl::build_graph() {
    ggml_init_params gp{};
    gp.mem_size = ggml_tensor_overhead() * VIS_GRAPH_SIZE
                + ggml_graph_overhead_custom(VIS_GRAPH_SIZE, false);
    gp.no_alloc = true;
    gctx = ggml_init(gp);
    gf = ggml_new_graph_custom(gctx, VIS_GRAPH_SIZE, false);

    const int K = patch * patch * 3;
    const float kq_scale = 1.0f / std::sqrt((float) d_head);

    // input: im2col'd patches in 2x2-block-contiguous order, (ic,ky,kx) per column
    ggml_tensor * inp = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, K, n_patch);
    ggml_set_input(inp); ggml_set_name(inp, "vis_inp");

    auto layer_norm = [&](ggml_tensor * x, ggml_tensor * lw, ggml_tensor * lb) {
        x = ggml_norm(gctx, x, ln_eps);
        return ggml_add(gctx, ggml_mul(gctx, x, lw), lb);
    };

    // patch + position embedding
    ggml_tensor * cur = ggml_mul_mat(gctx, W("patch_w"), inp);          // [dim, n_patch]
    cur = ggml_add(gctx, cur, W("patch_b"));
    cur = ggml_add(gctx, cur, W("pos_r"));

    // M-RoPE applied per head: x*C + rot_half(x)*S, with rot_half(x) =
    // concat(-x_hi, x_lo) along the d_head axis (NeoX pairing d <-> d+dh/2)
    ggml_tensor * rc3 = ggml_reshape_3d(gctx, W("rope_cos"), d_head, 1, n_patch);
    ggml_tensor * rs3 = ggml_reshape_3d(gctx, W("rope_sin"), d_head, 1, n_patch);
    auto mrope = [&](ggml_tensor * x) {   // x: [d_head, n_head, n_patch]
        const int half = d_head / 2;
        ggml_tensor * lo = ggml_view_3d(gctx, x, half, n_head, n_patch,
                x->nb[1], x->nb[2], 0);
        ggml_tensor * hi = ggml_view_3d(gctx, x, half, n_head, n_patch,
                x->nb[1], x->nb[2], half * ggml_element_size(x));
        ggml_tensor * rot = ggml_concat(gctx, ggml_scale(gctx, ggml_cont(gctx, hi), -1.0f),
                                        ggml_cont(gctx, lo), 0);        // [d_head, nh, np]
        return ggml_add(gctx, ggml_mul(gctx, x, rc3), ggml_mul(gctx, rot, rs3));
    };

    for (int il = 0; il < n_layer; ++il) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        ggml_tensor * resid = cur;

        ggml_tensor * x = layer_norm(cur, W(p + "ln1.w"), W(p + "ln1.b"));
        ggml_tensor * qkv = ggml_add(gctx, ggml_mul_mat(gctx, W(p + "qkv.w"), x),
                                     W(p + "qkv.b"));                   // [3*dim, np]
        const size_t es = ggml_element_size(qkv);
        auto head_view = [&](size_t off) {
            return ggml_view_3d(gctx, qkv, d_head, n_head, n_patch,
                                (size_t) d_head * es, qkv->nb[1], off * es);
        };
        ggml_tensor * q = mrope(head_view(0));
        ggml_tensor * k = mrope(head_view((size_t) dim));
        ggml_tensor * v = head_view((size_t) 2 * dim);

        // full (unmasked) attention, batched over heads
        q = ggml_cont(gctx, ggml_permute(gctx, q, 0, 2, 1, 3));         // [dh, np, nh]
        k = ggml_cont(gctx, ggml_permute(gctx, k, 0, 2, 1, 3));
        ggml_tensor * kq = ggml_mul_mat(gctx, k, q);                    // [np(k), np(q), nh]
        kq = ggml_soft_max(gctx, ggml_scale(gctx, kq, kq_scale));
        ggml_tensor * vt = ggml_cont(gctx, ggml_permute(gctx, v, 1, 2, 0, 3)); // [np, dh, nh]
        ggml_tensor * kqv = ggml_mul_mat(gctx, vt, kq);                 // [dh, np(q), nh]
        kqv = ggml_cont(gctx, ggml_permute(gctx, kqv, 0, 2, 1, 3));     // [dh, nh, np]
        ggml_tensor * att = ggml_reshape_2d(gctx, kqv, dim, n_patch);

        att = ggml_add(gctx, ggml_mul_mat(gctx, W(p + "attn_out.w"), att),
                       W(p + "attn_out.b"));
        cur = ggml_add(gctx, att, resid);

        resid = cur;
        x = layer_norm(cur, W(p + "ln2.w"), W(p + "ln2.b"));
        x = ggml_add(gctx, ggml_mul_mat(gctx, W(p + "up.w"), x), W(p + "up.b"));
        x = ggml_gelu(gctx, x);
        x = ggml_add(gctx, ggml_mul_mat(gctx, W(p + "down.w"), x), W(p + "down.b"));
        cur = ggml_add(gctx, x, resid);
    }

    cur = layer_norm(cur, W("post_ln.w"), W("post_ln.b"));

    // spatial merge: tokens are 2x2-block contiguous, so this is just a reshape
    cur = ggml_cont(gctx, cur);
    cur = ggml_reshape_2d(gctx, cur, merge * merge * dim, n_tokens);    // [4*dim, 576]

    cur = ggml_add(gctx, ggml_mul_mat(gctx, W("mm0.w"), cur), W("mm0.b"));
    cur = ggml_gelu(gctx, cur);
    cur = ggml_add(gctx, ggml_mul_mat(gctx, W("mm2.w"), cur), W("mm2.b")); // [proj_dim, 576]

    ggml_set_output(cur); ggml_set_name(cur, "vis_out");
    ggml_build_forward_expand(gf, cur);

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf))
        throw std::runtime_error("mmproj: graph alloc failed");
}

std::vector<float> VisionEncoder::Impl::forward(const std::vector<float> & patches) {
    ggml_tensor * inp = ggml_graph_get_tensor(gf, "vis_inp");
    ggml_backend_tensor_set(inp, patches.data(), 0, patches.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("mmproj: compute failed");
    ggml_tensor * outt = ggml_graph_get_tensor(gf, "vis_out");
    std::vector<float> out((size_t) n_tokens * proj_dim);
    ggml_backend_tensor_get(outt, out.data(), 0, out.size() * sizeof(float));
    return out;
}

// ---- public API -----------------------------------------------------------

VisionEncoder::VisionEncoder() : impl_(new Impl()) {}
VisionEncoder::~VisionEncoder() = default;
int VisionEncoder::n_image_tokens() const { return impl_->n_tokens; }
int VisionEncoder::n_embd() const         { return impl_->proj_dim; }

std::unique_ptr<VisionEncoder> VisionEncoder::load(const std::string & path,
                                                   ggml_backend_t backend) {
    auto enc = std::unique_ptr<VisionEncoder>(new VisionEncoder());
    enc->impl_->backend = backend;
    enc->impl_->load_weights(path);
    enc->impl_->build_graph();
    return enc;
}

std::vector<float> VisionEncoder::encode_rgb(const uint8_t * rgb, int w_px, int h_px) {
    Impl & im = *impl_;
    const int IS = im.img_size, P = im.patch, N = im.n_side;
    const int K = P * P * 3;

    // resize to the trained resolution (plain stretch, like the reference impl)
    std::vector<uint8_t> resized((size_t) IS * IS * 3);
    if (w_px == IS && h_px == IS) {
        memcpy(resized.data(), rgb, resized.size());
    } else {
        if (!stbir_resize_uint8_srgb(rgb, w_px, h_px, 0,
                                     resized.data(), IS, IS, 0, STBIR_RGB))
            throw std::runtime_error("vision: image resize failed");
    }

    // normalize to (x/255 - 0.5) / 0.5 and im2col into 2x2-block-contiguous
    // patch order, (ic, ky, kx) within each column
    std::vector<float> patches((size_t) K * im.n_patch);
    for (int py = 0; py < N; ++py)
        for (int px = 0; px < N; ++px) {
            float * col = &patches[(size_t) im.reorder_index(py, px) * K];
            for (int ic = 0; ic < 3; ++ic)
                for (int ky = 0; ky < P; ++ky)
                    for (int kx = 0; kx < P; ++kx) {
                        const int y = py * P + ky, x = px * P + kx;
                        const uint8_t u = resized[((size_t) y * IS + x) * 3 + ic];
                        col[ic * P * P + ky * P + kx] = ((float) u / 255.0f - 0.5f) / 0.5f;
                    }
        }
    return im.forward(patches);
}

std::vector<float> VisionEncoder::encode_image(const std::string & image_path) {
    int w_px = 0, h_px = 0, comp = 0;
    uint8_t * rgb = stbi_load(image_path.c_str(), &w_px, &h_px, &comp, 3);
    if (!rgb) throw std::runtime_error("vision: failed to load image: " + image_path);
    std::vector<float> out;
    try {
        out = encode_rgb(rgb, w_px, h_px);
    } catch (...) {
        stbi_image_free(rgb);
        throw;
    }
    stbi_image_free(rgb);
    return out;
}

} // namespace qwencpp
