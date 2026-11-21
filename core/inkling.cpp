// Inkling 2 engine. Graph follows llama.cpp's src/models/inkling2.cpp
// (PR ggml-org/llama.cpp#32695 for the looped variant) op for op, so the two
// can be diffed tensor by tensor.

#include "inkling.h"
#include "model.h"
#include "expert_cache.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace questwend {

namespace {

constexpr int GRAPH_NODES = 65536;
constexpr int UBATCH      = 512;    // prefill chunk; the SWA ring is sized for it

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

} // namespace

struct InklingEngine::Impl {
    Model &             model;
    RuntimeConfig       cfg;
    const HParams &     hp;
    const InklingParams & ik;

    ggml_backend_t gpu = nullptr;
    ggml_backend_t cpu = nullptr;
    std::vector<ggml_backend_buffer_t> wbufs_gpu, wbufs_cpu;
    bool offload = false;

    // ---- per-visit state: KV (F16 rows of n_embd_k_gqa) and conv history ----
    ggml_context *        st_ctx = nullptr;
    ggml_backend_buffer_t st_buf = nullptr;
    std::vector<ggml_tensor *> k_l, v_l, conv_l;
    int n_ctx = 0, ring = 0;   // full-attention capacity, SWA ring capacity
    int n_embd_kv = 0, n_head = 0, n_head_kv = 0, n_hd = 0, n_ch = 0;

    int n_past_ = 0;
    // Flash attention with the relative bias folded into a per-head mask (the
    // default); INK_NO_FLASH=1 falls back to the manual path.
    bool flash = getenv("INK_NO_FLASH") == nullptr;
    std::vector<int32_t> kv_toks;
    std::vector<float>   logits;
    std::function<void(int, int)> progress_cb;

    ggml_gallocr_t galloc = nullptr;

    // ---- RAM offload ----
    // Routed experts live in pinned host memory; a VRAM slot cache (keyed by
    // PHYSICAL layer, so every visit of a looped layer shares it) holds a
    // subset. Decode runs per visit: graph A (attention .. router) on the GPU,
    // the selection is read back, cached experts run on the GPU (graph B) and
    // the rest on the CPU from the host copy, and the next visit's graph A
    // starts by finishing this one (post-conv + residual).
    std::unique_ptr<ExpertCache> ec;
    bool cpu_miss = true;       // false: fetch every miss into VRAM before computing
    // Static placement (INK_PLACE=core|lru|front): the pool is filled once, by
    // priority, and never churns -- decode and prefill both compute what is not
    // resident on the CPU. "core" puts the looped layers first: each of their
    // bytes serves n_pass visits per token.
    bool place_static = false;
    std::string place = "lru";
    // Static placement's prefill streams a layer that is not fully resident into
    // this one-layer scratch and runs it on the GPU (as llama.cpp's op offload
    // does): a 512-token chunk reuses every expert dozens of times, so moving the
    // weights beats computing on the CPU, and the pinned pool is never disturbed.
    ggml_context *        scr_ctx = nullptr;
    ggml_backend_buffer_t scr_buf = nullptr;
    std::vector<ggml_tensor *> scr[3];   // per physical layer: gate / up / down views of scr_buf
    // Static placement never moves an expert, so a fully resident layer needs no
    // host routing: its router picks are turned into slot ids inside the graph
    // through this fixed map, and a run of such visits -- with "core" placement,
    // every visit of the looped core, which are contiguous -- decodes as one
    // graph. Only the other visits come back to the host (INK_FUSE=0: none fused).
    std::vector<uint8_t> full_res;       // per physical layer
    ggml_context *        map_ctx = nullptr;
    ggml_backend_buffer_t map_buf = nullptr;
    ggml_tensor *         p_slotmap = nullptr;   // i32 [n_exp, 3 * n_phys]: (role, layer) rows
    bool fuse = false;
    int  refill   = 4;          // most-missed experts made resident after each token
    ggml_context *        car_ctx = nullptr;
    ggml_backend_buffer_t car_buf = nullptr;
    std::vector<ggml_tensor *> slots;   // [E, UBATCH] f32 carries
    ggml_tensor * p_tok = nullptr;      // [UBATCH] i32
    ggml_tensor * p_tau = nullptr;      // [UBATCH] f32
    ggml_tensor * p_rel_full = nullptr, * p_mask_full = nullptr;   // n_ctx * UBATCH
    ggml_tensor * p_rel_swa  = nullptr, * p_mask_swa  = nullptr;   // ring  * UBATCH
    ggml_tensor * p_idx_full = nullptr, * p_idx_swa = nullptr;     // [UBATCH] i64
    ggml_tensor * p_vidx_full = nullptr, * p_vidx_swa = nullptr;   // [n_embd_kv * UBATCH] i64
    ggml_tensor * p_sink0 = nullptr;
    ggml_tensor * p_sel = nullptr, * p_w = nullptr;                 // router out [n_used, UBATCH]
    ggml_tensor * p_gpu = nullptr, * p_cpu = nullptr;               // expert sums [E, UBATCH]
    ggml_tensor * p_bpack = nullptr, * p_bw = nullptr;              // graph B ids (packed) / weights
    bool p_cpu_zero = false;
    // Decode does one blocking round trip per visit: graph A's selection,
    // weights and expert input come back in one read (p_rout), and everything
    // the host sends (graph B ids / weights, the CPU sum, zeros) goes up as
    // async copies from pinned memory, ordered on the GPU stream.
    ggml_tensor * p_rout = nullptr;                                 // [2*n_used + E] f32 (decode)
    ggml_backend_buffer_t pin_buf = nullptr;
    float   * pin_rout = nullptr, * pin_w = nullptr, * pin_cpu = nullptr, * pin_zero = nullptr;
    int32_t * pin_ids = nullptr;
    // host copies of the per-expert scales (the CPU expert graphs read them)
    ggml_context * hs_ctx = nullptr; ggml_backend_buffer_t hs_buf = nullptr;
    std::vector<ggml_tensor *> hs_gu, hs_dn;   // per physical layer, CPU tensors
    // one segment graph
    struct Seg {
        ggml_context * ctx = nullptr; ggml_cgraph * gf = nullptr; ggml_gallocr_t ga = nullptr;
        ggml_tensor * out = nullptr;   // logits (head graph)
        ggml_tensor * sel = nullptr, * w = nullptr, * x = nullptr;   // graph A outputs (slot views)
        ggml_tensor * xin = nullptr, * ids = nullptr, * win = nullptr;  // CPU expert graph inputs
        int key = -1;
    };
    void free_seg(Seg & sg) {
        if (sg.ga) ggml_gallocr_free(sg.ga);
        if (sg.ctx) ggml_free(sg.ctx);
        sg = Seg{};
    }
    std::vector<Seg> segA;      // per visit (+1: head), kept for the current KV bucket
    std::vector<Seg> segB;      // per (visit, n resident) decode experts
    std::vector<Seg> segC;      // per (physical layer, n missed) CPU experts
    int seg_bucket = 0;
    // per-visit St snapshot after graph A, captured on the first build (deterministic)
    struct Stats {
        uint64_t sel = 0, gpu = 0, cpu = 0, fetched = 0;
        uint64_t core_sel = 0, core_repeat = 0;   // core-layer selections; already chosen by an earlier pass this token
        uint64_t tokens = 0;
        uint64_t streamed_layers = 0, streamed_bytes = 0;   // static prefill
        double t_a = 0, t_b = 0, t_c = 0, t_fetch = 0, t_head = 0;
    } stx;
    std::vector<uint64_t> l_sel, l_gpu;   // per physical layer
    void init_offload(size_t budget, size_t st_bytes);
    void forward_offload(const int32_t * toks, int T, bool want_logits);
    void upload_inputs(int T, int n_kv_full);
    void dump_offload_stats();

    Impl(Model & m, const RuntimeConfig & c) : model(m), cfg(c), hp(m.hparams()), ik(m.hparams().ink) {}
    ~Impl();

    void init();
    ggml_tensor * W(const char * fmt, int il) const {
        char nm[160]; snprintf(nm, sizeof(nm), fmt, il);
        ggml_tensor * t = model.tensor(nm);
        if (!t) throw std::runtime_error(std::string("inkling2: missing tensor ") + nm);
        return t;
    }
    ggml_tensor * Wopt(const char * fmt, int il) const {
        char nm[160]; snprintf(nm, sizeof(nm), fmt, il);
        return model.tensor(nm);
    }
    ggml_tensor * G(const char * nm) const {
        ggml_tensor * t = model.tensor(nm);
        if (!t) throw std::runtime_error(std::string("inkling2: missing tensor ") + nm);
        return t;
    }

    // ---- graph pieces ----
    struct Gr {
        ggml_context * ctx = nullptr;
        ggml_cgraph *  gf  = nullptr;
        int T = 0, n_kv_full = 0, n_kv_swa = 0;
        ggml_tensor * tau = nullptr;                    // [1,1,T]
        ggml_tensor * rel_full = nullptr, * rel_swa = nullptr;    // [n_kv, T] i32
        ggml_tensor * mask_full = nullptr, * mask_swa = nullptr;  // [n_kv, T] f32
        ggml_tensor * idx_full = nullptr, * idx_swa = nullptr;    // [T] i64 write rows
        ggml_tensor * vidx_full = nullptr, * vidx_swa = nullptr;  // [n_embd_kv * T] i64 transposed-V elements
        ggml_tensor * sink0 = nullptr;                  // zero sinks [n_head]
    };
    // depth-attention bookkeeping (the PR's blocks / prelude / bank lists)
    struct St {
        std::vector<ggml_tensor *> blocks, prelude_blocks, bank;
        ggml_tensor * prelude_cur = nullptr, * bank_last = nullptr, * cur = nullptr;
    };

    ggml_tensor * clip(Gr & g, ggml_tensor * cur, ggml_tensor * scale, float thr);
    ggml_tensor * mix(Gr & g, const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query);
    ggml_tensor * control(Gr & g, ggml_tensor * cur, ggml_tensor * msg, int ip, int module);
    ggml_tensor * convolve(Gr & g, int iv, ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * dyn, int64_t offset);
    // Visit iv up to the MoE input. Leaves st.cur unchanged; returns ffn_inp
    // (the residual) and *x (normed expert input).
    ggml_tensor * visit_pre(Gr & g, St & st, int iv, ggml_tensor ** x);
    void route(Gr & g, int ip, ggml_tensor * x, ggml_tensor ** sel, ggml_tensor ** w);
    ggml_tensor * situ(Gr & g, ggml_tensor * gate, ggml_tensor * up);
    ggml_tensor * experts(Gr & g, int ip, ggml_tensor * x, ggml_tensor * sel, ggml_tensor * w,
                          ggml_tensor * tg, ggml_tensor * tu, ggml_tensor * td,
                          ggml_tensor * ig = nullptr, ggml_tensor * iu = nullptr, ggml_tensor * id = nullptr,
                          ggml_tensor * sc_gu = nullptr, ggml_tensor * sc_dn = nullptr);
    ggml_tensor * visit_post(Gr & g, int iv, ggml_tensor * ffn_inp, ggml_tensor * moe_out);
    ggml_tensor * head(Gr & g, St & st, bool last_only);

    Gr   new_graph(int T, int n_kv_full);
    Gr   seg_graph(int T, int n_kv_full, int nodes = 8192);   // offload: inputs are views of the persistent p_* tensors
    void set_inputs(Gr & g, int T);
    // the whole stack for T tokens on the GPU (resident weights)
    void build_full(Gr & g, ggml_tensor * inp, bool want_logits, ggml_tensor ** out);
    void forward(const int32_t * toks, int T, bool want_logits);

    // Decode graph kept across tokens: sized for a KV bucket (rows past
    // n_past are masked), so its topology and shapes stay fixed and ggml-cuda
    // can replay it as a CUDA graph instead of launching ~11k kernels.
    struct Kept {
        Gr g; ggml_tensor * inp = nullptr, * out = nullptr;
        ggml_gallocr_t ga = nullptr; int bucket = 0;
    } kept;
    void free_kept() {
        if (kept.ga) ggml_gallocr_free(kept.ga);
        if (kept.g.ctx) ggml_free(kept.g.ctx);
        kept = Kept{};
    }

    // element indices of T new V columns in the transposed caches (channel-major)
    void fill_vidx(int T, std::vector<int64_t> & vf, std::vector<int64_t> & vs) const {
        vf.resize((size_t) n_embd_kv * T);
        vs.resize((size_t) n_embd_kv * T);
        for (int t = 0; t < T; ++t) {
            const int64_t cf = n_past_ + t, cs = (n_past_ + t) % ring;
            for (int c = 0; c < n_embd_kv; ++c) {
                vf[(size_t) t * n_embd_kv + c] = (int64_t) c * n_ctx + cf;
                vs[(size_t) t * n_embd_kv + c] = (int64_t) c * ring  + cs;
            }
        }
    }
    int  pos_of_ring(int cell, int n_total) const {
        // newest position p < n_total with p % ring == cell
        if (cell >= n_total) return -1;
        return cell + ((n_total - 1 - cell) / ring) * ring;
    }
};

InklingEngine::Impl::~Impl() {
    free_kept();
    if (galloc) ggml_gallocr_free(galloc);
    if (st_buf) ggml_backend_buffer_free(st_buf);
    if (st_ctx) ggml_free(st_ctx);
    for (auto b : wbufs_gpu) ggml_backend_buffer_free(b);
    for (auto b : wbufs_cpu) ggml_backend_buffer_free(b);
    for (auto & sg : segA) free_seg(sg);
    for (auto & sg : segB) free_seg(sg);
    for (auto & sg : segC) free_seg(sg);
    ec.reset();
    if (car_buf) ggml_backend_buffer_free(car_buf);
    if (pin_buf) ggml_backend_buffer_free(pin_buf);
    if (scr_buf) ggml_backend_buffer_free(scr_buf);
    if (map_buf) ggml_backend_buffer_free(map_buf);
    if (map_ctx) ggml_free(map_ctx);
    if (scr_ctx) ggml_free(scr_ctx);
    if (car_ctx) ggml_free(car_ctx);
    if (hs_buf) ggml_backend_buffer_free(hs_buf);
    if (hs_ctx) ggml_free(hs_ctx);
    if (cpu && cpu != gpu) ggml_backend_free(cpu);
    if (gpu) ggml_backend_free(gpu);
}

void InklingEngine::Impl::init() {
    n_head    = (int) hp.head_count(0);
    n_head_kv = (int) hp.head_count_kv(0);
    n_hd      = (int) hp.n_embd_head;
    n_embd_kv = n_hd * n_head_kv;
    n_ch      = 2 * n_embd_kv + 2 * (int) hp.n_embd;

    // ---- backends ----
    std::vector<ggml_backend_dev_t> devs = cfg.use_cuda ? gpu_devices() : std::vector<ggml_backend_dev_t>{};
    size_t budget = 0;
    if (!devs.empty()) {
        const int di = cfg.gpus.empty() ? 0 : cfg.gpus[0].device;
        if (di < 0 || (size_t) di >= devs.size()) throw std::runtime_error("--gpus: no such device");
        gpu = ggml_backend_dev_init(devs[(size_t) di], nullptr);
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(devs[(size_t) di], &fr, &tot);
        size_t mb = cfg.vram_budget_mb;
        if (!cfg.gpus.empty() && cfg.gpus[0].budget_mb != VRAM_BUDGET_AUTO) mb = cfg.gpus[0].budget_mb;
        budget = mb > 0 ? mb * 1024ull * 1024ull : fr;
        fprintf(stderr, "inkling2: GPU %s (%zu MB free, budget %zu MB)\n",
                ggml_backend_dev_description(devs[(size_t) di]), fr >> 20, budget >> 20);
    }
    cpu = ggml_backend_cpu_init();
    int nth = cfg.n_threads > 0 ? cfg.n_threads : (int) std::thread::hardware_concurrency() / 2;
    if (nth <= 0) nth = 4;
    if (const char * e = getenv("INK_CPU_THREADS")) nth = std::max(1, atoi(e));
    ggml_backend_cpu_set_n_threads(cpu, nth);
    if (!gpu) { gpu = cpu; fprintf(stderr, "inkling2: CPU only (%d threads)\n", nth); }

    // ---- state sizes ----
    n_ctx = cfg.n_ctx > 0 ? cfg.n_ctx : 4096;
    ring  = std::min<int>(n_ctx, (int) ik.n_swa + UBATCH);
    size_t st_bytes = 0;
    for (uint32_t iv = 0; iv < ik.n_visit; ++iv)
        st_bytes += 2ull * n_embd_kv * (ik.is_swa[iv] ? ring : n_ctx) * 2 + (size_t) (ik.n_conv - 1) * n_ch * 4;

    model.set_embd_q8(cfg.embd_q8);
    const size_t wbytes = model.weight_bytes(gpu);
    const size_t reserve = 1200ull << 20;   // compute buffers (prefill attention bias dominates)
    offload = gpu != cpu && wbytes + st_bytes + reserve > budget;
    fprintf(stderr, "inkling2: weights %zu MB, state %zu MB (KV: %u full-attn visits x %d + %u SWA x %d ring)%s\n",
            wbytes >> 20, st_bytes >> 20,
            (unsigned) std::count(ik.is_swa.begin(), ik.is_swa.end(), 0), n_ctx,
            (unsigned) std::count(ik.is_swa.begin(), ik.is_swa.end(), 1), ring,
            offload ? " -> experts offloaded to RAM" : "");
    if (getenv("INK_FORCE_OFFLOAD")) offload = gpu != cpu;

    if (offload) {
        ggml_backend_dev_t gdev = ggml_backend_get_device(gpu);
        ggml_backend_buffer_type_t host_bt = ggml_backend_dev_host_buffer_type(gdev);
        if (!host_bt) host_bt = ggml_backend_cpu_buffer_type();
        model.load_weights_split(gpu, host_bt, wbufs_gpu, wbufs_cpu);
    } else {
        model.load_weights(gpu);   // Model owns the buffer
    }

    // ---- state buffers ----
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * (ik.n_visit * 3 + 4);
    p.no_alloc = true;
    st_ctx = ggml_init(p);
    for (uint32_t iv = 0; iv < ik.n_visit; ++iv) {
        const int cap = ik.is_swa[iv] ? ring : n_ctx;
        k_l.push_back(ggml_new_tensor_2d(st_ctx, GGML_TYPE_F16, n_embd_kv, cap));
        v_l.push_back(flash ? ggml_new_tensor_2d(st_ctx, GGML_TYPE_F16, n_embd_kv, cap)
                            : ggml_new_tensor_2d(st_ctx, GGML_TYPE_F16, cap, n_embd_kv));   // manual path: transposed
        conv_l.push_back(ggml_new_tensor_2d(st_ctx, GGML_TYPE_F32, ik.n_conv - 1, n_ch));
    }
    st_buf = ggml_backend_alloc_ctx_tensors(st_ctx, gpu);
    if (!st_buf) throw std::runtime_error("inkling2: cannot allocate KV / conv state");
    ggml_backend_buffer_clear(st_buf, 0);

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    if (offload) init_offload(budget, st_bytes);
}

// ---------------------------------------------------------------------------
// graph pieces
// ---------------------------------------------------------------------------

// scale each row down to an L2 norm of at most thr, after the optional per-channel scale
ggml_tensor * InklingEngine::Impl::clip(Gr & g, ggml_tensor * cur, ggml_tensor * scale, float thr) {
    ggml_context * ctx = g.ctx;
    if (scale) cur = ggml_mul(ctx, cur, scale);
    ggml_tensor * norm = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, cur)));
    norm = ggml_scale_bias(ctx, norm, 1.0f, 1e-6f);
    ggml_tensor * factor = ggml_div(ctx, ggml_fill(ctx, norm, thr), norm);
    return ggml_mul(ctx, cur, ggml_clamp(ctx, factor, 0.0f, 1.0f));
}

// attention over depth: mix the earlier block outputs and cur with learned queries
ggml_tensor * InklingEngine::Impl::mix(Gr & g, const std::vector<ggml_tensor *> & blocks,
                                       ggml_tensor * cur, ggml_tensor * query) {
    ggml_context * ctx = g.ctx;
    const int64_t E = hp.n_embd, T = cur->ne[1];
    ggml_tensor * states = ggml_reshape_3d(ctx, cur, E, 1, T);
    for (ggml_tensor * b : blocks)
        states = ggml_concat(ctx, ggml_reshape_3d(ctx, b, E, 1, T), states, 1);
    states = ggml_rms_norm(ctx, states, 1e-6f);
    const int64_t nq = query->ne[1];
    ggml_tensor * queries = ggml_repeat(ctx, query, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, E, nq, T));
    ggml_tensor * scores = ggml_mul_mat(ctx, states, queries);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / sqrtf((float) E), 0.0f);
    return ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, states)), scores);   // [E, nq, T]
}

// loop message controller: (1 - tanh(F h))*cur + tanh(G h)*msg, h = D [rms(cur); rms(msg)]
ggml_tensor * InklingEngine::Impl::control(Gr & g, ggml_tensor * cur, ggml_tensor * msg, int ip, int module) {
    static const char * names[4] = { "attn", "k", "v", "mlp" };
    ggml_context * ctx = g.ctx;
    const int64_t E = hp.n_embd, T = cur->ne[1];
    char fd[96], fg[96], ff[96];
    snprintf(fd, sizeof(fd), "blk.%%d.loop_%s_down.weight",   names[module]);
    snprintf(fg, sizeof(fg), "blk.%%d.loop_%s_gain.weight",   names[module]);
    snprintf(ff, sizeof(ff), "blk.%%d.loop_%s_forget.weight", names[module]);
    ggml_tensor * h = ggml_concat(ctx, ggml_reshape_3d(ctx, cur, E, 1, T), ggml_reshape_3d(ctx, msg, E, 1, T), 1);
    h = ggml_reshape_2d(ctx, ggml_rms_norm(ctx, h, 1e-6f), 2 * E, T);
    h = ggml_mul_mat(ctx, W(fd, ip), h);
    ggml_tensor * gain   = ggml_tanh(ctx, ggml_mul_mat(ctx, W(fg, ip), h));
    ggml_tensor * forget = ggml_tanh(ctx, ggml_mul_mat(ctx, W(ff, ip), h));
    return ggml_add(ctx, ggml_mul(ctx, ggml_scale_bias(ctx, forget, -1.0f, 1.0f), cur), ggml_mul(ctx, gain, msg));
}

// causal short convolution with a residual; `dyn` adds input-dependent taps per head.
// The last n_conv-1 inputs live in the visit's conv state at channel `offset`.
ggml_tensor * InklingEngine::Impl::convolve(Gr & g, int iv, ggml_tensor * cur, ggml_tensor * weight,
                                            ggml_tensor * dyn, int64_t offset) {
    ggml_context * ctx = g.ctx;
    const int64_t nc = ik.n_conv, C = cur->ne[0], T = cur->ne[1];
    ggml_tensor * state = conv_l[(size_t) iv];
    ggml_tensor * past = ggml_view_2d(ctx, state, nc - 1, C, state->nb[1], (size_t) offset * state->nb[1]);
    ggml_tensor * x = ggml_concat(ctx, past, ggml_transpose(ctx, cur), 0);          // [nc-1+T, C]
    ggml_tensor * last = ggml_view_2d(ctx, x, nc - 1, C, x->nb[1], (size_t) T * x->nb[0]);
    ggml_build_forward_expand(g.gf, ggml_cpy(ctx, last, past));

    ggml_tensor * out = ggml_ssm_conv(ctx, ggml_reshape_3d(ctx, x, nc - 1 + T, C, 1), weight);
    out = ggml_reshape_2d(ctx, out, C, T);
    if (dyn) {
        const int64_t nh = dyn->ne[0] / nc;
        ggml_tensor * d3 = ggml_reshape_3d(ctx, dyn, nc, nh, T);
        for (int64_t lag = 0; lag < nc; ++lag) {
            ggml_tensor * tap = ggml_view_3d(ctx, d3, 1, nh, T, d3->nb[1], d3->nb[2], (size_t) lag * d3->nb[0]);
            ggml_tensor * win = ggml_view_2d(ctx, x, T, C, x->nb[1], (size_t) (nc - 1 - lag) * x->nb[0]);
            win = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, win)), C / nh, nh, T);
            out = ggml_add(ctx, out, ggml_reshape_2d(ctx, ggml_mul(ctx, win, tap), C, T));
        }
    }
    return ggml_scale(ctx, ggml_add(ctx, cur, out), 1.0f / sqrtf((float) ((dyn ? 2 : 1) * nc + 1)));
}

ggml_tensor * InklingEngine::Impl::visit_pre(Gr & g, St & st, int iv, ggml_tensor ** x_out) {
    ggml_context * ctx = g.ctx;
    const int64_t E = hp.n_embd, T = g.T;
    const int  ip  = (int) ik.phys((uint32_t) iv);
    const bool swa = ik.is_swa[(size_t) iv] != 0;
    const float eps = hp.rms_eps;

    // ---- loop bookkeeping: later passes restart from the prelude state ----
    ggml_tensor * messages[4] = { nullptr, nullptr, nullptr, nullptr };
    if (ik.in_core((uint32_t) ip)) {
        const int pass = (int) ik.pass((uint32_t) iv);
        if (ip == (int) ik.core_start) {
            if (pass == 0) {
                st.prelude_blocks = st.blocks;
                st.prelude_cur    = st.cur;
            } else {
                st.bank.assign(st.blocks.begin() + st.prelude_blocks.size() + 1, st.blocks.end());
                st.bank_last = st.cur;
                st.blocks    = st.prelude_blocks;
                st.cur       = st.prelude_cur;
            }
        }
        if (pass > 0) {
            const size_t row = 2 * (size_t) (ip - (int) ik.core_start);
            ggml_tensor * lrq = G("loop_res_query.weight"), * lkq = G("loop_kv_res_query.weight");
            ggml_tensor * q_attn = ggml_view_2d(ctx, lrq, E, 2, lrq->nb[1], row * lrq->nb[1]);
            ggml_tensor * q_kv   = ggml_view_2d(ctx, lkq, E, 2, lkq->nb[1], row * lkq->nb[1]);
            ggml_tensor * m_attn = mix(g, st.bank, st.bank_last, q_attn);
            ggml_tensor * m_kv   = mix(g, st.bank, st.bank_last, q_kv);
            for (int j = 0; j < 2; ++j) {
                messages[j == 0 ? 0 : 3] = ggml_cont(ctx, ggml_view_2d(ctx, m_attn, E, T, m_attn->nb[2], j * m_attn->nb[1]));
                messages[j == 0 ? 1 : 2] = ggml_cont(ctx, ggml_view_2d(ctx, m_kv,   E, T, m_kv->nb[2],   j * m_kv->nb[1]));
            }
        }
    }

    ggml_tensor * inpSA = st.cur;
    ggml_tensor * inputs[3] = { st.cur, st.cur, st.cur };
    if (ggml_tensor * rq = Wopt("blk.%d.attn_res_query.weight", ip)) {
        ggml_tensor * mixed = mix(g, st.blocks, st.cur, rq);
        for (int j = 0; j < 3; ++j)
            inputs[j] = ggml_cont(ctx, ggml_view_2d(ctx, mixed, E, T, mixed->nb[2], (size_t) (swa ? 0 : j) * mixed->nb[1]));
    }
    if (messages[0])
        for (int j = 0; j < 3; ++j) inputs[j] = control(g, inputs[j], messages[j], ip, j);

    auto norm = [&](ggml_tensor * t, ggml_tensor * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, t, eps), w); };
    if (swa && messages[0]) {
        for (int j = 0; j < 3; ++j) inputs[j] = norm(inputs[j], W("blk.%d.attn_norm.weight", ip));
    } else if (swa) {
        inputs[0] = norm(inputs[0], W("blk.%d.attn_norm.weight", ip));
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = norm(inputs[0], W("blk.%d.attn_q_inp_norm.weight", ip));
        inputs[1] = norm(inputs[1], W("blk.%d.attn_k_inp_norm.weight", ip));
        inputs[2] = norm(inputs[2], W("blk.%d.attn_v_inp_norm.weight", ip));
    }

    // ---- self-attention ----
    ggml_tensor * cur;
    {
        ggml_tensor * Q = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", ip), inputs[0]);
        ggml_tensor * K = convolve(g, iv, ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", ip), inputs[1]),
                                   W("blk.%d.attn_k_conv.weight", ip),
                                   swa ? nullptr : ggml_mul_mat(ctx, W("blk.%d.attn_k_conv_proj.weight", ip), inputs[1]), 0);
        ggml_tensor * V = convolve(g, iv, ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", ip), inputs[2]),
                                   W("blk.%d.attn_v_conv.weight", ip),
                                   swa ? nullptr : ggml_mul_mat(ctx, W("blk.%d.attn_v_conv_proj.weight", ip), inputs[2]),
                                   n_embd_kv);
        ggml_tensor * R = ggml_mul_mat(ctx, W("blk.%d.attn_rel.weight", ip), inputs[0]);

        Q = clip(g, ggml_reshape_3d(ctx, Q, n_hd, n_head,    T), W("blk.%d.attn_q_clip.weight", ip), ik.qk_clip);
        K = clip(g, ggml_reshape_3d(ctx, K, n_hd, n_head_kv, T), nullptr, ik.qk_clip);
        V = ggml_rms_norm(ctx, ggml_reshape_3d(ctx, V, n_hd, n_head_kv, T), eps);
        R = clip(g, ggml_reshape_3d(ctx, R, ik.n_rel, n_head, T), W("blk.%d.attn_rel_clip.weight", ip), ik.rel_clip);
        if (!swa) {
            Q = ggml_mul(ctx, Q, g.tau);
            R = ggml_mul(ctx, R, g.tau);
        }

        // write K / V rows into the visit's cache, then attend over the cached cells
        const int n_kv = swa ? g.n_kv_swa : g.n_kv_full;
        ggml_tensor * idx = swa ? g.idx_swa : g.idx_full;
        ggml_tensor * kc = ggml_set_rows(ctx, k_l[(size_t) iv], ggml_reshape_2d(ctx, K, n_embd_kv, T), idx);
        ggml_tensor * k3 = ggml_view_3d(ctx, kc, n_hd, n_head_kv, n_kv, (size_t) n_hd * ggml_element_size(kc), kc->nb[1], 0);
        ggml_tensor * vl = v_l[(size_t) iv];
        ggml_tensor * v_att = nullptr;   // flash: [hd, n_kv, n_head_kv]; manual: [n_kv, hd, n_head_kv]
        if (flash) {
            ggml_tensor * vc = ggml_set_rows(ctx, vl, ggml_reshape_2d(ctx, V, n_embd_kv, T), idx);
            v_att = ggml_permute(ctx, ggml_view_3d(ctx, vc, n_hd, n_head_kv, n_kv, (size_t) n_hd * 2, vc->nb[1], 0), 0, 2, 1, 3);
        } else {
            // V cached transposed ([cap, n_embd_kv]) so the manual path reads it
            // without a copy: one element per index, like llama.cpp's v_trans cache
            const int64_t cap = vl->ne[0];
            ggml_tensor * vc = ggml_set_rows(ctx, ggml_reshape_2d(ctx, vl, 1, ggml_nelements(vl)),
                                             ggml_reshape_2d(ctx, ggml_cont(ctx, V), 1, (int64_t) n_embd_kv * T),
                                             swa ? g.vidx_swa : g.vidx_full);
            v_att = ggml_view_3d(ctx, vc, n_kv, n_hd, n_head_kv, (size_t) cap * 2, (size_t) cap * 2 * n_hd, 0);
        }

        // relative bias: project R on the bias table, subtract the per-query sink
        // logit, then gather by distance -> [n_kv, T, n_head]
        ggml_tensor * sinks = ggml_mul_mat(ctx, W("blk.%d.attn_sink_proj.weight", ip), inputs[0]);   // [n_head, T]
        ggml_tensor * table = ggml_mul_mat(ctx, ggml_pad(ctx, W("blk.%d.attn_rel_bias.weight", ip), 0, 1, 0, 0), R);
        table = ggml_sub(ctx, table, ggml_reshape_3d(ctx, sinks, 1, n_head, T));
        table = ggml_cont(ctx, ggml_permute(ctx, table, 1, 0, 2, 3));                // [n_head, ext+1, T]
        ggml_tensor * kq_b = ggml_get_rows(ctx, table, swa ? g.rel_swa : g.rel_full);  // [n_head, n_kv, T]
        kq_b = ggml_cont(ctx, ggml_permute(ctx, kq_b, 2, 0, 1, 3));                  // [n_kv, T, n_head]

        ggml_tensor * q  = ggml_permute(ctx, Q, 0, 2, 1, 3);                         // [hd, T, n_head]
        ggml_tensor * k  = ggml_permute(ctx, k3, 0, 2, 1, 3);                        // [hd, n_kv, n_head_kv]
        if (flash) {
            // the bias rides in a per-head F16 mask, as llama.cpp does for this model
            ggml_tensor * m  = ggml_cast(ctx, ggml_add(ctx, kq_b, swa ? g.mask_swa : g.mask_full), GGML_TYPE_F16);
            ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v_att, m, 1.0f / sqrtf((float) n_hd), 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(fa, g.sink0);
            cur = ggml_reshape_2d(ctx, fa, n_hd * n_head, T);                        // fa: [hd, n_head, T]
        } else {
            ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                              // [n_kv, T, n_head]
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float) n_hd));
            kq = ggml_add(ctx, ggml_add(ctx, kq, kq_b), swa ? g.mask_swa : g.mask_full);
            kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
            ggml_soft_max_add_sinks(kq, g.sink0);
            ggml_tensor * kqv = ggml_mul_mat(ctx, v_att, kq);                        // [hd, T, n_head]
            cur = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), n_hd * n_head, T);
        }

        cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, ggml_mul_mat(ctx, W("blk.%d.attn_gate.weight", ip), inputs[0])));
        cur = convolve(g, iv, ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", ip), cur),
                       W("blk.%d.attn_post_conv.weight", ip), nullptr, 2 * n_embd_kv);
    }

    // the first layer of each block starts a new residual stream; the old one joins the mixed blocks
    if (ip % (int) ik.block_size == 0) st.blocks.push_back(inpSA);
    else                               cur = ggml_add(ctx, inpSA, cur);
    ggml_tensor * ffn_inp = cur;

    ggml_tensor * x = ggml_reshape_2d(ctx, mix(g, st.blocks, cur, W("blk.%d.ffn_res_query.weight", ip)), E, T);
    if (messages[3]) x = control(g, x, messages[3], ip, 3);
    *x_out = norm(x, W("blk.%d.ffn_norm.weight", ip));
    return ffn_inp;
}

void InklingEngine::Impl::route(Gr & g, int ip, ggml_tensor * x, ggml_tensor ** sel, ggml_tensor ** w) {
    ggml_context * ctx = g.ctx;
    const int n_exp = (int) hp.n_expert, n_used = (int) hp.n_expert_used, T = (int) x->ne[1];
    ggml_tensor * logits = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp.weight", ip), x);   // [n_exp, T]
    ggml_prec_set_acc(logits, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sigmoid(ctx, logits);
    ggml_tensor * rank = probs;
    if (ggml_tensor * b = Wopt("blk.%d.exp_probs_b.bias", ip)) rank = ggml_add(ctx, probs, b);
    *sel = ggml_argsort_top_k(ctx, rank, n_used);                                        // [n_used, T]
    ggml_tensor * wt = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_exp, T), *sel); // [1, n_used, T]
    wt = ggml_reshape_2d(ctx, wt, n_used, T);
    ggml_tensor * ws = ggml_clamp(ctx, ggml_sum_rows(ctx, wt), 1e-20f, INFINITY);
    wt = ggml_div(ctx, wt, ws);
    *w = ggml_scale(ctx, ggml_reshape_3d(ctx, wt, 1, n_used, T), hp.expert_weights_scale);
}

// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
ggml_tensor * InklingEngine::Impl::situ(Gr & g, ggml_tensor * gate, ggml_tensor * up) {
    ggml_context * ctx = g.ctx;
    const float b = ik.situ_beta, lb = ik.situ_lin_beta;
    ggml_tensor * act = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, gate, 1.0f / b)), b);
    act = ggml_mul(ctx, act, ggml_sigmoid(ctx, gate));
    if (lb > 0.0f) up = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, up, 1.0f / lb)), lb);
    return ggml_mul(ctx, act, up);
}

// Routed experts over (tg, tu, td) indexed by `slots` (null: the model's own
// tensors, indexed by `sel`). Scales are per logical expert, so they always
// gather by `sel`. Returns the weighted sum [E, T].
ggml_tensor * InklingEngine::Impl::experts(Gr & g, int ip, ggml_tensor * x, ggml_tensor * sel, ggml_tensor * w,
                                           ggml_tensor * tg, ggml_tensor * tu, ggml_tensor * td,
                                           ggml_tensor * ig, ggml_tensor * iu, ggml_tensor * id,
                                           ggml_tensor * sc_gu, ggml_tensor * sc_dn) {
    ggml_context * ctx = g.ctx;
    const int64_t E = hp.n_embd, T = x->ne[1], n_exp = hp.n_expert;
    if (!ig) ig = sel;
    if (!iu) iu = sel;
    if (!id) id = sel;
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, E, 1, T);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, tg, x3, ig);   // [ff, n_used, T]
    ggml_tensor * up   = ggml_mul_mat_id(ctx, tu, x3, iu);
    static const bool no_scale = getenv("INK_NO_EXP_SCALE") != nullptr;
    auto scale_of = [&](ggml_tensor * s) {
        ggml_tensor * r = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, s, 1, n_exp, 1), 1, n_exp, T, 1);
        return ggml_get_rows(ctx, r, sel);   // [1, n_used, T]
    };
    if (!sc_gu) sc_gu = Wopt("blk.%d.ffn_gate_up_exps.scale", ip);
    if (!sc_dn) sc_dn = Wopt("blk.%d.ffn_down_exps.scale", ip);
    if (!no_scale) {
        if (ggml_tensor * s = sc_gu) {
            ggml_tensor * sr = scale_of(s);
            gate = ggml_mul(ctx, gate, sr);
            up   = ggml_mul(ctx, up,   sr);
        }
    }
    ggml_tensor * ex = ggml_mul_mat_id(ctx, td, situ(g, gate, up), id);    // [E, n_used, T]
    // as llama.cpp does for NVFP4 weights (build_lora_mm_id): BF16 accumulation is allowed
    static const bool nvfp4_bf16 = getenv("INK_NVFP4_F32") == nullptr;
    if (nvfp4_bf16)
        for (ggml_tensor * t : { gate, up, ex })
            if (t->src[0]->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(t, GGML_PREC_BF16);
    if (!no_scale)
        if (ggml_tensor * s = sc_dn) ex = ggml_mul(ctx, ex, scale_of(s));
    ex = ggml_mul(ctx, ex, w);
    ex = ggml_cont(ctx, ggml_permute(ctx, ex, 1, 0, 2, 3));                 // [n_used, E, T]
    return ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ex), E, T);
}

ggml_tensor * InklingEngine::Impl::visit_post(Gr & g, int iv, ggml_tensor * ffn_inp, ggml_tensor * moe_out) {
    const int ip = (int) ik.phys((uint32_t) iv);
    ggml_tensor * cur = convolve(g, iv, moe_out, W("blk.%d.ffn_post_conv.weight", ip), nullptr,
                                 2 * n_embd_kv + (int64_t) hp.n_embd);
    return ggml_add(g.ctx, ffn_inp, cur);
}

ggml_tensor * InklingEngine::Impl::head(Gr & g, St & st, bool last_only) {
    ggml_context * ctx = g.ctx;
    const int64_t E = hp.n_embd;
    std::vector<ggml_tensor *> blocks = st.blocks;
    ggml_tensor * cur = st.cur;
    int64_t T = cur->ne[1];
    if (last_only && T > 1) {
        auto last = [&](ggml_tensor * t) { return ggml_view_2d(ctx, t, E, 1, t->nb[1], (size_t) (T - 1) * t->nb[1]); };
        for (auto & b : blocks) b = ggml_cont(ctx, last(b));
        cur = ggml_cont(ctx, last(cur));
        T = 1;
    }
    cur = ggml_reshape_2d(ctx, mix(g, blocks, cur, G("output_res_query.weight")), E, T);
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.rms_eps), G("output_norm.weight"));
    cur = ggml_mul_mat(ctx, G("output.weight"), cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    if (ik.logit_cap != 0.0f)
        cur = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, cur, 1.0f / ik.logit_cap)), ik.logit_cap);
    return ggml_cont(ctx, ggml_view_2d(ctx, cur, ik.n_vocab, T, cur->nb[1], 0));
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

InklingEngine::Impl::Gr InklingEngine::Impl::new_graph(int T, int n_kv_full) {
    Gr g;
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false);
    p.no_alloc = true;
    g.ctx = ggml_init(p);
    g.gf  = ggml_new_graph_custom(g.ctx, GRAPH_NODES, false);
    g.T   = T;
    g.n_kv_full = n_kv_full;
    g.n_kv_swa  = std::min(ring, n_kv_full);
    auto in = [&](ggml_tensor * t, const char * nm) { ggml_set_input(t); ggml_set_name(t, nm); return t; };
    g.tau       = in(ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, 1, T), "ink_tau");
    g.rel_full  = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, g.n_kv_full, T), "ink_rel_full");
    g.rel_swa   = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, g.n_kv_swa, T), "ink_rel_swa");
    g.mask_full = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, g.n_kv_full, T), "ink_mask_full");
    g.mask_swa  = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, g.n_kv_swa, T), "ink_mask_swa");
    g.idx_full  = in(ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, T), "ink_idx_full");
    g.idx_swa   = in(ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, T), "ink_idx_swa");
    g.vidx_full = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_I64, (int64_t) n_embd_kv * T, 1), "ink_vidx_full");
    g.vidx_swa  = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_I64, (int64_t) n_embd_kv * T, 1), "ink_vidx_swa");
    g.sink0     = in(ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_head), "ink_sink0");
    return g;
}

void InklingEngine::Impl::set_inputs(Gr & g, int T) {
    const int n_total = n_past_ + T;
    std::vector<float> tau(T);
    for (int i = 0; i < T; ++i) {
        const double p = (double) (n_past_ + i) + 1.0;
        tau[(size_t) i] = (float) (1.0 + ik.log_alpha * std::log(std::max(1.0, p / ik.log_floor)));
    }
    ggml_backend_tensor_set(g.tau, tau.data(), 0, tau.size() * sizeof(float));

    const int ext = (int) ik.rel_extent;
    auto fill = [&](int n_kv, bool swa, ggml_tensor * rel, ggml_tensor * mask) {
        std::vector<int32_t> r((size_t) n_kv * T);
        std::vector<float>   m((size_t) n_kv * T);
        for (int i = 0; i < T; ++i) {
            const int pi = n_past_ + i;
            for (int j = 0; j < n_kv; ++j) {
                const int pj = swa ? pos_of_ring(j, n_total) : j;
                const int d  = pi - pj;
                const bool vis = pj >= 0 && d >= 0 && (!swa || d < (int) ik.n_swa);
                r[(size_t) i * n_kv + j] = (d >= 0 && d < ext) ? d : ext;
                m[(size_t) i * n_kv + j] = vis ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(rel,  r.data(), 0, r.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    };
    fill(g.n_kv_full, false, g.rel_full, g.mask_full);
    fill(g.n_kv_swa,  true,  g.rel_swa,  g.mask_swa);

    std::vector<int64_t> idf(T), ids(T);
    for (int i = 0; i < T; ++i) { idf[(size_t) i] = n_past_ + i; ids[(size_t) i] = (n_past_ + i) % ring; }
    ggml_backend_tensor_set(g.idx_full, idf.data(), 0, idf.size() * sizeof(int64_t));
    ggml_backend_tensor_set(g.idx_swa,  ids.data(), 0, ids.size() * sizeof(int64_t));
    if (!flash) {   // only the manual path's transposed V cache reads these
        std::vector<int64_t> vf, vs;
        fill_vidx(T, vf, vs);
        ggml_backend_tensor_set(g.vidx_full, vf.data(), 0, vf.size() * sizeof(int64_t));
        ggml_backend_tensor_set(g.vidx_swa,  vs.data(), 0, vs.size() * sizeof(int64_t));
    }
    std::vector<float> z((size_t) n_head, 0.0f);
    ggml_backend_tensor_set(g.sink0, z.data(), 0, z.size() * sizeof(float));
}

void InklingEngine::Impl::build_full(Gr & g, ggml_tensor * inp, bool want_logits, ggml_tensor ** out_p) {
    ggml_context * ctx = g.ctx;
    St st;
    st.cur = ggml_get_rows(ctx, model.tok_embd_rows(), inp);
    st.cur = ggml_mul(ctx, ggml_rms_norm(ctx, st.cur, hp.rms_eps), G("token_embd_norm.weight"));

    for (int iv = 0; iv < (int) ik.n_visit; ++iv) {
        const int ip = (int) ik.phys((uint32_t) iv);
        ggml_tensor * x = nullptr, * sel = nullptr, * w = nullptr;
        ggml_tensor * ffn_inp = visit_pre(g, st, iv, &x);
        route(g, ip, x, &sel, &w);
        ggml_build_forward_expand(g.gf, w);
        ggml_tensor * moe = experts(g, ip, x, sel, w, W("blk.%d.ffn_gate_exps.weight", ip),
                                    W("blk.%d.ffn_up_exps.weight", ip), W("blk.%d.ffn_down_exps.weight", ip), nullptr);
        st.cur = visit_post(g, iv, ffn_inp, moe);
    }
    ggml_tensor * out = nullptr;
    if (want_logits) {
        out = head(g, st, true);
        ggml_set_output(out);
        ggml_build_forward_expand(g.gf, out);
    } else {
        ggml_build_forward_expand(g.gf, st.cur);
    }
    *out_p = out;
}

void InklingEngine::Impl::forward(const int32_t * toks, int T, bool want_logits) {
    if (n_past_ + T > n_ctx) throw std::runtime_error("inkling2: context full");
    if (offload) { forward_offload(toks, T, want_logits); return; }
    static const bool prof = getenv("INK_PROF") != nullptr;
    static const bool no_keep = getenv("INK_NO_KEEP") != nullptr;
    auto t0 = clk::now();
    const bool use_kept = T == 1 && want_logits && !no_keep;
    Gr g; ggml_tensor * inp = nullptr, * out = nullptr;
    if (use_kept) {
        const int bucket = std::min(n_ctx, (n_past_ + 1 + 255) / 256 * 256);
        if (kept.bucket != bucket) {
            free_kept();
            kept.g = new_graph(1, bucket);
            kept.inp = ggml_new_tensor_1d(kept.g.ctx, GGML_TYPE_I32, 1);
            ggml_set_input(kept.inp); ggml_set_name(kept.inp, "ink_tokens");
            build_full(kept.g, kept.inp, true, &kept.out);
            kept.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
            if (!ggml_gallocr_alloc_graph(kept.ga, kept.g.gf)) throw std::runtime_error("inkling2: graph alloc failed");
            kept.bucket = bucket;
        }
        g = kept.g; inp = kept.inp; out = kept.out;
    } else {
        g = new_graph(T, n_past_ + T);
        inp = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
        ggml_set_input(inp); ggml_set_name(inp, "ink_tokens");
        build_full(g, inp, want_logits, &out);
    }
    auto t1 = clk::now();
    if (!use_kept && !ggml_gallocr_alloc_graph(galloc, g.gf)) throw std::runtime_error("inkling2: graph alloc failed");
    auto t2 = clk::now();
    ggml_backend_tensor_set(inp, toks, 0, (size_t) T * sizeof(int32_t));
    set_inputs(g, T);
    auto t3 = clk::now();
    if (ggml_backend_graph_compute(gpu, g.gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("inkling2: compute failed");
    if (prof)
        fprintf(stderr, "[ink T=%d nodes=%d build %.1f alloc %.1f inputs %.1f compute %.1f ms]\n", T,
                ggml_graph_n_nodes(g.gf), std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count(),
                std::chrono::duration<double, std::milli>(t3 - t2).count(), ms_since(t3));
    if (out) {
        logits.resize(ik.n_vocab);
        ggml_backend_tensor_get(out, logits.data(), 0, logits.size() * sizeof(float));
    }
    if (!use_kept) ggml_free(g.ctx);
    n_past_ += T;
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// RAM offload
// ---------------------------------------------------------------------------

namespace {
// Depth-attention state across segment graphs, as carry-slot ids (-1 = none).
struct SSt {
    std::vector<int> blocks, prelude_blocks, bank;
    int prelude_cur = -1, bank_last = -1, cur = -1, ffn_inp = -1, x = -1;
};
}

void InklingEngine::Impl::init_offload(size_t budget, size_t st_bytes) {
    const int E = (int) hp.n_embd, n_used = (int) hp.n_expert_used;
    if (const char * e = getenv("INK_CPU_MISS")) cpu_miss = atoi(e) != 0;
    if (const char * e = getenv("INK_REFILL"))   refill   = atoi(e);

    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * 64;
    p.no_alloc = true;
    car_ctx = ggml_init(p);
    for (int i = 0; i < 16; ++i) slots.push_back(ggml_new_tensor_2d(car_ctx, GGML_TYPE_F32, E, UBATCH));
    p_tok       = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I32, UBATCH);
    p_tau       = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, UBATCH);
    p_rel_full  = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I32, (int64_t) n_ctx * UBATCH);
    p_mask_full = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) n_ctx * UBATCH);
    p_rel_swa   = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I32, (int64_t) ring * UBATCH);
    p_mask_swa  = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) ring * UBATCH);
    p_idx_full  = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I64, UBATCH);
    p_idx_swa   = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I64, UBATCH);
    p_vidx_full = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I64, (int64_t) n_embd_kv * UBATCH);
    p_vidx_swa  = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I64, (int64_t) n_embd_kv * UBATCH);
    p_sink0     = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, n_head);
    p_sel       = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I32, (int64_t) n_used * UBATCH);
    p_w         = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) n_used * UBATCH);
    p_gpu       = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) E * UBATCH);
    p_cpu       = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) E * UBATCH);
    // graph B's integer inputs, packed so one upload covers them: [gate|up|down|logical] x (k*T)
    p_bpack     = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I32, (int64_t) 4 * n_used * UBATCH);
    p_bw        = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) n_used * UBATCH);
    p_rout      = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) 2 * n_used + E);
    car_buf = ggml_backend_alloc_ctx_tensors(car_ctx, gpu);
    if (!car_buf) throw std::runtime_error("inkling2: cannot allocate offload carries");
    ggml_backend_buffer_clear(car_buf, 0);
    p_cpu_zero = true;
    {
        ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu));
        if (!hb) hb = ggml_backend_cpu_buffer_type();
        const size_t n_f = (size_t) (2 * n_used + E) + n_used + 2 * (size_t) E;
        pin_buf = ggml_backend_buft_alloc_buffer(hb, n_f * 4 + (size_t) 4 * n_used * 4 + 256);
        if (!pin_buf) throw std::runtime_error("inkling2: cannot allocate pinned staging");
        float * base = (float *) ggml_backend_buffer_get_base(pin_buf);
        pin_rout = base;
        pin_w    = pin_rout + 2 * n_used + E;
        pin_cpu  = pin_w + n_used;
        pin_zero = pin_cpu + E;
        pin_ids  = (int32_t *) (pin_zero + E);
        std::fill(pin_zero, pin_zero + E, 0.0f);
    }

    // host copies of the per-expert scales for the CPU expert graphs
    const int n_phys = (int) hp.n_layer;
    ggml_init_params hp2{};
    hp2.mem_size = ggml_tensor_overhead() * (2 * n_phys + 2);
    hp2.no_alloc = true;
    hs_ctx = ggml_init(hp2);
    for (int ip = 0; ip < n_phys; ++ip) {
        hs_gu.push_back(ggml_new_tensor_1d(hs_ctx, GGML_TYPE_F32, hp.n_expert));
        hs_dn.push_back(ggml_new_tensor_1d(hs_ctx, GGML_TYPE_F32, hp.n_expert));
    }
    hs_buf = ggml_backend_alloc_ctx_tensors(hs_ctx, cpu);
    std::vector<float> tmp(hp.n_expert);
    for (int ip = 0; ip < n_phys; ++ip) {
        ggml_tensor * a = Wopt("blk.%d.ffn_gate_up_exps.scale", ip), * b = Wopt("blk.%d.ffn_down_exps.scale", ip);
        std::fill(tmp.begin(), tmp.end(), 1.0f);
        if (a) ggml_backend_tensor_get(a, tmp.data(), 0, tmp.size() * sizeof(float));
        ggml_backend_tensor_set(hs_gu[(size_t) ip], tmp.data(), 0, tmp.size() * sizeof(float));
        std::fill(tmp.begin(), tmp.end(), 1.0f);
        if (b) ggml_backend_tensor_get(b, tmp.data(), 0, tmp.size() * sizeof(float));
        ggml_backend_tensor_set(hs_dn[(size_t) ip], tmp.data(), 0, tmp.size() * sizeof(float));
    }

    // ---- VRAM expert pool: what the budget leaves ----
    size_t gpu_w = 0;
    for (auto b : wbufs_gpu) gpu_w += ggml_backend_buffer_get_size(b);
    size_t reserve = 900ull << 20;
    if (const char * e = getenv("INK_RESERVE_MB")) reserve = (size_t) atoll(e) << 20;
    if (const char * e = getenv("INK_PLACE")) place = e;
    size_t scr_bytes = 0;
    if (place != "lru") {
        static const char * fmt[3] = { "blk.%d.ffn_gate_exps.weight", "blk.%d.ffn_up_exps.weight", "blk.%d.ffn_down_exps.weight" };
        size_t role_max[3] = { 0, 0, 0 };
        for (int ip = 0; ip < n_phys; ++ip)
            for (int r = 0; r < 3; ++r) role_max[r] = std::max(role_max[r], ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(gpu), W(fmt[r], ip)));
        const size_t align = 256;
        scr_bytes = GGML_PAD(role_max[0], align) + GGML_PAD(role_max[1], align) + GGML_PAD(role_max[2], align);
        scr_buf = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(gpu), scr_bytes);
        if (!scr_buf) throw std::runtime_error("inkling2: cannot allocate the prefill expert scratch");
        ggml_init_params sp{};
        sp.mem_size = ggml_tensor_overhead() * (3 * n_phys + 4);
        sp.no_alloc = true;
        scr_ctx = ggml_init(sp);
        char * base = (char *) ggml_backend_buffer_get_base(scr_buf);
        for (int ip = 0; ip < n_phys; ++ip) {
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                ggml_tensor * src = W(fmt[r], ip);
                ggml_tensor * t = ggml_new_tensor(scr_ctx, src->type, ggml_n_dims(src), src->ne);
                if (ggml_backend_tensor_alloc(scr_buf, t, base + off) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("inkling2: scratch view alloc failed");
                scr[r].push_back(t);
                off += GGML_PAD(role_max[r], align);
            }
        }
    }
    const size_t used = gpu_w + st_bytes + ggml_backend_buffer_get_size(car_buf) + reserve + scr_bytes;
    const size_t pool = budget > used ? budget - used : 0;
    fprintf(stderr, "inkling2: offload: GPU weights %zu MB + state %zu MB + carries %zu MB + reserve %zu MB"
                    " + prefill scratch %zu MB -> expert pool %zu MB (%s)\n",
            gpu_w >> 20, st_bytes >> 20, ggml_backend_buffer_get_size(car_buf) >> 20, reserve >> 20,
            scr_bytes >> 20, pool >> 20,
            cpu_miss ? "misses computed on the CPU" : "misses fetched into VRAM");
    ec = std::make_unique<ExpertCache>(gpu, model, n_phys, (int) hp.n_expert, n_used, pool);
    l_sel.assign((size_t) n_phys, 0);
    l_gpu.assign((size_t) n_phys, 0);

    if (place != "lru") {
        place_static = true;
        cpu_miss = true;
        refill = 0;
        // Fill in reverse priority: the slot pools evict least-recently installed,
        // so whatever goes in last is what stays when a pool runs out.
        std::vector<int> order;   // highest priority first
        if (place == "core") {
            for (int ip = (int) ik.core_start; ip < (int) ik.core_end; ++ip) order.push_back(ip);
            for (int ip = 0; ip < n_phys; ++ip) if (!ik.in_core((uint32_t) ip)) order.push_back(ip);
        } else if (place == "front") {   // what --n-cpu-moe's complement looks like: last layers first
            for (int ip = n_phys - 1; ip >= 0; --ip) order.push_back(ip);
        } else {
            throw std::runtime_error("INK_PLACE: core, front or lru");
        }
        auto t0 = clk::now();
        for (auto it = order.rbegin(); it != order.rend(); ++it)
            for (int e = 0; e < (int) hp.n_expert; ++e) ec->ensure_resident(*it, e);
        ggml_backend_synchronize(gpu);
        fprintf(stderr, "inkling2: static placement '%s' filled in %.0f ms; resident experts per layer:",
                place.c_str(), ms_since(t0));
        for (int ip = 0; ip < n_phys; ++ip) {
            int n = 0;
            for (int e = 0; e < (int) hp.n_expert; ++e) {
                bool r = true;
                for (int k = 0; k < 3 && r; ++k) r = ec->resident((ExpertCache::Role) k, ip, e);
                n += r;
            }
            fprintf(stderr, " %d%s:%d", ip, ik.in_core((uint32_t) ip) ? "*" : "", n);
            full_res.push_back(n == (int) hp.n_expert ? 1 : 0);
        }
        fprintf(stderr, "\n");

        const char * fz = getenv("INK_FUSE");
        fuse = !(fz && atoi(fz) == 0);
        if (fuse) {
            ggml_init_params mp{};
            mp.mem_size = ggml_tensor_overhead() * 2;
            mp.no_alloc = true;
            map_ctx = ggml_init(mp);
            p_slotmap = ggml_new_tensor_2d(map_ctx, GGML_TYPE_I32, hp.n_expert, 3 * n_phys);
            map_buf = ggml_backend_alloc_ctx_tensors(map_ctx, gpu);
            if (!map_buf) throw std::runtime_error("inkling2: cannot allocate the slot map");
            std::vector<int32_t> mv((size_t) hp.n_expert * 3 * n_phys, 0);
            for (int r = 0; r < 3; ++r)
                for (int ip = 0; ip < n_phys; ++ip) {
                    const int32_t * row = ec->slot_of_row((ExpertCache::Role) r, ip);
                    std::copy(row, row + hp.n_expert, mv.begin() + ((size_t) r * n_phys + ip) * hp.n_expert);
                }
            ggml_backend_tensor_set(p_slotmap, mv.data(), 0, mv.size() * 4);
            int nf = 0;
            for (uint32_t iv = 0; iv < ik.n_visit; ++iv) nf += full_res[ik.phys(iv)];
            fprintf(stderr, "inkling2: %d of %u visits fully resident -> decoded inside fused graphs\n", nf, ik.n_visit);
        }
    }
}

void InklingEngine::Impl::upload_inputs(int T, int n_kv_full) {
    const int n_total = n_past_ + T;
    const int n_kv_swa = std::min(ring, n_kv_full);
    std::vector<float> tau(T);
    for (int i = 0; i < T; ++i) {
        const double p = (double) (n_past_ + i) + 1.0;
        tau[(size_t) i] = (float) (1.0 + ik.log_alpha * std::log(std::max(1.0, p / ik.log_floor)));
    }
    ggml_backend_tensor_set(p_tau, tau.data(), 0, tau.size() * sizeof(float));
    const int ext = (int) ik.rel_extent;
    auto fill = [&](int n_kv, bool swa, ggml_tensor * rel, ggml_tensor * mask) {
        std::vector<int32_t> r((size_t) n_kv * T);
        std::vector<float>   m((size_t) n_kv * T);
        for (int i = 0; i < T; ++i) {
            const int pi = n_past_ + i;
            for (int j = 0; j < n_kv; ++j) {
                const int pj = swa ? pos_of_ring(j, n_total) : j;
                const int d  = pi - pj;
                const bool vis = pj >= 0 && d >= 0 && (!swa || d < (int) ik.n_swa);
                r[(size_t) i * n_kv + j] = (d >= 0 && d < ext) ? d : ext;
                m[(size_t) i * n_kv + j] = vis ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(rel,  r.data(), 0, r.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    };
    fill(n_kv_full, false, p_rel_full, p_mask_full);
    fill(n_kv_swa,  true,  p_rel_swa,  p_mask_swa);
    std::vector<int64_t> idf(T), ids(T);
    for (int i = 0; i < T; ++i) { idf[(size_t) i] = n_past_ + i; ids[(size_t) i] = (n_past_ + i) % ring; }
    ggml_backend_tensor_set(p_idx_full, idf.data(), 0, idf.size() * sizeof(int64_t));
    ggml_backend_tensor_set(p_idx_swa,  ids.data(), 0, ids.size() * sizeof(int64_t));
    std::vector<int64_t> vf, vs;
    fill_vidx(T, vf, vs);
    ggml_backend_tensor_set(p_vidx_full, vf.data(), 0, vf.size() * sizeof(int64_t));
    ggml_backend_tensor_set(p_vidx_swa,  vs.data(), 0, vs.size() * sizeof(int64_t));
}

InklingEngine::Impl::Gr InklingEngine::Impl::seg_graph(int T, int n_kv_full, int nodes) {
    Gr g;
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false);
    p.no_alloc = true;
    g.ctx = ggml_init(p);
    g.gf  = ggml_new_graph_custom(g.ctx, nodes, false);
    g.T   = T;
    g.n_kv_full = n_kv_full;
    g.n_kv_swa  = std::min(ring, n_kv_full);
    ggml_context * c = g.ctx;
    g.tau       = ggml_view_3d(c, p_tau, 1, 1, T, 4, 4, 0);
    g.rel_full  = ggml_view_2d(c, p_rel_full,  g.n_kv_full, T, (size_t) g.n_kv_full * 4, 0);
    g.mask_full = ggml_view_2d(c, p_mask_full, g.n_kv_full, T, (size_t) g.n_kv_full * 4, 0);
    g.rel_swa   = ggml_view_2d(c, p_rel_swa,   g.n_kv_swa,  T, (size_t) g.n_kv_swa * 4, 0);
    g.mask_swa  = ggml_view_2d(c, p_mask_swa,  g.n_kv_swa,  T, (size_t) g.n_kv_swa * 4, 0);
    g.idx_full  = ggml_view_1d(c, p_idx_full, T, 0);
    g.idx_swa   = ggml_view_1d(c, p_idx_swa,  T, 0);
    g.vidx_full = ggml_view_2d(c, p_vidx_full, (int64_t) n_embd_kv * T, 1, (size_t) n_embd_kv * T * 8, 0);
    g.vidx_swa  = ggml_view_2d(c, p_vidx_swa,  (int64_t) n_embd_kv * T, 1, (size_t) n_embd_kv * T * 8, 0);
    g.sink0     = p_sink0;
    return g;
}

void InklingEngine::Impl::forward_offload(const int32_t * toks, int T, bool want_logits) {
    const int E = (int) hp.n_embd, n_used = (int) hp.n_expert_used, n_exp = (int) hp.n_expert;
    const int NV = (int) ik.n_visit;
    static const bool no_keep = getenv("INK_NO_KEEP") != nullptr;
    const bool decode = T == 1;
    const bool keep   = decode && want_logits && !no_keep;
    const int  n_kv_full = keep ? std::min(n_ctx, (n_past_ + 1 + 255) / 256 * 256) : n_past_ + T;

    if (keep && seg_bucket != n_kv_full) {
        for (auto & sg : segA) free_seg(sg);
        segA.assign((size_t) NV + 1, Seg{});
        seg_bucket = n_kv_full;
    }
    static std::vector<SSt> sst_after;   // per visit, for kept graphs (deterministic)
    if (sst_after.size() != (size_t) NV + 1) sst_after.assign((size_t) NV + 1, SSt{});
    if (segB.size() != (size_t) NV * (n_used + 1)) segB.assign((size_t) NV * (n_used + 1), Seg{});
    if (segC.size() != (size_t) hp.n_layer * (n_used + 1)) segC.assign((size_t) hp.n_layer * (n_used + 1), Seg{});

    ggml_backend_tensor_set(p_tok, toks, 0, (size_t) T * sizeof(int32_t));
    upload_inputs(T, n_kv_full);

    auto t_tok = clk::now();
    SSt sst;
    std::vector<int32_t> sel((size_t) n_used * T), slot3[3];
    std::vector<float>   w((size_t) n_used * T), xh(E), cpu_out(E);
    for (auto & v : slot3) v.resize((size_t) n_used * T);
    std::vector<std::vector<uint8_t>> chosen(hp.n_layer, std::vector<uint8_t>(n_exp, 0));   // this token, per physical layer
    std::vector<std::pair<int, int>> missed;   // (ip, e) computed on the CPU this token

    // Build graph A for visit iv (iv == NV: the head), starting from `in`.
    // Visits [iv_from, iv) are fully resident and run complete inside this graph.
    auto build_A = [&](Seg & sg, int iv_from, int iv, const SSt & in, SSt & outs) {
        Gr g = seg_graph(T, n_kv_full, 8192 + 640 * (iv - iv_from));
        sg.ctx = g.ctx; sg.gf = g.gf;
        ggml_context * c = g.ctx;
        std::vector<ggml_tensor *> id2v(slots.size(), nullptr);
        auto view = [&](int id) -> ggml_tensor * {
            if (id < 0) return nullptr;
            if (!id2v[(size_t) id]) id2v[(size_t) id] = ggml_view_2d(c, slots[(size_t) id], E, T, slots[(size_t) id]->nb[1], 0);
            return id2v[(size_t) id];
        };
        St st;
        for (int b : in.blocks)         st.blocks.push_back(view(b));
        for (int b : in.prelude_blocks) st.prelude_blocks.push_back(view(b));
        for (int b : in.bank)           st.bank.push_back(view(b));
        st.prelude_cur = view(in.prelude_cur);
        st.bank_last   = view(in.bank_last);
        if (iv_from == 0) {
            ggml_tensor * tk = ggml_view_1d(c, p_tok, T, 0);
            st.cur = ggml_get_rows(c, model.tok_embd_rows(), tk);
            st.cur = ggml_mul(c, ggml_rms_norm(c, st.cur, hp.rms_eps), G("token_embd_norm.weight"));
        } else {
            ggml_tensor * moe = ggml_view_2d(c, p_gpu, E, T, (size_t) E * 4, 0);
            if (decode || place_static) moe = ggml_add(c, moe, ggml_view_2d(c, p_cpu, E, T, (size_t) E * 4, 0));
            st.cur = visit_post(g, iv_from - 1, view(in.ffn_inp), moe);
        }
        for (int v = iv_from; v < iv; ++v) {
            const int vp = (int) ik.phys((uint32_t) v);
            ggml_tensor * vx = nullptr, * vs = nullptr, * vw = nullptr;
            ggml_tensor * vin = visit_pre(g, st, v, &vx);
            route(g, vp, vx, &vs, &vw);
            ggml_tensor * ids[3];
            for (int r = 0; r < 3; ++r) {
                ggml_tensor * row = ggml_view_2d(c, p_slotmap, 1, n_exp, 4,
                                                 ((size_t) r * hp.n_layer + vp) * n_exp * 4);
                ids[r] = ggml_reshape_2d(c, ggml_get_rows(c, row, vs), n_used, T);
            }
            ggml_tensor * moe = experts(g, vp, vx, vs, vw, ec->gate(vp), ec->up(vp), ec->down(vp), ids[0], ids[1], ids[2]);
            st.cur = visit_post(g, v, vin, moe);
        }
        if (iv == NV) {
            sg.out = head(g, st, true);
            ggml_set_output(sg.out);
            ggml_build_forward_expand(g.gf, sg.out);
            outs = SSt{};
            return;
        }
        const int ip = (int) ik.phys((uint32_t) iv);
        ggml_tensor * x = nullptr, * s = nullptr, * wt = nullptr;
        ggml_tensor * ffn_inp = visit_pre(g, st, iv, &x);
        route(g, ip, x, &s, &wt);
        sg.sel = ggml_view_2d(c, p_sel, n_used, T, (size_t) n_used * 4, 0);
        sg.w   = ggml_view_2d(c, p_w,   n_used, T, (size_t) n_used * 4, 0);
        ggml_build_forward_expand(g.gf, ggml_cpy(c, s, sg.sel));
        ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_reshape_2d(c, wt, n_used, T), sg.w));
        if (T == 1) {
            ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_cast(c, s, GGML_TYPE_F32), ggml_view_1d(c, p_rout, n_used, 0)));
            ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_reshape_1d(c, wt, n_used), ggml_view_1d(c, p_rout, n_used, (size_t) n_used * 4)));
            ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_reshape_1d(c, x, E), ggml_view_1d(c, p_rout, E, (size_t) 2 * n_used * 4)));
        }

        // materialise every live value into a carry slot
        std::vector<char> busy(slots.size(), 0);
        auto mark = [&](int id) { if (id >= 0) busy[(size_t) id] = 1; };
        for (int b : in.blocks) mark(b);
        for (int b : in.prelude_blocks) mark(b);
        for (int b : in.bank) mark(b);
        mark(in.prelude_cur); mark(in.bank_last); mark(in.ffn_inp); mark(in.cur);
        std::vector<std::pair<ggml_tensor *, int>> done;
        auto slot_of = [&](ggml_tensor * t) -> int {
            if (!t) return -1;
            for (size_t i = 0; i < id2v.size(); ++i) if (id2v[i] == t) return (int) i;
            for (auto & d : done) if (d.first == t) return d.second;
            for (size_t i = 0; i < busy.size(); ++i) {
                if (busy[i]) continue;
                busy[i] = 1;
                ggml_build_forward_expand(g.gf, ggml_cpy(c, t, view((int) i)));
                done.push_back({ t, (int) i });
                return (int) i;
            }
            throw std::runtime_error("inkling2: out of carry slots");
        };
        outs = SSt{};
        for (auto * b : st.blocks)         outs.blocks.push_back(slot_of(b));
        for (auto * b : st.prelude_blocks) outs.prelude_blocks.push_back(slot_of(b));
        for (auto * b : st.bank)           outs.bank.push_back(slot_of(b));
        outs.prelude_cur = slot_of(st.prelude_cur);
        outs.bank_last   = slot_of(st.bank_last);
        outs.ffn_inp     = slot_of(ffn_inp);
        outs.x           = slot_of(x);
        sg.x = view(outs.x);
    };

    // Graph B: k experts per token on VRAM slots, ids / weights from p_bids / p_bw.
    auto build_B = [&](Seg & sg, int iv, int k, int x_id, bool accumulate) {
        Gr g = seg_graph(T, n_kv_full);
        sg.ctx = g.ctx; sg.gf = g.gf;
        ggml_context * c = g.ctx;
        const int ip = (int) ik.phys((uint32_t) iv);
        ggml_tensor * x  = ggml_view_2d(c, slots[(size_t) x_id], E, T, slots[(size_t) x_id]->nb[1], 0);
        const size_t blk = (size_t) k * T * 4;
        ggml_tensor * ig = ggml_view_2d(c, p_bpack, k, T, (size_t) k * 4, 0 * blk);
        ggml_tensor * iu = ggml_view_2d(c, p_bpack, k, T, (size_t) k * 4, 1 * blk);
        ggml_tensor * id = ggml_view_2d(c, p_bpack, k, T, (size_t) k * 4, 2 * blk);
        ggml_tensor * bs = ggml_view_2d(c, p_bpack, k, T, (size_t) k * 4, 3 * blk);
        ggml_tensor * bw = ggml_view_3d(c, p_bw, 1, k, T, 4, (size_t) k * 4, 0);
        ggml_tensor * r = experts(g, ip, x, bs, bw, ec->gate(ip), ec->up(ip), ec->down(ip), ig, iu, id);
        ggml_tensor * dst = ggml_view_2d(c, p_gpu, E, T, (size_t) E * 4, 0);
        if (accumulate) r = ggml_add(c, r, dst);
        ggml_build_forward_expand(g.gf, ggml_cpy(c, r, dst));
    };

    auto alloc_run = [&](Seg & sg, bool kept, ggml_backend_t be) {
        if (!sg.ga) {
            sg.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            if (!ggml_gallocr_alloc_graph(sg.ga, sg.gf)) throw std::runtime_error("inkling2: segment alloc failed");
        } else if (!kept) {
            if (!ggml_gallocr_alloc_graph(sg.ga, sg.gf)) throw std::runtime_error("inkling2: segment alloc failed");
        }
    };

    // the visits that come back to the host (+ NV, the head); all of them unless fusing
    std::vector<int> routed;
    const bool fused = keep && fuse && place_static;
    for (int v = 0; v < NV; ++v)
        if (!fused || !full_res[ik.phys((uint32_t) v)]) routed.push_back(v);
    routed.push_back(NV);

    for (size_t j = 0; j < routed.size(); ++j) {
        const int iv = routed[j];
        const int iv_from = j == 0 ? 0 : routed[j - 1] + 1;
        if (iv > iv_from) {   // fused visits: every pick is on the GPU
            stx.sel += (uint64_t) n_used * (iv - iv_from);
            stx.gpu += (uint64_t) n_used * (iv - iv_from);
            for (int v = iv_from; v < iv; ++v) {
                l_sel[ik.phys((uint32_t) v)] += n_used;
                l_gpu[ik.phys((uint32_t) v)] += n_used;
            }
        }
        // ---- graph A ----
        auto t0 = clk::now();
        Seg tmpA;
        Seg * A = &tmpA;
        SSt next;
        if (keep) {
            A = &segA[(size_t) iv];
            if (!A->gf) {
                build_A(*A, iv_from, iv, sst, next);
                sst_after[(size_t) iv] = next;
                alloc_run(*A, true, gpu);
            }
            next = sst_after[(size_t) iv];
        } else {
            build_A(*A, iv_from, iv, sst, next);
            if (!ggml_gallocr_alloc_graph(galloc, A->gf)) throw std::runtime_error("inkling2: graph A alloc failed");
        }
        if (decode && iv < NV) {
            if (ggml_backend_graph_compute_async(gpu, A->gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("inkling2: graph A failed");
            ggml_backend_tensor_get_async(gpu, p_rout, pin_rout, 0, (size_t) (2 * n_used + E) * 4);
            ggml_backend_synchronize(gpu);
        } else if (ggml_backend_graph_compute(gpu, A->gf) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("inkling2: graph A failed");
        }
        stx.t_a += ms_since(t0);
        if (iv == NV) {
            if (want_logits) {
                logits.resize(ik.n_vocab);
                ggml_backend_tensor_get(A->out, logits.data(), 0, logits.size() * sizeof(float));
            }
            if (!keep) ggml_free(A->ctx);
            break;
        }
        sst = next;
        const int ip = (int) ik.phys((uint32_t) iv);
        if (!keep) ggml_free(A->ctx);

        if (decode) {
            for (int k = 0; k < n_used; ++k) {
                sel[(size_t) k] = (int32_t) pin_rout[k];
                w[(size_t) k]   = pin_rout[n_used + k];
            }
        } else {
            ggml_backend_tensor_get(p_sel, sel.data(), 0, sel.size() * sizeof(int32_t));
            ggml_backend_tensor_get(p_w,   w.data(),   0, w.size() * sizeof(float));
        }

        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < n_used; ++k) {
                const int e = sel[(size_t) t * n_used + k];
                if (decode && ik.in_core((uint32_t) ip)) {
                    stx.core_sel++;
                    if (chosen[(size_t) ip][(size_t) e]) stx.core_repeat++;
                }
                if (decode) chosen[(size_t) ip][(size_t) e] = 1;
            }
        }

        if (decode) {
            // ---- split the selection: resident on the GPU, the rest on the CPU ----
            auto t1 = clk::now();
            if (!cpu_miss) {
                const uint64_t m0 = ec->stats().misses;
                ec->ensure(ip, sel.data(), n_used, slot3[0].data(), slot3[1].data(), slot3[2].data());
                stx.fetched += (ec->stats().misses - m0) / 3;
                stx.t_fetch += ms_since(t1);
            }
            std::vector<int32_t> hg, hu, hd, hs, ms;
            std::vector<float>   hw, mw;
            for (int k = 0; k < n_used; ++k) {
                const int e = sel[(size_t) k];
                bool res = true;
                for (int r = 0; r < 3 && res; ++r) res = ec->resident((ExpertCache::Role) r, ip, e);
                if (!cpu_miss) res = true;
                if (res) {
                    if (cpu_miss) {
                        for (int r = 0; r < 3; ++r) ec->touch((ExpertCache::Role) r, ip, e);
                        hg.push_back(ec->slot_of_row(ExpertCache::GATE, ip)[e]);
                        hu.push_back(ec->slot_of_row(ExpertCache::UP,   ip)[e]);
                        hd.push_back(ec->slot_of_row(ExpertCache::DOWN, ip)[e]);
                    } else {
                        hg.push_back(slot3[0][(size_t) k]); hu.push_back(slot3[1][(size_t) k]); hd.push_back(slot3[2][(size_t) k]);
                    }
                    hs.push_back(e); hw.push_back(w[(size_t) k]);
                } else {
                    ms.push_back(e); mw.push_back(w[(size_t) k]);
                }
            }
            const int nr = (int) hs.size(), nm = (int) ms.size();
            stx.sel += n_used; stx.gpu += nr; stx.cpu += nm;
            l_sel[(size_t) ip] += n_used; l_gpu[(size_t) ip] += nr;

            if (nr > 0) {
                std::copy(hg.begin(), hg.end(), pin_ids);
                std::copy(hu.begin(), hu.end(), pin_ids + nr);
                std::copy(hd.begin(), hd.end(), pin_ids + 2 * nr);
                std::copy(hs.begin(), hs.end(), pin_ids + 3 * nr);
                std::copy(hw.begin(), hw.end(), pin_w);
                ggml_backend_tensor_set_async(gpu, p_bpack, pin_ids, 0, (size_t) 4 * nr * 4);
                ggml_backend_tensor_set_async(gpu, p_bw,    pin_w,   0, (size_t) nr * 4);
                Seg tmpB;
                Seg * B = &tmpB;
                if (keep) {
                    B = &segB[(size_t) iv * (n_used + 1) + nr];
                    if (!B->gf) { build_B(*B, iv, nr, sst.x, false); alloc_run(*B, true, gpu); }
                } else {
                    build_B(*B, iv, nr, sst.x, false);
                    if (!ggml_gallocr_alloc_graph(galloc, B->gf)) throw std::runtime_error("inkling2: graph B alloc failed");
                }
                ggml_backend_graph_compute_async(gpu, B->gf);
                if (!keep) { ggml_backend_synchronize(gpu); ggml_free(B->ctx); }
            } else {
                ggml_backend_tensor_set_async(gpu, p_gpu, pin_zero, 0, (size_t) E * 4);
            }
            auto t2 = clk::now();
            stx.t_b += std::chrono::duration<double, std::milli>(t2 - t1).count();

            if (nm > 0) {
                std::copy(pin_rout + 2 * n_used, pin_rout + 2 * n_used + E, xh.begin());
                Seg & C = segC[(size_t) ip * (n_used + 1) + nm];
                if (!C.gf) {
                    Gr g;
                    ggml_init_params cp{};
                    cp.mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false);
                    cp.no_alloc = true;
                    g.ctx = ggml_init(cp);
                    g.gf  = ggml_new_graph_custom(g.ctx, 512, false);
                    g.T = 1;
                    C.ctx = g.ctx; C.gf = g.gf;
                    C.xin = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, E, 1);  ggml_set_input(C.xin);
                    C.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, nm, 1); ggml_set_input(C.ids);
                    C.win = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, nm, 1); ggml_set_input(C.win);
                    C.out = experts(g, ip, C.xin, C.ids, C.win, W("blk.%d.ffn_gate_exps.weight", ip),
                                    W("blk.%d.ffn_up_exps.weight", ip), W("blk.%d.ffn_down_exps.weight", ip),
                                    nullptr, nullptr, nullptr, hs_gu[(size_t) ip], hs_dn[(size_t) ip]);
                    ggml_set_output(C.out);
                    ggml_build_forward_expand(g.gf, C.out);
                    C.ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
                    if (!ggml_gallocr_alloc_graph(C.ga, C.gf)) throw std::runtime_error("inkling2: CPU expert alloc failed");
                }
                ggml_backend_tensor_set(C.xin, xh.data(), 0, (size_t) E * 4);
                ggml_backend_tensor_set(C.ids, ms.data(), 0, (size_t) nm * 4);
                ggml_backend_tensor_set(C.win, mw.data(), 0, (size_t) nm * 4);
                ggml_backend_graph_compute(cpu, C.gf);
                ggml_backend_tensor_get(C.out, pin_cpu, 0, (size_t) E * 4);
                ggml_backend_tensor_set_async(gpu, p_cpu, pin_cpu, 0, (size_t) E * 4);
                p_cpu_zero = false;
                for (int e : ms) missed.push_back({ ip, e });
            } else if (!p_cpu_zero) {
                ggml_backend_tensor_set_async(gpu, p_cpu, pin_zero, 0, (size_t) E * 4);
                p_cpu_zero = true;
            }
            // no sync: the next graph A is queued behind graph B and these copies
            stx.t_c += ms_since(t2);
        } else if (place_static) {
            // ---- prefill, static pool: a fully resident layer runs from its slots;
            // any other layer is streamed whole into the scratch and runs there ----
            auto t1 = clk::now();
            bool full = true;
            for (int e = 0; e < n_exp && full; ++e)
                for (int r = 0; r < 3 && full; ++r) full = ec->resident((ExpertCache::Role) r, ip, e);
            Seg B;
            if (full) {
                for (size_t i = 0; i < sel.size(); ++i) {
                    slot3[0][i] = ec->slot_of_row(ExpertCache::GATE, ip)[sel[i]];
                    slot3[1][i] = ec->slot_of_row(ExpertCache::UP,   ip)[sel[i]];
                    slot3[2][i] = ec->slot_of_row(ExpertCache::DOWN, ip)[sel[i]];
                }
                std::vector<int32_t> pk;
                for (int r = 0; r < 3; ++r) pk.insert(pk.end(), slot3[r].begin(), slot3[r].end());
                pk.insert(pk.end(), sel.begin(), sel.end());
                ggml_backend_tensor_set(p_bpack, pk.data(), 0, pk.size() * 4);
                ggml_backend_tensor_set(p_bw, w.data(), 0, w.size() * 4);
                build_B(B, iv, n_used, sst.x, false);
            } else {
                static const char * fmt[3] = { "blk.%d.ffn_gate_exps.weight", "blk.%d.ffn_up_exps.weight", "blk.%d.ffn_down_exps.weight" };
                for (int r = 0; r < 3; ++r) {
                    ggml_tensor * src = W(fmt[r], ip);
                    ggml_backend_tensor_set_async(gpu, scr[r][(size_t) ip], src->data, 0, ggml_nbytes(src));
                    stx.streamed_bytes += ggml_nbytes(src);
                }
                stx.streamed_layers++;
                Gr g = seg_graph(T, n_kv_full);
                B.ctx = g.ctx; B.gf = g.gf;
                ggml_context * c = g.ctx;
                ggml_tensor * x  = ggml_view_2d(c, slots[(size_t) sst.x], E, T, slots[(size_t) sst.x]->nb[1], 0);
                ggml_tensor * s2 = ggml_view_2d(c, p_sel, n_used, T, (size_t) n_used * 4, 0);
                ggml_tensor * w3 = ggml_view_3d(c, p_w, 1, n_used, T, 4, (size_t) n_used * 4, 0);
                ggml_tensor * r2 = experts(g, ip, x, s2, w3, scr[0][(size_t) ip], scr[1][(size_t) ip], scr[2][(size_t) ip]);
                ggml_build_forward_expand(g.gf, ggml_cpy(c, r2, ggml_view_2d(c, p_gpu, E, T, (size_t) E * 4, 0)));
            }
            if (!ggml_gallocr_alloc_graph(galloc, B.gf)) throw std::runtime_error("inkling2: graph B alloc failed");
            if (ggml_backend_graph_compute(gpu, B.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("inkling2: graph B failed");
            ggml_free(B.ctx);
            if (!p_cpu_zero) {   // the static tail adds p_cpu; nothing went to the CPU here
                ggml_backend_tensor_memset(p_cpu, 0, 0, (size_t) E * UBATCH * 4);
                p_cpu_zero = true;
            }
            stx.t_fetch += ms_since(t1);
        } else {
            // ---- prefill: the chunk's union of experts, in groups the pool can hold ----
            auto t1 = clk::now();
            std::vector<int> uni;
            std::vector<uint8_t> in_u(n_exp, 0);
            for (int32_t e : sel) if (!in_u[(size_t) e]) { in_u[(size_t) e] = 1; uni.push_back(e); }
            const int cap = std::max(1, ec->capacity(ip));
            const uint64_t m0 = ec->stats().misses;
            bool first = true;
            // Balanced groups, each of at least n_used experts: a token's
            // out-of-group picks need distinct stand-in slots (this ggml's CUDA
            // mul_mat_id does not take a repeated id within a row).
            if (cap < n_used) throw std::runtime_error("inkling2: expert pool smaller than one token's experts");
            const size_t n_groups = (uni.size() + cap - 1) / cap;
            const size_t gsz = (uni.size() + n_groups - 1) / n_groups;
            for (size_t g0 = 0; g0 < uni.size(); g0 += gsz) {
                const int ng = (int) std::min(uni.size() - g0, gsz);
                std::vector<int32_t> gs[3];
                for (auto & v : gs) v.resize((size_t) ng);
                ec->ensure(ip, uni.data() + g0, ng, gs[0].data(), gs[1].data(), gs[2].data());
                static const bool dbg = getenv("INK_DBG") != nullptr;
                if (dbg) {
                    int bad = 0;
                    for (int r = 0; r < 3; ++r)
                        for (int i = 0; i < ng; ++i)
                            bad += gs[r][(size_t) i] < 0 || gs[r][(size_t) i] >= ec->tensor((ExpertCache::Role) r, ip)->ne[2];
                    fprintf(stderr, "[ink prefill iv %d ip %d union %zu cap %d group %zu+%d bad %d]\n",
                            iv, ip, uni.size(), cap, g0, ng, bad);
                }
                std::vector<int> pos_in_group(n_exp, -1);
                for (int i = 0; i < ng; ++i) pos_in_group[(size_t) uni[g0 + (size_t) i]] = i;
                std::vector<float> bw((size_t) n_used * T);
                for (int t = 0; t < T; ++t) {
                    // group members this row already uses, then fill the rest with unused ones
                    std::vector<uint8_t> used_in_row((size_t) ng, 0);
                    for (int k = 0; k < n_used; ++k) {
                        const int pi = pos_in_group[(size_t) sel[(size_t) t * n_used + k]];
                        if (pi >= 0) used_in_row[(size_t) pi] = 1;
                    }
                    int next_free = 0;
                    for (int k = 0; k < n_used; ++k) {
                        const size_t i = (size_t) t * n_used + k;
                        int pi = pos_in_group[(size_t) sel[i]];
                        bw[i] = pi >= 0 ? w[i] : 0.0f;
                        if (pi < 0) {
                            while (used_in_row[(size_t) next_free]) ++next_free;
                            pi = next_free;
                            used_in_row[(size_t) pi] = 1;
                        }
                        for (int r = 0; r < 3; ++r) slot3[r][i] = gs[r][(size_t) pi];
                    }
                }
                std::vector<int32_t> pk;
                for (int r = 0; r < 3; ++r) pk.insert(pk.end(), slot3[r].begin(), slot3[r].end());
                pk.insert(pk.end(), sel.begin(), sel.end());
                ggml_backend_tensor_set(p_bpack, pk.data(), 0, pk.size() * 4);
                ggml_backend_tensor_set(p_bw, bw.data(), 0, bw.size() * 4);
                Seg B;
                build_B(B, iv, n_used, sst.x, !first);
                if (!ggml_gallocr_alloc_graph(galloc, B.gf)) throw std::runtime_error("inkling2: graph B alloc failed");
                if (ggml_backend_graph_compute(gpu, B.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("inkling2: graph B failed");
                ggml_free(B.ctx);
                first = false;
            }
            stx.fetched += (ec->stats().misses - m0) / 3;
            stx.t_fetch += ms_since(t1);
        }
    }

    if (decode) {
        stx.tokens++;
        // refill: make a few of this token's CPU experts resident (most recent first)
        int left = refill;
        for (auto it = missed.rbegin(); it != missed.rend() && left > 0; ++it, --left)
            ec->ensure_resident(it->first, it->second);
    }
    (void) t_tok;
    n_past_ += T;
}

void InklingEngine::Impl::dump_offload_stats() {
    if (!stx.sel) return;
    const double n = (double) std::max<uint64_t>(1, stx.tokens);
    fprintf(stderr, "inkling2 offload: %llu decode tokens, expert visits %.1f/token: GPU %.1f%%, CPU %.1f%%;"
                    " core-layer picks already made by an earlier pass this token %.1f%%\n",
            (unsigned long long) stx.tokens, stx.sel / n, 100.0 * stx.gpu / stx.sel, 100.0 * stx.cpu / stx.sel,
            stx.core_sel ? 100.0 * stx.core_repeat / stx.core_sel : 0.0);
    fprintf(stderr, "  per token: graph A %.2f ms, GPU experts %.2f ms, CPU experts+sync %.2f ms, fetch %.2f ms;"
                    " prefill/fetch total %.0f ms, %llu experts fetched\n",
            stx.t_a / n, stx.t_b / n, stx.t_c / n, 0.0, stx.t_fetch, (unsigned long long) stx.fetched);
    if (stx.streamed_layers)
        fprintf(stderr, "  prefill streamed %llu layer visits into the scratch (%.1f GB over PCIe)\n",
                (unsigned long long) stx.streamed_layers, stx.streamed_bytes / 1e9);
    fprintf(stderr, "  GPU share by physical layer:");
    for (size_t ip = 0; ip < l_sel.size(); ++ip)
        fprintf(stderr, " %zu%s:%.0f", ip, ik.in_core((uint32_t) ip) ? "*" : "", l_sel[ip] ? 100.0 * l_gpu[ip] / l_sel[ip] : 0.0);
    fprintf(stderr, "  (* = looped)\n");
}

InklingEngine::InklingEngine(Model & model, const RuntimeConfig & cfg)
    : impl_(std::make_unique<Impl>(model, cfg)) { impl_->init(); }
InklingEngine::~InklingEngine() {
    if (impl_ && impl_->ec) impl_->dump_offload_stats();
}

const std::vector<float> & InklingEngine::decode(const std::vector<int32_t> & tokens) {
    Impl & m = *impl_;
    const int P = (int) tokens.size();
    for (int i = 0; i < P; i += UBATCH) {
        const int T = std::min(UBATCH, P - i);
        m.forward(tokens.data() + i, T, i + T == P);
        if (m.progress_cb && P > UBATCH) m.progress_cb(i + T, P);
    }
    m.kv_toks.insert(m.kv_toks.end(), tokens.begin(), tokens.end());
    return m.logits;
}
void InklingEngine::prefill(const std::vector<int32_t> & tokens) {
    Impl & m = *impl_;
    const int P = (int) tokens.size();
    for (int i = 0; i < P; i += UBATCH) {
        const int T = std::min(UBATCH, P - i);
        m.forward(tokens.data() + i, T, false);
        if (m.progress_cb) m.progress_cb(i + T, P);
    }
    m.kv_toks.insert(m.kv_toks.end(), tokens.begin(), tokens.end());
}
void InklingEngine::reset() {
    impl_->free_kept();
    impl_->n_past_ = 0;
    impl_->kv_toks.clear();
    ggml_backend_buffer_clear(impl_->st_buf, 0);
}
int InklingEngine::n_past() const { return impl_->n_past_; }
const std::vector<int32_t> & InklingEngine::kv_tokens() const { return impl_->kv_toks; }
bool InklingEngine::has_expert_cache() const { return impl_->ec != nullptr; }
Runtime::CacheStats InklingEngine::cache_stats() const {
    Runtime::CacheStats c;
    if (impl_->ec) {
        c.hits   = impl_->stx.gpu;
        c.misses = impl_->stx.cpu + impl_->stx.fetched;
        c.fetch_bytes = impl_->ec->stats().fetch_bytes;
        c.fetch_ms    = impl_->ec->stats().fetch_ms;
    }
    return c;
}
void InklingEngine::set_progress_cb(std::function<void(int, int)> cb) { impl_->progress_cb = std::move(cb); }

} // namespace questwend
