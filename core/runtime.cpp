#include "runtime.h"
#include "model.h"
#include "expert_cache.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <memory>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace qwencpp {

static const int GRAPH_SIZE = 16384;

struct Runtime::Impl {
    Model & model;
    RuntimeConfig cfg;

    // Primary compute backend (GPU or CPU).
    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t weights_buf = nullptr;

    // Phase B: CPU backend + sched for expert weight offloading.
    // When active, expert tensors live in expert_cpu_buf (CPU pinned memory)
    // and the rest of the weights are in weights_buf (GPU).
    // ggml_backend_sched handles routing ops to the right backend.
    ggml_backend_t        cpu_backend    = nullptr;
    // Expert weights live in one or more pinned host buffers. Multiple buffers are
    // used because a single cudaHostAlloc is capped (~15.5 GB on WDDM); chunking
    // lets the full expert set be page-locked for fast/overlappable H2D.
    std::vector<ggml_backend_buffer_t> expert_cpu_bufs;
    ggml_backend_sched_t  sched          = nullptr;

    // Phase B v2: dynamic per-expert VRAM cache (single-token decode hot path).
    // When active, decode runs layer-by-layer on GPU: each layer's router is
    // computed, its selected experts are made resident in the cache (streaming
    // misses from CPU/SSD), then the expert matmuls run on GPU slot tensors.
    std::unique_ptr<ExpertCache> ecache;
    bool                  ssd_mode = false;     // experts streamed from SSD (no RAM copy)
    ggml_gallocr_t        cache_galloc = nullptr;
    // persistent "carry" tensors that bridge the per-layer graph segments
    ggml_context *        cctx       = nullptr;
    ggml_backend_buffer_t cbuf       = nullptr;
    ggml_tensor *         p_h        = nullptr;  // [n_embd]   hidden state across layers
    // Carry tensors are double-buffered (parity by layer) so that a fused
    // segB(L)+segA(L+1) graph has no write-after-read hazard: segB(L) reads
    // carry[L%2] while segA(L+1) writes carry[(L+1)%2].
    ggml_tensor *         p_ffn_in   = nullptr;  // [n_embd]   normed FFN input (seg A -> B)
    ggml_tensor *         p_resid    = nullptr;  // [n_embd]   FFN residual base (seg A -> B)
    ggml_tensor *         p_weights  = nullptr;  // [n_used]   normalized expert weights
    ggml_tensor *         p_ffn_in2  = nullptr;
    ggml_tensor *         p_resid2   = nullptr;
    ggml_tensor *         p_weights2 = nullptr;
    ggml_tensor *         p_slot_g   = nullptr;  // [n_used]   gate slot ids (host -> GPU)
    ggml_tensor *         p_slot_u   = nullptr;  // [n_used]   up   slot ids
    ggml_tensor *         p_slot_d   = nullptr;  // [n_used]   down slot ids

    // Phase B v2-fast: optimistic single-graph decode over the VRAM cache.
    // One persistent (CUDA-graph friendly) graph runs the whole token; per-layer
    // MoE remaps logical expert ids to cache slots in-graph via g2s_all. After a
    // speculative run we verify residency (sel_all read back once) and, on a
    // miss, restore recurrent state and fall back to the slow decode_cached.
    bool                  cache_fast_build   = false;   // set while building the fast graph
    bool                  cache_fast_enabled = false;    // experimental; opt-in via QWEN_FASTCACHE
    ggml_tensor *         g2s_all = nullptr;  // [1, n_expert, 3*n_layer] i32  (host-filled remap)
    ggml_tensor *         sel_all = nullptr;  // [n_used, n_layer]        i32  (selected readback)
    ggml_context *        f_ctx    = nullptr;
    ggml_cgraph *         f_gf     = nullptr;
    ggml_gallocr_t        f_galloc = nullptr;
    int                   f_nkv    = 0;
    std::vector<int32_t>  g2s_host;
    std::vector<int32_t>  sel_host;
    // recurrent-state backup (for rolling back a speculative miss)
    ggml_context *        bak_ctx = nullptr;
    ggml_backend_buffer_t bak_buf = nullptr;
    std::vector<ggml_tensor *> conv_bak, ssm_bak;

    // recurrent / KV state (persistent across decode steps)
    ggml_context *        st_ctx = nullptr;
    ggml_backend_buffer_t st_buf = nullptr;
    std::vector<ggml_tensor *> k_cache;     // [n_embd_gqa, n_ctx]  (attention layers)
    std::vector<ggml_tensor *> v_cache;     // [n_embd_gqa, n_ctx]  (attention layers, non-transposed)
    std::vector<ggml_tensor *> conv_state;  // [d_conv-1, conv_ch]  (GDN layers)
    std::vector<ggml_tensor *> ssm_state;   // [S, S, H_v]          (GDN layers)

    ggml_gallocr_t galloc = nullptr;

    // MTP (multi-token prediction): the trailing nextn block drafts the next-next
    // token from the main model's last hidden state. Used for self-speculative
    // decoding. The MTP block has its own KV cache slot (k/v_cache[n_main]).
    bool                  capture_hidden = false;   // main forward exposes its last hidden
    std::vector<float>    mtp_hidden;               // host copy of the last main hidden [n_embd]
    std::vector<float>    mtp_logits;
    int                   mtp_past = 0;             // MTP KV write position
    ggml_gallocr_t        mtp_galloc = nullptr;
    // 2-token verify outputs (logits + hidden for both positions)
    // k+1-token verify outputs: per-position logits (vL) and main hidden (vH).
    std::vector<std::vector<float>> vL, vH;
    std::vector<float>    mtp_block_hidden;   // MTP block output hidden (for chaining drafts)

    int n_ctx  = 0;
    int n_past = 0;

    // decode_cached profiling (QWEN_PROF_DC): wall vs GPU-compute time
    double dc_wall_ms = 0, dc_gpu_ms = 0;
    uint64_t dc_tokens = 0;

    // persistent single-token decode graph (built once, reused -> enables CUDA graphs)
    // NOTE: disabled when sched is active (expert offload mode)
    bool                  persistent = false;     // true while building/using the decode graph
    ggml_context *        dctx    = nullptr;
    ggml_cgraph *         dgf     = nullptr;
    ggml_gallocr_t        dgalloc = nullptr;
    ggml_tensor *         d_kvidx = nullptr;       // I64 [1] : current write position (n_past)
    int                   d_nkv   = 0;             // n_kv the decode graph was built for (bucketed)
    bool                  reuse_graph = true;
    bool                  use_flash = false;       // fused flash-attention (GPU)
    static const int      KV_BUCKET = 32;          // rebuild decode graph only when crossing a bucket

    std::vector<float> logits;

    Impl(Model & m, const RuntimeConfig & c) : model(m), cfg(c) {}
    ~Impl() {
        if (ecache) {
            const auto & s = ecache->stats();
            const uint64_t acc = s.hits + s.misses;
            fprintf(stderr,
                    "expert cache stats: %llu accesses, %.1f%% hit, %llu misses, %llu evictions\n",
                    (unsigned long long) acc,
                    acc ? 100.0 * (double) s.hits / (double) acc : 0.0,
                    (unsigned long long) s.misses, (unsigned long long) s.evictions);
            if (dc_tokens)
                fprintf(stderr,
                    "decode_cached prof: %llu tok, wall %.2f ms/tok, gpu-compute %.2f ms/tok (%.0f%%), host %.2f ms/tok\n",
                    (unsigned long long) dc_tokens, dc_wall_ms / dc_tokens, dc_gpu_ms / dc_tokens,
                    dc_wall_ms > 0 ? 100.0 * dc_gpu_ms / dc_wall_ms : 0.0,
                    (dc_wall_ms - dc_gpu_ms) / dc_tokens);
            if (s.fetch_bytes && s.fetch_ms > 1.0)
                fprintf(stderr,
                    "expert cache fetch: %.0f ms total, %.1f MB (%.1f GB/s effective)\n",
                    s.fetch_ms, s.fetch_bytes / 1024.0 / 1024.0,
                    (s.fetch_bytes / 1024.0 / 1024.0 / 1024.0) / (s.fetch_ms / 1000.0));
            else if (s.fetch_bytes)
                fprintf(stderr, "expert cache fetch: %.1f MB (async H2D)\n",
                    s.fetch_bytes / 1024.0 / 1024.0);
            if (cfg.cache_profile_save && !cfg.cache_profile.empty() && ecache->save_profile(cfg.cache_profile))
                fprintf(stderr, "expert cache: saved profile to '%s'\n", cfg.cache_profile.c_str());
        }
        ecache.reset();
        if (mtp_galloc)     ggml_gallocr_free(mtp_galloc);
        if (f_galloc)       ggml_gallocr_free(f_galloc);
        if (f_ctx)          ggml_free(f_ctx);
        if (bak_buf)        ggml_backend_buffer_free(bak_buf);
        if (bak_ctx)        ggml_free(bak_ctx);
        if (cache_galloc)   ggml_gallocr_free(cache_galloc);
        if (cbuf)           ggml_backend_buffer_free(cbuf);
        if (cctx)           ggml_free(cctx);
        if (sched)          ggml_backend_sched_free(sched);
        if (galloc)         ggml_gallocr_free(galloc);
        if (dgalloc)        ggml_gallocr_free(dgalloc);
        if (dctx)           ggml_free(dctx);
        if (st_buf)         ggml_backend_buffer_free(st_buf);
        if (st_ctx)         ggml_free(st_ctx);
        // expert_cpu_bufs and weights_buf are owned here (not by Model in split mode)
        for (auto b : expert_cpu_bufs) if (b) ggml_backend_buffer_free(b);
        if (cpu_backend)    ggml_backend_free(cpu_backend);
        // In single-backend mode, weights_buf is owned by Model; in split mode it's ours.
        // Model::~Model() also tries to free weights_buf_ -- that one is the original
        // Model-side buf which is nullptr in split mode. Safe.
        if (backend)        ggml_backend_free(backend);
    }

    ggml_tensor * W(const char * fmt, int il) {
        char name[256];
        snprintf(name, sizeof(name), fmt, il);
        ggml_tensor * t = model.tensor(name);
        if (!t) throw std::runtime_error(std::string("missing tensor: ") + name);
        return t;
    }
    ggml_tensor * Wopt(const char * fmt, int il) {
        char name[256];
        snprintf(name, sizeof(name), fmt, il);
        return model.tensor(name);
    }

    void init();
    void zero_states();
    ggml_cgraph * build_graph(ggml_context * ctx, int n_tokens, int n_kv);
    ggml_tensor * build_attn(ggml_context * ctx, ggml_cgraph * gf, int il,
                             ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V,
                             ggml_tensor * mask, int n_tokens, int n_kv);
    ggml_tensor * build_gdn(ggml_context * ctx, ggml_cgraph * gf, int il,
                            ggml_tensor * x, int n_tokens);
    ggml_tensor * build_moe(ggml_context * ctx, ggml_cgraph * gf, int il,
                            ggml_tensor * x, int n_tokens);
    const std::vector<float> & decode(const std::vector<int32_t> & tokens);
    const std::vector<float> & decode_reuse(int32_t token);

    // MTP draft: predict the token after `token`, given the captured main hidden.
    ggml_tensor * build_mtp(ggml_context * ctx, ggml_cgraph * gf,
                            ggml_tensor * h, ggml_tensor * tok, ggml_tensor * pos, ggml_tensor * mask);
    const std::vector<float> & mtp_draft(int32_t token);
    void capture_main_hidden(ggml_cgraph * gf, int col);
    void decode_verify(const std::vector<int32_t> & toks);        // T-token forward -> vL/vH
    void decode_verify_cached(const std::vector<int32_t> & toks); // offload variant (decode_cached_batch)
    void generate_mtp(const std::vector<int32_t> & prompt, int max_new, int n_draft,
                      const std::function<bool(int32_t)> & on_token);

    // ---- Phase B v2 dynamic-cache decode (single token) ----
    void init_cache();
    void init_state_backup();
    void backup_states();
    void restore_states();
    const std::vector<float> & decode_cached_fast(int32_t token);
    ggml_tensor * build_router(ggml_context * ctx, ggml_cgraph * gf, int il,
                               ggml_tensor * ffn_in, ggml_tensor * & weights_out);
    ggml_tensor * build_moe_cached(ggml_context * ctx, ggml_cgraph * gf, int il,
                                   ggml_tensor * ffn_in, ggml_tensor * slot_g,
                                   ggml_tensor * slot_u, ggml_tensor * slot_d,
                                   ggml_tensor * weights);
    const std::vector<float> & decode_cached(int32_t token);
    // Batched prefill over the cache: process up to a pool-sized chunk of tokens
    // in one segmented forward (instead of token-by-token).
    void decode_cached_batch(const int32_t * toks, int n_tokens, bool want_logits,
                             bool verify = false);
};

void Runtime::Impl::init() {
    // Prefer a GPU device (CUDA/Metal/etc.) when requested and available.
    if (cfg.use_cuda) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                backend = ggml_backend_dev_init(dev, nullptr);
                if (backend) {
                    fprintf(stderr, "backend: GPU [%s] %s\n",
                            ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
                    break;
                }
            }
        }
        if (!backend) fprintf(stderr, "backend: no GPU device found, falling back to CPU\n");
    }
    if (!backend) {
        backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("failed to init CPU backend");
        int nth = cfg.n_threads > 0 ? cfg.n_threads : (int) std::thread::hardware_concurrency();
        if (nth <= 0) nth = 4;
        ggml_backend_cpu_set_n_threads(backend, nth);
        fprintf(stderr, "backend: CPU (%d threads)\n", nth);
    }

    // ---- Phase B: Expert weight offload via CPU backend + sched ----
    const bool use_expert_offload =
        cfg.vram_budget_mb > 0 && model.has_expert_tensors() && cfg.use_cuda;

    // Keep the MTP (nextn) block's experts VRAM-resident only when MTP is in use;
    // otherwise let them offload with the rest (saves VRAM). Must be set before
    // any load_weights_* call.
    model.set_keep_nextn_resident(cfg.use_mtp && model.hparams().has_mtp());

    if (use_expert_offload && cfg.experts_ssd) {
        // ---- SSD tier: experts stay on disk; non-expert weights -> GPU ----
        ssd_mode = true;
        model.load_weights_ssd(backend, weights_buf);
        reuse_graph = false;   // every token goes through the per-token cache path
        fprintf(stderr, "expert offload: ON (SSD tier, decode via VRAM cache; prefill token-by-token)\n");
    } else if (use_expert_offload) {
        // Create a CPU backend for expert weights.
        cpu_backend = ggml_backend_cpu_init();
        if (!cpu_backend) throw std::runtime_error("failed to init CPU backend for expert offload");
        int nth = cfg.n_threads > 0 ? cfg.n_threads : (int) std::thread::hardware_concurrency();
        if (nth <= 0) nth = 4;
        ggml_backend_cpu_set_n_threads(cpu_backend, nth);

        // Try to use pinned host memory for expert weights (faster GPU<->CPU transfers).
        ggml_backend_buffer_type_t cpu_buft = nullptr;
        ggml_backend_dev_t gpu_dev = ggml_backend_get_device(backend);
        if (gpu_dev) {
            cpu_buft = ggml_backend_dev_host_buffer_type(gpu_dev);
        }
        if (!cpu_buft) {
            cpu_buft = ggml_backend_get_default_buffer_type(cpu_backend);
        }

        // Load weights: non-expert → GPU, expert → CPU (pinned).
        model.load_weights_split(backend, cpu_buft, weights_buf, expert_cpu_bufs);

        // Create backend scheduler: GPU first (higher priority), CPU fallback.
        // The sched routes ops to GPU for GPU-backend tensors and CPU for CPU-backend tensors.
        ggml_backend_t   sched_be[2]   = { backend, cpu_backend };
        ggml_backend_buffer_type_t sched_bt[2] = {
            ggml_backend_get_default_buffer_type(backend),
            ggml_backend_get_default_buffer_type(cpu_backend),
        };
        sched = ggml_backend_sched_new(sched_be, sched_bt, 2, GRAPH_SIZE, false, false);
        if (!sched) throw std::runtime_error("failed to create backend scheduler");

        // Disable persistent reuse graph — sched alloc is incompatible with it.
        reuse_graph = false;

        fprintf(stderr, "expert offload: ON (prefill on CPU experts, decode via VRAM cache)\n");
    } else {
        weights_buf = model.load_weights(backend);
    }

    // fused flash attention on GPU (disable with QWEN_NO_FLASH)
    use_flash = cfg.use_cuda && getenv("QWEN_NO_FLASH") == nullptr;
    if (getenv("QWEN_NO_REUSE")) reuse_graph = false;

    const auto & hp = model.hparams();
    n_ctx = cfg.n_ctx;
    const int n_layer    = hp.n_layer;
    const int n_embd_gqa = hp.n_head_kv * hp.n_embd_head;

    const int conv_ch = hp.ssm_d_inner + 2 * hp.ssm_n_group * hp.ssm_d_state;
    const int S       = hp.ssm_d_state;
    const int H_v     = hp.ssm_dt_rank;

    ggml_init_params kp{};
    kp.mem_size   = (size_t) ggml_tensor_overhead() * n_layer * 4 + 4096;
    kp.no_alloc   = true;
    st_ctx = ggml_init(kp);

    k_cache.assign(n_layer, nullptr);
    v_cache.assign(n_layer, nullptr);
    conv_state.assign(n_layer, nullptr);
    ssm_state.assign(n_layer, nullptr);

    for (int il = 0; il < n_layer; ++il) {
        if (hp.is_recurrent(il)) {
            conv_state[il] = ggml_new_tensor_2d(st_ctx, GGML_TYPE_F32, hp.ssm_d_conv - 1, conv_ch);
            ssm_state[il]  = ggml_new_tensor_3d(st_ctx, GGML_TYPE_F32, S, S, H_v);
            ggml_set_name(conv_state[il], ("conv_" + std::to_string(il)).c_str());
            ggml_set_name(ssm_state[il],  ("ssm_"  + std::to_string(il)).c_str());
        } else {
            k_cache[il] = ggml_new_tensor_2d(st_ctx, GGML_TYPE_F32, n_embd_gqa, n_ctx);
            v_cache[il] = ggml_new_tensor_2d(st_ctx, GGML_TYPE_F32, n_embd_gqa, n_ctx);
            ggml_set_name(k_cache[il], ("k_" + std::to_string(il)).c_str());
            ggml_set_name(v_cache[il], ("v_" + std::to_string(il)).c_str());
        }
    }
    st_buf = ggml_backend_alloc_ctx_tensors(st_ctx, backend);
    if (!st_buf) throw std::runtime_error("failed to alloc state buffer");

    if (!sched) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    if (sched || ssd_mode) {
        init_cache();   // dynamic per-expert VRAM cache for the decode hot path
    }
    // MTP: main forward exposes its last hidden, drafted by the nextn block
    capture_hidden = model.hparams().has_mtp();
    if (capture_hidden) {
        mtp_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        mtp_hidden.assign(model.hparams().n_embd, 0.0f);
        init_state_backup();   // MTP reject needs GDN rollback even without the cache
    }
    zero_states();
}

// Allocate the VRAM slot pools and the persistent per-layer carry tensors.
void Runtime::Impl::init_cache() {
    const auto & hp = model.hparams();
    const int n_embd = hp.n_embd;
    const int n_used = hp.n_expert_used;

    // VRAM left for the expert slot pools after non-expert weights + headroom.
    const size_t budget   = cfg.vram_budget_mb * 1024ull * 1024ull;
    const size_t gpu_w    = ggml_backend_buffer_get_size(weights_buf);
    const size_t headroom = 1024ull * 1024ull * 1024ull;   // KV cache + compute buffers
    size_t avail = (budget > gpu_w + headroom) ? budget - gpu_w - headroom : 0;

    // Only the main stack's experts are offloaded; the trailing MTP (nextn) block
    // stays fully VRAM-resident, so the cache covers n_main() layers (not n_layer).
    ecache = std::make_unique<ExpertCache>(backend, model, hp.n_main(), hp.n_expert, n_used, avail, ssd_mode);

    // persistent carry tensors (bridge per-layer graph segments) + fast-path
    // in-graph remap table (g2s_all) and selection readback (sel_all).
    ggml_init_params cp{};
    cp.mem_size = ggml_tensor_overhead() * 16 + 256;
    cp.no_alloc = true;
    cctx = ggml_init(cp);
    p_h       = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, n_embd);
    p_ffn_in  = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, n_embd);
    p_resid   = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, n_embd);
    p_weights = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, n_used);
    p_ffn_in2 = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, n_embd);
    p_resid2  = ggml_new_tensor_1d(cctx, GGML_TYPE_F32, n_embd);
    p_weights2= ggml_new_tensor_1d(cctx, GGML_TYPE_F32, n_used);
    p_slot_g  = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, n_used, 1);
    p_slot_u  = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, n_used, 1);
    p_slot_d  = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, n_used, 1);
    g2s_all   = ggml_new_tensor_3d(cctx, GGML_TYPE_I32, 1, hp.n_expert, 3 * hp.n_layer);
    sel_all   = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, n_used, hp.n_layer);
    ggml_set_name(p_h, "carry.h");
    ggml_set_name(p_ffn_in, "carry.ffn_in");
    ggml_set_name(p_resid, "carry.resid");
    ggml_set_name(p_weights, "carry.weights");
    ggml_set_name(p_ffn_in2, "carry.ffn_in2");
    ggml_set_name(p_resid2, "carry.resid2");
    ggml_set_name(p_weights2, "carry.weights2");
    ggml_set_name(p_slot_g, "carry.slot_g");
    ggml_set_name(p_slot_u, "carry.slot_u");
    ggml_set_name(p_slot_d, "carry.slot_d");
    ggml_set_name(g2s_all, "carry.g2s");
    ggml_set_name(sel_all, "carry.sel");
    cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);
    if (!cbuf) throw std::runtime_error("init_cache: failed to alloc carry buffer");

    cache_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    f_galloc     = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    g2s_host.assign((size_t) 3 * hp.n_layer * hp.n_expert, 0);
    sel_host.assign((size_t) n_used * hp.n_layer, 0);

    if (getenv("QWEN_FASTCACHE")) cache_fast_enabled = true;   // experimental single-graph path

    init_state_backup();   // GDN rollback buffers (speculative miss / MTP reject)

    // warm restart: pre-fill VRAM slots from a saved hot-expert profile
    if (!cfg.cache_profile.empty()) {
        size_t n = ecache->load_prefetch(cfg.cache_profile);
        if (n > 0)
            fprintf(stderr, "expert cache: prefetched %zu experts from profile '%s'\n",
                    n, cfg.cache_profile.c_str());
    }
}

void Runtime::Impl::zero_states() {
    // GDN recurrent states must start at zero each session. KV cache is also
    // zeroed so the persistent decode graph (which attends over the full n_ctx)
    // never sees NaN/garbage at not-yet-written, masked-out positions.
    std::vector<float> zeros;
    auto zero = [&](ggml_tensor * t) {
        if (!t) return;
        const size_t n = ggml_nbytes(t);
        if (zeros.size() * sizeof(float) < n) zeros.assign(n / sizeof(float), 0.0f);
        ggml_backend_tensor_set(t, zeros.data(), 0, n);
    };
    for (auto * t : conv_state) zero(t);
    for (auto * t : ssm_state)  zero(t);
    for (auto * t : k_cache)    zero(t);
    for (auto * t : v_cache)    zero(t);
}

// ---- gated attention (shared by qwen3 plain and qwen35 gated paths) ----
ggml_tensor * Runtime::Impl::build_attn(ggml_context * ctx, ggml_cgraph * gf, int il,
        ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V,
        ggml_tensor * mask, int n_tokens, int n_kv) {
    const auto & hp = model.hparams();
    const int n_head      = hp.n_head;
    const int n_head_kv   = hp.n_head_kv;
    const int n_embd_head = hp.n_embd_head;
    const int n_embd_gqa  = n_head_kv * n_embd_head;
    const float kq_scale  = 1.0f / sqrtf((float) n_embd_head);

    // store K, V into cache
    ggml_tensor * Kflat = ggml_reshape_2d(ctx, K, n_embd_gqa, n_tokens);
    ggml_tensor * Vflat = ggml_reshape_2d(ctx, V, n_embd_gqa, n_tokens);
    if (persistent) {
        // dynamic write position via index input -> graph stays identical each step
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache[il], Kflat, d_kvidx));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache[il], Vflat, d_kvidx));
    } else {
        ggml_tensor * k_dst = ggml_view_2d(ctx, k_cache[il], n_embd_gqa, n_tokens,
                                           k_cache[il]->nb[1], (size_t) n_past * k_cache[il]->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kflat, k_dst));
        ggml_tensor * v_dst = ggml_view_2d(ctx, v_cache[il], n_embd_gqa, n_tokens,
                                           v_cache[il]->nb[1], (size_t) n_past * v_cache[il]->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Vflat, v_dst));
    }

    // K, V views from cache: [head_dim, n_kv, n_head_kv]
    ggml_tensor * Kc = ggml_view_2d(ctx, k_cache[il], n_embd_gqa, n_kv, k_cache[il]->nb[1], 0);
    Kc = ggml_permute(ctx, ggml_reshape_3d(ctx, Kc, n_embd_head, n_head_kv, n_kv), 0, 2, 1, 3);
    ggml_tensor * Vc = ggml_view_2d(ctx, v_cache[il], n_embd_gqa, n_kv, v_cache[il]->nb[1], 0);
    Vc = ggml_permute(ctx, ggml_reshape_3d(ctx, Vc, n_embd_head, n_head_kv, n_kv), 0, 2, 1, 3);

    if (use_flash) {
        // fused: q=[head,n_tokens,n_head], k/v=[head,n_kv,n_head_kv], mask F16
        ggml_tensor * Qf = ggml_permute(ctx, Q, 0, 2, 1, 3);              // [head, n_tokens, n_head]
        ggml_tensor * r = ggml_flash_attn_ext(ctx, Qf, Kc, Vc, mask, kq_scale, 0.0f, 0.0f);
        return ggml_reshape_2d(ctx, r, n_embd_head * n_head, n_tokens);   // r: [head, n_head, n_tokens]
    }

    ggml_tensor * Qp = ggml_permute(ctx, Q, 0, 2, 1, 3);
    ggml_tensor * Kcc = ggml_cont(ctx, Kc);
    ggml_tensor * kq = ggml_mul_mat(ctx, Kcc, Qp);
    kq = ggml_soft_max_ext(ctx, kq, mask, kq_scale, 0.0f);

    // V for the manual path: [n_kv, head, head_kv]
    ggml_tensor * Vslice = ggml_view_2d(ctx, v_cache[il], n_embd_gqa, n_kv, v_cache[il]->nb[1], 0);
    ggml_tensor * Vt = ggml_cont(ctx, ggml_transpose(ctx, Vslice));   // [n_kv, n_embd_gqa]
    ggml_tensor * Vm = ggml_reshape_3d(ctx, Vt, n_kv, n_embd_head, n_head_kv);
    ggml_tensor * kqv = ggml_mul_mat(ctx, Vm, kq);
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    kqv = ggml_cont_2d(ctx, kqv, n_embd_head * n_head, n_tokens);
    return kqv;  // [n_embd_head*n_head, n_tokens]
}

// ---- Gated DeltaNet layer (qwen35 / qwen3next) ----
ggml_tensor * Runtime::Impl::build_gdn(ggml_context * ctx, ggml_cgraph * gf, int il,
        ggml_tensor * x, int n_tokens) {
    const auto & hp = model.hparams();
    const int S        = hp.ssm_d_state;         // 128
    const int H_k      = hp.ssm_n_group;         // 16
    const int H_v      = hp.ssm_dt_rank;         // 16
    const int key_dim  = S * H_k;                // 2048
    const int conv_ch  = hp.ssm_d_inner + 2 * H_k * S; // 6144
    const float eps    = hp.rms_eps;
    const size_t el    = sizeof(float);

    // projections
    ggml_tensor * qkv_mixed = ggml_mul_mat(ctx, W("blk.%d.attn_qkv.weight", il), x);   // [conv_ch, n_tokens]
    ggml_tensor * z         = ggml_mul_mat(ctx, W("blk.%d.attn_gate.weight", il), x);  // [d_inner, n_tokens]

    ggml_tensor * beta = ggml_mul_mat(ctx, W("blk.%d.ssm_beta.weight", il), x);        // [H_v, n_tokens]
    beta = ggml_sigmoid(ctx, beta);
    beta = ggml_reshape_4d(ctx, beta, 1, H_v, n_tokens, 1);

    ggml_tensor * alpha = ggml_mul_mat(ctx, W("blk.%d.ssm_alpha.weight", il), x);      // [H_v, n_tokens]
    alpha = ggml_add(ctx, alpha, W("blk.%d.ssm_dt.bias", il));
    alpha = ggml_softplus(ctx, alpha);
    ggml_tensor * g = ggml_mul(ctx, alpha, W("blk.%d.ssm_a", il));                     // [H_v, n_tokens]
    g = ggml_reshape_4d(ctx, g, 1, H_v, n_tokens, 1);

    // causal conv1d with state
    ggml_tensor * conv_kernel = W("blk.%d.ssm_conv1d.weight", il);   // [d_conv, conv_ch]
    ggml_tensor * cs = ggml_reshape_3d(ctx, conv_state[il], hp.ssm_d_conv - 1, conv_ch, 1);
    ggml_tensor * qkv_t = ggml_transpose(ctx, qkv_mixed);           // [n_tokens, conv_ch]
    ggml_tensor * conv_input = ggml_concat(ctx, cs, qkv_t, 0);      // [d_conv-1+n_tokens, conv_ch, 1]

    // write back last (d_conv-1) timesteps to conv_state
    ggml_tensor * cs_last = ggml_view_3d(ctx, conv_input, hp.ssm_d_conv - 1, conv_ch, 1,
            conv_input->nb[1], conv_input->nb[2],
            ggml_row_size(conv_input->type, n_tokens));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, cs_last, conv_state[il]));

    ggml_tensor * conv_out = ggml_ssm_conv(ctx, conv_input, conv_kernel);  // [conv_ch, n_tokens, 1]
    conv_out = ggml_silu(ctx, conv_out);

    const int64_t nb1q = ggml_row_size(conv_out->type, conv_ch);
    ggml_tensor * q = ggml_view_4d(ctx, conv_out, S, H_k, n_tokens, 1,
            ggml_row_size(conv_out->type, S), nb1q, nb1q * n_tokens, 0);
    ggml_tensor * k = ggml_view_4d(ctx, conv_out, S, H_k, n_tokens, 1,
            ggml_row_size(conv_out->type, S), nb1q, nb1q * n_tokens, (size_t) key_dim * el);
    ggml_tensor * v = ggml_view_4d(ctx, conv_out, S, H_v, n_tokens, 1,
            ggml_row_size(conv_out->type, S), nb1q, nb1q * n_tokens, (size_t) 2 * key_dim * el);

    q = ggml_l2_norm(ctx, q, eps);
    k = ggml_l2_norm(ctx, k, eps);
    q = ggml_cont(ctx, q);
    k = ggml_cont(ctx, k);
    v = ggml_cont(ctx, v);

    // fused gated delta net: returns output + new state packed in one tensor
    ggml_tensor * s_in  = ggml_reshape_3d(ctx, ssm_state[il], S * S * H_v, 1, 1);
    ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, s_in);

    ggml_tensor * output = ggml_view_4d(ctx, result, S, H_v, n_tokens, 1,
            ggml_row_size(result->type, S),
            ggml_row_size(result->type, S * H_v),
            ggml_row_size(result->type, S * H_v * n_tokens), 0);

    ggml_tensor * new_state = ggml_view_4d(ctx, result, S, S, H_v, 1,
            ggml_row_size(result->type, S),
            ggml_row_size(result->type, S * S),
            ggml_row_size(result->type, S * S * H_v),
            ggml_row_size(result->type, S * H_v * n_tokens));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state,
            ggml_reshape_3d(ctx, ssm_state[il], S, S, H_v)));

    // gated RMSNorm with z: rms_norm(output)*ssm_norm * silu(z)
    output = ggml_cont(ctx, output);
    output = ggml_rms_norm(ctx, output, eps);
    output = ggml_mul(ctx, output, W("blk.%d.ssm_norm.weight", il));   // broadcast [S]
    ggml_tensor * zr = ggml_reshape_4d(ctx, z, S, H_v, n_tokens, 1);
    output = ggml_mul(ctx, output, ggml_silu(ctx, zr));

    output = ggml_reshape_2d(ctx, output, S * H_v, n_tokens);
    ggml_tensor * cur = ggml_mul_mat(ctx, W("blk.%d.ssm_out.weight", il), output);  // [n_embd, n_tokens]
    return cur;
}

// ---- MoE FFN (qwen3moe / qwen35moe): softmax gating, top-k, normalized weights ----
ggml_tensor * Runtime::Impl::build_moe(ggml_context * ctx, ggml_cgraph * gf, int il,
        ggml_tensor * x, int n_tokens) {
    const auto & hp = model.hparams();
    const int n_embd  = hp.n_embd;
    const int n_exp   = hp.n_expert;
    const int n_used  = hp.n_expert_used;

    ggml_tensor * logits = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp.weight", il), x); // [n_exp, n_tokens]
    ggml_tensor * probs  = ggml_soft_max(ctx, logits);

    ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, n_used);   // [n_used, n_tokens] i32

    ggml_tensor * probs3 = ggml_reshape_3d(ctx, probs, 1, n_exp, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx, probs3, selected);      // [1, n_used, n_tokens]

    // normalize weights over selected experts
    weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);
    ggml_tensor * wsum = ggml_sum_rows(ctx, weights);                  // [1, n_tokens]
    wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, wsum);
    weights = ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);
    if (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f)
        weights = ggml_scale(ctx, weights, hp.expert_weights_scale);

    // expand early so the CUDA top-k-moe path matches (mirrors llama.cpp)
    ggml_build_forward_expand(gf, weights);

    ggml_tensor * moe_out;
    if (cache_fast_build) {
        // ---- optimistic VRAM-cache experts with in-graph slot remap ----
        // Record this layer's selection so the host can verify residency after
        // the (speculative) single-graph run.
        ggml_tensor * sel_col = ggml_view_2d(ctx, sel_all, n_used, 1,
                                             sel_all->nb[1], (size_t) il * sel_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, selected, sel_col));

        // logical expert id -> physical slot id, gathered in-graph from the
        // per-(role,layer) remap table filled from host before each run.
        auto remap = [&](int role) {
            const int idx = il * 3 + role;
            ggml_tensor * g2s = ggml_view_2d(ctx, g2s_all, 1, n_exp,
                                             g2s_all->nb[1], (size_t) idx * g2s_all->nb[2]);
            ggml_tensor * s = ggml_get_rows(ctx, g2s, selected);  // [1, n_used, 1] i32
            return ggml_reshape_2d(ctx, s, n_used, 1);
        };
        ggml_tensor * slot_g = remap(ExpertCache::GATE);
        ggml_tensor * slot_u = remap(ExpertCache::UP);
        ggml_tensor * slot_d = remap(ExpertCache::DOWN);

        ggml_tensor * x3   = ggml_reshape_3d(ctx, x, n_embd, 1, 1);
        ggml_tensor * up   = ggml_mul_mat_id(ctx, ecache->up(il),   x3,  slot_u);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, ecache->gate(il), x3,  slot_g);
        ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, ecache->down(il), act, slot_d);
        ggml_tensor * et = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, experts, n_embd, n_used)));
        ggml_tensor * w  = ggml_reshape_2d(ctx, weights, n_used, 1);
        moe_out = ggml_mul_mat(ctx, et, w);   // [n_embd, 1]
    } else {
        ggml_tensor * x3 = ggml_reshape_3d(ctx, x, n_embd, 1, n_tokens);

        ggml_tensor * up   = ggml_mul_mat_id(ctx, W("blk.%d.ffn_up_exps.weight",   il), x3, selected);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, W("blk.%d.ffn_gate_exps.weight", il), x3, selected);
        ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);            // silu(gate)*up [ff_exp, n_used, n_tokens]
        ggml_tensor * experts = ggml_mul_mat_id(ctx, W("blk.%d.ffn_down_exps.weight", il), act, selected); // [n_embd, n_used, n_tokens]

        if (n_tokens == 1) {
            // weighted sum of the n_used experts as a single GEMV:
            //   moe_out[e] = sum_k experts[e,k] * weights[k]
            ggml_tensor * et = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, experts, n_embd, n_used))); // [n_used, n_embd]
            ggml_tensor * w  = ggml_reshape_2d(ctx, weights, n_used, 1);
            moe_out = ggml_mul_mat(ctx, et, w);   // [n_embd, 1]
        } else {
            experts = ggml_mul(ctx, experts, weights);
            ggml_build_forward_expand(gf, experts);
            ggml_tensor * cur_experts[256] = { nullptr };
            for (int i = 0; i < n_used; ++i) {
                cur_experts[i] = ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2], (size_t) i * experts->nb[1]);
                ggml_build_forward_expand(gf, cur_experts[i]);
            }
            moe_out = cur_experts[0];
            for (int i = 1; i < n_used; ++i) {
                moe_out = ggml_add(ctx, moe_out, cur_experts[i]);
                ggml_build_forward_expand(gf, moe_out);
            }
            if (n_used == 1) moe_out = ggml_cont(ctx, moe_out);
        }
    }

    // shared expert (qwen35moe / qwen3next): gated SwiGLU added to the MoE output
    if (ggml_tensor * up_sh = Wopt("blk.%d.ffn_up_shexp.weight", il)) {
        ggml_tensor * g  = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_shexp.weight", il), x);
        ggml_tensor * u  = ggml_mul_mat(ctx, up_sh, x);
        ggml_tensor * sh = ggml_mul_mat(ctx, W("blk.%d.ffn_down_shexp.weight", il),
                                        ggml_mul(ctx, ggml_silu(ctx, g), u));
        ggml_tensor * sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp_shexp.weight", il), x));
        sh = ggml_mul(ctx, sh, sg);
        moe_out = ggml_add(ctx, moe_out, sh);
    }
    return moe_out;
}

ggml_cgraph * Runtime::Impl::build_graph(ggml_context * ctx, int n_tokens, int n_kv) {
    const auto & hp = model.hparams();
    const int n_embd      = hp.n_embd;
    const int n_head      = hp.n_head;
    const int n_head_kv   = hp.n_head_kv;
    const int n_embd_head = hp.n_embd_head;
    const float eps       = hp.rms_eps;
    const bool  gated     = hp.has_gdn;   // qwen35 attention layers use gated Q + post norm
    const char * post_norm_name = "blk.%d.post_attention_norm.weight";

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens); ggml_set_name(inp_tokens, "inp_tokens");
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_pos); ggml_set_name(inp_pos, "inp_pos");
    ggml_tensor * inp_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_set_input(inp_mask); ggml_set_name(inp_mask, "inp_mask");

    if (persistent) {
        d_kvidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        ggml_set_input(d_kvidx); ggml_set_name(d_kvidx, "inp_kvidx");
    }

    ggml_tensor * cur;
    ggml_tensor * inpL = ggml_get_rows(ctx, model.tok_embd_rows(), inp_tokens);

    for (int il = 0; il < (int) hp.n_main(); ++il) {
        ggml_tensor * inpSA = inpL;

        cur = ggml_rms_norm(ctx, inpL, eps);
        cur = ggml_mul(ctx, cur, W("blk.%d.attn_norm.weight", il));

        if (hp.is_recurrent(il)) {
            cur = build_gdn(ctx, gf, il, cur, n_tokens);
        } else {
            // attention (plain for qwen3, gated for qwen35)
            ggml_tensor * Q, * K, * V, * gate = nullptr;
            if (gated) {
                ggml_tensor * Qf = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur); // [2*hd*nh, T]
                const size_t es = ggml_element_size(Qf);
                Q = ggml_view_3d(ctx, Qf, n_embd_head, n_head, n_tokens,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head, 0);
                gate = ggml_view_3d(ctx, Qf, n_embd_head, n_head, n_tokens,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head, es * n_embd_head);
                gate = ggml_cont_2d(ctx, gate, n_embd_head * n_head, n_tokens);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, n_tokens);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, n_tokens);
            } else {
                Q = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head,    n_tokens);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, n_tokens);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, n_tokens);
            }

            Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, eps), W("blk.%d.attn_q_norm.weight", il));
            K = ggml_mul(ctx, ggml_rms_norm(ctx, K, eps), W("blk.%d.attn_k_norm.weight", il));

            Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX,
                              0, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(ctx, K, inp_pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX,
                              0, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor * att = build_attn(ctx, gf, il, Q, K, V, inp_mask, n_tokens, n_kv);
            if (gated) att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate));
            cur = ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", il), att);
        }

        cur = ggml_add(ctx, cur, inpSA);

        // FFN with (qwen35) post-attention norm placement
        ggml_tensor * ffn_res = cur;
        ggml_tensor * ffn_in;
        if (gated) {
            ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), W(post_norm_name, il));
        } else {
            ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), W("blk.%d.ffn_norm.weight", il));
            ffn_res = cur;  // same residual for qwen3 (norm of cur, add cur)
        }

        ggml_tensor * ff;
        if (hp.is_moe()) {
            ff = build_moe(ctx, gf, il, ffn_in, n_tokens);
        } else {
            ggml_tensor * gt = ggml_mul_mat(ctx, W("blk.%d.ffn_gate.weight", il), ffn_in);
            ggml_tensor * up = ggml_mul_mat(ctx, W("blk.%d.ffn_up.weight",   il), ffn_in);
            ff = ggml_mul_mat(ctx, W("blk.%d.ffn_down.weight", il), ggml_mul(ctx, ggml_silu(ctx, gt), up));
        }

        cur = ggml_add(ctx, ff, ffn_res);
        inpL = cur;
    }

    // expose the main stack's last hidden (pre-output-norm) for the MTP module
    if (capture_hidden) {
        ggml_set_name(inpL, "main_hidden");
        ggml_set_output(inpL);
        ggml_build_forward_expand(gf, inpL);
    }

    cur = ggml_rms_norm(ctx, inpL, eps);
    cur = ggml_mul(ctx, cur, model.tensor("output_norm.weight"));

    ggml_tensor * output_w = model.tensor("output.weight");
    if (!output_w) output_w = model.tensor("token_embd.weight");
    cur = ggml_mul_mat(ctx, output_w, cur);
    ggml_set_name(cur, "logits");

    ggml_build_forward_expand(gf, cur);
    return gf;
}

// Read column `col` of the graph's "main_hidden" tensor into mtp_hidden (host).
void Runtime::Impl::capture_main_hidden(ggml_cgraph * gf, int col) {
    if (!capture_hidden) return;
    ggml_tensor * h = ggml_graph_get_tensor(gf, "main_hidden");
    if (!h) return;
    const int n_embd = (int) h->ne[0];
    mtp_hidden.resize(n_embd);
    ggml_backend_tensor_get(h, mtp_hidden.data(), (size_t) col * h->nb[1], n_embd * sizeof(float));
}

// MTP (nextn) block: combine the main hidden with the embedding of `tok`, run one
// transformer block (its own KV at layer n_main), then the shared head -> logits
// for the token *after* `tok`.
ggml_tensor * Runtime::Impl::build_mtp(ggml_context * ctx, ggml_cgraph * gf,
        ggml_tensor * h, ggml_tensor * tok, ggml_tensor * pos, ggml_tensor * mask) {
    const auto & hp = model.hparams();
    const int L           = (int) hp.n_main();      // MTP block index
    const int n_embd      = hp.n_embd;
    const int n_head      = hp.n_head;
    const int n_head_kv   = hp.n_head_kv;
    const int n_embd_head = hp.n_embd_head;
    const float eps       = hp.rms_eps;
    const int n_kv        = mtp_past + 1;

    // h' = eh_proj( concat( hnorm(h), enorm(emb(tok)) ) )
    ggml_tensor * emb = ggml_get_rows(ctx, model.tok_embd_rows(), tok);            // [n_embd,1]
    ggml_tensor * e   = ggml_mul(ctx, ggml_rms_norm(ctx, emb, eps), W("blk.%d.nextn.enorm.weight", L));
    ggml_tensor * h2  = ggml_reshape_2d(ctx, h, n_embd, 1);
    ggml_tensor * hn  = ggml_mul(ctx, ggml_rms_norm(ctx, h2, eps), W("blk.%d.nextn.hnorm.weight", L));
    // eh_proj expects [ enorm(emb) ; hnorm(hidden) ]  (embedding first)
    ggml_tensor * combined = ggml_concat(ctx, e, hn, 0);                            // [2*n_embd,1]
    ggml_tensor * cur = ggml_mul_mat(ctx, W("blk.%d.nextn.eh_proj.weight", L), combined); // [n_embd,1]

    // transformer block (gated attention + dense FFN), gated like qwen35
    ggml_tensor * inpSA = cur;
    ggml_tensor * x = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), W("blk.%d.attn_norm.weight", L));

    ggml_tensor * Qf = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", L), x);
    const size_t es = ggml_element_size(Qf);
    ggml_tensor * Q = ggml_view_3d(ctx, Qf, n_embd_head, n_head, 1,
            es * n_embd_head * 2, es * n_embd_head * 2 * n_head, 0);
    ggml_tensor * gate_t = ggml_view_3d(ctx, Qf, n_embd_head, n_head, 1,
            es * n_embd_head * 2, es * n_embd_head * 2 * n_head, es * n_embd_head);
    gate_t = ggml_cont_2d(ctx, gate_t, n_embd_head * n_head, 1);
    ggml_tensor * K = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", L), x), n_embd_head, n_head_kv, 1);
    ggml_tensor * V = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", L), x), n_embd_head, n_head_kv, 1);
    Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, eps), W("blk.%d.attn_q_norm.weight", L));
    K = ggml_mul(ctx, ggml_rms_norm(ctx, K, eps), W("blk.%d.attn_k_norm.weight", L));
    Q = ggml_rope_ext(ctx, Q, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX, 0, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX, 0, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * att = build_attn(ctx, gf, L, Q, K, V, mask, 1, n_kv);   // KV write uses n_past (= mtp_past, set by caller)
    att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate_t));
    cur = ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", L), att);
    cur = ggml_add(ctx, cur, inpSA);

    ggml_tensor * ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), W("blk.%d.post_attention_norm.weight", L));
    ggml_tensor * ff;
    if (hp.is_moe()) {
        // MoE nextn block (e.g. Qwen3.6-35B-A3B-MTP): experts are VRAM-resident.
        ff = build_moe(ctx, gf, L, ffn_in, 1);
    } else {
        ggml_tensor * gt = ggml_mul_mat(ctx, W("blk.%d.ffn_gate.weight", L), ffn_in);
        ggml_tensor * up = ggml_mul_mat(ctx, W("blk.%d.ffn_up.weight",   L), ffn_in);
        ff = ggml_mul_mat(ctx, W("blk.%d.ffn_down.weight", L), ggml_mul(ctx, ggml_silu(ctx, gt), up));
    }
    cur = ggml_add(ctx, ff, cur);

    // expose the block output hidden so drafts can be chained (use it as the next
    // step's "main hidden" proxy when drafting 2+ tokens ahead).
    ggml_set_name(cur, "mtp_blk_hidden");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    ggml_tensor * fin = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), W("blk.%d.nextn.shared_head_norm.weight", L));
    ggml_tensor * lm = model.tensor("output.weight");
    if (!lm) lm = model.tensor("token_embd.weight");
    ggml_tensor * logits = ggml_mul_mat(ctx, lm, fin);
    ggml_set_name(logits, "mtp_logits");
    ggml_build_forward_expand(gf, logits);
    return logits;
}

// Run the MTP block once: draft the token after `token`, advancing MTP KV.
// The nextn block is kept fully VRAM-resident (even in expert-offload mode), so
// this single-graph path serves both dense and MoE MTP blocks.
const std::vector<float> & Runtime::Impl::mtp_draft(int32_t token) {
    const auto & hp = model.hparams();
    const int n_embd = hp.n_embd;
    const int n_kv   = mtp_past + 1;

    ggml_init_params gp{};
    gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
    gp.no_alloc = true;
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);

    ggml_tensor * h_in  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd); ggml_set_input(h_in);
    ggml_tensor * t_in  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);      ggml_set_input(t_in);
    ggml_tensor * p_in  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);      ggml_set_input(p_in);
    ggml_tensor * m_in  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, 1); ggml_set_input(m_in);

    const int saved = n_past;
    n_past = mtp_past;                 // build_attn writes MTP KV at this position
    ggml_tensor * logits_t = build_mtp(ctx, gf, h_in, t_in, p_in, m_in);
    n_past = saved;

    if (!ggml_gallocr_alloc_graph(mtp_galloc, gf))
        throw std::runtime_error("mtp_draft: gallocr alloc failed");

    ggml_backend_tensor_set(h_in, mtp_hidden.data(), 0, n_embd * sizeof(float));
    ggml_backend_tensor_set(t_in, &token, 0, sizeof(int32_t));
    int32_t pos = mtp_past; ggml_backend_tensor_set(p_in, &pos, 0, sizeof(int32_t));
    std::vector<ggml_fp16_t> mask(n_kv, ggml_fp32_to_fp16(0.0f));   // all past positions visible
    ggml_backend_tensor_set(m_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("mtp_draft: compute failed");

    const int n_vocab = (int) logits_t->ne[0];
    mtp_logits.resize(n_vocab);
    ggml_backend_tensor_get(logits_t, mtp_logits.data(), 0, n_vocab * sizeof(float));
    // capture the block output hidden for chaining the next draft
    if (ggml_tensor * bh = ggml_graph_get_tensor(gf, "mtp_blk_hidden")) {
        mtp_block_hidden.resize(n_embd);
        ggml_backend_tensor_get(bh, mtp_block_hidden.data(), 0, n_embd * sizeof(float));
    }
    ggml_free(ctx);
    mtp_past += 1;
    return mtp_logits;
}
// Offload verify: run the 2-token main forward through the VRAM expert cache.
// Uses the batched cache path when the pools can hold both tokens' experts at
// once; otherwise falls back to two single-token decodes (still correct, just
// no shared-fetch amortization).
void Runtime::Impl::decode_verify_cached(const std::vector<int32_t> & toks) {
    const int T = (int) toks.size();
    const int n_used = model.hparams().n_expert_used;
    if (ecache->min_slots() >= T * n_used) {
        decode_cached_batch(toks.data(), T, /*want_logits=*/false, /*verify=*/true);
    } else {
        // pools too small to hold all tokens' experts at once: token-by-token.
        vL.assign(T, {}); vH.assign(T, {});
        for (int i = 0; i < T; ++i) { vL[i] = decode_cached(toks[i]); vH[i] = mtp_hidden; }
    }
}

// Verify forward: run the main model on `toks` (T tokens) at the current position,
// exposing per-position logits (vL[i]) and main hidden (vH[i]). Advances n_past+=T.
void Runtime::Impl::decode_verify(const std::vector<int32_t> & toks) {
    if (ecache) { decode_verify_cached(toks); return; }   // expert-offload mode

    const int n_tokens = (int) toks.size();
    const int n_kv = n_past + n_tokens;

    ggml_init_params gp{};
    gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
    gp.no_alloc = true;
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = build_graph(ctx, n_tokens, n_kv);
    if (!ggml_gallocr_alloc_graph(galloc, gf))
        throw std::runtime_error("decode_verify: gallocr alloc failed");

    ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    ggml_tensor * inp_pos    = ggml_graph_get_tensor(gf, "inp_pos");
    ggml_tensor * inp_mask   = ggml_graph_get_tensor(gf, "inp_mask");
    ggml_backend_tensor_set(inp_tokens, toks.data(), 0, n_tokens * sizeof(int32_t));
    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = n_past + i;
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, n_tokens * sizeof(int32_t));
    std::vector<ggml_fp16_t> mask((size_t) n_kv * n_tokens);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n_tokens; ++i) {
        const int abs_i = n_past + i;
        for (int j = 0; j < n_kv; ++j) mask[(size_t) i * n_kv + j] = (j <= abs_i) ? z : ninf;
    }
    ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("decode_verify: compute failed");

    ggml_tensor * lt = ggml_graph_get_tensor(gf, "logits");
    ggml_tensor * h  = ggml_graph_get_tensor(gf, "main_hidden");
    const int n_vocab = (int) lt->ne[0];
    const int n_embd  = (int) h->ne[0];
    vL.assign(n_tokens, std::vector<float>(n_vocab));
    vH.assign(n_tokens, std::vector<float>(n_embd));
    for (int i = 0; i < n_tokens; ++i) {
        ggml_backend_tensor_get(lt, vL[i].data(), (size_t) i * lt->nb[1], n_vocab * sizeof(float));
        ggml_backend_tensor_get(h,  vH[i].data(), (size_t) i * h->nb[1],  n_embd  * sizeof(float));
    }

    n_past += n_tokens;
    ggml_free(ctx);
}

// MTP self-speculative greedy decode. Drafts `n_draft` tokens by chaining the
// single nextn block (each draft feeds the block's own hidden into the next),
// verifies them with one (n_draft+1)-token main forward, and accepts the longest
// matching prefix. n_draft=1 reduces exactly to the 2-token verify.
void Runtime::Impl::generate_mtp(const std::vector<int32_t> & prompt, int max_new, int n_draft,
                                 const std::function<bool(int32_t)> & on_token) {
    const auto & hp = model.hparams();
    const bool gdn = hp.has_gdn;
    const int  K   = n_draft < 1 ? 1 : n_draft;
    static const bool no_accept = getenv("QWEN_MTP_NOACCEPT") != nullptr;
    auto argmax = [](const std::vector<float> & v) {
        int b = 0; for (int i = 1; i < (int) v.size(); ++i) if (v[i] > v[b]) b = i; return b;
    };

    // prefill token-by-token: builds main KV + MTP KV (so drafts have history)
    std::vector<float> mlog;
    const int P = (int) prompt.size();
    for (int i = 0; i < P; ++i) {
        mlog = decode({ prompt[i] });
        if (i + 1 < P) mtp_draft(prompt[i + 1]);
    }
    int32_t x = argmax(mlog);            // first generated token
    // invariant at loop top: n_past = pos(x), mtp_past = pos(x)-1, mtp_hidden = h_{pos(x)-1}
    int generated = 0;
    long steps = 0, draft_forwards = 0, accepted_drafts = 0;

    while (generated < max_new) {
        if (!on_token(x)) break;
        if (++generated >= max_new) break;

        const int p   = n_past;            // x lands at main position p
        const int mp0 = mtp_past;          // = p-1

        // ---- draft K tokens by chaining the MTP block ----
        std::vector<int32_t> drafts; drafts.reserve(K);
        {
            int32_t t = x;                 // first draft uses the true main hidden in mtp_hidden
            for (int j = 0; j < K; ++j) {
                int32_t dj = argmax(mtp_draft(t));   // writes MTP KV, advances mtp_past, sets mtp_block_hidden
                drafts.push_back(dj);
                ++draft_forwards;
                if (j + 1 < K) { mtp_hidden = mtp_block_hidden; t = dj; }   // chain on the block hidden
            }
        }

        // ---- verify [x, d_1..d_K] in one (K+1)-token main forward ----
        std::vector<int32_t> vtoks; vtoks.reserve(K + 1);
        vtoks.push_back(x);
        for (int j = 0; j < K; ++j) vtoks.push_back(drafts[j]);
        if (gdn) backup_states();
        decode_verify(vtoks);              // fills vL[0..K], vH[0..K]; n_past += K+1
        ++steps;

        // ---- accept the longest matching draft prefix ----
        int a = 0;
        if (!no_accept) while (a < K && argmax(vL[a]) == drafts[a]) ++a;
        accepted_drafts += a;
        const int32_t x_new = argmax(vL[a]);   // correction (or bonus token if a==K)

        bool stop = false;
        for (int j = 0; j < a; ++j) {          // emit accepted drafts
            if (!on_token(drafts[j])) { stop = true; break; }
            if (++generated >= max_new) { stop = true; break; }
        }
        if (stop) break;

        // ---- settle main KV / recurrent state to the a+1 confirmed tokens ----
        if (a == K) {
            // full accept: the verify forward already left the correct state
        } else if (gdn) {
            restore_states();
            n_past = p;
            std::vector<int32_t> conf; conf.reserve(a + 1);
            conf.push_back(x);
            for (int j = 0; j < a; ++j) conf.push_back(drafts[j]);
            decode(conf);                      // redo a+1 tokens for correct GDN state
        } else {
            n_past = p + a + 1;                // KV[p..p+a] from verify is valid; drop the rest
        }

        // ---- re-sync MTP KV for the confirmed tokens using the true main hiddens ----
        mtp_past = mp0 + 1;                    // keep index p-1 (draft 1 used the true hidden)
        for (int j = 0; j < a; ++j) {
            mtp_hidden = vH[j];                // true h_{p+j}
            mtp_draft(drafts[j]);              // rewrite MTP KV index p+j with the true hidden
        }
        mtp_hidden = vH[a];                    // h_{p+a}: draft context for x_new
        x = x_new;
    }
    if (steps > 0)
        fprintf(stderr, "[MTP: %d tokens, %ld verify forwards, %ld/%ld drafts accepted (%.0f%%), %.2f tok/forward]\n",
                generated, steps, accepted_drafts, draft_forwards,
                draft_forwards ? 100.0 * accepted_drafts / draft_forwards : 0.0,
                (double) generated / steps);
}

// Single-token decode using a persistent graph built once and reused, so that
// ggml-cuda can capture & replay a CUDA graph (eliminating per-kernel launch
// overhead). Attends over the full n_ctx; the mask hides not-yet-written slots.
// NOTE: only available in single-backend (non-sched) mode.
const std::vector<float> & Runtime::Impl::decode_reuse(int32_t token) {
    const bool prof = getenv("QWEN_PROF") != nullptr;
    auto pnow = []{ return std::chrono::steady_clock::now(); };
    auto pt0 = pnow();

    // bucket n_kv so the decode graph stays identical (reusable) within a bucket,
    // while keeping attention cost proportional to the actual sequence length.
    const int want_nkv = std::min(((n_past + 1 + KV_BUCKET - 1) / KV_BUCKET) * KV_BUCKET, n_ctx);
    if (!dgf || want_nkv != d_nkv) {
        if (dgalloc) { ggml_gallocr_free(dgalloc); dgalloc = nullptr; }
        if (dctx)    { ggml_free(dctx); dctx = nullptr; }
        d_nkv = want_nkv;
        ggml_init_params gp{};
        gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
        gp.no_alloc = true;
        dctx = ggml_init(gp);
        persistent = true;
        dgf = build_graph(dctx, /*n_tokens=*/1, /*n_kv=*/d_nkv);
        persistent = false;
        dgalloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(dgalloc, dgf))
            throw std::runtime_error("persistent gallocr alloc failed");
    }
    auto pt_build = pnow();

    ggml_tensor * inp_tokens = ggml_graph_get_tensor(dgf, "inp_tokens");
    ggml_tensor * inp_pos    = ggml_graph_get_tensor(dgf, "inp_pos");
    ggml_tensor * inp_mask   = ggml_graph_get_tensor(dgf, "inp_mask");
    ggml_tensor * inp_kvidx  = ggml_graph_get_tensor(dgf, "inp_kvidx");

    ggml_backend_tensor_set(inp_tokens, &token, 0, sizeof(int32_t));
    int32_t pos = n_past;
    ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
    int64_t kvidx = n_past;
    ggml_backend_tensor_set(inp_kvidx, &kvidx, 0, sizeof(int64_t));

    std::vector<ggml_fp16_t> mask(d_nkv);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int j = 0; j < d_nkv; ++j) mask[j] = (j <= n_past) ? z : ninf;
    ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    auto pt_input = pnow();

    if (ggml_backend_graph_compute(backend, dgf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("persistent graph compute failed");
    auto pt_compute = pnow();

    ggml_tensor * logits_t = ggml_graph_get_tensor(dgf, "logits");
    const int n_vocab = (int) logits_t->ne[0];
    logits.resize(n_vocab);
    ggml_backend_tensor_get(logits_t, logits.data(), 0, n_vocab * sizeof(float));
    capture_main_hidden(dgf, 0);

    if (prof) {
        auto ms = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
        fprintf(stderr, "[prof] build=%.2f input=%.2f compute=%.2f ms\n",
                ms(pt0, pt_build), ms(pt_build, pt_input), ms(pt_input, pt_compute));
    }
    n_past += 1;
    return logits;
}

// ---- MoE router: softmax gating + top-k + normalized weights (no expert matmul) ----
ggml_tensor * Runtime::Impl::build_router(ggml_context * ctx, ggml_cgraph * gf, int il,
        ggml_tensor * ffn_in, ggml_tensor * & weights_out) {
    const auto & hp = model.hparams();
    const int n_exp  = hp.n_expert;
    const int n_used = hp.n_expert_used;

    ggml_tensor * logits   = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp.weight", il), ffn_in); // [n_exp,1]
    ggml_tensor * probs    = ggml_soft_max(ctx, logits);
    ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, n_used);     // [n_used,1] i32

    ggml_tensor * probs3   = ggml_reshape_3d(ctx, probs, 1, n_exp, 1);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs3, selected);       // [1,n_used,1]
    weights = ggml_reshape_2d(ctx, weights, n_used, 1);
    ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
    wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, wsum);
    if (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f)
        weights = ggml_scale(ctx, weights, hp.expert_weights_scale);

    ggml_set_output(selected);             // preserve its buffer for host readback
    ggml_build_forward_expand(gf, selected);
    weights_out = weights;                 // [n_used,1]
    return selected;
}

// ---- expert matmul over the VRAM slot cache (single token) ----
ggml_tensor * Runtime::Impl::build_moe_cached(ggml_context * ctx, ggml_cgraph * gf, int il,
        ggml_tensor * ffn_in, ggml_tensor * slot_g, ggml_tensor * slot_u,
        ggml_tensor * slot_d, ggml_tensor * weights) {
    const auto & hp = model.hparams();
    const int n_embd = hp.n_embd;
    const int n_used = hp.n_expert_used;

    ggml_tensor * x3   = ggml_reshape_3d(ctx, ffn_in, n_embd, 1, 1);
    ggml_tensor * up   = ggml_mul_mat_id(ctx, ecache->up(il),   x3,  slot_u);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, ecache->gate(il), x3,  slot_g);
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);              // [ff_exp,n_used,1]
    ggml_tensor * experts = ggml_mul_mat_id(ctx, ecache->down(il), act, slot_d); // [n_embd,n_used,1]

    // weighted sum of the n_used experts as one GEMV
    ggml_tensor * et = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, experts, n_embd, n_used))); // [n_used,n_embd]
    ggml_tensor * w  = ggml_reshape_2d(ctx, weights, n_used, 1);
    ggml_tensor * moe_out = ggml_mul_mat(ctx, et, w);                  // [n_embd,1]

    // shared expert (qwen35moe): GPU-resident, runs every token
    if (ggml_tensor * up_sh = Wopt("blk.%d.ffn_up_shexp.weight", il)) {
        ggml_tensor * g  = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_shexp.weight", il), ffn_in);
        ggml_tensor * u  = ggml_mul_mat(ctx, up_sh, ffn_in);
        ggml_tensor * sh = ggml_mul_mat(ctx, W("blk.%d.ffn_down_shexp.weight", il),
                                        ggml_mul(ctx, ggml_silu(ctx, g), u));
        ggml_tensor * sg = ggml_sigmoid(ctx, ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp_shexp.weight", il), ffn_in));
        sh = ggml_mul(ctx, sh, sg);
        moe_out = ggml_add(ctx, moe_out, sh);
    }
    return moe_out;
}

// Batched prefill over the cache: run n_tokens through one segmented forward
// (seg A attention/router for all tokens -> ensure the union of selected experts
//  -> seg B expert matmuls for all tokens). Far fewer graph dispatches than the
// token-by-token path. n_tokens is bounded so a layer's distinct experts fit the
// pool. Only the last token's logits are produced (when want_logits).
void Runtime::Impl::decode_cached_batch(const int32_t * toks, int n_tokens, bool want_logits,
                                        bool verify) {
    const auto & hp = model.hparams();
    const int n_embd      = hp.n_embd;
    const int n_exp       = hp.n_expert;
    const int n_used      = hp.n_expert_used;
    const int n_head      = hp.n_head;
    const int n_head_kv   = hp.n_head_kv;
    const int n_embd_head = hp.n_embd_head;
    const float eps       = hp.rms_eps;
    const bool  gated     = hp.has_gdn;
    const int   T         = n_tokens;
    const int   n_kv      = n_past + T;

    // temp carry tensors sized for this batch (bridge seg A->B and layer->layer)
    ggml_init_params tp{};
    tp.mem_size = ggml_tensor_overhead() * 8 + 256;
    tp.no_alloc = true;
    ggml_context * tctx = ggml_init(tp);
    ggml_tensor * h_b      = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * ffn_in_b = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * resid_b  = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * weights_b= ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_used, T);
    ggml_tensor * slot_g_b = ggml_new_tensor_2d(tctx, GGML_TYPE_I32, n_used, T);
    ggml_tensor * slot_u_b = ggml_new_tensor_2d(tctx, GGML_TYPE_I32, n_used, T);
    ggml_tensor * slot_d_b = ggml_new_tensor_2d(tctx, GGML_TYPE_I32, n_used, T);
    ggml_backend_buffer_t tbuf = ggml_backend_alloc_ctx_tensors(tctx, backend);
    if (!tbuf) throw std::runtime_error("decode_cached_batch: temp alloc failed");

    auto new_ctx = [&]() {
        ggml_init_params gp{};
        gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
        gp.no_alloc = true;
        return ggml_init(gp);
    };
    auto run = [&](ggml_context * ctx, ggml_cgraph * gf) {
        if (!ggml_gallocr_alloc_graph(cache_galloc, gf))
            throw std::runtime_error("decode_cached_batch: gallocr alloc failed");
    };

    // ---- seg 0: token embeddings -> h_b ----
    {
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(inp); ggml_set_name(inp, "inp_tok");
        ggml_tensor * emb = ggml_get_rows(ctx, model.tok_embd_rows(), inp);  // [n_embd, T]
        ggml_build_forward_expand(gf, ggml_cpy(ctx, emb, h_b));
        run(ctx, gf);
        ggml_backend_tensor_set(inp, toks, 0, T * sizeof(int32_t));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("decode_cached_batch: embed compute failed");
        ggml_free(ctx);
    }

    std::vector<int32_t> sel(n_used * T), sg(n_used * T), su(n_used * T), sd(n_used * T);

    for (int il = 0; il < (int) hp.n_main(); ++il) {
        const bool recurrent = hp.is_recurrent(il);

        // ===== seg A: attention/GDN + router (all T tokens) =====
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);

        ggml_tensor * inp_pos = nullptr, * inp_mask = nullptr;
        if (!recurrent) {
            inp_pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
            ggml_set_input(inp_pos);  ggml_set_name(inp_pos, "inp_pos");
            inp_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, T);
            ggml_set_input(inp_mask); ggml_set_name(inp_mask, "inp_mask");
        }

        ggml_tensor * cur = ggml_rms_norm(ctx, h_b, eps);
        cur = ggml_mul(ctx, cur, W("blk.%d.attn_norm.weight", il));

        if (recurrent) {
            cur = build_gdn(ctx, gf, il, cur, T);
        } else {
            ggml_tensor * Q, * K, * V, * gate_t = nullptr;
            if (gated) {
                ggml_tensor * Qf = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur);
                const size_t es = ggml_element_size(Qf);
                Q = ggml_view_3d(ctx, Qf, n_embd_head, n_head, T,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head, 0);
                gate_t = ggml_view_3d(ctx, Qf, n_embd_head, n_head, T,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head, es * n_embd_head);
                gate_t = ggml_cont_2d(ctx, gate_t, n_embd_head * n_head, T);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, T);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, T);
            } else {
                Q = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head,    T);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, T);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, T);
            }
            Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, eps), W("blk.%d.attn_q_norm.weight", il));
            K = ggml_mul(ctx, ggml_rms_norm(ctx, K, eps), W("blk.%d.attn_k_norm.weight", il));
            Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX,
                              0, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(ctx, K, inp_pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX,
                              0, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * att = build_attn(ctx, gf, il, Q, K, V, inp_mask, T, n_kv);
            if (gated) att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate_t));
            cur = ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", il), att);
        }

        ggml_tensor * attn_resid = ggml_add(ctx, cur, h_b);          // [n_embd, T]
        ggml_tensor * ffn_in;
        if (gated) ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.post_attention_norm.weight", il));
        else       ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.ffn_norm.weight", il));

        // multi-token router
        ggml_tensor * logits = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp.weight", il), ffn_in);  // [n_exp, T]
        ggml_tensor * probs   = ggml_soft_max(ctx, logits);
        // argsort_top_k returns a STRIDED view (nb[1] = n_exp*4); make it contiguous
        // so the [n_used, T] host readback (ggml_backend_tensor_get) is not corrupted
        // for columns >= 1 (it ignores strides). Single-token path is T=1 so unaffected.
        ggml_tensor * selected = ggml_cont(ctx, ggml_argsort_top_k(ctx, probs, n_used));  // [n_used, T]
        ggml_tensor * probs3  = ggml_reshape_3d(ctx, probs, 1, n_exp, T);
        ggml_tensor * weights = ggml_get_rows(ctx, probs3, selected);             // [1, n_used, T]
        weights = ggml_reshape_2d(ctx, weights, n_used, T);
        ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
        wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx, weights, wsum);
        if (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f)
            weights = ggml_scale(ctx, weights, hp.expert_weights_scale);
        ggml_set_output(selected);
        ggml_build_forward_expand(gf, selected);

        ggml_build_forward_expand(gf, ggml_cpy(ctx, ffn_in, ffn_in_b));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, attn_resid, resid_b));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, weights, weights_b));

        run(ctx, gf);
        if (!recurrent) {
            std::vector<int32_t> pos(T);
            for (int i = 0; i < T; ++i) pos[i] = n_past + i;
            ggml_backend_tensor_set(inp_pos, pos.data(), 0, T * sizeof(int32_t));
            std::vector<ggml_fp16_t> mask((size_t) n_kv * T);
            const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
            for (int i = 0; i < T; ++i) {
                const int abs_i = n_past + i;
                for (int j = 0; j < n_kv; ++j)
                    mask[(size_t) i * n_kv + j] = (j <= abs_i) ? z : ninf;
            }
            ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("decode_cached_batch: seg A compute failed");

        ggml_backend_tensor_get(selected, sel.data(), 0, (size_t) n_used * T * sizeof(int32_t));
        ggml_free(ctx);

        // ===== host: make the union of selected experts resident =====
        ecache->ensure(il, sel.data(), n_used * T, sg.data(), su.data(), sd.data());
        ggml_backend_tensor_set(slot_g_b, sg.data(), 0, (size_t) n_used * T * sizeof(int32_t));
        ggml_backend_tensor_set(slot_u_b, su.data(), 0, (size_t) n_used * T * sizeof(int32_t));
        ggml_backend_tensor_set(slot_d_b, sd.data(), 0, (size_t) n_used * T * sizeof(int32_t));

        // ===== seg B: expert matmuls + residual (per token, one graph) =====
        ctx = new_ctx();
        gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * up_sh = Wopt("blk.%d.ffn_up_shexp.weight", il);
        for (int t = 0; t < T; ++t) {
            ggml_tensor * ffn_t = ggml_view_2d(ctx, ffn_in_b, n_embd, 1, ffn_in_b->nb[1], (size_t) t * ffn_in_b->nb[1]);
            ggml_tensor * sg_t  = ggml_view_2d(ctx, slot_g_b, n_used, 1, slot_g_b->nb[1], (size_t) t * slot_g_b->nb[1]);
            ggml_tensor * su_t  = ggml_view_2d(ctx, slot_u_b, n_used, 1, slot_u_b->nb[1], (size_t) t * slot_u_b->nb[1]);
            ggml_tensor * sd_t  = ggml_view_2d(ctx, slot_d_b, n_used, 1, slot_d_b->nb[1], (size_t) t * slot_d_b->nb[1]);
            ggml_tensor * w_t   = ggml_view_2d(ctx, weights_b, n_used, 1, weights_b->nb[1], (size_t) t * weights_b->nb[1]);
            ggml_tensor * r_t   = ggml_view_2d(ctx, resid_b,  n_embd, 1, resid_b->nb[1],  (size_t) t * resid_b->nb[1]);

            ggml_tensor * x3   = ggml_reshape_3d(ctx, ggml_cont(ctx, ffn_t), n_embd, 1, 1);
            ggml_tensor * up   = ggml_mul_mat_id(ctx, ecache->up(il),   x3, su_t);
            ggml_tensor * gate = ggml_mul_mat_id(ctx, ecache->gate(il), x3, sg_t);
            ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
            ggml_tensor * experts = ggml_mul_mat_id(ctx, ecache->down(il), act, sd_t);  // [n_embd, n_used, 1]
            ggml_tensor * et = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, experts, n_embd, n_used)));
            ggml_tensor * w  = ggml_reshape_2d(ctx, w_t, n_used, 1);
            ggml_tensor * moe_t = ggml_mul_mat(ctx, et, w);   // [n_embd, 1]

            if (up_sh) {
                ggml_tensor * ffc = ggml_cont(ctx, ffn_t);
                ggml_tensor * g  = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_shexp.weight", il), ffc);
                ggml_tensor * u  = ggml_mul_mat(ctx, up_sh, ffc);
                ggml_tensor * sh = ggml_mul_mat(ctx, W("blk.%d.ffn_down_shexp.weight", il),
                                                ggml_mul(ctx, ggml_silu(ctx, g), u));
                ggml_tensor * sgt = ggml_sigmoid(ctx, ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp_shexp.weight", il), ffc));
                moe_t = ggml_add(ctx, moe_t, ggml_mul(ctx, sh, sgt));
            }
            ggml_tensor * h_t = ggml_add(ctx, moe_t, r_t);
            ggml_tensor * dst = ggml_view_2d(ctx, h_b, n_embd, 1, h_b->nb[1], (size_t) t * h_b->nb[1]);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, h_t, dst));
        }
        run(ctx, gf);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("decode_cached_batch: seg B compute failed");
        ggml_free(ctx);
    }

    // ---- final norm + output projection (last token only) ----
    if (want_logits) {
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * last = ggml_view_2d(ctx, h_b, n_embd, 1, h_b->nb[1], (size_t) (T - 1) * h_b->nb[1]);
        ggml_tensor * cur = ggml_rms_norm(ctx, last, eps);
        cur = ggml_mul(ctx, cur, model.tensor("output_norm.weight"));
        ggml_tensor * output_w = model.tensor("output.weight");
        if (!output_w) output_w = model.tensor("token_embd.weight");
        cur = ggml_mul_mat(ctx, output_w, cur);
        ggml_set_name(cur, "logits");
        ggml_build_forward_expand(gf, cur);
        run(ctx, gf);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("decode_cached_batch: output compute failed");
        const int n_vocab = (int) cur->ne[0];
        logits.resize(n_vocab);
        ggml_backend_tensor_get(cur, logits.data(), 0, n_vocab * sizeof(float));
        ggml_free(ctx);
    }

    // MTP verify: expose per-position logits (vL[i]) and main hidden (vH[i]) for
    // all T tokens so the caller can accept the longest matching draft prefix.
    if (verify) {
        vH.assign(T, std::vector<float>(n_embd));
        for (int i = 0; i < T; ++i)
            ggml_backend_tensor_get(h_b, vH[i].data(), (size_t) i * n_embd * sizeof(float), n_embd * sizeof(float));

        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * cur = ggml_rms_norm(ctx, h_b, eps);                        // [n_embd, T]
        cur = ggml_mul(ctx, cur, model.tensor("output_norm.weight"));
        ggml_tensor * output_w = model.tensor("output.weight");
        if (!output_w) output_w = model.tensor("token_embd.weight");
        cur = ggml_mul_mat(ctx, output_w, cur);                                 // [n_vocab, T]
        ggml_set_name(cur, "vlogits");
        ggml_build_forward_expand(gf, cur);
        run(ctx, gf);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("decode_cached_batch: verify output compute failed");
        const int n_vocab = (int) cur->ne[0];
        vL.assign(T, std::vector<float>(n_vocab));
        for (int i = 0; i < T; ++i)
            ggml_backend_tensor_get(cur, vL[i].data(), (size_t) i * cur->nb[1], n_vocab * sizeof(float));
        ggml_free(ctx);
    }

    ggml_backend_buffer_free(tbuf);
    ggml_free(tctx);
    n_past += T;
}

// Single-token decode using the dynamic VRAM expert cache.
// Each layer is run as two GPU graph segments around a host sync point:
//   seg A: attention/GDN + router  -> read back selected experts
//   (host) ensure experts resident in VRAM cache (stream misses)
//   seg B: expert matmuls on cache slots + shared expert + residual
// Correctness follows the *actual* routing; the cache only changes where the
// expert weights are fetched from (VRAM hit vs CPU/SSD miss).
const std::vector<float> & Runtime::Impl::decode_cached(int32_t token) {
    const auto & hp = model.hparams();
    const int n_embd      = hp.n_embd;
    const int n_used      = hp.n_expert_used;
    const int n_head      = hp.n_head;
    const int n_head_kv   = hp.n_head_kv;
    const int n_embd_head = hp.n_embd_head;
    const float eps       = hp.rms_eps;
    const bool  gated     = hp.has_gdn;
    const int   n_kv      = n_past + 1;

    auto new_ctx = [&]() {
        ggml_init_params gp{};
        gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
        gp.no_alloc = true;
        return ggml_init(gp);
    };
    auto run = [&](ggml_context * ctx, ggml_cgraph * gf) {
        if (!ggml_gallocr_alloc_graph(cache_galloc, gf))
            throw std::runtime_error("decode_cached: gallocr alloc failed");
    };
    const bool prof_dc = getenv("QWEN_PROF_DC") != nullptr;
    auto wall0 = std::chrono::steady_clock::now();
    auto compute = [&](ggml_cgraph * gf, const char * msg) {
        auto t = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error(std::string("decode_cached: ") + msg);
        if (prof_dc) dc_gpu_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t).count();
    };

    // ---- seg 0: token embedding -> p_h ----
    {
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_set_input(inp); ggml_set_name(inp, "inp_tok");
        ggml_tensor * emb = ggml_get_rows(ctx, model.tok_embd_rows(), inp);  // [n_embd,1]
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_1d(ctx, emb, n_embd), p_h));
        run(ctx, gf);
        ggml_backend_tensor_set(inp, &token, 0, sizeof(int32_t));
        compute(gf, "embed compute failed");
        ggml_free(ctx);
    }

    std::vector<int32_t> sel(n_used), slot_g(n_used), slot_u(n_used), slot_d(n_used);

    // Double-buffered carry tensors (parity by layer) so a fused segB(L)+segA(L+1)
    // graph has no write-after-read hazard on the carry buffers.
    ggml_tensor * carry_ffn[2] = { p_ffn_in, p_ffn_in2 };
    ggml_tensor * carry_res[2] = { p_resid,  p_resid2  };
    ggml_tensor * carry_wgt[2] = { p_weights, p_weights2 };

    // Append seg A (attention/GDN + router) for layer `il` to (ctx,gf). Writes the
    // normed FFN input / residual / router weights into the layer's parity carry,
    // and exposes `selected` (router top-k) for host readback. Returns selected.
    auto build_segA = [&](ggml_context * ctx, ggml_cgraph * gf, int il) -> ggml_tensor * {
        const bool recurrent = hp.is_recurrent(il);
        ggml_tensor * inp_pos = nullptr, * inp_mask = nullptr;
        if (!recurrent) {
            inp_pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            ggml_set_input(inp_pos);  ggml_set_name(inp_pos, "inp_pos");
            inp_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, 1);
            ggml_set_input(inp_mask); ggml_set_name(inp_mask, "inp_mask");
        }
        ggml_tensor * cur = ggml_rms_norm(ctx, p_h, eps);
        cur = ggml_mul(ctx, cur, W("blk.%d.attn_norm.weight", il));
        if (recurrent) {
            cur = build_gdn(ctx, gf, il, cur, 1);
        } else {
            ggml_tensor * Q, * K, * V, * gate_t = nullptr;
            if (gated) {
                ggml_tensor * Qf = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur);
                const size_t es = ggml_element_size(Qf);
                Q = ggml_view_3d(ctx, Qf, n_embd_head, n_head, 1,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head, 0);
                gate_t = ggml_view_3d(ctx, Qf, n_embd_head, n_head, 1,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head, es * n_embd_head);
                gate_t = ggml_cont_2d(ctx, gate_t, n_embd_head * n_head, 1);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, 1);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, 1);
            } else {
                Q = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head,    1);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, 1);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, 1);
            }
            Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, eps), W("blk.%d.attn_q_norm.weight", il));
            K = ggml_mul(ctx, ggml_rms_norm(ctx, K, eps), W("blk.%d.attn_k_norm.weight", il));
            Q = ggml_rope_ext(ctx, Q, inp_pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX,
                              0, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            K = ggml_rope_ext(ctx, K, inp_pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX,
                              0, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * att = build_attn(ctx, gf, il, Q, K, V, inp_mask, 1, n_kv);
            if (gated) att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate_t));
            cur = ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", il), att);
        }
        ggml_tensor * attn_resid = ggml_add(ctx, cur, p_h);
        ggml_tensor * ffn_in;
        if (gated) ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.post_attention_norm.weight", il));
        else       ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.ffn_norm.weight", il));
        ggml_tensor * weights = nullptr;
        ggml_tensor * selected = build_router(ctx, gf, il, ffn_in, weights);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ffn_in, carry_ffn[il & 1]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, attn_resid, carry_res[il & 1]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_1d(ctx, weights, n_used), carry_wgt[il & 1]));
        return selected;
    };
    // Append seg B (cached expert matmul + residual) for layer `il`; writes p_h.
    auto build_segB = [&](ggml_context * ctx, ggml_cgraph * gf, int il) {
        ggml_tensor * moe_out = build_moe_cached(ctx, gf, il, carry_ffn[il & 1],
                                                 p_slot_g, p_slot_u, p_slot_d, carry_wgt[il & 1]);
        ggml_tensor * h_new = ggml_add(ctx, moe_out, carry_res[il & 1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_1d(ctx, h_new, n_embd), p_h));
    };
    // Set the attention pos/mask inputs of a graph (no-op if it has none, e.g. a
    // segB-only graph or a GDN-only seg A).
    auto set_attn_inputs = [&](ggml_cgraph * gf) {
        if (ggml_tensor * ip = ggml_graph_get_tensor(gf, "inp_pos")) {
            int32_t pos = n_past; ggml_backend_tensor_set(ip, &pos, 0, sizeof(int32_t));
        }
        if (ggml_tensor * im = ggml_graph_get_tensor(gf, "inp_mask")) {
            std::vector<ggml_fp16_t> mask(n_kv);
            const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
            for (int j = 0; j < n_kv; ++j) mask[j] = (j <= n_past) ? z : ninf;
            ggml_backend_tensor_set(im, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
    };
    // Read back layer `il`'s router selection and make those experts resident.
    auto ensure_layer = [&](ggml_tensor * selected, int il) {
        ggml_backend_tensor_get(selected, sel.data(), 0, n_used * sizeof(int32_t));
        ecache->ensure(il, sel.data(), n_used, slot_g.data(), slot_u.data(), slot_d.data());
        ggml_backend_tensor_set(p_slot_g, slot_g.data(), 0, n_used * sizeof(int32_t));
        ggml_backend_tensor_set(p_slot_u, slot_u.data(), 0, n_used * sizeof(int32_t));
        ggml_backend_tensor_set(p_slot_d, slot_d.data(), 0, n_used * sizeof(int32_t));
    };

    const int N = (int) hp.n_main();

    // seg A(0) on its own, then fuse segB(L)+segA(L+1) per step so each layer
    // boundary is a single GPU submit instead of two (~halves the dispatches).
    {
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * selected = build_segA(ctx, gf, 0);
        run(ctx, gf);
        set_attn_inputs(gf);
        compute(gf, "seg A0 compute failed");
        ensure_layer(selected, 0);
        ggml_free(ctx);
    }
    for (int il = 0; il < N; ++il) {
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        build_segB(ctx, gf, il);
        ggml_tensor * nsel = (il + 1 < N) ? build_segA(ctx, gf, il + 1) : nullptr;
        run(ctx, gf);
        set_attn_inputs(gf);
        compute(gf, "fused segB/segA compute failed");
        if (nsel) ensure_layer(nsel, il + 1);
        ggml_free(ctx);
    }

    // ---- final norm + output projection ----
    std::vector<float> out;
    {
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * cur = ggml_rms_norm(ctx, p_h, eps);
        cur = ggml_mul(ctx, cur, model.tensor("output_norm.weight"));
        ggml_tensor * output_w = model.tensor("output.weight");
        if (!output_w) output_w = model.tensor("token_embd.weight");
        cur = ggml_mul_mat(ctx, output_w, cur);
        ggml_set_name(cur, "logits");
        ggml_build_forward_expand(gf, cur);
        run(ctx, gf);
        compute(gf, "output compute failed");
        const int n_vocab = (int) cur->ne[0];
        logits.resize(n_vocab);
        ggml_backend_tensor_get(cur, logits.data(), 0, n_vocab * sizeof(float));
        ggml_free(ctx);
    }

    // MTP: expose this token's main hidden (pre-output-norm) for the nextn block.
    if (capture_hidden) {
        mtp_hidden.resize(n_embd);
        ggml_backend_tensor_get(p_h, mtp_hidden.data(), 0, n_embd * sizeof(float));
    }

    if (prof_dc) {
        dc_wall_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall0).count();
        dc_tokens++;
    }
    n_past += 1;
    return logits;
}

// Allocate the GDN recurrent-state backup buffers (idempotent). Needed to roll
// back conv/ssm state on a speculative miss (cache fast path) or MTP reject.
void Runtime::Impl::init_state_backup() {
    const auto & hp = model.hparams();
    if (!hp.has_gdn || bak_buf) return;
    ggml_init_params bp{};
    bp.mem_size = ggml_tensor_overhead() * hp.n_layer * 2 + 256;
    bp.no_alloc = true;
    bak_ctx = ggml_init(bp);
    conv_bak.assign(hp.n_layer, nullptr);
    ssm_bak.assign(hp.n_layer, nullptr);
    for (int il = 0; il < (int) hp.n_layer; ++il) {
        if (!conv_state[il]) continue;
        conv_bak[il] = ggml_new_tensor(bak_ctx, conv_state[il]->type,
                                       ggml_n_dims(conv_state[il]), conv_state[il]->ne);
        ssm_bak[il]  = ggml_new_tensor(bak_ctx, ssm_state[il]->type,
                                       ggml_n_dims(ssm_state[il]), ssm_state[il]->ne);
    }
    bak_buf = ggml_backend_alloc_ctx_tensors(bak_ctx, backend);
    if (!bak_buf) throw std::runtime_error("failed to alloc state backup");
}

void Runtime::Impl::backup_states() {
    if (!bak_buf) return;
    for (int il = 0; il < (int) model.hparams().n_layer; ++il) {
        if (!conv_bak[il]) continue;
        ggml_backend_tensor_copy(conv_state[il], conv_bak[il]);
        ggml_backend_tensor_copy(ssm_state[il],  ssm_bak[il]);
    }
}
void Runtime::Impl::restore_states() {
    if (!bak_buf) return;
    for (int il = 0; il < (int) model.hparams().n_layer; ++il) {
        if (!conv_bak[il]) continue;
        ggml_backend_tensor_copy(conv_bak[il], conv_state[il]);
        ggml_backend_tensor_copy(ssm_bak[il],  ssm_state[il]);
    }
}

// Optimistic single-graph decode: run the whole token in one persistent
// (CUDA-graph friendly) graph that reads experts from the VRAM cache via an
// in-graph slot remap, then verify residency. On a miss, roll back recurrent
// state and fall back to the (always-correct) slow per-layer decode_cached.
const std::vector<float> & Runtime::Impl::decode_cached_fast(int32_t token) {
    const auto & hp = model.hparams();
    const int n_used  = hp.n_expert_used;
    const int n_exp   = hp.n_expert;
    const int n_layer = (int) hp.n_main();   // MTP blocks excluded from the main stack

    // (re)build the persistent fast graph when the KV bucket changes
    const int want_nkv = std::min(((n_past + 1 + KV_BUCKET - 1) / KV_BUCKET) * KV_BUCKET, n_ctx);
    if (!f_gf || want_nkv != f_nkv) {
        if (f_ctx) { ggml_free(f_ctx); f_ctx = nullptr; }
        f_nkv = want_nkv;
        ggml_init_params gp{};
        gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
        gp.no_alloc = true;
        f_ctx = ggml_init(gp);
        persistent = true;
        cache_fast_build = true;
        f_gf = build_graph(f_ctx, /*n_tokens=*/1, /*n_kv=*/f_nkv);
        cache_fast_build = false;
        persistent = false;
        if (!ggml_gallocr_alloc_graph(f_galloc, f_gf))
            throw std::runtime_error("decode_cached_fast: gallocr alloc failed");
    }

    // graph inputs
    ggml_tensor * inp_tokens = ggml_graph_get_tensor(f_gf, "inp_tokens");
    ggml_tensor * inp_pos    = ggml_graph_get_tensor(f_gf, "inp_pos");
    ggml_tensor * inp_mask   = ggml_graph_get_tensor(f_gf, "inp_mask");
    ggml_tensor * inp_kvidx  = ggml_graph_get_tensor(f_gf, "inp_kvidx");
    ggml_backend_tensor_set(inp_tokens, &token, 0, sizeof(int32_t));
    int32_t pos = n_past;   ggml_backend_tensor_set(inp_pos, &pos, 0, sizeof(int32_t));
    int64_t kvidx = n_past; ggml_backend_tensor_set(inp_kvidx, &kvidx, 0, sizeof(int64_t));
    std::vector<ggml_fp16_t> mask(f_nkv);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int j = 0; j < f_nkv; ++j) mask[j] = (j <= n_past) ? z : ninf;
    ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    // refresh the in-graph remap table from current residency
    auto fill_g2s = [&]() {
        for (int il = 0; il < n_layer; ++il)
            for (int r = 0; r < 3; ++r) {
                const int32_t * row = ecache->slot_of_row((ExpertCache::Role) r, il);
                memcpy(&g2s_host[(size_t) (il * 3 + r) * n_exp], row, n_exp * sizeof(int32_t));
            }
        ggml_backend_tensor_set(g2s_all, g2s_host.data(), 0, g2s_host.size() * sizeof(int32_t));
    };
    // verify residency of every selected expert; returns true if all resident
    auto verify = [&]() {
        ggml_backend_tensor_get(sel_all, sel_host.data(), 0, sel_host.size() * sizeof(int32_t));
        bool ok = true;
        for (int il = 0; il < n_layer; ++il)
            for (int k = 0; k < n_used; ++k) {
                const int e = sel_host[(size_t) il * n_used + k];
                if (!ecache->resident(ExpertCache::GATE, il, e) ||
                    !ecache->resident(ExpertCache::UP,   il, e) ||
                    !ecache->resident(ExpertCache::DOWN, il, e)) ok = false;
            }
        return ok;
    };

    backup_states();   // so a speculative miss can be rolled back

    fill_g2s();
    if (ggml_backend_graph_compute(backend, f_gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("decode_cached_fast: compute failed");

    // On a miss the speculative result is wrong: roll back recurrent state,
    // warm the cache, and recompute correctly via the slow per-layer path.
    if (!verify()) {
        restore_states();
        return decode_cached(token);
    }

    // accept: bump LRU/frequency for the resident accesses, then read logits
    for (int il = 0; il < n_layer; ++il)
        for (int k = 0; k < n_used; ++k) {
            const int e = sel_host[(size_t) il * n_used + k];
            ecache->touch(ExpertCache::GATE, il, e);
            ecache->touch(ExpertCache::UP,   il, e);
            ecache->touch(ExpertCache::DOWN, il, e);
        }

    ggml_tensor * logits_t = ggml_graph_get_tensor(f_gf, "logits");
    const int n_vocab = (int) logits_t->ne[0];
    logits.resize(n_vocab);
    ggml_backend_tensor_get(logits_t, logits.data(), 0, n_vocab * sizeof(float));
    n_past += 1;
    return logits;
}

const std::vector<float> & Runtime::Impl::decode(const std::vector<int32_t> & tokens) {
    const int n_tokens = (int) tokens.size();
    if (n_tokens == 0) throw std::runtime_error("decode: empty tokens");
    if (n_past + n_tokens > n_ctx) throw std::runtime_error("decode: context overflow");

    // single-token decode via the dynamic VRAM expert cache (expert-offload mode)
    if (n_tokens == 1 && ecache)
        return cache_fast_enabled ? decode_cached_fast(tokens[0]) : decode_cached(tokens[0]);

    // SSD tier has no CPU-expert sched path: run prefill as batched chunks via
    // the cache (chunk bounded so a layer's distinct experts fit the pools).
    if (ecache && !sched) {
        // Batched prefill (QWEN_BATCH) processes a chunk of tokens through one
        // segmented forward (seg A attention/GDN/router for all tokens -> one
        // expert-residency sync -> seg B expert matmuls). Now argmax-equivalent to
        // the token-by-token path for both plain-attn (qwen3moe) and GDN MoE models
        // (the prior divergence was a router-readback bug, since fixed). It is opt-in
        // because in the SSD tier it is *slower* than token-by-token: seg B is still
        // per-token, and the per-chunk union-of-experts fetch defeats the async
        // prefetch pipeline. The code path is retained for the MTP 2-token verify,
        // where the union is tiny and mostly already resident.
        if (getenv("QWEN_BATCH")) {
            int chunk = ecache->min_slots() / (model.hparams().n_expert_used > 0 ? model.hparams().n_expert_used : 1);
            if (chunk < 1)   chunk = 1;
            if (chunk > 256) chunk = 256;
            if (const char * c = getenv("QWEN_BATCH_CHUNK")) { int v = atoi(c); if (v >= 1) chunk = v; }
            int i = 0;
            while (i < n_tokens) {
                const int t = std::min(chunk, n_tokens - i);
                decode_cached_batch(&tokens[i], t, /*want_logits=*/ i + t >= n_tokens);
                i += t;
            }
            return logits;
        }
        for (int i = 0; i < n_tokens; ++i) decode_cached(tokens[i]);
        return logits;
    }

    // fast path: single-token decode with a reusable (CUDA-graph friendly) graph
    // (only when not using the backend scheduler for expert offload)
    if (n_tokens == 1 && reuse_graph && !sched) return decode_reuse(tokens[0]);

    const int n_kv = n_past + n_tokens;
    ggml_init_params gp{};
    gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
    gp.no_alloc = true;
    ggml_context * ctx = ggml_init(gp);

    ggml_cgraph * gf = build_graph(ctx, n_tokens, n_kv);

    if (sched) {
        // Expert offload path: use backend scheduler.
        // Reset scheduler state from any previous graph, then allocate and compute.
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            ggml_free(ctx);
            throw std::runtime_error("sched alloc failed");
        }
    } else {
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            ggml_free(ctx);
            throw std::runtime_error("gallocr alloc failed");
        }
    }

    ggml_tensor * inp_tokens_t = ggml_graph_get_tensor(gf, "inp_tokens");
    ggml_tensor * inp_pos_t    = ggml_graph_get_tensor(gf, "inp_pos");
    ggml_tensor * inp_mask_t   = ggml_graph_get_tensor(gf, "inp_mask");

    ggml_backend_tensor_set(inp_tokens_t, tokens.data(), 0, n_tokens * sizeof(int32_t));

    std::vector<int32_t> pos(n_tokens);
    for (int i = 0; i < n_tokens; ++i) pos[i] = n_past + i;
    ggml_backend_tensor_set(inp_pos_t, pos.data(), 0, n_tokens * sizeof(int32_t));

    std::vector<ggml_fp16_t> mask((size_t) n_kv * n_tokens);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n_tokens; ++i) {
        const int abs_i = n_past + i;
        for (int j = 0; j < n_kv; ++j)
            mask[(size_t) i * n_kv + j] = (j <= abs_i) ? z : ninf;
    }
    ggml_backend_tensor_set(inp_mask_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    enum ggml_status status;
    if (sched) {
        status = ggml_backend_sched_graph_compute(sched, gf);
    } else {
        status = ggml_backend_graph_compute(backend, gf);
    }
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("graph compute failed");
    }

    ggml_tensor * logits_t = ggml_graph_get_tensor(gf, "logits");
    const int n_vocab = (int) logits_t->ne[0];
    logits.resize(n_vocab);
    const size_t off = (size_t) (n_tokens - 1) * logits_t->nb[1];
    ggml_backend_tensor_get(logits_t, logits.data(), off, n_vocab * sizeof(float));
    capture_main_hidden(gf, n_tokens - 1);

    n_past += n_tokens;
    ggml_free(ctx);
    return logits;
}

// ---- public wrappers ----
Runtime::Runtime(Model & model, const RuntimeConfig & cfg)
    : impl_(std::make_unique<Impl>(model, cfg)) {
    impl_->init();
}
Runtime::~Runtime() = default;

const std::vector<float> & Runtime::decode(const std::vector<int32_t> & tokens) {
    return impl_->decode(tokens);
}
const std::vector<float> & Runtime::mtp_draft(int32_t token) { return impl_->mtp_draft(token); }
bool Runtime::has_mtp() const { return impl_->model.hparams().has_mtp(); }
void Runtime::generate_mtp(const std::vector<int32_t> & prompt, int max_new, int n_draft,
                           const std::function<bool(int32_t)> & on_token) {
    impl_->generate_mtp(prompt, max_new, n_draft, on_token);
}
void Runtime::reset()        { impl_->n_past = 0; impl_->mtp_past = 0; impl_->zero_states(); }
int  Runtime::n_past() const { return impl_->n_past; }

} // namespace qwencpp
