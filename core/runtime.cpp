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

namespace questwend {

static const int GRAPH_SIZE = 16384;

struct Runtime::Impl {
    Model & model;
    RuntimeConfig cfg;

    // Primary compute backend (GPU or CPU).
    ggml_backend_t        backend     = nullptr;
    ggml_backend_buffer_t weights_buf = nullptr;
    bool weights_buf_owned = false;   // split/ssd: ours; single-backend: Model frees it

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
    std::vector<int32_t>  vA;   // per-position argmax (GPU-computed in the fast path)
    bool                  v_from_batch = false;   // last verify ran the batched (ckpt-capable) path
    std::vector<float>    mtp_block_hidden;   // MTP block output hidden (for chaining drafts)

    // persistent MTP draft graph on a dedicated backend instance (own CUDA-graph
    // slot, so alternating with the main verify graph doesn't thrash capture)
    ggml_backend_t        backend_mtp = nullptr;
    ggml_context *        m_ctx    = nullptr;
    ggml_cgraph *         m_gf     = nullptr;
    ggml_gallocr_t        m_galloc = nullptr;
    int                   m_nkv    = 0;
    // headless variant for MTP KV resync (skips the ~1GB shared-head matmul)
    bool                  mtp_headless = false;    // set while building the resync graph
    ggml_context *        r_ctx    = nullptr;
    ggml_cgraph *         r_gf     = nullptr;
    ggml_gallocr_t        r_galloc = nullptr;
    int                   r_nkv    = 0;

    // persistent (K+1)-token verify graph
    ggml_context *        v_ctx    = nullptr;
    ggml_cgraph *         v_gf     = nullptr;
    ggml_gallocr_t        v_galloc = nullptr;
    int                   v_nkv    = 0;
    int                   v_ntok   = 0;

    // GDN state checkpoints: in the verify graph the delta-net runs per-token so
    // the state after each verify token can be snapshotted; a partial accept then
    // restores checkpoint[a] instead of re-decoding the accepted tokens.
    int                   gdn_ckpt = 0;        // = n_tokens while building a ckpt verify graph
    ggml_context *        ckpt_ctx = nullptr;
    ggml_backend_buffer_t ckpt_buf = nullptr;
    int                   ckpt_T   = 0;
    std::vector<std::vector<ggml_tensor *>> ckpt_conv, ckpt_ssm;   // [t][il]

    int n_ctx  = 0;
    int n_past = 0;

    // M-RoPE: next text rope position. Equals n_past for text-only, but an
    // image span advances it by max(grid_h, grid_w) instead of its token count,
    // so it diverges from n_past after images. Persisted with the cache state.
    int mrope_next = 0;

    // prompt prefix cache bookkeeping: the tokens behind n_past / the recurrent
    // state. Appended by the public decode() wrapper and by generate_mtp's
    // confirmed tokens; cleared by reset().
    std::vector<int32_t> kv_toks;

    // vision: embedding overrides applied to the next batched decode (consumed
    // by build_graph; the input tensors are filled in decode() after alloc)
    std::vector<Runtime::EmbdOverride> embd_ovr;

    // optional prefill progress callback (tokens_done, tokens_total) per chunk
    std::function<void(int, int)> progress_cb;

    // MTP batched prefill: while set, batched decodes append every token's final
    // hidden (pre-output-norm) to bh_all so the nextn KV can be built in batches.
    bool               want_bh_all = false;
    std::vector<float> bh_all;     // [n_captured * n_embd], appended per batch

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
        if (m_galloc)       ggml_gallocr_free(m_galloc);
        if (m_ctx)          ggml_free(m_ctx);
        if (r_galloc)       ggml_gallocr_free(r_galloc);
        if (r_ctx)          ggml_free(r_ctx);
        if (v_galloc)       ggml_gallocr_free(v_galloc);
        if (v_ctx)          ggml_free(v_ctx);
        if (ckpt_buf)       ggml_backend_buffer_free(ckpt_buf);
        if (ckpt_ctx)       ggml_free(ckpt_ctx);
        if (backend_mtp && backend_mtp != backend) ggml_backend_free(backend_mtp);
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
        // expert_cpu_bufs and weights_buf are owned here (not by Model) in
        // split/ssd mode; in single-backend mode Model owns and frees weights_buf.
        for (auto b : expert_cpu_bufs) if (b) ggml_backend_buffer_free(b);
        if (cpu_backend)    ggml_backend_free(cpu_backend);
        if (weights_buf_owned && weights_buf) ggml_backend_buffer_free(weights_buf);
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
    // M-RoPE helpers (no-op when !hp.use_mrope): rope_dim returns the inp_pos
    // length for a graph, apply_rope picks ggml_rope_multi vs ggml_rope_ext,
    // fill_rope_pos computes the per-token (sequential or 2D-grid) positions
    // for the main stack and returns the next text rope position.
    int rope_dim(int n_tokens) const {
        return model.hparams().use_mrope ? 4 * n_tokens : n_tokens;
    }
    ggml_tensor * apply_rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos) {
        const auto & hp = model.hparams();
        if (hp.use_mrope) {
            int sec[4] = { hp.rope_sections[0], hp.rope_sections[1],
                           hp.rope_sections[2], hp.rope_sections[3] };
            return ggml_rope_multi(ctx, x, pos, nullptr, hp.n_rot, sec,
                                   GGML_ROPE_TYPE_MROPE, 0, hp.rope_freq_base,
                                   1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }
        return ggml_rope_ext(ctx, x, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX,
                             0, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    int fill_rope_pos(std::vector<int32_t> & dst, int n_tokens, int rope_start);
    int fill_rope_pos_spans(std::vector<int32_t> & dst, int n_tokens, int rope_start,
                            const Runtime::EmbdOverride * spans, int n_spans);
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
    // n_tokens > 1 runs the nextn block over a whole batch (KV prefill); ovr/n_ovr
    // splice image embeddings over the token embeddings (spans relative to `tok`).
    ggml_tensor * build_mtp(ggml_context * ctx, ggml_cgraph * gf,
                            ggml_tensor * h, ggml_tensor * tok, ggml_tensor * pos, ggml_tensor * mask,
                            int n_kv, int n_tokens = 1,
                            const Runtime::EmbdOverride * ovr = nullptr, int n_ovr = 0);
    const std::vector<float> & mtp_draft(int32_t token);
    int32_t mtp_draft_fast(int32_t token, bool need_hidden);   // persistent graph, argmax-only readback
    void mtp_resync(int32_t token);                            // KV-write only (headless, no readback)
    void mtp_prefill_batch(const int32_t * toks, const float * hiddens, int T,
                           const std::vector<Runtime::EmbdOverride> & ovr);
    void init_ckpts(int T);
    void restore_ckpt(int t);
    void capture_main_hidden(ggml_cgraph * gf, int col);
    void decode_verify(const std::vector<int32_t> & toks);        // T-token forward -> vL/vH
    void decode_verify_cached(const std::vector<int32_t> & toks); // offload variant (decode_cached_batch)
    void generate_mtp(const std::vector<int32_t> & prompt, int max_new, int n_draft,
                      const std::function<bool(int32_t)> & on_token,
                      int32_t * out_pending = nullptr);
    void prefill(const std::vector<int32_t> & toks, bool mtp_kv);

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
    const std::vector<float> & decode_cached(int32_t token, const float * embd_override = nullptr);
    // Batched prefill over the cache: process up to a pool-sized chunk of tokens
    // in one segmented forward (instead of token-by-token). ovr/n_ovr overwrite
    // image-span rows of the embed output (spans relative to `toks`).
    void decode_cached_batch(const int32_t * toks, int n_tokens, bool want_logits,
                             bool verify = false,
                             const Runtime::EmbdOverride * ovr = nullptr, int n_ovr = 0);
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
    model.set_embd_q8(cfg.embd_q8);

    if (use_expert_offload && cfg.experts_ssd) {
        // ---- SSD tier: experts stay on disk; non-expert weights -> GPU ----
        ssd_mode = true;
        model.load_weights_ssd(backend, weights_buf);
        weights_buf_owned = true;
        reuse_graph = false;   // every token goes through the per-token cache path
        fprintf(stderr, "expert offload: ON (SSD tier, decode via VRAM cache; prefill in batched chunks)\n");
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
        weights_buf_owned = true;

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

    // VRAM left for the expert slot pools after non-expert weights, the KV
    // cache, and a compute-buffer headroom. The KV cache (st_buf, already
    // allocated) scales with n_ctx -- ~6 GB at n_ctx=36000 for a 40-layer
    // model -- so a fixed headroom would oversize the pool and overcommit VRAM.
    // On Windows the driver then silently spills allocations to shared system
    // memory (paged over PCIe), uniformly slowing prefill and decode; sizing
    // the pool against the real KV bytes keeps everything VRAM-resident.
    const size_t budget   = cfg.vram_budget_mb * 1024ull * 1024ull;
    const size_t gpu_w    = ggml_backend_buffer_get_size(weights_buf);
    const size_t kv_bytes = st_buf ? ggml_backend_buffer_get_size(st_buf) : 0;
    const size_t compute  = 1024ull * 1024ull * 1024ull;   // gallocr graph buffers
    const size_t reserve  = gpu_w + kv_bytes + compute;
    size_t avail = budget > reserve ? budget - reserve : 0;
    fprintf(stderr, "VRAM budget %zu MB = weights %zu + KV %zu + compute %zu + expert pool %zu MB\n",
            cfg.vram_budget_mb, gpu_w >> 20, kv_bytes >> 20, compute >> 20, avail >> 20);

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

// Fill the main-stack rope positions for `n_tokens` tokens starting at rope
// position `rope_start`. Without M-RoPE, dst[i] = n_past + i (unchanged 1D
// behavior; rope_start ignored). With M-RoPE, dst is section-major
// [t.. , h.. , w.. , e..] of size 4*n_tokens: text tokens get t=h=w=running,
// image spans (embd_ovr, indices batch-local) get t=base, h=base+row,
// w=base+col over their post-merge patch grid, advancing the counter by
// max(grid) instead of the token count. Returns the rope position after the
// batch (the caller commits it to mrope_next for real decodes).
int Runtime::Impl::fill_rope_pos(std::vector<int32_t> & dst, int n_tokens, int rope_start) {
    return fill_rope_pos_spans(dst, n_tokens, rope_start,
                               embd_ovr.data(), (int) embd_ovr.size());
}

int Runtime::Impl::fill_rope_pos_spans(std::vector<int32_t> & dst, int n_tokens, int rope_start,
                                       const Runtime::EmbdOverride * spans, int n_spans) {
    const auto & hp = model.hparams();
    if (!hp.use_mrope) {
        dst.resize(n_tokens);
        for (int i = 0; i < n_tokens; ++i) dst[i] = n_past + i;
        return rope_start;
    }
    dst.assign((size_t) 4 * n_tokens, 0);
    int32_t * pt = dst.data();
    int32_t * ph = pt + n_tokens;
    int32_t * pw = ph + n_tokens;          // pe (4th section) stays 0
    int cur = rope_start;
    int i = 0;
    while (i < n_tokens) {
        const Runtime::EmbdOverride * span = nullptr;
        for (int s = 0; s < n_spans; ++s) if (spans[s].first == i) { span = &spans[s]; break; }
        if (span) {
            const int cnt = span->count;
            int gw = span->grid_w > 0 ? span->grid_w
                                      : (int) std::lround(std::sqrt((double) cnt));
            if (gw < 1) gw = 1;
            const int gh = span->grid_h > 0 ? span->grid_h : (cnt + gw - 1) / gw;
            const int base = cur;
            for (int k = 0; k < cnt && i < n_tokens; ++k, ++i) {
                pt[i] = base; ph[i] = base + k / gw; pw[i] = base + k % gw;
            }
            cur = base + std::max(gh, gw);
        } else {
            pt[i] = ph[i] = pw[i] = cur; ++cur; ++i;
        }
    }
    return cur;
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

    // verify-mode checkpoints: conv state after token t = timesteps [t+1 .. t+d_conv-1]
    if (gdn_ckpt == n_tokens && n_tokens > 1) {
        for (int t = 0; t + 1 < n_tokens; ++t) {
            ggml_tensor * ck = ggml_view_3d(ctx, conv_input, hp.ssm_d_conv - 1, conv_ch, 1,
                    conv_input->nb[1], conv_input->nb[2],
                    ggml_row_size(conv_input->type, t + 1));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ck, ckpt_conv[t][il]));
        }
    }

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
    ggml_tensor * output;
    if (gdn_ckpt == n_tokens && n_tokens > 1) {
        // verify mode: run the scan per token (same sequential math, same FLOPs)
        // and snapshot the state after each token so a partial accept can restore
        // an intermediate state without re-decoding.
        auto rs = [&](int64_t n){ return ggml_row_size(GGML_TYPE_F32, n); };
        output = nullptr;
        for (int t = 0; t < n_tokens; ++t) {
            auto slice = [&](ggml_tensor * x4) {   // [a,b,n_tokens,1] -> [a,b,1,1] at t
                return ggml_view_4d(ctx, x4, x4->ne[0], x4->ne[1], 1, 1,
                        x4->nb[1], x4->nb[2], x4->nb[3], (size_t) t * x4->nb[2]);
            };
            ggml_tensor * rt = ggml_gated_delta_net(ctx,
                    slice(q), slice(k), slice(v), slice(g), slice(beta), s_in);
            ggml_tensor * out_t = ggml_view_4d(ctx, rt, S, H_v, 1, 1,
                    rs(S), rs(S * H_v), rs(S * H_v), 0);
            ggml_tensor * st_t = ggml_view_4d(ctx, rt, S, S, H_v, 1,
                    rs(S), rs(S * S), rs(S * S * H_v), rs(S * H_v));
            if (t + 1 < n_tokens)
                ggml_build_forward_expand(gf, ggml_cpy(ctx, st_t, ckpt_ssm[t][il]));
            if (t + 1 == n_tokens)
                ggml_build_forward_expand(gf, ggml_cpy(ctx, st_t,
                        ggml_reshape_3d(ctx, ssm_state[il], S, S, H_v)));
            s_in = ggml_view_3d(ctx, rt, S * S * H_v, 1, 1,
                    rs(S * S * H_v), rs(S * S * H_v), rs(S * H_v));   // chain
            output = output ? ggml_concat(ctx, output, out_t, 2) : out_t;
        }
    } else {
        ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, s_in);

        output = ggml_view_4d(ctx, result, S, H_v, n_tokens, 1,
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
    }

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
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rope_dim(n_tokens));
    ggml_set_input(inp_pos); ggml_set_name(inp_pos, "inp_pos");
    ggml_tensor * inp_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_set_input(inp_mask); ggml_set_name(inp_mask, "inp_mask");

    if (persistent) {
        d_kvidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
        ggml_set_input(d_kvidx); ggml_set_name(d_kvidx, "inp_kvidx");
    }

    ggml_tensor * cur;
    ggml_tensor * inpL = ggml_get_rows(ctx, model.tok_embd_rows(), inp_tokens);

    // vision: splice precomputed image embeddings over the <|image_pad|> spans
    if (!embd_ovr.empty() && !persistent) {
        inpL = ggml_cont(ctx, inpL);   // ggml_set needs a writable contiguous dst
        for (size_t k = 0; k < embd_ovr.size(); ++k) {
            const auto & o = embd_ovr[k];
            ggml_tensor * ov = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, o.count);
            ggml_set_input(ov);
            ggml_set_name(ov, ("inp_embd_ovr_" + std::to_string(k)).c_str());
            inpL = ggml_set_2d(ctx, inpL, ov, inpL->nb[1], (size_t) o.first * inpL->nb[1]);
        }
    }

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

            Q = apply_rope(ctx, Q, inp_pos);
            K = apply_rope(ctx, K, inp_pos);

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
        ggml_tensor * h, ggml_tensor * tok, ggml_tensor * pos, ggml_tensor * mask,
        int n_kv, int n_tokens, const Runtime::EmbdOverride * ovr, int n_ovr) {
    const auto & hp = model.hparams();
    const int L           = (int) hp.n_main();      // MTP block index
    const int n_embd      = hp.n_embd;
    const int n_head      = hp.n_head;
    const int n_head_kv   = hp.n_head_kv;
    const int n_embd_head = hp.n_embd_head;
    const float eps       = hp.rms_eps;
    const int   T         = n_tokens;

    // h' = eh_proj( concat( hnorm(h), enorm(emb(tok)) ) )
    ggml_tensor * emb = ggml_get_rows(ctx, model.tok_embd_rows(), tok);            // [n_embd,T]
    // vision: splice the image embeddings over the <|image_pad|> token rows so
    // the nextn KV sees the same inputs as the main stack
    if (n_ovr > 0) {
        emb = ggml_cont(ctx, emb);
        for (int k = 0; k < n_ovr; ++k) {
            ggml_tensor * ov = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, ovr[k].count);
            ggml_set_input(ov);
            ggml_set_name(ov, ("mtp_embd_ovr_" + std::to_string(k)).c_str());
            emb = ggml_set_2d(ctx, emb, ov, emb->nb[1], (size_t) ovr[k].first * emb->nb[1]);
        }
    }
    ggml_tensor * e   = ggml_mul(ctx, ggml_rms_norm(ctx, emb, eps), W("blk.%d.nextn.enorm.weight", L));
    ggml_tensor * h2  = ggml_reshape_2d(ctx, h, n_embd, T);
    ggml_tensor * hn  = ggml_mul(ctx, ggml_rms_norm(ctx, h2, eps), W("blk.%d.nextn.hnorm.weight", L));
    // eh_proj expects [ enorm(emb) ; hnorm(hidden) ]  (embedding first)
    ggml_tensor * combined = ggml_concat(ctx, e, hn, 0);                            // [2*n_embd,T]
    ggml_tensor * cur = ggml_mul_mat(ctx, W("blk.%d.nextn.eh_proj.weight", L), combined); // [n_embd,T]

    // transformer block (gated attention + dense FFN), gated like qwen35
    ggml_tensor * inpSA = cur;
    ggml_tensor * x = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), W("blk.%d.attn_norm.weight", L));

    ggml_tensor * Qf = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", L), x);
    const size_t es = ggml_element_size(Qf);
    ggml_tensor * Q = ggml_view_3d(ctx, Qf, n_embd_head, n_head, T,
            es * n_embd_head * 2, es * n_embd_head * 2 * n_head, 0);
    ggml_tensor * gate_t = ggml_view_3d(ctx, Qf, n_embd_head, n_head, T,
            es * n_embd_head * 2, es * n_embd_head * 2 * n_head, es * n_embd_head);
    gate_t = ggml_cont_2d(ctx, gate_t, n_embd_head * n_head, T);
    ggml_tensor * K = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", L), x), n_embd_head, n_head_kv, T);
    ggml_tensor * V = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", L), x), n_embd_head, n_head_kv, T);
    Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, eps), W("blk.%d.attn_q_norm.weight", L));
    K = ggml_mul(ctx, ggml_rms_norm(ctx, K, eps), W("blk.%d.attn_k_norm.weight", L));
    Q = ggml_rope_ext(ctx, Q, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX, 0, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx, K, pos, nullptr, hp.n_rot, GGML_ROPE_TYPE_NEOX, 0, hp.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * att = build_attn(ctx, gf, L, Q, K, V, mask, T, n_kv);   // KV write uses n_past (= mtp_past, set by caller)
    att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate_t));
    cur = ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", L), att);
    cur = ggml_add(ctx, cur, inpSA);

    ggml_tensor * ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), W("blk.%d.post_attention_norm.weight", L));
    ggml_tensor * ff;
    if (hp.is_moe()) {
        // MoE nextn block (e.g. Qwen3.6-35B-A3B-MTP): experts are VRAM-resident.
        ff = build_moe(ctx, gf, L, ffn_in, T);
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

    if (mtp_headless) return cur;   // KV-write only (resync); skip the shared head

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
    ggml_tensor * logits_t = build_mtp(ctx, gf, h_in, t_in, p_in, m_in, n_kv);
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

// Fast MTP draft: persistent single-token graph on a dedicated backend instance.
// The dedicated backend gives the draft graph its own CUDA-graph slot, so
// alternating draft/verify computes don't evict each other's capture. Reads back
// only the argmax token (4 bytes) and, when chaining, the block hidden.
int32_t Runtime::Impl::mtp_draft_fast(int32_t token, bool need_hidden) {
    const auto & hp = model.hparams();
    const int n_embd = hp.n_embd;
    const int want_nkv = std::min(((mtp_past + 1 + KV_BUCKET - 1) / KV_BUCKET) * KV_BUCKET, n_ctx);

    if (!backend_mtp) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev) backend_mtp = ggml_backend_dev_init(dev, nullptr);
        if (!backend_mtp) backend_mtp = backend;
    }

    if (!m_gf || want_nkv != m_nkv) {
        if (m_ctx) { ggml_free(m_ctx); m_ctx = nullptr; }
        m_nkv = want_nkv;
        ggml_init_params gp{};
        gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
        gp.no_alloc = true;
        m_ctx = ggml_init(gp);
        m_gf = ggml_new_graph_custom(m_ctx, GRAPH_SIZE, false);

        ggml_tensor * h_in = ggml_new_tensor_1d(m_ctx, GGML_TYPE_F32, n_embd);
        ggml_set_input(h_in); ggml_set_name(h_in, "mtp_h");
        ggml_tensor * t_in = ggml_new_tensor_1d(m_ctx, GGML_TYPE_I32, 1);
        ggml_set_input(t_in); ggml_set_name(t_in, "mtp_tok");
        ggml_tensor * p_in = ggml_new_tensor_1d(m_ctx, GGML_TYPE_I32, 1);
        ggml_set_input(p_in); ggml_set_name(p_in, "mtp_pos");
        ggml_tensor * m_in = ggml_new_tensor_2d(m_ctx, GGML_TYPE_F16, m_nkv, 1);
        ggml_set_input(m_in); ggml_set_name(m_in, "mtp_mask");
        d_kvidx = ggml_new_tensor_1d(m_ctx, GGML_TYPE_I64, 1);
        ggml_set_input(d_kvidx); ggml_set_name(d_kvidx, "inp_kvidx");

        persistent = true;
        ggml_tensor * logits_t = build_mtp(m_ctx, m_gf, h_in, t_in, p_in, m_in, m_nkv);
        persistent = false;

        ggml_tensor * am = ggml_argmax(m_ctx, logits_t);
        ggml_set_name(am, "mtp_argmax"); ggml_set_output(am);
        ggml_build_forward_expand(m_gf, am);

        if (!m_galloc) m_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(m_galloc, m_gf))
            throw std::runtime_error("mtp_draft_fast: gallocr alloc failed");
    }

    ggml_tensor * h_in = ggml_graph_get_tensor(m_gf, "mtp_h");
    ggml_tensor * t_in = ggml_graph_get_tensor(m_gf, "mtp_tok");
    ggml_tensor * p_in = ggml_graph_get_tensor(m_gf, "mtp_pos");
    ggml_tensor * m_in = ggml_graph_get_tensor(m_gf, "mtp_mask");
    ggml_tensor * kvix = ggml_graph_get_tensor(m_gf, "inp_kvidx");
    ggml_backend_tensor_set(h_in, mtp_hidden.data(), 0, n_embd * sizeof(float));
    ggml_backend_tensor_set(t_in, &token, 0, sizeof(int32_t));
    int32_t pos = mtp_past; ggml_backend_tensor_set(p_in, &pos, 0, sizeof(int32_t));
    int64_t kvidx = mtp_past; ggml_backend_tensor_set(kvix, &kvidx, 0, sizeof(int64_t));
    std::vector<ggml_fp16_t> mask(m_nkv);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int j = 0; j < m_nkv; ++j) mask[j] = (j <= mtp_past) ? z : ninf;
    ggml_backend_tensor_set(m_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    static const bool prof2 = getenv("QWEN_PROF_MTP2") != nullptr;
    static double pf_set = 0, pf_cmp = 0, pf_get = 0; static long pf_n = 0;
    auto pnow = []{ return std::chrono::steady_clock::now(); };
    auto pms = [](std::chrono::steady_clock::duration d){ return std::chrono::duration<double, std::milli>(d).count(); };
    auto t1 = pnow();

    if (ggml_backend_graph_compute(backend_mtp, m_gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("mtp_draft_fast: compute failed");
    auto t2 = pnow();

    int32_t out = 0;
    ggml_backend_tensor_get(ggml_graph_get_tensor(m_gf, "mtp_argmax"), &out, 0, sizeof(int32_t));
    if (prof2) {
        pf_cmp += pms(t2 - t1); pf_get += pms(pnow() - t2); ++pf_n;
        if (pf_n % 100 == 0)
            fprintf(stderr, "[mtp_draft_fast: n=%ld compute %.2f ms, readback %.2f ms avg]\n",
                    pf_n, pf_cmp / pf_n, pf_get / pf_n);
    }
    if (need_hidden) {
        ggml_tensor * bh = ggml_graph_get_tensor(m_gf, "mtp_blk_hidden");
        mtp_block_hidden.resize(n_embd);
        ggml_backend_tensor_get(bh, mtp_block_hidden.data(), 0, n_embd * sizeof(float));
    }
    mtp_past += 1;
    return out;
}

// MTP KV resync: run the nextn block for `token` (with mtp_hidden as the main
// hidden) only to write its KV row -- no shared head, no logits, no readback.
// Used to rewrite accepted-draft KV entries with the true main hiddens, and to
// build MTP KV over the prompt during prefill.
void Runtime::Impl::mtp_resync(int32_t token) {
    const auto & hp = model.hparams();
    const int n_embd = hp.n_embd;
    const int want_nkv = std::min(((mtp_past + 1 + KV_BUCKET - 1) / KV_BUCKET) * KV_BUCKET, n_ctx);

    if (!backend_mtp) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev) backend_mtp = ggml_backend_dev_init(dev, nullptr);
        if (!backend_mtp) backend_mtp = backend;
    }

    if (!r_gf || want_nkv != r_nkv) {
        if (r_ctx) { ggml_free(r_ctx); r_ctx = nullptr; }
        r_nkv = want_nkv;
        ggml_init_params gp{};
        gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
        gp.no_alloc = true;
        r_ctx = ggml_init(gp);
        r_gf = ggml_new_graph_custom(r_ctx, GRAPH_SIZE, false);

        ggml_tensor * h_in = ggml_new_tensor_1d(r_ctx, GGML_TYPE_F32, n_embd);
        ggml_set_input(h_in); ggml_set_name(h_in, "mtp_h");
        ggml_tensor * t_in = ggml_new_tensor_1d(r_ctx, GGML_TYPE_I32, 1);
        ggml_set_input(t_in); ggml_set_name(t_in, "mtp_tok");
        ggml_tensor * p_in = ggml_new_tensor_1d(r_ctx, GGML_TYPE_I32, 1);
        ggml_set_input(p_in); ggml_set_name(p_in, "mtp_pos");
        ggml_tensor * m_in = ggml_new_tensor_2d(r_ctx, GGML_TYPE_F16, r_nkv, 1);
        ggml_set_input(m_in); ggml_set_name(m_in, "mtp_mask");
        d_kvidx = ggml_new_tensor_1d(r_ctx, GGML_TYPE_I64, 1);
        ggml_set_input(d_kvidx); ggml_set_name(d_kvidx, "inp_kvidx");

        persistent   = true;
        mtp_headless = true;
        build_mtp(r_ctx, r_gf, h_in, t_in, p_in, m_in, r_nkv);
        mtp_headless = false;
        persistent   = false;

        if (!r_galloc) r_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(r_galloc, r_gf))
            throw std::runtime_error("mtp_resync: gallocr alloc failed");
    }

    ggml_tensor * h_in = ggml_graph_get_tensor(r_gf, "mtp_h");
    ggml_tensor * t_in = ggml_graph_get_tensor(r_gf, "mtp_tok");
    ggml_tensor * p_in = ggml_graph_get_tensor(r_gf, "mtp_pos");
    ggml_tensor * m_in = ggml_graph_get_tensor(r_gf, "mtp_mask");
    ggml_tensor * kvix = ggml_graph_get_tensor(r_gf, "inp_kvidx");
    ggml_backend_tensor_set(h_in, mtp_hidden.data(), 0, n_embd * sizeof(float));
    ggml_backend_tensor_set(t_in, &token, 0, sizeof(int32_t));
    int32_t pos = mtp_past; ggml_backend_tensor_set(p_in, &pos, 0, sizeof(int32_t));
    int64_t kvidx = mtp_past; ggml_backend_tensor_set(kvix, &kvidx, 0, sizeof(int64_t));
    std::vector<ggml_fp16_t> mask(r_nkv);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int j = 0; j < r_nkv; ++j) mask[j] = (j <= mtp_past) ? z : ninf;
    ggml_backend_tensor_set(m_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_graph_compute(backend_mtp, r_gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("mtp_resync: compute failed");
    mtp_past += 1;
}

// Batched MTP KV prefill: one headless nextn forward over T tokens (tok[i] paired
// with the main hidden of the previous prompt position, hiddens + i*n_embd),
// writing MTP KV entries mtp_past..mtp_past+T-1. Equivalent to T mtp_resync
// calls but with a single graph build/submit. `ovr` spans are relative to `toks`.
void Runtime::Impl::mtp_prefill_batch(const int32_t * toks, const float * hiddens, int T,
                                      const std::vector<Runtime::EmbdOverride> & ovr) {
    const int n_embd = model.hparams().n_embd;
    const int n_kv   = mtp_past + T;

    ggml_init_params gp{};
    gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
    gp.no_alloc = true;
    ggml_context * ctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);

    ggml_tensor * h_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, T); ggml_set_input(h_in);
    ggml_tensor * t_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);         ggml_set_input(t_in);
    ggml_tensor * p_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);         ggml_set_input(p_in);
    ggml_tensor * m_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, T);   ggml_set_input(m_in);

    const int saved = n_past;
    n_past = mtp_past;                 // build_attn writes MTP KV at this position
    mtp_headless = true;
    build_mtp(ctx, gf, h_in, t_in, p_in, m_in, n_kv, T, ovr.data(), (int) ovr.size());
    mtp_headless = false;
    n_past = saved;

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        throw std::runtime_error("mtp_prefill_batch: gallocr alloc failed");
    }

    ggml_backend_tensor_set(h_in, hiddens, 0, (size_t) T * n_embd * sizeof(float));
    ggml_backend_tensor_set(t_in, toks, 0, T * sizeof(int32_t));
    std::vector<int32_t> pos(T);
    for (int i = 0; i < T; ++i) pos[i] = mtp_past + i;
    ggml_backend_tensor_set(p_in, pos.data(), 0, T * sizeof(int32_t));
    std::vector<ggml_fp16_t> mask((size_t) n_kv * T);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < T; ++i) {
        const int abs_i = mtp_past + i;
        for (int j = 0; j < n_kv; ++j)
            mask[(size_t) i * n_kv + j] = (j <= abs_i) ? z : ninf;
    }
    ggml_backend_tensor_set(m_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    for (size_t k = 0; k < ovr.size(); ++k) {
        ggml_tensor * ov = ggml_graph_get_tensor(gf, ("mtp_embd_ovr_" + std::to_string(k)).c_str());
        if (ov) ggml_backend_tensor_set(ov, ovr[k].data, 0, ggml_nbytes(ov));
    }

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    if (st != GGML_STATUS_SUCCESS)
        throw std::runtime_error("mtp_prefill_batch: compute failed");
    mtp_past += T;
}
// Offload verify: run the 2-token main forward through the VRAM expert cache.
// Uses the batched cache path when the pools can hold both tokens' experts at
// once; otherwise falls back to two single-token decodes (still correct, just
// no shared-fetch amortization).
void Runtime::Impl::decode_verify_cached(const std::vector<int32_t> & toks) {
    const int T = (int) toks.size();
    const int n_used = model.hparams().n_expert_used;
    const bool gdn = model.hparams().has_gdn;
    if (ecache->min_slots() >= T * n_used) {
        // batched path: fills vH and vA (GPU argmax, no logits readback).
        // GDN states are checkpointed per verify token so a partial accept can
        // restore an intermediate state instead of re-decoding (same as the
        // resident decode_verify path).
        if (gdn) init_ckpts(T);
        gdn_ckpt = gdn ? T : 0;
        decode_cached_batch(toks.data(), T, /*want_logits=*/false, /*verify=*/true);
        gdn_ckpt = 0;
        v_from_batch = true;
    } else {
        v_from_batch = false;
        // pools too small to hold all tokens' experts at once: token-by-token,
        // with host argmax over the full logits.
        vL.assign(T, {}); vH.assign(T, {});
        vA.assign(T, 0);
        for (int i = 0; i < T; ++i) {
            vL[i] = decode_cached(toks[i]);
            vH[i] = mtp_hidden;
            const auto & v = vL[i];
            int b = 0;
            for (int j = 1; j < (int) v.size(); ++j) if (v[j] > v[b]) b = j;
            vA[i] = b;
        }
    }
}

// Verify forward: run the main model on `toks` (T tokens) at the current position,
// exposing per-position logits (vL[i]) and main hidden (vH[i]). Advances n_past+=T.
// Uses a persistent graph (rebuilt only when the KV bucket changes) with the GDN
// scan split per token so intermediate states are checkpointed for partial accept.
void Runtime::Impl::decode_verify(const std::vector<int32_t> & toks) {
    if (ecache) { decode_verify_cached(toks); return; }   // expert-offload mode

    const int n_tokens = (int) toks.size();
    const bool gdn = model.hparams().has_gdn;
    const int want_nkv = std::min(((n_past + n_tokens + KV_BUCKET - 1) / KV_BUCKET) * KV_BUCKET, n_ctx);

    if (!v_gf || want_nkv != v_nkv || n_tokens != v_ntok) {
        if (v_ctx) { ggml_free(v_ctx); v_ctx = nullptr; }
        v_nkv  = want_nkv;
        v_ntok = n_tokens;
        ggml_init_params gp{};
        gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
        gp.no_alloc = true;
        v_ctx = ggml_init(gp);
        if (gdn) init_ckpts(n_tokens);
        persistent = true;
        gdn_ckpt   = gdn ? n_tokens : 0;
        v_gf = build_graph(v_ctx, n_tokens, v_nkv);
        gdn_ckpt   = 0;
        persistent = false;
        // accept only needs per-position argmax: compute it on GPU instead of
        // reading back n_tokens x n_vocab logits
        ggml_tensor * am = ggml_argmax(v_ctx, ggml_graph_get_tensor(v_gf, "logits"));
        ggml_set_name(am, "verify_argmax"); ggml_set_output(am);
        ggml_build_forward_expand(v_gf, am);
        if (!v_galloc) v_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(v_galloc, v_gf))
            throw std::runtime_error("decode_verify: gallocr alloc failed");
    }

    ggml_tensor * inp_tokens = ggml_graph_get_tensor(v_gf, "inp_tokens");
    ggml_tensor * inp_pos    = ggml_graph_get_tensor(v_gf, "inp_pos");
    ggml_tensor * inp_mask   = ggml_graph_get_tensor(v_gf, "inp_mask");
    ggml_tensor * inp_kvidx  = ggml_graph_get_tensor(v_gf, "inp_kvidx");
    ggml_backend_tensor_set(inp_tokens, toks.data(), 0, n_tokens * sizeof(int32_t));
    std::vector<int32_t> pos;
    fill_rope_pos(pos, n_tokens, mrope_next);   // verify tokens are text: sequential
    std::vector<int64_t> kvi(n_tokens);
    for (int i = 0; i < n_tokens; ++i) kvi[i] = n_past + i;
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(inp_kvidx, kvi.data(), 0, n_tokens * sizeof(int64_t));
    std::vector<ggml_fp16_t> mask((size_t) v_nkv * n_tokens);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n_tokens; ++i) {
        const int abs_i = n_past + i;
        for (int j = 0; j < v_nkv; ++j) mask[(size_t) i * v_nkv + j] = (j <= abs_i) ? z : ninf;
    }
    ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_graph_compute(backend, v_gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("decode_verify: compute failed");

    ggml_tensor * h  = ggml_graph_get_tensor(v_gf, "main_hidden");
    const int n_embd = (int) h->ne[0];
    vA.assign(n_tokens, 0);
    ggml_backend_tensor_get(ggml_graph_get_tensor(v_gf, "verify_argmax"),
                            vA.data(), 0, n_tokens * sizeof(int32_t));
    vH.assign(n_tokens, std::vector<float>(n_embd));
    for (int i = 0; i < n_tokens; ++i)
        ggml_backend_tensor_get(h, vH[i].data(), (size_t) i * h->nb[1], n_embd * sizeof(float));

    n_past += n_tokens;
}

// MTP self-speculative greedy decode. Drafts `n_draft` tokens by chaining the
// single nextn block (each draft feeds the block's own hidden into the next),
// verifies them with one (n_draft+1)-token main forward, and accepts the longest
// matching prefix. n_draft=1 reduces exactly to the 2-token verify.
// Advance the main KV (and, when mtp_kv, the nextn KV + mtp_hidden) over
// `toks` without sampling: one chunk of a (possibly preemptible) prefill.
// Consumes embd_ovr (indices relative to `toks`). Leaves the last token's
// logits in `logits`; with mtp_kv the invariants mtp_past == n_past - 1 and
// mtp_hidden == h(last) hold afterwards, so generation or another chunk can
// follow seamlessly.
void Runtime::Impl::prefill(const std::vector<int32_t> & toks, bool mtp_kv) {
    const int P = (int) toks.size();
    if (P == 0) return;
    if (!mtp_kv) { decode(toks); return; }

    const int n_embd = model.hparams().n_embd;
    static const bool no_batch_prefill = getenv("QWEN_MTP_NO_BATCH_PREFILL") != nullptr;

    // continuation from earlier tokens (cached prefix or a previous chunk):
    // mtp_hidden holds the last token's hidden; pair it with the first new
    // token so the nextn KV chain stays gapless.
    if (n_past > 0) {
        std::vector<Runtime::EmbdOverride> bovr;
        for (const auto & o : embd_ovr)
            if (o.first == 0) bovr.push_back({ 0, 1, o.data });
        if (!bovr.empty()) mtp_prefill_batch(&toks[0], mtp_hidden.data(), 1, bovr);
        else               mtp_resync(toks[0]);
    }

    if (P > 1 && (!no_batch_prefill || !embd_ovr.empty())) {
        const std::vector<Runtime::EmbdOverride> povr = embd_ovr;   // decode() consumes the member
        bh_all.clear();
        want_bh_all = true;
        decode(toks);                      // chunking + embd overrides handled inside
        want_bh_all = false;
        // MTP KV for positions 0..P-2: token i+1 paired with hidden h_i
        const int MCHUNK = 512;            // bounds the [n_kv, T] mask allocation
        int i = 0;
        while (i < P - 1) {
            const int t = std::min(MCHUNK, P - 1 - i);
            // image spans clipped to this chunk, shifted to the nextn token
            // array (its token j is toks[i+1+j])
            std::vector<Runtime::EmbdOverride> ovr;
            for (const auto & o : povr) {
                const int lo = std::max(o.first, i + 1);
                const int hi = std::min(o.first + o.count, i + 1 + t);
                if (lo < hi) ovr.push_back({ lo - (i + 1), hi - lo,
                                             o.data + (size_t) (lo - o.first) * n_embd });
            }
            mtp_prefill_batch(&toks[i + 1], bh_all.data() + (size_t) i * n_embd, t, ovr);
            i += t;
        }
        mtp_hidden.assign(bh_all.begin() + (size_t) (P - 1) * n_embd,
                          bh_all.begin() + (size_t) P * n_embd);
        bh_all.clear();
        bh_all.shrink_to_fit();
    } else {
        for (int i = 0; i < P; ++i) {
            decode({ toks[i] });
            if (i + 1 < P) mtp_resync(toks[i + 1]);   // KV only, no head
        }
    }
}

void Runtime::Impl::generate_mtp(const std::vector<int32_t> & prompt, int max_new, int n_draft,
                                 const std::function<bool(int32_t)> & on_token,
                                 int32_t * out_pending) {
    const auto & hp = model.hparams();
    const bool gdn = hp.has_gdn;
    const int  K   = n_draft < 1 ? 1 : n_draft;
    static const bool no_accept = getenv("QWEN_MTP_NOACCEPT") != nullptr;
    static const bool prof      = getenv("QWEN_PROF_MTP") != nullptr;
    double ms_draft = 0, ms_verify = 0, ms_settle = 0, ms_resync = 0;
    long n_settle = 0;
    using pclk = std::chrono::steady_clock;
    auto msec = [](pclk::duration d){ return std::chrono::duration<double, std::milli>(d).count(); };
    auto argmax = [](const std::vector<float> & v) {
        int b = 0; for (int i = 1; i < (int) v.size(); ++i) if (v[i] > v[b]) b = i; return b;
    };

    // checkpoint mode: the verify graph snapshots per-token GDN states, so a
    // partial accept restores a checkpoint instead of re-decoding (offload mode
    // keeps the backup + re-decode path: its verify runs through the cache).
    const bool use_ckpt = gdn && !ecache;

    // prefill: one batched main forward captures every token's hidden (bh_all),
    // then batched headless nextn forwards build the MTP KV (so drafts have
    // history). See Impl::prefill. The runtime may already hold a cached
    // prefix (n_past > 0); prefill bridges the nextn KV across the boundary.
    const int P = (int) prompt.size();
    if (P > 0) prefill(prompt, /*mtp_kv=*/true);
    std::vector<float> mlog = logits;    // logits of the last prefilled token
    kv_toks.insert(kv_toks.end(), prompt.begin(), prompt.end());
    int32_t x = argmax(mlog);            // first generated token
    // invariant at loop top: n_past = pos(x), mtp_past = pos(x)-1, mtp_hidden = h_{pos(x)-1}
    int generated = 0;
    long steps = 0, draft_forwards = 0, accepted_drafts = 0;

    while (generated < max_new) {
        if (!on_token(x)) { if (out_pending) *out_pending = x; break; }
        if (++generated >= max_new) break;

        const int p   = n_past;            // x lands at main position p
        const int mp0 = mtp_past;          // = p-1
        const int m0  = mrope_next;        // rope position of x (all gen tokens are text)

        // ---- draft K tokens by chaining the MTP block ----
        auto td0 = pclk::now();
        std::vector<int32_t> drafts; drafts.reserve(K);
        {
            int32_t t = x;                 // first draft uses the true main hidden in mtp_hidden
            for (int j = 0; j < K; ++j) {
                int32_t dj = mtp_draft_fast(t, j + 1 < K);   // writes MTP KV, advances mtp_past
                drafts.push_back(dj);
                ++draft_forwards;
                if (j + 1 < K) { mtp_hidden = mtp_block_hidden; t = dj; }   // chain on the block hidden
            }
        }
        ms_draft += msec(pclk::now() - td0);

        // ---- verify [x, d_1..d_K] in one (K+1)-token main forward ----
        std::vector<int32_t> vtoks; vtoks.reserve(K + 1);
        vtoks.push_back(x);
        for (int j = 0; j < K; ++j) vtoks.push_back(drafts[j]);
        auto tv0 = pclk::now();
        if (gdn && !use_ckpt) backup_states();
        decode_verify(vtoks);              // fills vL[0..K], vH[0..K]; n_past += K+1
        ms_verify += msec(pclk::now() - tv0);
        ++steps;

        // ---- accept the longest matching draft prefix ----
        int a = 0;
        if (!no_accept) while (a < K && vA[a] == drafts[a]) ++a;
        accepted_drafts += a;
        const int32_t x_new = vA[a];           // correction (or bonus token if a==K)

        // ---- settle main KV / recurrent state to the a+1 confirmed tokens ----
        auto ts0 = pclk::now();
        if (a == K) {
            // full accept: the verify forward already left the correct state
        } else if (use_ckpt || (gdn && v_from_batch)) {
            restore_ckpt(a);                   // GDN state after verify token a (= x,d_1..d_a)
            n_past = p + a + 1;                // KV[p..p+a] from verify is valid; drop the rest
        } else if (gdn) {
            restore_states();
            n_past = p;
            std::vector<int32_t> conf; conf.reserve(a + 1);
            conf.push_back(x);
            for (int j = 0; j < a; ++j) conf.push_back(drafts[j]);
            decode(conf);                      // redo a+1 tokens for correct GDN state
            ++n_settle;
        } else {
            n_past = p + a + 1;                // KV[p..p+a] from verify is valid; drop the rest
        }
        ms_settle += msec(pclk::now() - ts0);

        // ---- re-sync MTP KV for the confirmed tokens using the true main hiddens ----
        auto tr0 = pclk::now();
        mtp_past = mp0 + 1;                    // keep index p-1 (draft 1 used the true hidden)
        for (int j = 0; j < a; ++j) {
            mtp_hidden = vH[j];                // true h_{p+j}
            mtp_resync(drafts[j]);             // rewrite MTP KV index p+j (headless: KV only)
        }
        mtp_hidden = vH[a];                    // h_{p+a}: draft context for x_new
        ms_resync += msec(pclk::now() - tr0);

        kv_toks.push_back(x);                  // the a+1 confirmed tokens now in KV
        for (int j = 0; j < a; ++j) kv_toks.push_back(drafts[j]);
        mrope_next = m0 + a + 1;               // a+1 confirmed text tokens (overrides any
                                               // advance the GDN re-decode settle made)

        // emit accepted drafts AFTER settle/resync, so an early exit (stop
        // token, budget) leaves the state consistent with kv_toks for reuse
        bool stop = false;
        for (int j = 0; j < a; ++j) {
            if (!on_token(drafts[j])) {
                // drafts[j..a-1] are confirmed in kv_toks but were never
                // delivered; x_new is the next undecoded token after them
                if (out_pending) *out_pending = x_new;
                stop = true; break;
            }
            if (++generated >= max_new) { stop = true; break; }
        }
        if (stop) break;
        x = x_new;
    }
    if (prof && steps > 0)
        fprintf(stderr, "[MTP prof: draft %.0fms verify %.0fms settle %.0fms (%ld re-decodes) resync %.0fms | per-cycle: draft %.1f verify %.1f settle %.1f resync %.1f ms]\n",
                ms_draft, ms_verify, ms_settle, n_settle, ms_resync,
                ms_draft / steps, ms_verify / steps, ms_settle / steps, ms_resync / steps);
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
    std::vector<int32_t> posv;
    fill_rope_pos(posv, 1, mrope_next);   // generation token: text, sequential
    ggml_backend_tensor_set(inp_pos, posv.data(), 0, posv.size() * sizeof(int32_t));
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
    mrope_next += 1;   // generation token is text: t=h=w advance by 1
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
                                        bool verify,
                                        const Runtime::EmbdOverride * ovr, int n_ovr) {
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

    // temp carry tensors sized for this batch (bridge seg A->B and layer->layer).
    // ffn_in/resid/weights are double-buffered (parity by layer) so the fused
    // segB(L)+segA(L+1) graph has no write-after-read hazard (mirrors decode_cached).
    ggml_init_params tp{};
    tp.mem_size = ggml_tensor_overhead() * 12 + 256;
    tp.no_alloc = true;
    ggml_context * tctx = ggml_init(tp);
    ggml_tensor * h_b       = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * ffn_in_b  = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * resid_b   = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * weights_b = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_used, T);
    ggml_tensor * ffn_in_b2 = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * resid_b2  = ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * weights_b2= ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_used, T);
    ggml_tensor * slot_g_b  = ggml_new_tensor_2d(tctx, GGML_TYPE_I32, n_used, T);
    ggml_tensor * slot_u_b  = ggml_new_tensor_2d(tctx, GGML_TYPE_I32, n_used, T);
    ggml_tensor * slot_d_b  = ggml_new_tensor_2d(tctx, GGML_TYPE_I32, n_used, T);
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

    // vision: overwrite the image-span rows of the embed output with the
    // precomputed image embeddings (spans are relative to `toks`)
    for (int k = 0; k < n_ovr; ++k)
        ggml_backend_tensor_set(h_b, ovr[k].data,
            (size_t) ovr[k].first * n_embd * sizeof(float),
            (size_t) ovr[k].count * n_embd * sizeof(float));

    std::vector<int32_t> sel(n_used * T), sg(n_used * T), su(n_used * T), sd(n_used * T);

    ggml_tensor * carry_ffn[2] = { ffn_in_b,  ffn_in_b2  };
    ggml_tensor * carry_res[2] = { resid_b,   resid_b2   };
    ggml_tensor * carry_wgt[2] = { weights_b, weights_b2 };

    // Append seg A (attention/GDN + router, all T tokens) for layer `il`; writes
    // the layer's parity carry and exposes `selected` for host readback.
    auto build_segA = [&](ggml_context * ctx, ggml_cgraph * gf, int il) -> ggml_tensor * {
        const bool recurrent = hp.is_recurrent(il);
        ggml_tensor * inp_pos = nullptr, * inp_mask = nullptr;
        if (!recurrent) {
            inp_pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rope_dim(T));
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
            Q = apply_rope(ctx, Q, inp_pos);
            K = apply_rope(ctx, K, inp_pos);
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

        ggml_build_forward_expand(gf, ggml_cpy(ctx, ffn_in, carry_ffn[il & 1]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, attn_resid, carry_res[il & 1]));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, weights, carry_wgt[il & 1]));
        return selected;
    };

    // Append seg B (batched expert matmuls + residual) for layer `il`; writes h_b.
    // mul_mat_id takes the whole batch at once (ids [n_used, T]); the weighted
    // sum over n_used mirrors build_moe's batched path.
    auto build_segB = [&](ggml_context * ctx, ggml_cgraph * gf, int il) {
        ggml_tensor * ffn_l = carry_ffn[il & 1];
        ggml_tensor * x3   = ggml_reshape_3d(ctx, ffn_l, n_embd, 1, T);
        ggml_tensor * up   = ggml_mul_mat_id(ctx, ecache->up(il),   x3, slot_u_b);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, ecache->gate(il), x3, slot_g_b);
        ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, ecache->down(il), act, slot_d_b); // [n_embd, n_used, T]
        experts = ggml_mul(ctx, experts, ggml_reshape_3d(ctx, carry_wgt[il & 1], 1, n_used, T));
        ggml_tensor * moe_out = nullptr;
        for (int i = 0; i < n_used; ++i) {
            ggml_tensor * v = ggml_view_2d(ctx, experts, n_embd, T, experts->nb[2], (size_t) i * experts->nb[1]);
            moe_out = i ? ggml_add(ctx, moe_out, v) : v;
        }
        if (n_used == 1) moe_out = ggml_cont(ctx, moe_out);

        if (ggml_tensor * up_sh = Wopt("blk.%d.ffn_up_shexp.weight", il)) {
            ggml_tensor * g  = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_shexp.weight", il), ffn_l);
            ggml_tensor * u  = ggml_mul_mat(ctx, up_sh, ffn_l);
            ggml_tensor * sh = ggml_mul_mat(ctx, W("blk.%d.ffn_down_shexp.weight", il),
                                            ggml_mul(ctx, ggml_silu(ctx, g), u));
            ggml_tensor * sgt = ggml_sigmoid(ctx, ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp_shexp.weight", il), ffn_l));
            moe_out = ggml_add(ctx, moe_out, ggml_mul(ctx, sh, sgt));
        }
        ggml_tensor * h_new = ggml_add(ctx, moe_out, carry_res[il & 1]);         // [n_embd, T]
        ggml_build_forward_expand(gf, ggml_cpy(ctx, h_new, h_b));
    };

    // Set the attention pos/mask inputs of a graph (no-op for GDN-only graphs).
    // M-RoPE positions use the chunk-relative image spans passed in ovr/n_ovr.
    auto set_attn_inputs = [&](ggml_cgraph * gf) {
        if (ggml_tensor * ip = ggml_graph_get_tensor(gf, "inp_pos")) {
            std::vector<int32_t> pos;
            fill_rope_pos_spans(pos, T, mrope_next, ovr, n_ovr);
            ggml_backend_tensor_set(ip, pos.data(), 0, pos.size() * sizeof(int32_t));
        }
        if (ggml_tensor * im = ggml_graph_get_tensor(gf, "inp_mask")) {
            std::vector<ggml_fp16_t> mask((size_t) n_kv * T);
            const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
            for (int i = 0; i < T; ++i) {
                const int abs_i = n_past + i;
                for (int j = 0; j < n_kv; ++j)
                    mask[(size_t) i * n_kv + j] = (j <= abs_i) ? z : ninf;
            }
            ggml_backend_tensor_set(im, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
    };

    // Read back layer `il`'s router selection, make the union of selected
    // experts resident, and upload the slot ids for seg B.
    auto ensure_layer = [&](ggml_tensor * selected, int il) {
        ggml_backend_tensor_get(selected, sel.data(), 0, (size_t) n_used * T * sizeof(int32_t));
        ecache->ensure(il, sel.data(), n_used * T, sg.data(), su.data(), sd.data());
        ggml_backend_tensor_set(slot_g_b, sg.data(), 0, (size_t) n_used * T * sizeof(int32_t));
        ggml_backend_tensor_set(slot_u_b, su.data(), 0, (size_t) n_used * T * sizeof(int32_t));
        ggml_backend_tensor_set(slot_d_b, sd.data(), 0, (size_t) n_used * T * sizeof(int32_t));
    };

    const int N = (int) hp.n_main();

    // seg A(0) on its own, then fuse segB(L)+segA(L+1) per step so each layer
    // boundary is a single GPU submit instead of two (mirrors decode_cached).
    {
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * selected = build_segA(ctx, gf, 0);
        run(ctx, gf);
        set_attn_inputs(gf);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("decode_cached_batch: seg A0 compute failed");
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
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("decode_cached_batch: fused segB/segA compute failed");
        if (nsel) ensure_layer(nsel, il + 1);
        ggml_free(ctx);
    }

    // MTP batched prefill: append every token's final hidden (pre-output-norm)
    if (want_bh_all) {
        const size_t base = bh_all.size();
        bh_all.resize(base + (size_t) T * n_embd);
        ggml_backend_tensor_get(h_b, bh_all.data() + base, 0, (size_t) T * n_embd * sizeof(float));
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

    // MTP verify: expose per-position argmax (vA[i], GPU-computed) and main
    // hidden (vH[i]) for all T tokens so the caller can accept the longest
    // matching draft prefix. Full logits are never read back (T x n_vocab
    // floats would dominate the readback cost on large-vocab models).
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
        ggml_tensor * am = ggml_argmax(ctx, cur);                               // [T] i32
        ggml_set_name(am, "verify_argmax");
        ggml_build_forward_expand(gf, am);
        run(ctx, gf);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("decode_cached_batch: verify output compute failed");
        vA.assign(T, 0);
        ggml_backend_tensor_get(am, vA.data(), 0, T * sizeof(int32_t));
        ggml_free(ctx);
    }

    ggml_backend_buffer_free(tbuf);
    ggml_free(tctx);
    n_past += T;
    if (!verify) {                       // prefill chunk: advance the rope counter
        std::vector<int32_t> tmp;        // (verify is speculative; caller manages it)
        mrope_next = fill_rope_pos_spans(tmp, T, mrope_next, ovr, n_ovr);
    }
}

// Single-token decode using the dynamic VRAM expert cache.
// Each layer is run as two GPU graph segments around a host sync point:
//   seg A: attention/GDN + router  -> read back selected experts
//   (host) ensure experts resident in VRAM cache (stream misses)
//   seg B: expert matmuls on cache slots + shared expert + residual
// Correctness follows the *actual* routing; the cache only changes where the
// expert weights are fetched from (VRAM hit vs CPU/SSD miss).
const std::vector<float> & Runtime::Impl::decode_cached(int32_t token, const float * embd_override) {
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
    if (embd_override) {
        // vision: this position's embedding is a precomputed image embedding
        ggml_backend_tensor_set(p_h, embd_override, 0, (size_t) n_embd * sizeof(float));
    } else {
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
            inp_pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rope_dim(1));
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
            Q = apply_rope(ctx, Q, inp_pos);
            K = apply_rope(ctx, K, inp_pos);
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
            std::vector<int32_t> posv;
            fill_rope_pos(posv, 1, mrope_next);
            ggml_backend_tensor_set(ip, posv.data(), 0, posv.size() * sizeof(int32_t));
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
    mrope_next += 1;   // single-token decode is a text generation step
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

// Allocate per-token GDN state checkpoints for a T-token verify forward
// (T-1 snapshots: state after verify token t, t = 0..T-2).
void Runtime::Impl::init_ckpts(int T) {
    const auto & hp = model.hparams();
    if (!hp.has_gdn || ckpt_T >= T) return;
    if (ckpt_buf) { ggml_backend_buffer_free(ckpt_buf); ckpt_buf = nullptr; }
    if (ckpt_ctx) { ggml_free(ckpt_ctx); ckpt_ctx = nullptr; }
    const int n_layer = (int) hp.n_layer;
    ggml_init_params cp{};
    cp.mem_size = ggml_tensor_overhead() * (size_t) n_layer * 2 * (T - 1) + 256;
    cp.no_alloc = true;
    ckpt_ctx = ggml_init(cp);
    ckpt_conv.assign(T - 1, std::vector<ggml_tensor *>(n_layer, nullptr));
    ckpt_ssm.assign(T - 1,  std::vector<ggml_tensor *>(n_layer, nullptr));
    for (int t = 0; t + 1 < T; ++t)
        for (int il = 0; il < n_layer; ++il) {
            if (!conv_state[il]) continue;
            ckpt_conv[t][il] = ggml_new_tensor(ckpt_ctx, conv_state[il]->type,
                    ggml_n_dims(conv_state[il]), conv_state[il]->ne);
            ckpt_ssm[t][il]  = ggml_new_tensor(ckpt_ctx, ssm_state[il]->type,
                    ggml_n_dims(ssm_state[il]), ssm_state[il]->ne);
        }
    ckpt_buf = ggml_backend_alloc_ctx_tensors(ckpt_ctx, backend);
    if (!ckpt_buf) throw std::runtime_error("failed to alloc GDN checkpoints");
    ckpt_T = T;
}

void Runtime::Impl::restore_ckpt(int t) {
    for (int il = 0; il < (int) model.hparams().n_layer; ++il) {
        if (!ckpt_conv[t][il]) continue;
        ggml_backend_tensor_copy(ckpt_conv[t][il], conv_state[il]);
        ggml_backend_tensor_copy(ckpt_ssm[t][il],  ssm_state[il]);
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
    std::vector<int32_t> posv;
    fill_rope_pos(posv, 1, mrope_next);   // generation token: text, sequential
    ggml_backend_tensor_set(inp_pos, posv.data(), 0, posv.size() * sizeof(int32_t));
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
    mrope_next += 1;   // generation token is text: t=h=w advance by 1
    return logits;
}

const std::vector<float> & Runtime::Impl::decode(const std::vector<int32_t> & tokens) {
    const int n_tokens = (int) tokens.size();
    if (n_tokens == 0) throw std::runtime_error("decode: empty tokens");
    if (n_past + n_tokens > n_ctx) throw std::runtime_error("decode: context overflow");

    // single-token decode via the dynamic VRAM expert cache (expert-offload mode)
    if (n_tokens == 1 && ecache)
        return cache_fast_enabled ? decode_cached_fast(tokens[0]) : decode_cached(tokens[0]);

    // Prefill as batched chunks through the VRAM expert cache (chunk bounded so
    // a layer's distinct experts fit the pools). Used for both offload tiers:
    // the experts run on the GPU (fetched from pinned RAM or SSD) instead of on
    // CPU via the scheduler, which otherwise leaves the GPU idle during prefill.
    // QWEN_CPU_PREFILL forces the old RAM-tier behavior (experts on CPU/sched).
    static const bool cpu_prefill = getenv("QWEN_CPU_PREFILL") != nullptr;
    if (ecache && !(sched && cpu_prefill)) {
        // Batched prefill: each chunk runs one segmented forward (seg A
        // attention/GDN/router for all tokens -> one expert-residency sync per
        // layer -> batched seg B expert matmuls). Argmax-equivalent to the
        // token-by-token path; one fetch per distinct expert per layer serves
        // the whole chunk, and per-layer GPU submits replace per-token ones.
        // Opt out with QWEN_NO_BATCH_PREFILL (not available while capturing
        // per-token hiddens for the MTP batched prefill).
        static const bool no_batch = getenv("QWEN_NO_BATCH_PREFILL") != nullptr;
        if (no_batch && !want_bh_all) {
            // token-by-token, with per-token image-embedding override lookup
            auto override_for = [&](int i) -> const float * {
                for (const auto & o : embd_ovr)
                    if (i >= o.first && i < o.first + o.count)
                        return o.data + (size_t) (i - o.first) * model.hparams().n_embd;
                return nullptr;
            };
            for (int i = 0; i < n_tokens; ++i) decode_cached(tokens[i], override_for(i));
            embd_ovr.clear();
            return logits;
        }
        const int n_embd = model.hparams().n_embd;
        int chunk = ecache->min_slots() / (model.hparams().n_expert_used > 0 ? model.hparams().n_expert_used : 1);
        if (chunk < 1)   chunk = 1;
        if (chunk > 256) chunk = 256;
        if (const char * c = getenv("QWEN_BATCH_CHUNK")) { int v = atoi(c); if (v >= 1) chunk = v; }
        int i = 0;
        while (i < n_tokens) {
            const int t = std::min(chunk, n_tokens - i);
            // image spans clipped to this chunk (chunk-relative)
            std::vector<Runtime::EmbdOverride> ovr;
            for (const auto & o : embd_ovr) {
                const int lo = std::max(o.first, i);
                const int hi = std::min(o.first + o.count, i + t);
                if (lo < hi) ovr.push_back({ lo - i, hi - lo,
                                             o.data + (size_t) (lo - o.first) * n_embd });
            }
            decode_cached_batch(&tokens[i], t, /*want_logits=*/ i + t >= n_tokens, /*verify=*/false,
                                ovr.empty() ? nullptr : ovr.data(), (int) ovr.size());
            i += t;
            if (progress_cb) progress_cb(i, n_tokens);
        }
        embd_ovr.clear();
        return logits;
    }

    // Long-prompt prefill through the build_graph path (resident, or RAM-tier
    // offload which uses the backend scheduler): a single graph would be a big
    // O(n^2) attention (the scores/mask scale with n_kv*n_tokens -> multi-GB
    // and OOM at ~8k tokens) and one uninterruptible compute that blocks
    // time-slicing. Split it into chunks (bounded buffers, progress reporting,
    // preemptible). The SSD-tier path (ecache && !sched) already chunks via the
    // decode_cached_batch loop above, so it is excluded here.
    static const int PF_CHUNK = []{ const char * c = getenv("QWEN_PREFILL_CHUNK");
                                    int v = c ? atoi(c) : 512; return v < 1 ? 512 : v; }();
    if (n_tokens > PF_CHUNK && (sched || !ecache)) {
        const int n_embd = model.hparams().n_embd;
        const std::vector<Runtime::EmbdOverride> all = embd_ovr;   // member is consumed per call
        int i = 0;
        while (i < n_tokens) {
            const int t = std::min(PF_CHUNK, n_tokens - i);
            std::vector<Runtime::EmbdOverride> ovr;                // image spans clipped to chunk
            for (const auto & o : all) {
                const int lo = std::max(o.first, i), hi = std::min(o.first + o.count, i + t);
                if (lo < hi) ovr.push_back({ lo - i, hi - lo,
                                             o.data + (size_t) (lo - o.first) * n_embd });
            }
            embd_ovr = std::move(ovr);
            decode(std::vector<int32_t>(tokens.begin() + i, tokens.begin() + i + t));
            i += t;
            if (progress_cb) progress_cb(i, n_tokens);
        }
        embd_ovr.clear();
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

    std::vector<int32_t> pos;
    const int new_mrope = fill_rope_pos(pos, n_tokens, mrope_next);
    ggml_backend_tensor_set(inp_pos_t, pos.data(), 0, pos.size() * sizeof(int32_t));

    std::vector<ggml_fp16_t> mask((size_t) n_kv * n_tokens);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int i = 0; i < n_tokens; ++i) {
        const int abs_i = n_past + i;
        for (int j = 0; j < n_kv; ++j)
            mask[(size_t) i * n_kv + j] = (j <= abs_i) ? z : ninf;
    }
    ggml_backend_tensor_set(inp_mask_t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    // vision: upload the image embeddings for each override span
    for (size_t k = 0; k < embd_ovr.size(); ++k) {
        ggml_tensor * ov = ggml_graph_get_tensor(gf, ("inp_embd_ovr_" + std::to_string(k)).c_str());
        if (ov) ggml_backend_tensor_set(ov, embd_ovr[k].data, 0, ggml_nbytes(ov));
    }
    embd_ovr.clear();   // one-shot: applies to this batch only

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

    // MTP batched prefill: append every token's final hidden (pre-output-norm)
    if (want_bh_all) {
        ggml_tensor * h = ggml_graph_get_tensor(gf, "main_hidden");
        if (h) {
            const int ne = (int) h->ne[0];
            const size_t base = bh_all.size();
            bh_all.resize(base + (size_t) n_tokens * ne);
            for (int i = 0; i < n_tokens; ++i)
                ggml_backend_tensor_get(h, bh_all.data() + base + (size_t) i * ne,
                                        (size_t) i * h->nb[1], ne * sizeof(float));
        }
    }

    n_past += n_tokens;
    mrope_next = new_mrope;
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
    const std::vector<float> & l = impl_->decode(tokens);
    impl_->kv_toks.insert(impl_->kv_toks.end(), tokens.begin(), tokens.end());
    return l;
}
void Runtime::set_embd_overrides(std::vector<EmbdOverride> ovr) {
    impl_->embd_ovr = std::move(ovr);
}
const std::vector<float> & Runtime::mtp_draft(int32_t token) { return impl_->mtp_draft(token); }
bool Runtime::has_mtp() const { return impl_->model.hparams().has_mtp(); }
void Runtime::generate_mtp(const std::vector<int32_t> & prompt, int max_new, int n_draft,
                           const std::function<bool(int32_t)> & on_token,
                           int32_t * out_pending) {
    impl_->generate_mtp(prompt, max_new, n_draft, on_token, out_pending);
}
void Runtime::prefill(const std::vector<int32_t> & tokens, bool mtp_kv) {
    impl_->prefill(tokens, mtp_kv);
    impl_->kv_toks.insert(impl_->kv_toks.end(), tokens.begin(), tokens.end());
}
void Runtime::reset()        { impl_->n_past = 0; impl_->mtp_past = 0; impl_->mrope_next = 0; impl_->kv_toks.clear(); impl_->zero_states(); }
int  Runtime::n_past() const { return impl_->n_past; }
const std::vector<int32_t> & Runtime::kv_tokens() const { return impl_->kv_toks; }
bool Runtime::has_expert_cache() const { return impl_->ecache != nullptr; }
Runtime::CacheStats Runtime::cache_stats() const {
    CacheStats c;
    if (impl_->ecache) {
        const auto & s = impl_->ecache->stats();
        c.hits = s.hits; c.misses = s.misses; c.fetch_ms = s.fetch_ms; c.fetch_bytes = s.fetch_bytes;
    }
    return c;
}
void Runtime::set_progress_cb(std::function<void(int, int)> cb) { impl_->progress_cb = std::move(cb); }

// ---- prompt-cache slot state save/load ----
// Sequential stream: header, kv_toks, mtp_hidden, then per layer either the
// KV prefix rows (attention layers; the trailing nextn layers use mtp_past)
// or the full conv/ssm states (GDN layers).
namespace {
struct StateHeader {
    uint32_t magic;      // 'QSS2'
    int32_t  n_layer, n_ctx, n_embd_gqa;
    int32_t  n_past, mtp_past, n_toks, n_hidden, mrope_next;
};
constexpr uint32_t STATE_MAGIC = 0x32535351;   // "QSS2" little-endian
}

size_t Runtime::state_bytes() const {
    const auto & hp = impl_->model.hparams();
    size_t n = sizeof(StateHeader);
    n += impl_->kv_toks.size() * sizeof(int32_t);
    n += impl_->mtp_hidden.size() * sizeof(float);
    for (int il = 0; il < (int) hp.n_layer; ++il) {
        if (impl_->k_cache[il]) {
            const int rows = il >= (int) hp.n_main() ? impl_->mtp_past : impl_->n_past;
            n += 2 * (size_t) rows * impl_->k_cache[il]->nb[1];
        } else {
            n += ggml_nbytes(impl_->conv_state[il]) + ggml_nbytes(impl_->ssm_state[il]);
        }
    }
    return n;
}

void Runtime::save_state(const std::function<void(const void *, size_t)> & sink) const {
    const auto & hp = impl_->model.hparams();
    StateHeader h{};
    h.magic      = STATE_MAGIC;
    h.n_layer    = (int32_t) hp.n_layer;
    h.n_ctx      = impl_->n_ctx;
    h.n_embd_gqa = (int32_t) (hp.n_head_kv * hp.n_embd_head);
    h.n_past     = impl_->n_past;
    h.mtp_past   = impl_->mtp_past;
    h.n_toks     = (int32_t) impl_->kv_toks.size();
    h.n_hidden   = (int32_t) impl_->mtp_hidden.size();
    h.mrope_next = impl_->mrope_next;
    sink(&h, sizeof(h));
    if (h.n_toks)   sink(impl_->kv_toks.data(),    h.n_toks   * sizeof(int32_t));
    if (h.n_hidden) sink(impl_->mtp_hidden.data(), h.n_hidden * sizeof(float));

    std::vector<uint8_t> buf;
    auto dump = [&](ggml_tensor * t, size_t nbytes) {
        if (nbytes == 0) return;
        buf.resize(nbytes);
        ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
        sink(buf.data(), nbytes);
    };
    for (int il = 0; il < h.n_layer; ++il) {
        if (impl_->k_cache[il]) {
            const int rows = il >= (int) hp.n_main() ? h.mtp_past : h.n_past;
            dump(impl_->k_cache[il], (size_t) rows * impl_->k_cache[il]->nb[1]);
            dump(impl_->v_cache[il], (size_t) rows * impl_->v_cache[il]->nb[1]);
        } else {
            dump(impl_->conv_state[il], ggml_nbytes(impl_->conv_state[il]));
            dump(impl_->ssm_state[il],  ggml_nbytes(impl_->ssm_state[il]));
        }
    }
}

void Runtime::load_state(const std::function<void(void *, size_t)> & src) {
    const auto & hp = impl_->model.hparams();
    StateHeader h{};
    src(&h, sizeof(h));
    if (h.magic != STATE_MAGIC ||
        h.n_layer != (int32_t) hp.n_layer || h.n_ctx != impl_->n_ctx ||
        h.n_embd_gqa != (int32_t) (hp.n_head_kv * hp.n_embd_head) ||
        h.n_past < 0 || h.n_past > impl_->n_ctx || h.mtp_past < 0 ||
        h.n_toks < 0 || h.n_hidden < 0)
        throw std::runtime_error("load_state: header mismatch (different model/n_ctx or corrupt slot)");

    std::vector<int32_t> toks(h.n_toks);
    if (h.n_toks) src(toks.data(), h.n_toks * sizeof(int32_t));
    std::vector<float> hidden(h.n_hidden);
    if (h.n_hidden) src(hidden.data(), h.n_hidden * sizeof(float));

    std::vector<uint8_t> buf;
    auto fill = [&](ggml_tensor * t, size_t nbytes) {
        if (nbytes == 0) return;
        buf.resize(nbytes);
        src(buf.data(), nbytes);
        ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
    };
    for (int il = 0; il < h.n_layer; ++il) {
        if (impl_->k_cache[il]) {
            const int rows = il >= (int) hp.n_main() ? h.mtp_past : h.n_past;
            fill(impl_->k_cache[il], (size_t) rows * impl_->k_cache[il]->nb[1]);
            fill(impl_->v_cache[il], (size_t) rows * impl_->v_cache[il]->nb[1]);
        } else {
            fill(impl_->conv_state[il], ggml_nbytes(impl_->conv_state[il]));
            fill(impl_->ssm_state[il],  ggml_nbytes(impl_->ssm_state[il]));
        }
    }
    impl_->n_past     = h.n_past;
    impl_->mtp_past   = h.mtp_past;
    impl_->mrope_next = h.mrope_next;
    impl_->kv_toks    = std::move(toks);
    impl_->mtp_hidden = std::move(hidden);
}

} // namespace questwend
