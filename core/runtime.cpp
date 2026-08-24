#include "runtime.h"
#include "model.h"
#include "expert_cache.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <memory>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace questwend {

static const int GRAPH_SIZE = 16384;

// Longest token batch any single graph attends over. The SWA ring must hold the
// window plus one of these, so the sizing and the chunking have to read the same
// number -- deriving it twice is how the ring ends up a few rows short and the
// only symptom is an assert deep in a long prefill.
static int env_chunk(const char * key, int def) {
    const char * c = getenv(key);
    const int v = c ? atoi(c) : def;
    return v < 1 ? def : v;
}
// plain (resident / RAM-tier) prefill chunk
static int pf_chunk_len()   { static const int v = env_chunk("QWEN_PREFILL_CHUNK", 512); return v; }
// seg A sub-chunk of the batched offload prefill
static int sega_chunk_len() { static const int v = env_chunk("QWEN_SEGA_CHUNK",    256); return v; }
// the bound the ring has to respect
static int max_attn_batch() { return std::max(pf_chunk_len(), sega_chunk_len()); }

struct Runtime::Impl {
    Model & model;
    RuntimeConfig cfg;

    // Primary compute backend (GPU or CPU).
    ggml_backend_t        backend     = nullptr;
    // GPU weights, split across buffers when the backend caps a single one.
    std::vector<ggml_backend_buffer_t> weights_bufs;
    bool weights_buf_owned = false;   // split/ssd: ours; single-backend: Model frees it

    // Phase B: CPU backend + sched for expert weight offloading.
    // When active, expert tensors live in expert_cpu_bufs (CPU pinned memory)
    // and the rest of the weights are in weights_bufs (GPU).
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
    // Resident-only routing (QWEN_RESIDENT_DECODE): a per-layer bias added to
    // the router logits (-inf for non-resident experts) makes every selection a
    // cache hit by construction, so the single fused graph never misses — no
    // verify readback, no state backup, no fallback. Lossy: the router is
    // restricted to the experts the prefill/profile left resident.
    bool                  resident_decode = false;
    ggml_tensor *         resmask_all = nullptr;  // [n_expert, n_layer] f32 (0 / -inf)
    std::vector<float>    resmask_host;
    // Background refill: the fused graph also records what the *unmasked*
    // router would have picked (want_all); wanted-but-absent experts are
    // fetched off the critical path (budget per token) and join the mask on
    // the next token, so the frozen palette tracks topic drift.
    ggml_tensor *         want_all = nullptr;     // [n_used, n_layer] i32 (unmasked top-k)
    std::vector<int32_t>  want_host;
    int                   refill_cursor = 0;      // round-robin layer scan position
    uint64_t              fast_remap_stamp = ~0ull; // residency version the mask/g2s uploads reflect
    bool                  fast_mask_complete = false;
    int                   fast_warm_tokens = 0;     // fast-path decode calls so far (warmup deadline)
    int                   fast_last_floor = -1;     // palette floor the current mask was built with
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

    // ---- sliding-window KV ring ----
    // A sliding layer only ever attends the last n_swa positions, so its cache
    // does not need n_ctx rows -- it needs the window plus the batch in flight.
    // swa_ring is that row count (0 = no ring: the model has no sliding layers,
    // or n_ctx is already small enough that wrapping would never happen). On
    // step35 at n_ctx 32k this takes the KV from 6.0 GB to 1.8 GB, and on an
    // offload build every byte saved goes straight to the expert pool.
    int                   swa_ring = 0;
    ggml_tensor *         ring_idx = nullptr;   // per-graph ring write positions
    int  kv_rows(int il) const {
        return (swa_ring > 0 && model.hparams().is_swa(il)) ? swa_ring : n_ctx;
    }
    bool kv_is_ring(int il) const { return kv_rows(il) < n_ctx; }

    std::vector<float> logits;

    Impl(Model & m, const RuntimeConfig & c) : model(m), cfg(c) {}
    ~Impl() {
        if (ecache) {
            if (getenv("QWEN_LAYER_STATS")) ecache->dump_layer_stats("decode");
            // What quota would the generation's own routing have asked for?
            if (getenv("QWEN_LAYER_QUOTA_DRIFT")) ecache->dump_quota_plan("decode-fit");
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
        // expert_cpu_bufs and weights_bufs are owned here (not by Model) in
        // split/ssd mode; in single-backend mode Model owns and frees the weights.
        for (auto b : expert_cpu_bufs) if (b) ggml_backend_buffer_free(b);
        if (cpu_backend)    ggml_backend_free(cpu_backend);
        if (weights_buf_owned)
            for (auto b : weights_bufs) if (b) ggml_backend_buffer_free(b);
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
    // logits_all=false computes the output head for the LAST token only. The
    // head is [n_embd, n_vocab] -- on a large vocabulary it is the single most
    // expensive matmul in the graph -- and a prefill batch throws away every row
    // but the last, so paying for all of them is pure waste. MTP verify is the
    // one caller that really needs every row.
    ggml_cgraph * build_graph(ggml_context * ctx, int n_tokens, int n_kv,
                              bool logits_all = true);
    // M-RoPE helpers (no-op when !hp.use_mrope): rope_dim returns the inp_pos
    // length for a graph, apply_rope picks ggml_rope_multi vs ggml_rope_ext,
    // fill_rope_pos computes the per-token (sequential or 2D-grid) positions
    // for the main stack and returns the next text rope position.
    int rope_dim(int n_tokens) const {
        return model.hparams().use_mrope ? 4 * n_tokens : n_tokens;
    }
    // `il` picks the layer's rotary width, base and frequency factors. Every
    // Qwen layer is identical so any il gives the same rope there; step35
    // alternates -- full-attention layers rotate half the head with the baked
    // llama3 factors, sliding layers rotate the whole head without them.
    ggml_tensor * apply_rope(ggml_context * ctx, ggml_tensor * x, ggml_tensor * pos, int il) {
        const auto & hp = model.hparams();
        if (hp.use_mrope) {
            int sec[4] = { hp.rope_sections[0], hp.rope_sections[1],
                           hp.rope_sections[2], hp.rope_sections[3] };
            return ggml_rope_multi(ctx, x, pos, nullptr, hp.n_rot, sec,
                                   GGML_ROPE_TYPE_MROPE, 0, hp.rope_freq_base,
                                   1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }
        // rope_freqs.weight carries the llama3 frequency factors, already baked
        // by the converter. step35 stores one shared tensor and applies it to
        // full-attention layers only; other architectures do not have it.
        ggml_tensor * freqs = hp.is_swa(il) ? nullptr : model.tensor("rope_freqs.weight");
        return ggml_rope_ext(ctx, x, pos, freqs, (int) hp.rot_dims(il), GGML_ROPE_TYPE_NEOX,
                             0, hp.rope_base(il), 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    int fill_rope_pos(std::vector<int32_t> & dst, int n_tokens, int rope_start);
    int fill_rope_pos_spans(std::vector<int32_t> & dst, int n_tokens, int rope_start,
                            const Runtime::EmbdOverride * spans, int n_spans);
    // kv_pos: KV-cache write position for this batch (-1 = n_past). Sub-chunked
    // prefill passes n_past+offset so a layer's chunks land consecutively.
    ggml_tensor * build_attn(ggml_context * ctx, ggml_cgraph * gf, int il,
                             ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V,
                             ggml_tensor * mask, int n_tokens, int n_kv, int kv_pos = -1);
    // ---- graph tensor dump (QWEN_DUMP_LAYERS) ----
    // Pins named intermediates as graph outputs so they survive gallocr reuse,
    // then prints one whole-tensor sum each after the run. The names match
    // llama.cpp's cb() labels so a capture from llama-debug can be diffed
    // line for line (see tests/step-ref/README.md).
    std::vector<int>                                   dump_layers;   // -1 = all
    std::vector<std::pair<std::string, ggml_tensor *>> dump_list;
    bool dump_wants(int il) const {
        if (dump_layers.empty()) return false;
        if (il < 0 || dump_layers[0] < 0) return true;   // il < 0 = the final result_* tensors
        return std::find(dump_layers.begin(), dump_layers.end(), il) != dump_layers.end();
    }
    ggml_tensor * dbg(ggml_tensor * t, const char * name, int il) {
        if (!dump_wants(il)) return t;
        char buf[128];
        if (il >= 0) snprintf(buf, sizeof(buf), "%s-%d", name, il);
        else         snprintf(buf, sizeof(buf), "%s", name);
        ggml_set_name(t, buf);
        ggml_set_output(t);
        dump_list.emplace_back(buf, t);
        return t;
    }
    void dump_flush(const char * tag);

    // Upload the causal (and, on step35, the sliding-window) attention masks for
    // a batch of n_tokens whose first token sits at absolute position pos0.
    void fill_masks(ggml_cgraph * gf, int n_tokens, int n_kv, int pos0);
    // Create a graph's sliding-layer inputs: the mask, and (when the cache
    // wraps) the ring write index build_attn scatters through. Sets ring_idx.
    ggml_tensor * new_swa_inputs(ggml_context * ctx, int n_tokens, int n_kv);
    // step35 head-wise attention gate (a no-op elsewhere).
    ggml_tensor * apply_attn_gate(ggml_context * ctx, int il, ggml_tensor * att,
                                  ggml_tensor * x, int n_head_l, int n_tokens);
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
                      int32_t * out_pending = nullptr,
                      bool ckpt_after_prefill = false);
    void prefill(const std::vector<int32_t> & toks, bool mtp_kv);

    // ---- prompt-position checkpoints (prefix-cache rewind) ----
    // Attention KV rows survive a rewind in place, but the GDN recurrent state
    // only exists "as of n_past", so mid-prompt reuse needs host snapshots.
    struct PromptCkpt {
        int pos = 0, mtp_past = 0, mrope = 0;
        std::vector<float>   hidden;   // mtp_hidden at pos (nextn KV bridge)
        std::vector<uint8_t> blob;     // conv+ssm states, layer order
    };
    std::vector<PromptCkpt> pk;        // sorted by pos ascending
    void pk_snapshot();
    int  pk_best(int n) const;
    int  pk_rewind(int n);

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

std::vector<ggml_backend_dev_t> gpu_devices() {
    std::vector<ggml_backend_dev_t> devs;
    // `auto`, not `ggml_backend_dev_type`: ggml gives the enum and the getter
    // function the same name, and the function hides the type.
    for (auto want : { GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU }) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == want) devs.push_back(dev);
        }
    }
    return devs;
}

void Runtime::Impl::init() {
    if (const char * d = getenv("QWEN_DUMP_LAYERS")) {
        if (!strcmp(d, "all")) {
            dump_layers.push_back(-1);
        } else {
            for (const char * p = d; *p; ) {
                if (*p == ',') { ++p; continue; }
                dump_layers.push_back(atoi(p));
                while (*p && *p != ',') ++p;
            }
        }
    }

    // Prefer a GPU device (CUDA/Metal/etc.) when requested and available.
    if (cfg.use_cuda) {
        for (ggml_backend_dev_t dev : gpu_devices()) {
            backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                fprintf(stderr, "backend: GPU [%s] %s\n",
                        ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
                break;
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
        weights_buf_owned = true;   // set first: a throw mid-load still frees what was allocated
        model.load_weights_ssd(backend, weights_bufs);
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
        weights_buf_owned = true;   // set first: a throw mid-load still frees what was allocated
        model.load_weights_split(backend, cpu_buft, weights_bufs, expert_cpu_bufs);

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

        fprintf(stderr, "expert offload: ON (experts stream into the VRAM cache;"
                        " QWEN_CPU_PREFILL=1 runs prefill experts on CPU instead)\n");
    } else {
        weights_bufs.push_back(model.load_weights(backend));
    }

    // fused flash attention on GPU (disable with QWEN_NO_FLASH)
    use_flash = cfg.use_cuda && getenv("QWEN_NO_FLASH") == nullptr;
    if (getenv("QWEN_NO_REUSE")) reuse_graph = false;

    const auto & hp = model.hparams();
    n_ctx = cfg.n_ctx;
    const int n_layer    = hp.n_layer;
    const int n_embd_gqa = hp.n_head_kv * hp.n_embd_head;

    // Size the sliding-window ring: the window itself plus the largest batch any
    // graph will attend over at once, since those queries need keys reaching
    // n_swa back from the batch's *first* token. Both chunk sizes are env knobs,
    // so read them the same way their owners do. QWEN_SWA_RING=0 turns the ring
    // off (the A/B that proves it changes no output).
    if (hp.n_swa > 0 && getenv("QWEN_SWA_RING") != nullptr
                     && atoi(getenv("QWEN_SWA_RING")) == 0) {
        swa_ring = 0;
    } else if (hp.n_swa > 0) {
        const int r = ((int) hp.n_swa + max_attn_batch() + KV_BUCKET - 1) / KV_BUCKET * KV_BUCKET;
        // Only worth it when it actually saves rows.
        swa_ring = r < n_ctx ? r : 0;
        if (swa_ring > 0) {
            size_t full = 0, ring = 0;
            for (int il = 0; il < (int) hp.n_layer; ++il)
                (hp.is_swa(il) ? ring : full) += 1;
            fprintf(stderr, "SWA KV ring: %d rows on %zu sliding layers (%zu full at %d)"
                            " = %.2f GB instead of %.2f GB\n",
                    swa_ring, ring, full, n_ctx,
                    (double) (ring * swa_ring + full * n_ctx) * n_embd_gqa * 2 * 2 / 1e9,
                    (double) (ring + full) * n_ctx * n_embd_gqa * 2 * 2 / 1e9);
        }
    }

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
            // F16 halves the KV cache, which is what bounds the usable context
            // (and on offload builds it competes with the expert pool). Writes
            // convert from F32 via set_rows / cpy; flash attention takes F16
            // K/V natively, and the fallback path feeds them to mul_mat, which
            // handles an F16 src0 against F32 activations.
            const int rows = kv_rows(il);
            k_cache[il] = ggml_new_tensor_2d(st_ctx, GGML_TYPE_F16, n_embd_gqa, rows);
            v_cache[il] = ggml_new_tensor_2d(st_ctx, GGML_TYPE_F16, n_embd_gqa, rows);
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
    size_t gpu_w = 0;
    for (auto b : weights_bufs) gpu_w += ggml_backend_buffer_get_size(b);
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
    resmask_all = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, hp.n_expert, hp.n_layer);
    ggml_set_name(resmask_all, "carry.resmask");
    want_all = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, n_used, hp.n_layer);
    ggml_set_name(want_all, "carry.want");
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
    resmask_host.assign((size_t) hp.n_layer * hp.n_expert, 0.0f);
    want_host.assign((size_t) n_used * hp.n_layer, 0);

    if (getenv("QWEN_FASTCACHE")) cache_fast_enabled = true;   // experimental single-graph path
    if (getenv("QWEN_RESIDENT_DECODE")) { cache_fast_enabled = true; resident_decode = true; }

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
        if (zeros.size() * sizeof(float) < n)
            zeros.assign((n + sizeof(float) - 1) / sizeof(float), 0.0f);   // round up: n need not be a multiple of 4 (F16 KV)
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
void Runtime::Impl::dump_flush(const char * tag) {
    if (dump_list.empty()) return;
    std::vector<float> buf;
    for (auto & e : dump_list) {
        ggml_tensor * t = e.second;
        if (!t->buffer || t->type != GGML_TYPE_F32) continue;
        const size_t n = ggml_nelements(t);
        buf.resize(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) sum += buf[i];
        fprintf(stderr, "qw_dump[%s]: %28s = {%lld, %lld, %lld, %lld} sum = %f\n",
                tag, e.first.c_str(),
                (long long) t->ne[0], (long long) t->ne[1],
                (long long) t->ne[2], (long long) t->ne[3], sum);
    }
    dump_list.clear();
}

ggml_tensor * Runtime::Impl::new_swa_inputs(ggml_context * ctx, int n_tokens, int n_kv) {
    const int rows = swa_ring > 0 ? swa_ring : n_kv;
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, rows, n_tokens);
    ggml_set_input(m); ggml_set_name(m, "inp_mask_swa");
    ring_idx = nullptr;
    if (swa_ring > 0) {
        if (swa_ring < (int) model.hparams().n_swa + n_tokens) {
            throw std::runtime_error(
                "SWA ring holds " + std::to_string(swa_ring) + " rows but this batch of " +
                std::to_string(n_tokens) + " needs " +
                std::to_string((int) model.hparams().n_swa + n_tokens) +
                " -- raise QWEN_PREFILL_CHUNK/QWEN_SEGA_CHUNK before load, or QWEN_SWA_RING=0");
        }
        ring_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
        ggml_set_input(ring_idx); ggml_set_name(ring_idx, "inp_kvidx_swa");
    }
    return m;
}

void Runtime::Impl::fill_masks(ggml_cgraph * gf, int n_tokens, int n_kv, int pos0) {
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> mask((size_t) n_kv * n_tokens);

    for (int i = 0; i < n_tokens; ++i) {
        const int abs_i = pos0 + i;
        for (int j = 0; j < n_kv; ++j)
            mask[(size_t) i * n_kv + j] = (j <= abs_i) ? z : ninf;
    }
    if (ggml_tensor * t = ggml_graph_get_tensor(gf, "inp_mask"))
        ggml_backend_tensor_set(t, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    // The offload path builds one graph per layer and names its single mask
    // after that layer's attention type, so a graph may carry only this one.
    ggml_tensor * t_swa = ggml_graph_get_tensor(gf, "inp_mask_swa");
    if (!t_swa) return;

    // A sliding layer's query at position p sees keys in (p - n_swa, p]: the
    // window spans n_swa positions, the query's own included.
    const int n_swa = (int) model.hparams().n_swa;
    const int rows  = swa_ring > 0 ? swa_ring : n_kv;
    mask.assign((size_t) rows * n_tokens, ninf);

    if (swa_ring <= 0) {
        for (int i = 0; i < n_tokens; ++i) {
            const int abs_i = pos0 + i;
            const int lo    = abs_i - n_swa + 1;
            for (int j = 0; j < rows; ++j)
                mask[(size_t) i * rows + j] = (j <= abs_i && j >= lo) ? z : ninf;
        }
    } else {
        // Ring: a row's absolute position is only known modulo the ring size, so
        // recover it as the newest position <= the last one this batch writes
        // that lands on that row. Rows outside every query's window keep -inf,
        // which is also what an as-yet-unwritten row gets.
        const int last = pos0 + n_tokens - 1;
        std::vector<int> row_pos(rows);
        for (int r = 0; r < rows; ++r) {
            const int back = ((last - r) % rows + rows) % rows;
            row_pos[r] = last - back;                 // < 0 => never written
        }
        for (int i = 0; i < n_tokens; ++i) {
            const int abs_i = pos0 + i;
            for (int r = 0; r < rows; ++r) {
                const int p = row_pos[r];
                if (p >= 0 && p <= abs_i && abs_i - p < n_swa)
                    mask[(size_t) i * rows + r] = z;
            }
        }

        if (ggml_tensor * ki = ggml_graph_get_tensor(gf, "inp_kvidx_swa")) {
            std::vector<int64_t> idx(n_tokens);
            for (int i = 0; i < n_tokens; ++i) idx[i] = (pos0 + i) % rows;
            ggml_backend_tensor_set(ki, idx.data(), 0, idx.size() * sizeof(int64_t));
        }
    }
    ggml_backend_tensor_set(t_swa, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
}

ggml_tensor * Runtime::Impl::build_attn(ggml_context * ctx, ggml_cgraph * gf, int il,
        ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V,
        ggml_tensor * mask, int n_tokens, int n_kv, int kv_pos) {
    if (kv_pos < 0) kv_pos = n_past;
    const auto & hp = model.hparams();
    const int n_head      = (int) hp.head_count(il);      // per-layer on step35
    const int n_head_kv   = (int) hp.head_count_kv(il);
    const int n_embd_head = hp.n_embd_head;
    const int n_embd_gqa  = n_head_kv * n_embd_head;
    const float kq_scale  = 1.0f / sqrtf((float) n_embd_head);

    // Rows this layer's cache actually has, and how many the attention reads.
    // A ring layer always reads the whole ring: which rows are live for a given
    // query is the mask's job, because a row's absolute position is only
    // recoverable modulo the ring size.
    const int rows = kv_rows(il);
    const bool ring = kv_is_ring(il);
    if (ring) n_kv = rows;

    // store K, V into cache
    ggml_tensor * Kflat = ggml_reshape_2d(ctx, K, n_embd_gqa, n_tokens);
    ggml_tensor * Vflat = ggml_reshape_2d(ctx, V, n_embd_gqa, n_tokens);
    if (ring) {
        // A batch can straddle the wrap, so scatter rather than copy a span.
        GGML_ASSERT(ring_idx && "ring layer without inp_kvidx_swa");
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache[il], Kflat, ring_idx));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache[il], Vflat, ring_idx));
    } else if (persistent) {
        // dynamic write position via index input -> graph stays identical each step
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache[il], Kflat, d_kvidx));
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache[il], Vflat, d_kvidx));
    } else {
        ggml_tensor * k_dst = ggml_view_2d(ctx, k_cache[il], n_embd_gqa, n_tokens,
                                           k_cache[il]->nb[1], (size_t) kv_pos * k_cache[il]->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, Kflat, k_dst));
        ggml_tensor * v_dst = ggml_view_2d(ctx, v_cache[il], n_embd_gqa, n_tokens,
                                           v_cache[il]->nb[1], (size_t) kv_pos * v_cache[il]->nb[1]);
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

// step35 head-wise attention gate: sigmoid(g_proj . x) scales each head's slice
// of the attention output. `x` is the post-attn_norm input -- the same tensor
// Q/K/V are projected from -- not the attention output. This is a separate
// weight, unlike qwen35's gate, which is packed into attn_q. Returns `att`
// unchanged on models without blk.N.attn_gate.weight.
ggml_tensor * Runtime::Impl::apply_attn_gate(ggml_context * ctx, int il, ggml_tensor * att,
                                             ggml_tensor * x, int n_head_l, int n_tokens) {
    ggml_tensor * w = Wopt("blk.%d.attn_gate.weight", il);
    if (!w) return att;
    const int n_embd_head = model.hparams().n_embd_head;
    ggml_tensor * g = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w, x));   // [n_head_l, n_tokens]
    g = ggml_reshape_3d(ctx, g, 1, n_head_l, n_tokens);             // broadcast over the head dim
    ggml_tensor * a = ggml_reshape_3d(ctx, att, n_embd_head, n_head_l, n_tokens);
    return ggml_reshape_2d(ctx, ggml_mul(ctx, a, g), n_embd_head * n_head_l, n_tokens);
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

    // fused gated delta net: returns output + new state packed in one tensor.
    // The state input is the initial state s0 only, shaped [S, S, H_v, n_seqs];
    // the snapshot slot count K is an op param (1 = keep the final state only).
    ggml_tensor * s_in  = ggml_reshape_4d(ctx, ssm_state[il], S, S, H_v, 1);
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
                    slice(q), slice(k), slice(v), slice(g), slice(beta), s_in, /*K=*/1);
            ggml_tensor * out_t = ggml_view_4d(ctx, rt, S, H_v, 1, 1,
                    rs(S), rs(S * H_v), rs(S * H_v), 0);
            ggml_tensor * st_t = ggml_view_4d(ctx, rt, S, S, H_v, 1,
                    rs(S), rs(S * S), rs(S * S * H_v), rs(S * H_v));
            if (t + 1 < n_tokens)
                ggml_build_forward_expand(gf, ggml_cpy(ctx, st_t, ckpt_ssm[t][il]));
            if (t + 1 == n_tokens)
                ggml_build_forward_expand(gf, ggml_cpy(ctx, st_t,
                        ggml_reshape_3d(ctx, ssm_state[il], S, S, H_v)));
            s_in = ggml_view_4d(ctx, rt, S, S, H_v, 1,
                    rs(S), rs(S * S), rs(S * S * H_v), rs(S * H_v));   // chain
            output = output ? ggml_concat(ctx, output, out_t, 2) : out_t;
        }
    } else {
        ggml_tensor * result = ggml_gated_delta_net(ctx, q, k, v, g, beta, s_in, /*K=*/1);

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
// SwiGLU with step35's per-layer clamp: silu(gate) is clipped from above and up
// from both sides before the product. limit <= 0 means no clamp, which is every
// layer of every other architecture (and all but the last two of step35).
static ggml_tensor * swiglu_limited(ggml_context * ctx, ggml_tensor * gate,
                                    ggml_tensor * up, float limit) {
    if (limit <= 0.0f) return ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * u = ggml_clamp(ctx, up, -limit, limit);
    ggml_tensor * g = ggml_clamp(ctx, ggml_silu(ctx, gate), -INFINITY, limit);
    return ggml_mul(ctx, g, u);
}

ggml_tensor * Runtime::Impl::build_moe(ggml_context * ctx, ggml_cgraph * gf, int il,
        ggml_tensor * x, int n_tokens) {
    const auto & hp = model.hparams();
    const int n_embd  = hp.n_embd;
    const int n_exp   = hp.n_expert;
    const int n_used  = hp.n_expert_used;

    ggml_tensor * logits = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp.weight", il), x); // [n_exp, n_tokens]
    ggml_tensor * rm = nullptr;   // resident-only routing mask, when in force
    if (cache_fast_build && resident_decode) {
        // record the unmasked router preference (top-k of the raw logits) so
        // the host can refill wanted-but-absent experts in the background
        ggml_tensor * want = ggml_argsort_top_k(ctx, logits, n_used);      // [n_used, 1]
        ggml_tensor * want_col = ggml_view_2d(ctx, want_all, n_used, 1,
                                              want_all->nb[1], (size_t) il * want_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, want, want_col));
        // resident-only routing: bias non-resident experts to -inf before the
        // softmax/top-k so the fused graph can never select a cache miss
        rm = ggml_view_2d(ctx, resmask_all, n_exp, 1,
                          resmask_all->nb[1], (size_t) il * resmask_all->nb[1]);
        logits = ggml_add(ctx, logits, rm);
    }

    // Routing shape. Qwen normalises the logits with a softmax and ranks on the
    // result; step35 squashes each logit independently with a sigmoid and ranks
    // on probs + a per-expert selection bias, while the mixing weights come from
    // the *unbiased* probs (the bias only decides who gets picked).
    const bool sigmoid_gate = hp.expert_gating_func == 2;
    ggml_tensor * probs = sigmoid_gate ? ggml_sigmoid(ctx, logits) : ggml_soft_max(ctx, logits);
    dbg(logits, "ffn_moe_logits", il);
    dbg(probs,  "ffn_moe_probs",  il);

    ggml_tensor * rank = probs;
    if (ggml_tensor * bias = Wopt("blk.%d.exp_probs_b.bias", il)) {
        rank = dbg(ggml_add(ctx, probs, bias), "ffn_moe_probs_biased", il);
        // The residency mask drives the logits to -inf, which zeroes probs but
        // leaves the bias free to rank a non-resident expert first. Re-apply it
        // after the bias so a masked expert can never win a slot.
        if (rm) rank = ggml_add(ctx, rank, rm);
    }

    ggml_tensor * selected = ggml_argsort_top_k(ctx, rank, n_used);    // [n_used, n_tokens] i32

    ggml_tensor * probs3 = ggml_reshape_3d(ctx, probs, 1, n_exp, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx, probs3, selected);      // [1, n_used, n_tokens]

    // normalize weights over selected experts (step35 can turn this off)
    if (hp.arch != Arch::STEP35 || hp.expert_weights_norm) {
        weights = ggml_reshape_2d(ctx, weights, n_used, n_tokens);
        ggml_tensor * wsum = ggml_sum_rows(ctx, weights);              // [1, n_tokens]
        wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx, weights, wsum);
        weights = ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens);
    }
    if (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f)
        weights = ggml_scale(ctx, weights, hp.expert_weights_scale);
    dbg(weights, "ffn_moe_weights_scaled", il);

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
        ggml_tensor * act  = swiglu_limited(ctx, gate, up, hp.swiglu_limit_exp(il));
        ggml_tensor * experts = ggml_mul_mat_id(ctx, ecache->down(il), act, slot_d);
        ggml_tensor * et = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, experts, n_embd, n_used)));
        ggml_tensor * w  = ggml_reshape_2d(ctx, weights, n_used, 1);
        moe_out = ggml_mul_mat(ctx, et, w);   // [n_embd, 1]
    } else {
        ggml_tensor * x3 = ggml_reshape_3d(ctx, x, n_embd, 1, n_tokens);

        ggml_tensor * up   = ggml_mul_mat_id(ctx, W("blk.%d.ffn_up_exps.weight",   il), x3, selected);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, W("blk.%d.ffn_gate_exps.weight", il), x3, selected);
        ggml_tensor * act  = swiglu_limited(ctx, gate, up, hp.swiglu_limit_exp(il));  // silu(gate)*up [ff_exp, n_used, n_tokens]
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

    // llama.cpp reserves "ffn_moe_out" for the routed experts alone and calls the
    // routed + shared sum "ffn_out"; dump here so the two captures line up.
    dbg(moe_out, "ffn_moe_out", il);

    // shared expert (qwen35moe / qwen3next): gated SwiGLU added to the MoE output
    if (ggml_tensor * up_sh = Wopt("blk.%d.ffn_up_shexp.weight", il)) {
        ggml_tensor * g  = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_shexp.weight", il), x);
        ggml_tensor * u  = ggml_mul_mat(ctx, up_sh, x);
        ggml_tensor * sh = ggml_mul_mat(ctx, W("blk.%d.ffn_down_shexp.weight", il),
                                        swiglu_limited(ctx, g, u, hp.swiglu_limit_shexp(il)));
        // qwen35moe / qwen3next gate the shared expert with their own sigmoid;
        // step35 has no such weight and adds it unconditionally.
        if (ggml_tensor * gw = Wopt("blk.%d.ffn_gate_inp_shexp.weight", il))
            sh = ggml_mul(ctx, sh, ggml_sigmoid(ctx, ggml_mul_mat(ctx, gw, x)));
        dbg(sh, "ffn_shared_out", il);
        moe_out = ggml_add(ctx, moe_out, sh);
    }
    return moe_out;
}

ggml_cgraph * Runtime::Impl::build_graph(ggml_context * ctx, int n_tokens, int n_kv,
                                         bool logits_all) {
    const auto & hp = model.hparams();
    const int n_embd      = hp.n_embd;
    const int n_head      = hp.n_head;
    const int n_head_kv   = hp.n_head_kv;
    const int n_embd_head = hp.n_embd_head;
    const float eps       = hp.rms_eps;
    const bool  gated     = hp.has_gdn;   // qwen35 attention layers use gated Q + post norm
    const char * post_norm_name = "blk.%d.post_attention_norm.weight";

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
    dump_list.clear();
    ring_idx = nullptr;

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_tokens); ggml_set_name(inp_tokens, "inp_tokens");
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rope_dim(n_tokens));
    ggml_set_input(inp_pos); ggml_set_name(inp_pos, "inp_pos");
    ggml_tensor * inp_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens);
    ggml_set_input(inp_mask); ggml_set_name(inp_mask, "inp_mask");

    // Sliding-window layers need a second mask (causal AND inside the window).
    // Only step35 has them; every other architecture keeps the single causal
    // mask and never allocates this one.
    ggml_tensor * inp_mask_swa = nullptr;
    if (hp.n_swa > 0) inp_mask_swa = new_swa_inputs(ctx, n_tokens, n_kv);

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

        // step35 gives full-attention and sliding layers different head counts
        // (64 vs 96); everywhere else these equal the scalars.
        const int n_head_l    = (int) hp.head_count(il);
        const int n_head_kv_l = (int) hp.head_count_kv(il);

        dbg(inpL, "attn_norm_in", il);
        cur = ggml_rms_norm(ctx, inpL, eps);
        cur = dbg(ggml_mul(ctx, cur, W("blk.%d.attn_norm.weight", il)), "attn_norm", il);

        if (hp.is_recurrent(il)) {
            cur = build_gdn(ctx, gf, il, cur, n_tokens);
        } else {
            // attention (plain for qwen3, gated for qwen35)
            ggml_tensor * Q, * K, * V, * gate = nullptr;
            if (gated) {
                ggml_tensor * Qf = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur); // [2*hd*nh, T]
                const size_t es = ggml_element_size(Qf);
                Q = ggml_view_3d(ctx, Qf, n_embd_head, n_head_l, n_tokens,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head_l, 0);
                gate = ggml_view_3d(ctx, Qf, n_embd_head, n_head_l, n_tokens,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head_l, es * n_embd_head);
                gate = ggml_cont_2d(ctx, gate, n_embd_head * n_head_l, n_tokens);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv_l, n_tokens);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv_l, n_tokens);
            } else {
                Q = dbg(ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur), "Qcur", il);
                K = dbg(ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur), "Kcur", il);
                V = dbg(ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur), "Vcur", il);
                Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head_l,    n_tokens);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv_l, n_tokens);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv_l, n_tokens);
            }

            Q = dbg(ggml_mul(ctx, ggml_rms_norm(ctx, Q, eps), W("blk.%d.attn_q_norm.weight", il)), "Qcur_normed", il);
            K = dbg(ggml_mul(ctx, ggml_rms_norm(ctx, K, eps), W("blk.%d.attn_k_norm.weight", il)), "Kcur_normed", il);

            Q = dbg(apply_rope(ctx, Q, inp_pos, il), "Qcur_pos", il);
            K = dbg(apply_rope(ctx, K, inp_pos, il), "Kcur_pos", il);

            ggml_tensor * att = build_attn(ctx, gf, il,
                    Q, K, V, hp.is_swa(il) ? inp_mask_swa : inp_mask, n_tokens, n_kv);
            if (gated) att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate));
            dbg(att, "attn_out", il);
            att = dbg(apply_attn_gate(ctx, il, att, cur, n_head_l, n_tokens), "attn_gated", il);
            cur = dbg(ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", il), att), "attn_proj", il);
        }

        cur = dbg(ggml_add(ctx, cur, inpSA), "ffn_inp", il);

        // FFN with (qwen35) post-attention norm placement
        ggml_tensor * ffn_res = cur;
        ggml_tensor * ffn_in;
        if (gated) {
            ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), W(post_norm_name, il));
        } else {
            ffn_in = dbg(ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), W("blk.%d.ffn_norm.weight", il)), "ffn_norm", il);
            ffn_res = cur;  // same residual for qwen3 (norm of cur, add cur)
        }

        ggml_tensor * ff;
        if (hp.is_moe_layer(il)) {
            ff = build_moe(ctx, gf, il, ffn_in, n_tokens);
        } else {
            ggml_tensor * gt = ggml_mul_mat(ctx, W("blk.%d.ffn_gate.weight", il), ffn_in);
            ggml_tensor * up = ggml_mul_mat(ctx, W("blk.%d.ffn_up.weight",   il), ffn_in);
            ff = ggml_mul_mat(ctx, W("blk.%d.ffn_down.weight", il), ggml_mul(ctx, ggml_silu(ctx, gt), up));
        }

        dbg(ff, "ffn_out", il);
        cur = dbg(ggml_add(ctx, ff, ffn_res), "l_out", il);
        inpL = cur;
    }

    // expose the main stack's last hidden (pre-output-norm) for the MTP module
    if (capture_hidden) {
        ggml_set_name(inpL, "main_hidden");
        ggml_set_output(inpL);
        ggml_build_forward_expand(gf, inpL);
    }

    // Narrow to the last token before the output norm when the caller only wants
    // that row: everything downstream then runs once instead of n_tokens times.
    ggml_tensor * head_in = inpL;
    if (!logits_all && n_tokens > 1) {
        head_in = ggml_cont(ctx, ggml_view_2d(ctx, inpL, n_embd, 1, inpL->nb[1],
                                              (size_t) (n_tokens - 1) * inpL->nb[1]));
    }

    cur = ggml_rms_norm(ctx, head_in, eps);
    cur = dbg(ggml_mul(ctx, cur, model.tensor("output_norm.weight")), "result_norm", -1);

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
    fill_masks(v_gf, n_tokens, v_nkv, n_past);

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
                                 int32_t * out_pending, bool ckpt_after_prefill) {
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
    // prompt fully in KV, nothing generated yet: the state the next request
    // rewinds to when the client edits/drops parts of this turn's output
    if (ckpt_after_prefill && P > 0) pk_snapshot();
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

    fill_masks(dgf, 1, d_nkv, n_past);
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
    // Must mirror build_moe's routing exactly: step35 squashes each logit with a
    // sigmoid and ranks on probs + a per-expert selection bias, mixing with the
    // unbiased probs. Diverging here would pick different experts on the offload
    // path than on the resident one, and nothing would report it.
    ggml_tensor * probs    = hp.expert_gating_func == 2 ? ggml_sigmoid(ctx, logits)
                                                       : ggml_soft_max(ctx, logits);
    ggml_tensor * rank     = probs;
    if (ggml_tensor * bias = Wopt("blk.%d.exp_probs_b.bias", il))
        rank = ggml_add(ctx, probs, bias);
    ggml_tensor * selected = ggml_argsort_top_k(ctx, rank, n_used);      // [n_used,1] i32

    ggml_tensor * probs3   = ggml_reshape_3d(ctx, probs, 1, n_exp, 1);
    ggml_tensor * weights  = ggml_get_rows(ctx, probs3, selected);       // [1,n_used,1]
    weights = ggml_reshape_2d(ctx, weights, n_used, 1);
    if (hp.arch != Arch::STEP35 || hp.expert_weights_norm) {
        ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
        wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx, weights, wsum);
    }
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
    ggml_tensor * act  = swiglu_limited(ctx, gate, up, hp.swiglu_limit_exp(il));  // [ff_exp,n_used,1]
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
                                        swiglu_limited(ctx, g, u, hp.swiglu_limit_shexp(il)));
        // step35 has no shared-expert gate and adds it unconditionally.
        if (ggml_tensor * gw = Wopt("blk.%d.ffn_gate_inp_shexp.weight", il))
            sh = ggml_mul(ctx, sh, ggml_sigmoid(ctx, ggml_mul_mat(ctx, gw, ffn_in)));
        moe_out = ggml_add(ctx, moe_out, sh);
    }
    return moe_out;
}

// Batched prefill over the cache: run n_tokens through one segmented forward
// (seg A attention/router for all tokens -> ensure the union of selected experts
//  -> seg B expert matmuls for all tokens). Far fewer graph dispatches than the
// token-by-token path. n_tokens is bounded by activation memory only: a layer
// whose selection union exceeds its pool capacity runs seg B in token slices.
// Only the last token's logits are produced (when want_logits).
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
    std::vector<int32_t> pre_g, pre_u, pre_d;   // scratch for whole-union prefetch

    ggml_tensor * carry_ffn[2] = { ffn_in_b,  ffn_in_b2  };
    ggml_tensor * carry_res[2] = { resid_b,   resid_b2   };
    ggml_tensor * carry_wgt[2] = { weights_b, weights_b2 };

    // Append seg A (attention/GDN + router) for layer `il` over the token slice
    // [tc0, tc0+tlen) of the chunk; writes that slice of the layer's parity carry
    // and exposes `selected` ([n_used, tlen]) for host readback. Attention KV
    // lands at n_past+tc0 and the causal mask covers n_kv_c columns, so a layer's
    // sub-chunks processed in order are equivalent to one full-chunk pass (GDN
    // states likewise chain across sub-chunks).
    auto build_segA = [&](ggml_context * ctx, ggml_cgraph * gf, int il, int tc0, int tlen) -> ggml_tensor * {
        auto tslice = [&](ggml_tensor * t) {   // token-dim slice view [ne0, tlen]
            return ggml_view_2d(ctx, t, t->ne[0], tlen, t->nb[1], (size_t) tc0 * t->nb[1]);
        };
        const int n_kv_c = n_past + tc0 + tlen;
        const int n_head_l    = (int) hp.head_count(il);      // per-layer on step35
        const int n_head_kv_l = (int) hp.head_count_kv(il);
        const bool recurrent = hp.is_recurrent(il);
        ggml_tensor * inp_pos = nullptr, * inp_mask = nullptr;
        if (!recurrent) {
            inp_pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rope_dim(tlen));
            ggml_set_input(inp_pos);  ggml_set_name(inp_pos, "inp_pos");
            // One graph per layer here, so the layer's attention type picks the
            // input name and fill_masks() then uploads the matching mask.
            if (hp.is_swa(il)) {
                inp_mask = new_swa_inputs(ctx, tlen, n_kv_c);
            } else {
                inp_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_c, tlen);
                ggml_set_input(inp_mask); ggml_set_name(inp_mask, "inp_mask");
                ring_idx = nullptr;
            }
        }

        ggml_tensor * h_c = tslice(h_b);
        // h_c is a view of the persistent carry buffer, which this same graph
        // overwrites with the layer's output -- dumping it directly would read
        // back l_out. Copy it when (and only when) the dump is on.
        // ...and the copy has to be forced into the graph, or it is never computed.
        if (dump_wants(il))
            ggml_build_forward_expand(gf, dbg(ggml_cont(ctx, h_c), "attn_norm_in", il));
        ggml_tensor * cur = ggml_rms_norm(ctx, h_c, eps);
        cur = dbg(ggml_mul(ctx, cur, W("blk.%d.attn_norm.weight", il)), "attn_norm", il);

        if (recurrent) {
            cur = build_gdn(ctx, gf, il, cur, tlen);
        } else {
            ggml_tensor * Q, * K, * V, * gate_t = nullptr;
            if (gated) {
                ggml_tensor * Qf = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur);
                const size_t es = ggml_element_size(Qf);
                Q = ggml_view_3d(ctx, Qf, n_embd_head, n_head_l, tlen,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head_l, 0);
                gate_t = ggml_view_3d(ctx, Qf, n_embd_head, n_head_l, tlen,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head_l, es * n_embd_head);
                gate_t = ggml_cont_2d(ctx, gate_t, n_embd_head * n_head_l, tlen);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv_l, tlen);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv_l, tlen);
            } else {
                Q = dbg(ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur), "Qcur", il);
                K = dbg(ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur), "Kcur", il);
                V = dbg(ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur), "Vcur", il);
                Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head_l,    tlen);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv_l, tlen);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv_l, tlen);
            }
            Q = dbg(ggml_mul(ctx, ggml_rms_norm(ctx, Q, eps), W("blk.%d.attn_q_norm.weight", il)), "Qcur_normed", il);
            K = dbg(ggml_mul(ctx, ggml_rms_norm(ctx, K, eps), W("blk.%d.attn_k_norm.weight", il)), "Kcur_normed", il);
            Q = dbg(apply_rope(ctx, Q, inp_pos, il), "Qcur_pos", il);
            K = dbg(apply_rope(ctx, K, inp_pos, il), "Kcur_pos", il);
            ggml_tensor * att = build_attn(ctx, gf, il, Q, K, V, inp_mask, tlen, n_kv_c, n_past + tc0);
            if (gated) att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate_t));
            dbg(att, "attn_out", il);
            att = dbg(apply_attn_gate(ctx, il, att, cur, n_head_l, tlen), "attn_gated", il);
            cur = dbg(ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", il), att), "attn_proj", il);
        }

        ggml_tensor * attn_resid = dbg(ggml_add(ctx, cur, h_c), "ffn_inp", il);   // [n_embd, tlen]
        ggml_tensor * ffn_in;
        if (gated) ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.post_attention_norm.weight", il));
        else       ffn_in = dbg(ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.ffn_norm.weight", il)), "ffn_norm", il);

        // Leading dense block (step35 blk.0-2): no routed experts, so the layer
        // finishes here and its seg B is skipped entirely. Returning nullptr is
        // how the drivers below learn there is nothing to fetch.
        if (!hp.is_moe_layer(il)) {
            ggml_tensor * gt = ggml_mul_mat(ctx, W("blk.%d.ffn_gate.weight", il), ffn_in);
            ggml_tensor * up = ggml_mul_mat(ctx, W("blk.%d.ffn_up.weight",   il), ffn_in);
            ggml_tensor * ff = ggml_mul_mat(ctx, W("blk.%d.ffn_down.weight", il),
                                            swiglu_limited(ctx, gt, up, hp.swiglu_limit_exp(il)));
            dbg(ff, "ffn_out", il);
            ggml_tensor * h_new = dbg(ggml_add(ctx, ff, attn_resid), "l_out", il);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, h_new, tslice(h_b)));
            return (ggml_tensor *) nullptr;
        }

        // multi-token router
        ggml_tensor * logits = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp.weight", il), ffn_in);  // [n_exp, tlen]
        // Same routing as build_moe / build_router: sigmoid + selection bias on
        // step35, softmax elsewhere. All four copies have to agree or the paths
        // silently pick different experts.
        ggml_tensor * probs = hp.expert_gating_func == 2 ? ggml_sigmoid(ctx, logits)
                                                        : ggml_soft_max(ctx, logits);
        dbg(logits, "ffn_moe_logits", il);
        dbg(probs,  "ffn_moe_probs",  il);
        ggml_tensor * rank  = probs;
        if (ggml_tensor * bias = Wopt("blk.%d.exp_probs_b.bias", il))
            rank = dbg(ggml_add(ctx, probs, bias), "ffn_moe_probs_biased", il);
        // argsort_top_k returns a STRIDED view (nb[1] = n_exp*4); make it contiguous
        // so the [n_used, tlen] host readback (ggml_backend_tensor_get) is not corrupted
        // for columns >= 1 (it ignores strides). Single-token path is T=1 so unaffected.
        ggml_tensor * selected = ggml_cont(ctx, ggml_argsort_top_k(ctx, rank, n_used));  // [n_used, tlen]
        ggml_tensor * probs3  = ggml_reshape_3d(ctx, probs, 1, n_exp, tlen);
        ggml_tensor * weights = ggml_get_rows(ctx, probs3, selected);             // [1, n_used, tlen]
        weights = ggml_reshape_2d(ctx, weights, n_used, tlen);
        if (hp.arch != Arch::STEP35 || hp.expert_weights_norm) {
            ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
            wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
            weights = ggml_div(ctx, weights, wsum);
        }
        if (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f)
            weights = ggml_scale(ctx, weights, hp.expert_weights_scale);
        dbg(weights, "ffn_moe_weights_scaled", il);
        ggml_set_output(selected);
        ggml_build_forward_expand(gf, selected);

        ggml_build_forward_expand(gf, ggml_cpy(ctx, ffn_in, tslice(carry_ffn[il & 1])));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, attn_resid, tslice(carry_res[il & 1])));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, weights, tslice(carry_wgt[il & 1])));
        return selected;
    };

    // Append seg B (batched expert matmuls + residual) for layer `il`, over the
    // token slice [t0, t0+len) of the chunk; writes the matching rows of h_b.
    // mul_mat_id takes the slice at once (ids [n_used, len]); the weighted sum
    // over n_used mirrors build_moe's batched path. The FFN is pointwise over
    // tokens, so a slice sees exactly the tensors it would in a full pass.
    auto build_segB = [&](ggml_context * ctx, ggml_cgraph * gf, int il, int t0, int len) {
        auto tslice = [&](ggml_tensor * t) {   // token-dim slice view [ne0, len]
            return ggml_view_2d(ctx, t, t->ne[0], len, t->nb[1], (size_t) t0 * t->nb[1]);
        };
        ggml_tensor * ffn_l = tslice(carry_ffn[il & 1]);
        ggml_tensor * x3   = ggml_reshape_3d(ctx, ffn_l, n_embd, 1, len);
        ggml_tensor * up   = ggml_mul_mat_id(ctx, ecache->up(il),   x3, tslice(slot_u_b));
        ggml_tensor * gate = ggml_mul_mat_id(ctx, ecache->gate(il), x3, tslice(slot_g_b));
        ggml_tensor * act  = swiglu_limited(ctx, gate, up, hp.swiglu_limit_exp(il));
        ggml_tensor * experts = ggml_mul_mat_id(ctx, ecache->down(il), act, tslice(slot_d_b)); // [n_embd, n_used, len]
        experts = ggml_mul(ctx, experts, ggml_reshape_3d(ctx, tslice(carry_wgt[il & 1]), 1, n_used, len));
        ggml_tensor * moe_out = nullptr;
        for (int i = 0; i < n_used; ++i) {
            ggml_tensor * v = ggml_view_2d(ctx, experts, n_embd, len, experts->nb[2], (size_t) i * experts->nb[1]);
            moe_out = i ? ggml_add(ctx, moe_out, v) : v;
        }
        if (n_used == 1) moe_out = ggml_cont(ctx, moe_out);
        dbg(moe_out, "ffn_moe_out", il);

        if (ggml_tensor * up_sh = Wopt("blk.%d.ffn_up_shexp.weight", il)) {
            ggml_tensor * g  = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_shexp.weight", il), ffn_l);
            ggml_tensor * u  = ggml_mul_mat(ctx, up_sh, ffn_l);
            ggml_tensor * sh = ggml_mul_mat(ctx, W("blk.%d.ffn_down_shexp.weight", il),
                                            swiglu_limited(ctx, g, u, hp.swiglu_limit_shexp(il)));
            // step35 has no shared-expert gate and adds it unconditionally.
            if (ggml_tensor * gw = Wopt("blk.%d.ffn_gate_inp_shexp.weight", il))
                sh = ggml_mul(ctx, sh, ggml_sigmoid(ctx, ggml_mul_mat(ctx, gw, ffn_l)));
            dbg(sh, "ffn_shared_out", il);
            moe_out = dbg(ggml_add(ctx, moe_out, sh), "ffn_out", il);
        }
        ggml_tensor * h_new = dbg(ggml_add(ctx, moe_out, tslice(carry_res[il & 1])), "l_out", il);  // [n_embd, len]
        ggml_build_forward_expand(gf, ggml_cpy(ctx, h_new, tslice(h_b)));
    };

    // Set the attention pos/mask inputs of a graph (no-op for GDN-only graphs)
    // for the token slice [tc0, tc0+tlen). M-RoPE positions come from one
    // whole-chunk fill (section-major layout: slice each of the 4 planes).
    std::vector<int32_t> pos_all;
    fill_rope_pos_spans(pos_all, T, mrope_next, ovr, n_ovr);
    auto set_attn_inputs = [&](ggml_cgraph * gf, int tc0, int tlen) {
        if (ggml_tensor * ip = ggml_graph_get_tensor(gf, "inp_pos")) {
            if (hp.use_mrope) {
                std::vector<int32_t> pos((size_t) 4 * tlen);
                for (int p = 0; p < 4; ++p)
                    memcpy(pos.data() + (size_t) p * tlen,
                           pos_all.data() + (size_t) p * T + tc0, (size_t) tlen * sizeof(int32_t));
                ggml_backend_tensor_set(ip, pos.data(), 0, pos.size() * sizeof(int32_t));
            } else {
                ggml_backend_tensor_set(ip, pos_all.data() + tc0, 0, (size_t) tlen * sizeof(int32_t));
            }
        }
        fill_masks(gf, tlen, n_past + tc0 + tlen, n_past + tc0);
    };

    // QWEN_PREFILL_STATS=1 logs, per (chunk,layer), the distinct-expert union and
    // the fetch cost of ensure() — diagnoses "traffic = n_chunks x full sweep".
    // plan_slices() below flips quota enforcement per layer; restore whatever
    // the caller had on the way out so a batch cannot leave the mode changed.
    struct QuotaScope {
        ExpertCache * c; bool prev;
        explicit QuotaScope(ExpertCache * cc) : c(cc), prev(cc->quotas_active()) {}
        ~QuotaScope() { c->set_quotas(prev); }
    } quota_scope(ecache.get());

    static const bool pf_stats = getenv("QWEN_PREFILL_STATS") != nullptr;
    static int      pf_chunk_id = 0;
    const int       pf_chunk    = pf_chunk_id++;
    struct { uint64_t miss = 0, bytes = 0; double ms = 0; int union_min = 1 << 30, union_max = 0; long union_sum = 0; }
        pf_tot;
    const auto pf_t0 = std::chrono::steady_clock::now();

    // The chunk size T is bounded by activation memory, not by the expert pools:
    // one ensure() call must not reference more distinct experts than the layer's
    // pool capacity (it would evict its own slots), so when a layer's selection
    // union exceeds it, seg B runs in token slices with ensure()+slot upload
    // between slices. The FFN is pointwise over tokens, so slicing is lossless.
    std::vector<std::pair<int, int>> slices;   // (t0, len) per seg-B pass
    std::vector<uint8_t> seen((size_t) n_exp, 0);
    std::vector<int32_t> plan_list;            // the union as a list (whole-layer prefetch)
    int plan_union = 0;                        // distinct experts over the whole chunk (stats)
    // Length cap per seg-B pass: bounds the [n_ff, n_used, len] MoE activations
    // (a length-capped follow-up slice re-hits the same resident experts, so it
    // costs no extra fetch traffic).
    static const int max_slice = []{ const char * c = getenv("QWEN_SEGB_SLICE");
                                     int v = c ? atoi(c) : 1024; return v < 1 ? 1024 : v; }();
    auto plan_slices = [&](int il) {
        // Pass 1: the layer's expert union over the whole chunk.
        plan_union = 0;
        plan_list.clear();
        std::vector<uint8_t> useen((size_t) n_exp, 0);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < n_used; ++j) {
                const int e = sel[(size_t) t * n_used + j];
                if (!useen[e]) { useen[e] = 1; ++plan_union; plan_list.push_back(e); }
            }

        // Per-layer slot quotas are a decode-time policy. A batch whose union
        // does not fit the layer's quota is a prefill-like sweep: enforcing the
        // quota there is still correct (the slicing below honours capacity())
        // but pointless -- it would cut seg B into far more slices and re-fetch
        // the same experts. Such a batch gets the whole pool as one LRU stream.
        // A batch that does fit (MTP verify, T=2) keeps its quota, so it cannot
        // evict the decode palette. Deciding here rather than at the call site
        // makes prefill and verify come out right without either knowing.
        ecache->set_quotas(plan_union <= ecache->quota_of(il) - 8);

        // Pass 2: cut the chunk so no single ensure() references more distinct
        // experts than the capacity now in force (it would evict its own slots).
        slices.clear();
        const int cap = std::max(n_used, ecache->capacity(il) - 8);
        std::fill(seen.begin(), seen.end(), 0);
        int distinct = 0, t0 = 0;
        for (int t = 0; t < T; ++t) {
            int nnew = 0;
            for (int j = 0; j < n_used; ++j) {
                const int e = sel[(size_t) t * n_used + j];
                if (!seen[e]) { seen[e] = 1; ++nnew; }
            }
            if ((distinct + nnew > cap || t - t0 >= max_slice) && t > t0) {
                slices.push_back({ t0, t - t0 });
                std::fill(seen.begin(), seen.end(), 0);
                distinct = 0; nnew = 0;
                for (int j = 0; j < n_used; ++j) {
                    const int e = sel[(size_t) t * n_used + j];
                    if (!seen[e]) { seen[e] = 1; ++nnew; }
                }
                t0 = t;
            }
            distinct += nnew;
        }
        slices.push_back({ t0, T - t0 });
    };

    // Make the slice's selected experts resident and upload its slot-id rows.
    auto ensure_slice = [&](int il, int t0, int len) {
        const size_t off = (size_t) t0 * n_used;
        ecache->ensure(il, sel.data() + off, n_used * len,
                       sg.data() + off, su.data() + off, sd.data() + off);
        const size_t ob = off * sizeof(int32_t), nb = (size_t) n_used * len * sizeof(int32_t);
        ggml_backend_tensor_set(slot_g_b, sg.data() + off, ob, nb);
        ggml_backend_tensor_set(slot_u_b, su.data() + off, ob, nb);
        ggml_backend_tensor_set(slot_d_b, sd.data() + off, ob, nb);
    };

    const int N = (int) hp.n_main();

    auto log_layer = [&](int il, int nsl, const ExpertCache::Stats & s0) {
        if (!pf_stats) return;
        const auto s1 = ecache->stats();
        fprintf(stderr, "prefill-stats: chunk=%d T=%d layer=%d union=%d/%d slices=%d miss=%llu fetch_mb=%.1f fetch_ms=%.1f\n",
                pf_chunk, T, il, plan_union, n_exp, nsl,
                (unsigned long long) (s1.misses - s0.misses),
                (double) (s1.fetch_bytes - s0.fetch_bytes) / (1024.0 * 1024.0),
                s1.fetch_ms - s0.fetch_ms);
        pf_tot.miss  += s1.misses - s0.misses;
        pf_tot.bytes += s1.fetch_bytes - s0.fetch_bytes;
        pf_tot.ms    += s1.fetch_ms - s0.fetch_ms;
        pf_tot.union_min = std::min(pf_tot.union_min, plan_union);
        pf_tot.union_max = std::max(pf_tot.union_max, plan_union);
        pf_tot.union_sum += plan_union;
    };

    // Seg-A sub-chunk length for the layer-major path (bounds attention scores).
    const int sega_chunk = sega_chunk_len();

    // ---- optional prefill expert pruning (QWEN_PREFILL_PRUNE=<eps>) ----
    // Skip fetching non-resident experts whose aggregate router mass over the
    // whole chunk is negligible: cheapest-first, until the dropped mass reaches
    // eps of the layer's total. A dropped entry keeps its token slot (id remapped
    // to the token's heaviest kept expert) with weight 0, and the token's kept
    // weights are renormalized, so seg B computes the same weighted sum minus the
    // dropped contributions. Lossy — quality knob, prefill only.
    static const float prune_eps = []{ const char * c = getenv("QWEN_PREFILL_PRUNE");
                                       return c ? (float) atof(c) : 0.0f; }();
    std::vector<float> wgt_host;
    auto prune_layer = [&](int il) {
        if (prune_eps <= 0.0f) return;
        wgt_host.resize((size_t) n_used * T);
        ggml_backend_tensor_get(carry_wgt[il & 1], wgt_host.data(), 0, wgt_host.size() * sizeof(float));

        std::vector<double> mass((size_t) n_exp, 0.0);
        for (size_t k = 0; k < wgt_host.size(); ++k) mass[sel[k]] += wgt_host[k];
        struct Cand { int e; double m; };
        std::vector<Cand> cands;
        double total = 0.0;
        for (int e = 0; e < n_exp; ++e) {
            if (mass[e] <= 0.0) continue;
            total += mass[e];
            const bool res = ecache->resident(ExpertCache::GATE, il, e) &&
                             ecache->resident(ExpertCache::UP,   il, e) &&
                             ecache->resident(ExpertCache::DOWN, il, e);
            if (!res) cands.push_back({ e, mass[e] });
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand & a, const Cand & b) { return a.m < b.m; });
        std::vector<uint8_t> drop((size_t) n_exp, 0);
        double dropped = 0.0;
        int n_drop = 0;
        for (const auto & c : cands) {
            if (dropped + c.m > (double) prune_eps * total) break;
            drop[c.e] = 1; dropped += c.m; ++n_drop;
        }
        if (!n_drop) return;
        // Replacement ids for dropped entries: kept selected experts (they are
        // fetched for this layer anyway, so a zero-weight reference is free).
        std::vector<int> fillers;
        for (int e = 0; e < n_exp; ++e)
            if (mass[e] > 0.0 && !drop[e]) fillers.push_back(e);
        if (fillers.empty()) return;

        // Replace each dropped entry with a distinct zero-weight resident filler:
        // an id may not repeat within a token's row (duplicate ids corrupt the
        // CUDA mul_mat_id expert bookkeeping), and a resident id costs no fetch.
        int fi = 0;
        for (int t = 0; t < T; ++t) {
            float   * w = wgt_host.data() + (size_t) t * n_used;
            int32_t * s = sel.data()      + (size_t) t * n_used;
            float sum_all = 0.0f, sum_kept = 0.0f;
            bool changed = false;
            for (int j = 0; j < n_used; ++j) {
                sum_all += w[j];
                if (drop[s[j]]) { changed = true; continue; }
                sum_kept += w[j];
            }
            if (!changed || sum_kept <= 0.0f)
                continue;                  // untouched, or everything dropped: keep as-is
            const float rescale = sum_all / sum_kept;
            for (int j = 0; j < n_used; ++j) {
                if (!drop[s[j]]) { w[j] *= rescale; continue; }
                w[j] = 0.0f;
                for (int tries = 0; tries < (int) fillers.size(); ++tries) {
                    const int f = fillers[(fi + tries) % (int) fillers.size()];
                    bool dup = false;
                    for (int q = 0; q < n_used; ++q)
                        if (q != j && s[q] == f) { dup = true; break; }
                    if (!dup) {
                        s[j] = f;
                        fi = (fi + tries + 1) % (int) fillers.size();
                        break;
                    }
                }   // no filler found (n_used > distinct residents): keep original id, weight 0
            }
        }
        ggml_backend_tensor_set(carry_wgt[il & 1], wgt_host.data(), 0, wgt_host.size() * sizeof(float));
        if (pf_stats)
            fprintf(stderr, "prefill-prune: chunk=%d layer=%d dropped=%d mass=%.4f\n",
                    pf_chunk, il, n_drop, total > 0.0 ? dropped / total : 0.0);
    };

    if (T > sega_chunk) {
        // ---- Layer-major prefill: for each layer, run seg A over attention-sized
        // token sub-chunks (KV/GDN state chains within the layer), then seg B once
        // over the whole chunk (union-bounded slices). Each layer's expert union
        // is fetched once per chunk regardless of chunk length, so expert traffic
        // is amortized over all T tokens instead of per pool-sized mini-chunk.
        for (int il = 0; il < N; ++il) {
            for (int tc0 = 0; tc0 < T; tc0 += sega_chunk) {
                const int tlen = std::min(sega_chunk, T - tc0);
                ggml_context * ctx = new_ctx();
                ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
                ggml_tensor * selected = build_segA(ctx, gf, il, tc0, tlen);
                run(ctx, gf);
                set_attn_inputs(gf, tc0, tlen);
                if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("decode_cached_batch: seg A compute failed");
                dump_flush("segA");
                if (selected)
                    ggml_backend_tensor_get(selected, sel.data() + (size_t) tc0 * n_used, 0,
                                            (size_t) n_used * tlen * sizeof(int32_t));
                ggml_free(ctx);
            }
            if (!hp.is_moe_layer(il)) continue;   // dense block: seg A finished it
            prune_layer(il);
            plan_slices(il);
            const auto s0 = ecache->stats();
            const int nsl = (int) slices.size();
            // Prefetch the layer's whole union with one ensure() when it fits
            // the pool: the per-slice ensures below then only hit, so the SSD
            // sweep happens once per layer as one dense coalesced read instead
            // of sparse per-slice residual fetches (which re-read the tensor).
            if (nsl > 1 && plan_union <= ecache->capacity(il) - 8) {
                pre_g.resize(plan_list.size());
                pre_u.resize(plan_list.size());
                pre_d.resize(plan_list.size());
                ecache->ensure(il, plan_list.data(), (int) plan_list.size(),
                               pre_g.data(), pre_u.data(), pre_d.data());
            }
            for (int k = 0; k < nsl; ++k) {
                const auto [st0, slen] = slices[k];
                ensure_slice(il, st0, slen);
                ggml_context * ctx = new_ctx();
                ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
                build_segB(ctx, gf, il, st0, slen);
                run(ctx, gf);
                if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("decode_cached_batch: seg B compute failed");
                dump_flush("segB");
                ggml_free(ctx);
            }
            log_layer(il, nsl, s0);
        }
    } else {
    // seg A(0) on its own, then fuse segB(L)+segA(L+1) per step so each layer
    // boundary is a single GPU submit instead of two (mirrors decode_cached).
    // A layer whose union overflows its pool runs extra unfused seg-B slices first.
    {
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * selected = build_segA(ctx, gf, 0, 0, T);
        run(ctx, gf);
        set_attn_inputs(gf, 0, T);
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("decode_cached_batch: seg A0 compute failed");
                dump_flush("segA");
        if (selected)
            ggml_backend_tensor_get(selected, sel.data(), 0, (size_t) n_used * T * sizeof(int32_t));
        ggml_free(ctx);
    }
    for (int il = 0; il < N; ++il) {
        if (!hp.is_moe_layer(il)) {
            // Dense block: seg A already finished it, so there is no seg B to
            // fuse the next layer's seg A onto -- run that one on its own.
            if (il + 1 < N) {
                ggml_context * ctx = new_ctx();
                ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
                ggml_tensor * nsel = build_segA(ctx, gf, il + 1, 0, T);
                run(ctx, gf);
                set_attn_inputs(gf, 0, T);
                if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("decode_cached_batch: dense seg A compute failed");
                dump_flush("segA");
                if (nsel)
                    ggml_backend_tensor_get(nsel, sel.data(), 0, (size_t) n_used * T * sizeof(int32_t));
                ggml_free(ctx);
            }
            continue;
        }
        plan_slices(il);
        const auto s0 = ecache->stats();
        const int nsl = (int) slices.size();
        for (int k = 0; k < nsl; ++k) {
            const auto [st0, slen] = slices[k];
            ensure_slice(il, st0, slen);
            ggml_context * ctx = new_ctx();
            ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
            build_segB(ctx, gf, il, st0, slen);
            ggml_tensor * nsel = (k == nsl - 1 && il + 1 < N) ? build_segA(ctx, gf, il + 1, 0, T) : nullptr;
            run(ctx, gf);
            set_attn_inputs(gf, 0, T);
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("decode_cached_batch: fused segB/segA compute failed");
                dump_flush("segBA");
            if (nsel)
                ggml_backend_tensor_get(nsel, sel.data(), 0, (size_t) n_used * T * sizeof(int32_t));
            ggml_free(ctx);
        }
        log_layer(il, nsl, s0);
    }
    }

    if (pf_stats) {
        const double wall_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - pf_t0).count();
        fprintf(stderr, "prefill-stats: chunk=%d SUMMARY T=%d layers=%d union_min=%d union_avg=%.1f union_max=%d/%d "
                        "miss=%llu fetch_mb=%.1f fetch_ms=%.1f wall_ms=%.1f fetch_frac=%.2f\n",
                pf_chunk, T, N, pf_tot.union_min, (double) pf_tot.union_sum / N, pf_tot.union_max, n_exp,
                (unsigned long long) pf_tot.miss, (double) pf_tot.bytes / (1024.0 * 1024.0),
                pf_tot.ms, wall_ms, pf_tot.ms / (wall_ms > 0 ? wall_ms : 1));
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
        cur = dbg(ggml_mul(ctx, cur, model.tensor("output_norm.weight")), "result_norm", -1);
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
        dbg(cur, "result_output", -1);
        dump_flush("out");
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
        const int n_head_l    = (int) hp.head_count(il);      // per-layer on step35
        const int n_head_kv_l = (int) hp.head_count_kv(il);
        const bool recurrent = hp.is_recurrent(il);
        ggml_tensor * inp_pos = nullptr, * inp_mask = nullptr;
        if (!recurrent) {
            inp_pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rope_dim(1));
            ggml_set_input(inp_pos);  ggml_set_name(inp_pos, "inp_pos");
            if (hp.is_swa(il)) {
                inp_mask = new_swa_inputs(ctx, 1, n_kv);
            } else {
                inp_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, 1);
                ggml_set_input(inp_mask); ggml_set_name(inp_mask, "inp_mask");
                ring_idx = nullptr;
            }
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
                Q = ggml_view_3d(ctx, Qf, n_embd_head, n_head_l, 1,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head_l, 0);
                gate_t = ggml_view_3d(ctx, Qf, n_embd_head, n_head_l, 1,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head_l, es * n_embd_head);
                gate_t = ggml_cont_2d(ctx, gate_t, n_embd_head * n_head_l, 1);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv_l, 1);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv_l, 1);
            } else {
                Q = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head_l,    1);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv_l, 1);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv_l, 1);
            }
            Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, eps), W("blk.%d.attn_q_norm.weight", il));
            K = ggml_mul(ctx, ggml_rms_norm(ctx, K, eps), W("blk.%d.attn_k_norm.weight", il));
            Q = apply_rope(ctx, Q, inp_pos, il);
            K = apply_rope(ctx, K, inp_pos, il);
            ggml_tensor * att = build_attn(ctx, gf, il, Q, K, V, inp_mask, 1, n_kv);
            if (gated) att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate_t));
            att = apply_attn_gate(ctx, il, att, cur, n_head_l, 1);
            cur = ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", il), att);
        }
        ggml_tensor * attn_resid = ggml_add(ctx, cur, p_h);
        ggml_tensor * ffn_in;
        if (gated) ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.post_attention_norm.weight", il));
        else       ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.ffn_norm.weight", il));
        if (!hp.is_moe_layer(il)) {
            // Leading dense block: no experts, so there is no seg B for it.
            ggml_tensor * gt = ggml_mul_mat(ctx, W("blk.%d.ffn_gate.weight", il), ffn_in);
            ggml_tensor * up = ggml_mul_mat(ctx, W("blk.%d.ffn_up.weight",   il), ffn_in);
            ggml_tensor * ff = ggml_mul_mat(ctx, W("blk.%d.ffn_down.weight", il),
                                            swiglu_limited(ctx, gt, up, hp.swiglu_limit_exp(il)));
            ggml_tensor * h_new = ggml_add(ctx, ff, attn_resid);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_1d(ctx, h_new, n_embd), p_h));
            return (ggml_tensor *) nullptr;
        }
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
        fill_masks(gf, 1, n_kv, n_past);
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
        if (selected) ensure_layer(selected, 0);
        ggml_free(ctx);
    }
    for (int il = 0; il < N; ++il) {
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        // A dense block finished inside its own seg A, so it has no seg B; the
        // next layer's seg A then runs alone instead of fused onto one.
        if (hp.is_moe_layer(il)) build_segB(ctx, gf, il);
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

// ---- prompt-position checkpoints (prefix-cache rewind) ----
// A history edit (e.g. a client dropping the previous turn's reasoning) makes
// the new prompt diverge mid-way through the cached tokens. The attention KV
// prefix stays valid in place, but GDN states cannot be truncated — they are
// restored from the newest host snapshot at or before the divergence point,
// and only the tail after it is re-prefilled.
static size_t pk_max() {
    static const size_t v = []{
        const char * c = getenv("QWEN_PROMPT_CKPTS");
        return (size_t) (c ? atoi(c) : 8);
    }();
    return v;
}

void Runtime::Impl::pk_snapshot() {
    const auto & hp = model.hparams();
    if (!hp.has_gdn || pk_max() == 0) return;   // attention-only: truncation is free
    if (n_past <= 0) return;
    if (!pk.empty() && pk.back().pos == n_past) return;
    PromptCkpt c;
    c.pos      = n_past;
    c.mtp_past = mtp_past;
    c.mrope    = mrope_next;
    c.hidden   = mtp_hidden;
    size_t total = 0;
    for (int il = 0; il < (int) hp.n_layer; ++il)
        if (conv_state[il]) total += ggml_nbytes(conv_state[il]) + ggml_nbytes(ssm_state[il]);
    c.blob.resize(total);
    size_t off = 0;
    for (int il = 0; il < (int) hp.n_layer; ++il) {
        if (!conv_state[il]) continue;
        ggml_backend_tensor_get(conv_state[il], c.blob.data() + off, 0, ggml_nbytes(conv_state[il]));
        off += ggml_nbytes(conv_state[il]);
        ggml_backend_tensor_get(ssm_state[il], c.blob.data() + off, 0, ggml_nbytes(ssm_state[il]));
        off += ggml_nbytes(ssm_state[il]);
    }
    auto it = std::lower_bound(pk.begin(), pk.end(), c.pos,
                               [](const PromptCkpt & a, int p) { return a.pos < p; });
    if (it != pk.end() && it->pos == c.pos) *it = std::move(c);
    else pk.insert(it, std::move(c));
    // evict the checkpoint whose removal leaves the smallest gap to its
    // predecessor (never the newest) — keeps positions spread over the prefix
    while (pk.size() > pk_max()) {
        size_t worst = 0;
        int wgap = pk[0].pos;
        for (size_t i = 1; i + 1 < pk.size(); ++i) {
            const int gap = pk[i].pos - pk[i-1].pos;
            if (gap < wgap) { wgap = gap; worst = i; }
        }
        pk.erase(pk.begin() + worst);
    }
}

int Runtime::Impl::pk_best(int n) const {
    if (n <= 0) return -1;
    const auto & hp = model.hparams();
    if (!hp.has_gdn)   // attention-only: any position works (text-only lineage)
        return mrope_next == n_past ? std::min(n, n_past) : -1;
    int best = -1;
    for (const auto & c : pk) { if (c.pos > n) break; best = c.pos; }
    return best;
}

int Runtime::Impl::pk_rewind(int n) {
    const auto & hp = model.hparams();
    if (n >= (int) kv_toks.size()) return (int) kv_toks.size();   // nothing to rewind
    // A sliding layer's ring keeps only the last swa_ring positions, so once
    // generation has run far enough past `n` the window it would need has been
    // written over. Bail out and let the caller reset rather than attend to rows
    // holding positions from the future.
    if (swa_ring > 0 && n_past - n > swa_ring - (int) hp.n_swa) return -1;
    if (!hp.has_gdn) {
        if (mrope_next != n_past) return -1;   // image lineage: positions ambiguous
        n_past = n;
        mrope_next = n;
        if (mtp_past > n) mtp_past = n;
        kv_toks.resize(n);
        return n;
    }
    int bi = -1;
    for (int i = 0; i < (int) pk.size(); ++i) { if (pk[i].pos > n) break; bi = i; }
    if (bi < 0) return -1;
    const PromptCkpt & c = pk[bi];
    size_t off = 0;
    for (int il = 0; il < (int) hp.n_layer; ++il) {
        if (!conv_state[il]) continue;
        ggml_backend_tensor_set(conv_state[il], c.blob.data() + off, 0, ggml_nbytes(conv_state[il]));
        off += ggml_nbytes(conv_state[il]);
        ggml_backend_tensor_set(ssm_state[il], c.blob.data() + off, 0, ggml_nbytes(ssm_state[il]));
        off += ggml_nbytes(ssm_state[il]);
    }
    n_past     = c.pos;
    mtp_past   = c.mtp_past;
    mrope_next = c.mrope;
    mtp_hidden = c.hidden;
    kv_toks.resize(c.pos);
    pk.erase(pk.begin() + bi + 1, pk.end());   // later snapshots: dead lineage
    return c.pos;
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
    ggml_tensor * inp_kvidx  = ggml_graph_get_tensor(f_gf, "inp_kvidx");
    ggml_backend_tensor_set(inp_tokens, &token, 0, sizeof(int32_t));
    std::vector<int32_t> posv;
    fill_rope_pos(posv, 1, mrope_next);   // generation token: text, sequential
    ggml_backend_tensor_set(inp_pos, posv.data(), 0, posv.size() * sizeof(int32_t));
    int64_t kvidx = n_past; ggml_backend_tensor_set(inp_kvidx, &kvidx, 0, sizeof(int64_t));
    fill_masks(f_gf, 1, f_nkv, n_past);   // causal, sliding-window and ring inputs

    // refresh the in-graph remap table from current residency
    auto fill_g2s = [&]() {
        for (int il = 0; il < n_layer; ++il) {
            if (!hp.is_moe_layer(il)) continue;   // dense block: no expert pools
            for (int r = 0; r < 3; ++r) {
                const int32_t * row = ecache->slot_of_row((ExpertCache::Role) r, il);
                memcpy(&g2s_host[(size_t) (il * 3 + r) * n_exp], row, n_exp * sizeof(int32_t));
            }
        }
        ggml_backend_tensor_set(g2s_all, g2s_host.data(), 0, g2s_host.size() * sizeof(int32_t));
    };
    // Refresh the router residency mask; true iff every layer is masked. A
    // layer joins the mask only once it has a decent resident palette
    // (QWEN_RESIDENT_MIN, default 32) — until then it stays unmasked so its
    // misses keep warming the cache via the fallback path. Layers with
    // concentrated routing may never reach that palette on a short prompt, so
    // after QWEN_RESIDENT_WARMUP fast-decode tokens (default 16) the floor
    // drops to n_used and whatever is resident gets frozen ("warm a few
    // tokens, then pin" — the original design intent).
    static const int res_min = []{ const char * c = getenv("QWEN_RESIDENT_MIN");
                                   const int v = c ? atoi(c) : 32; return v < 1 ? 32 : v; }();
    static const int warm_tok = []{ const char * c = getenv("QWEN_RESIDENT_WARMUP");
                                    const int v = c ? atoi(c) : 32; return v < 0 ? 32 : v; }();
    const int floor_res = fast_warm_tokens > warm_tok
                              ? n_used
                              : std::max(n_used, std::min(res_min, n_exp));
    auto fill_resmask = [&]() {
        bool complete = true;
        for (int il = 0; il < n_layer; ++il) {
            if (!hp.is_moe_layer(il)) continue;
            float * row = &resmask_host[(size_t) il * n_exp];
            int n_res = 0;
            for (int e = 0; e < n_exp; ++e) {
                const bool res = ecache->resident(ExpertCache::GATE, il, e) &&
                                 ecache->resident(ExpertCache::UP,   il, e) &&
                                 ecache->resident(ExpertCache::DOWN, il, e);
                row[e] = res ? 0.0f : -INFINITY;
                n_res += res;
            }
            if (n_res < floor_res) {   // palette too thin: leave the layer unmasked
                for (int e = 0; e < n_exp; ++e) row[e] = 0.0f;
                complete = false;
            }
        }
        ggml_backend_tensor_set(resmask_all, resmask_host.data(), 0,
                                resmask_host.size() * sizeof(float));
        return complete;
    };
    // verify residency of every selected expert; returns true if all resident
    auto verify = [&]() {
        ggml_backend_tensor_get(sel_all, sel_host.data(), 0, sel_host.size() * sizeof(int32_t));
        bool ok = true;
        for (int il = 0; il < n_layer; ++il) {
            if (!hp.is_moe_layer(il)) continue;
            for (int k = 0; k < n_used; ++k) {
                const int e = sel_host[(size_t) il * n_used + k];
                if (!ecache->resident(ExpertCache::GATE, il, e) ||
                    !ecache->resident(ExpertCache::UP,   il, e) ||
                    !ecache->resident(ExpertCache::DOWN, il, e)) ok = false;
            }
        }
        return ok;
    };

    // Residency only changes through fetches/evictions, so the mask and remap
    // uploads are versioned by the miss+eviction count (and the warmup floor)
    // and skipped while stale-free — steady-state resident decode uploads
    // nothing but token/pos/mask.
    ++fast_warm_tokens;
    const uint64_t stamp = ecache->stats().misses + ecache->stats().evictions;
    if (stamp != fast_remap_stamp || floor_res != fast_last_floor) {
        fast_mask_complete = resident_decode && fill_resmask();
        fill_g2s();
        fast_remap_stamp = stamp;
        fast_last_floor  = floor_res;
    }
    // Resident-only routing with a complete mask cannot miss: the run is not
    // speculative, so state backup, the sel_all verify readback and the
    // fallback are all skipped — one graph submit + one logits readback.
    const bool masked = resident_decode && fast_mask_complete;

    if (!masked) backup_states();   // so a speculative miss can be rolled back
    if (ggml_backend_graph_compute(backend, f_gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("decode_cached_fast: compute failed");

    if (!masked) {
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
    } else {
        // ---- background refill: keep the frozen palette tracking the input ----
        // The graph recorded what the unmasked router wanted (want_all). Fetch
        // wanted-but-absent experts within a per-token budget (they join the
        // mask next token via the stamp), and touch the actually-used experts
        // so the refill's LRU evictions land on stale entries, not hot ones.
        // per-token fetch budget, shared across all layers (see the round-robin
        // scan below). One default for both tiers: the SSD tier reads
        // synchronously and used to default lower (4), but that starves the
        // palette after a mid-generation domain shift (docs/resident_decode.md)
        static const int refill_budget = []{ const char * c = getenv("QWEN_RESIDENT_REFILL");
                                             const int v = c ? atoi(c) : 8;
                                             return v < 0 ? 0 : v; }();
        if (refill_budget > 0) {
            ggml_backend_tensor_get(sel_all,  sel_host.data(),  0, sel_host.size()  * sizeof(int32_t));
            ggml_backend_tensor_get(want_all, want_host.data(), 0, want_host.size() * sizeof(int32_t));
            for (int il = 0; il < n_layer; ++il) {
                if (!hp.is_moe_layer(il)) continue;
                for (int k = 0; k < n_used; ++k) {
                    const int e = sel_host[(size_t) il * n_used + k];
                    ecache->touch(ExpertCache::GATE, il, e);
                    ecache->touch(ExpertCache::UP,   il, e);
                    ecache->touch(ExpertCache::DOWN, il, e);
                }
            }
            // Instrumentation: how often the layer's frozen palette failed to
            // hold what the *unmasked* router wanted. This is the quality cost
            // of the layer's slot share (masked decode never stalls on a miss).
            static const bool lstats = getenv("QWEN_LAYER_STATS") != nullptr;
            if (lstats) {
                for (int il = 0; il < n_layer; ++il) {
                    if (!hp.is_moe_layer(il)) continue;
                    for (int k = 0; k < n_used; ++k) {
                        const int e = want_host[(size_t) il * n_used + k];
                        const bool res = ecache->resident(ExpertCache::GATE, il, e) &&
                                         ecache->resident(ExpertCache::UP,   il, e) &&
                                         ecache->resident(ExpertCache::DOWN, il, e);
                        ecache->note_want(il, !res);
                    }
                }
            }
            int budget = refill_budget;
            for (int step = 0; step < n_layer && budget > 0; ++step) {
                const int il = (refill_cursor + step) % n_layer;
                if (!hp.is_moe_layer(il)) continue;
                for (int k = 0; k < n_used && budget > 0; ++k) {
                    const int e = want_host[(size_t) il * n_used + k];
                    if (ecache->resident(ExpertCache::GATE, il, e) &&
                        ecache->resident(ExpertCache::UP,   il, e) &&
                        ecache->resident(ExpertCache::DOWN, il, e)) continue;
                    ecache->ensure_resident(il, e);   // async H2D (RAM tier); joins mask next token
                    --budget;
                }
            }
            refill_cursor = (refill_cursor + 1) % n_layer;
        }
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
        // Chunk size is an activation bound only: per-layer expert residency is
        // handled inside decode_cached_batch (layer-major seg A sub-chunking +
        // union-bounded seg B slicing), so the chunk no longer needs to fit the
        // expert pools. Bigger chunks amortize each layer's expert fetch over
        // more tokens: expert traffic scales with the number of chunks, not T.
        // 4096 tokens ≈ 100 MB of carry tensors (7 x n_embd x T floats).
        int chunk = 4096;
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
        if (getenv("QWEN_LAYER_STATS")) {
            ecache->dump_layer_stats("prefill");
            ecache->reset_layer_stats();   // the destructor dump is decode-only
        }
        // How the pool's slots are split across layers. "quota" re-shapes it
        // from the prompt's own routing (the prefill sweep otherwise leaves the
        // pool holding only the last layers it touched, which is the worst
        // possible split); "lru" leaves the split to the global LRU.
        //
        // auto (the default) = quota iff resident-only decode. Measured on
        // Qwen3.6-35B-A3B at 54% residency: under the resident mask the reshape
        // takes the share of router picks that miss the palette from 18% to 10%
        // and decode from 33.5 to 40.4 tok/s, while on the miss-driven decode
        // path it buys nothing (+0.9%, and -1.6% at 25% residency) and still
        // costs ~0.7 s of prefill.
        static const int alloc_mode = []{
            const char * c = getenv("QWEN_EXPERT_ALLOC");
            if (!c || !strcmp(c, "auto")) return 0;
            if (!strcmp(c, "lru"))        return 1;
            if (!strcmp(c, "quota"))      return 2;
            fprintf(stderr, "warning: QWEN_EXPERT_ALLOC=%s not understood "
                            "(expected lru, quota or auto); using auto\n", c);
            return 0;
        }();
        if (alloc_mode == 2 || (alloc_mode == 0 && resident_decode)) {
            const char * c = getenv("QWEN_QUOTA_PREFETCH");
            const long long v = c ? atoll(c) : 0;
            ecache->rebalance(v > 0 ? (size_t) v : (size_t) -1);
            // Forget the prompt's history so the exit dump measures how far the
            // ideal shape drifts once generation takes over.
            if (getenv("QWEN_LAYER_QUOTA_DRIFT")) ecache->clear_counts();
            if (getenv("QWEN_LAYER_STATS")) {
                ecache->dump_layer_stats("rebalanced");
                ecache->reset_layer_stats();
            }
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
    const int PF_CHUNK = pf_chunk_len();
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

    // Prefill/decode only ever reads the last row of the logits.
    ggml_cgraph * gf = build_graph(ctx, n_tokens, n_kv, /*logits_all=*/false);

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

    ggml_backend_tensor_set(inp_tokens_t, tokens.data(), 0, n_tokens * sizeof(int32_t));

    std::vector<int32_t> pos;
    const int new_mrope = fill_rope_pos(pos, n_tokens, mrope_next);
    ggml_backend_tensor_set(inp_pos_t, pos.data(), 0, pos.size() * sizeof(int32_t));

    fill_masks(gf, n_tokens, n_kv, n_past);

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
    dbg(logits_t, "result_output", -1);   // renames it, so resolve the name first
    dump_flush("decode");

    const int n_vocab = (int) logits_t->ne[0];
    logits.resize(n_vocab);
    // one row when the head was narrowed, n_tokens rows otherwise
    const int row = logits_t->ne[1] == 1 ? 0 : n_tokens - 1;
    const size_t off = (size_t) row * logits_t->nb[1];
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
                           int32_t * out_pending, bool ckpt_after_prefill) {
    impl_->generate_mtp(prompt, max_new, n_draft, on_token, out_pending, ckpt_after_prefill);
}
void Runtime::prefill(const std::vector<int32_t> & tokens, bool mtp_kv) {
    impl_->prefill(tokens, mtp_kv);
    impl_->kv_toks.insert(impl_->kv_toks.end(), tokens.begin(), tokens.end());
}
void Runtime::snapshot_ckpt()          { impl_->pk_snapshot(); }
int  Runtime::best_ckpt(int n) const   { return impl_->pk_best(n); }
int  Runtime::rewind_to(int n)         { return impl_->pk_rewind(n); }
void Runtime::reset()        { impl_->n_past = 0; impl_->mtp_past = 0; impl_->mrope_next = 0; impl_->kv_toks.clear(); impl_->pk.clear(); impl_->zero_states(); }
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
    uint32_t magic;      // 'QSS3'
    int32_t  n_layer, n_ctx, n_embd_gqa;
    int32_t  n_past, mtp_past, n_toks, n_hidden, mrope_next;
};
// Bumped to QSS3 when the KV cache became F16: the per-row byte count halved,
// so a QSS2 slot file on disk would pass the header check and then be misread.
constexpr uint32_t STATE_MAGIC = 0x33535351;   // "QSS3" little-endian
}

size_t Runtime::state_bytes() const {
    const auto & hp = impl_->model.hparams();
    size_t n = sizeof(StateHeader);
    n += impl_->kv_toks.size() * sizeof(int32_t);
    n += impl_->mtp_hidden.size() * sizeof(float);
    for (int il = 0; il < (int) hp.n_layer; ++il) {
        if (impl_->k_cache[il]) {
            int rows = il >= (int) hp.n_main() ? impl_->mtp_past : impl_->n_past;
            rows = std::min(rows, impl_->kv_rows(il));   // ring layers hold only a window
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
            int rows = il >= (int) hp.n_main() ? h.mtp_past : h.n_past;
            rows = std::min(rows, impl_->kv_rows(il));
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
            int rows = il >= (int) hp.n_main() ? h.mtp_past : h.n_past;
            rows = std::min(rows, impl_->kv_rows(il));
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
    impl_->pk.clear();   // checkpoints belong to the replaced lineage
}

size_t parse_vram_budget_mb(const std::string & arg, bool * legacy_mb) {
    if (legacy_mb) *legacy_mb = false;

    const char * s = arg.c_str();
    char * end = nullptr;
    const double v = strtod(s, &end);
    if (end == s || v <= 0.0) return 0;

    std::string suf;
    for (const char * p = end; *p; ++p)
        if (!isspace((unsigned char) *p)) suf += (char) tolower((unsigned char) *p);

    double mb;
    if (suf == "m" || suf == "mb") {
        mb = v;
    } else if (suf == "g" || suf == "gb" || suf.empty()) {
        // bare values large enough to be nonsense as GB are the pre-GB spelling
        const bool legacy = suf.empty() && v >= 512.0;
        if (legacy && legacy_mb) *legacy_mb = true;
        mb = legacy ? v : v * 1024.0;
    } else {
        return 0;   // unknown suffix: caller reports it as a bad value
    }
    return (size_t) (mb + 0.5);
}

} // namespace questwend
