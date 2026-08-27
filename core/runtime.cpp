#include "runtime.h"
#include "model.h"
#include "ngram_table.h"
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

// Allocate exactly `ts` into one fresh buffer on `buft`. ggml_backend_alloc_ctx_tensors
// cannot be used where tensors from a single context are spread over several
// devices: it claims every unallocated tensor in the context for one buffer.
static ggml_backend_buffer_t alloc_tensor_list(ggml_backend_buffer_type_t buft,
                                               const std::vector<ggml_tensor *> & ts,
                                               const char * what) {
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t sz = 0;
    for (auto * t : ts) sz += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
    ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft, sz > 0 ? sz : 1);
    if (!b) throw std::runtime_error(std::string("failed to allocate ") + what);
    ggml_tallocr ta = ggml_tallocr_new(b);
    for (auto * t : ts)
        if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS)
            throw std::runtime_error(std::string("alloc failed for ") + ggml_get_name(t));
    return b;
}

struct Runtime::Impl {
    Model & model;
    RuntimeConfig cfg;

    // Primary compute backend (GPU or CPU).
    ggml_backend_t        backend     = nullptr;

    // ---- Multi-GPU (layer split) ----
    // Every GPU backend in the plan; gpus[0] == backend, the primary device that
    // owns everything not tied to a transformer layer (embeddings, output head,
    // final norm) and runs the sampling tail. A layer's weights, KV/recurrent
    // state and expert pool all live on the same device, so the only traffic
    // crossing the PCIe bus is the hidden state at a device boundary (~n_embd
    // floats per token). With one device this vector has a single entry and
    // multi_gpu() is false, which keeps every path below on its original code.
    std::vector<ggml_backend_t> gpus;
    std::vector<int>            layer_dev;      // layer -> index into gpus
    // Where each layer's EXPERT POOL lives, which need not be where the layer
    // computes its attention. A device given --gpu-split 0 holds no layers at
    // all, so its whole budget is surplus; pool_dev hands it the expert pools of
    // layers whose attention runs elsewhere, and that layer's expert matmul then
    // runs there too (compute follows the weights -- the alternative, shipping
    // expert weights device-to-device, is PCIe-bound and pointless). Equal to
    // layer_dev unless a device is pool-only.
    std::vector<int>            pool_dev;
    std::vector<size_t>         dev_budget;     // per-device VRAM budget, bytes (0 = unset)
    std::vector<size_t>         dev_weight_bytes;  // per-device weight bytes actually allocated
    std::vector<size_t>         dev_kv_bytes;      // per-device KV/state bytes

    bool multi_gpu() const { return gpus.size() > 1; }
    int  dev_of(int il) const {
        return layer_dev.empty() ? 0 : layer_dev[(size_t) il];
    }
    // Device holding `il`'s expert pool (and therefore running its expert matmul).
    int  pdev_of(int il) const {
        return pool_dev.empty() ? dev_of(il) : pool_dev[(size_t) il];
    }
    ggml_backend_t be_of(int il) const {
        return gpus.empty() ? backend : gpus[(size_t) dev_of(il)];
    }
    // Placement plan handed to the weight loaders.
    DevicePlan device_plan() const {
        DevicePlan p;
        for (ggml_backend_t be : gpus) p.bufts.push_back(ggml_backend_get_default_buffer_type(be));
        p.layer_dev = layer_dev;
        p.pool_dev  = pool_dev;
        return p;
    }

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
    // One expert cache per GPU (a device with no layers gets a null entry).
    // `ecache` aliases the primary's and doubles as the "offload is on" flag,
    // so single-GPU code reads exactly as it did before the split.
    std::vector<std::unique_ptr<ExpertCache>> ecaches;
    ExpertCache *         ecache = nullptr;
    // The cache holding `il`'s experts: the one on the device that computes it.
    // Prefill runs its expert matmuls at the pool, like decode does. Routing
    // prefill to the layer's compute device instead was tried and measured: it
    // does speed prefill up (+3-6%, the heavy matmul lands on the fast card),
    // but prefill is also what WARMS the pools, so the pool device starts decode
    // cold and has to fault its share in one miss at a time. Net -10% at 128
    // generated tokens and still -2% at 512, where the transient has mostly
    // washed out. The two are not separable; don't try it again.
    ExpertCache * ec_of(int il) const {
        if (ecaches.size() <= 1) return ecache;
        const size_t d = (size_t) pdev_of(il);
        return d < ecaches.size() ? ecaches[d].get() : nullptr;
    }
    // Every live cache, for the whole-model operations (stats, quota mode).
    std::vector<ExpertCache *> all_ecaches() const {
        std::vector<ExpertCache *> v;
        for (const auto & c : ecaches) if (c) v.push_back(c.get());
        return v;
    }
    // Model-wide cache counters. fetch_ms adds up per-device fetch time, which
    // on a split runs partly in parallel -- it is a cost total, not a wall time.
    ExpertCache::Stats ec_stats_sum() const {
        ExpertCache::Stats t;
        for (const auto & c : ecaches) {
            if (!c) continue;
            const auto & s = c->stats();
            t.hits += s.hits; t.misses += s.misses; t.evictions += s.evictions;
            t.fetch_ms += s.fetch_ms; t.fetch_bytes += s.fetch_bytes;
        }
        return t;
    }
    // ---- graph allocation / execution ----
    // With one GPU these stay on the historical gallocr + direct-compute path,
    // which is what makes reuse_graph and the CUDA-graph-friendly fused decode
    // possible. A layer split makes every graph span devices, so it goes to the
    // scheduler instead: sched assigns each op to the backend that owns its
    // weights (layer weights, KV rows and expert slots all sit on the layer's
    // device) and inserts the copies where the hidden state crosses a boundary.
    bool alloc_graph(ggml_gallocr_t ga, ggml_cgraph * gf) {
        if (!multi_gpu()) return ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_sched_reset(sched);
        return ggml_backend_sched_alloc_graph(sched, gf);
    }
    ggml_status compute_graph(ggml_cgraph * gf) {
        return multi_gpu() ? ggml_backend_sched_graph_compute(sched, gf)
                           : ggml_backend_graph_compute(backend, gf);
    }
    // A scheduler over every GPU in the plan plus the CPU. ggml requires the
    // last backend to be a CPU device: it is the fallback for ops no
    // accelerator claims.
    ggml_backend_sched_t make_sched() {
        std::vector<ggml_backend_t> be(gpus.begin(), gpus.end());
        std::vector<ggml_backend_buffer_type_t> bt;
        for (ggml_backend_t b : gpus) bt.push_back(ggml_backend_get_default_buffer_type(b));
        be.push_back(cpu_backend);
        bt.push_back(ggml_backend_get_default_buffer_type(cpu_backend));
        ggml_backend_sched_t s = ggml_backend_sched_new(be.data(), bt.data(), (int) be.size(),
                                                        GRAPH_SIZE, false, false);
        if (!s) throw std::runtime_error("failed to create backend scheduler");
        return s;
    }

    // Smallest pool across devices: what bounds a batch that must fit every layer.
    int ec_min_slots() const {
        int m = INT32_MAX;
        for (const auto & c : ecaches) if (c) m = std::min(m, c->min_slots());
        return m == INT32_MAX ? 0 : m;
    }
    bool                  ssd_mode = false;     // experts streamed from SSD (no RAM copy)
    ggml_gallocr_t        cache_galloc = nullptr;
    // One graph allocator per GPU: under a layer split each per-layer segment
    // graph is allocated and run on the device that owns the layer, so this path
    // never touches the scheduler at all.
    std::vector<ggml_gallocr_t> cache_gallocs;

    // Carry tensors exist once per device. A layer's segment graph must read and
    // write carries in the memory of the device that runs it; otherwise every
    // layer on a secondary device becomes a cross-device round trip with a
    // device sync, which is what made this path ~100x slower than the fused one.
    // The p_* members below point at the set for the layer currently being
    // built (use_carry); only the hidden state ever crosses, once per device
    // boundary (hop_carry).
    struct CarrySet {
        ggml_context *        ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor * h = nullptr, * ffn_in = nullptr, * resid = nullptr, * weights = nullptr;
        ggml_tensor * ffn_in2 = nullptr, * resid2 = nullptr, * weights2 = nullptr;
        ggml_tensor * slot_g = nullptr, * slot_u = nullptr, * slot_d = nullptr;
        // qwen4exp: `resid` has no meaning when the residual is hc streams wide
        // (seg B re-reads `h` instead); what seg B needs from seg A is the
        // scatter weights the ffn mixer produced. `ple` holds this token's
        // gathered n-gram rows.
        ggml_tensor * inject = nullptr, * inject2 = nullptr, * ple = nullptr;
    };
    std::vector<CarrySet> carries;
    std::vector<float>    hop_buf;      // staging for the boundary hidden-state hop

    void use_carry(int d);
    void hop_carry(int from, int to);            // the running hidden state
    void hop_carry_ab(int from, int to, int il); // seg A -> seg B carries for layer il
    void plan_pools();                           // fill pool_dev from per-device surplus
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
    ggml_tensor *         p_inject   = nullptr;  // [hc]       hc scatter weights (seg A -> B)
    ggml_tensor *         p_inject2  = nullptr;
    ggml_tensor *         p_ple      = nullptr;  // [n_embd]   this token's n-gram rows

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
    // Multi-GPU only: the fused graph's own scheduler, so that a reset from any
    // other graph cannot invalidate its one-time allocation.
    ggml_backend_sched_t  f_sched  = nullptr;
    int                   f_nkv    = 0;
    std::vector<int32_t>  g2s_host;
    std::vector<int32_t>  sel_host;
    // recurrent-state backup (for rolling back a speculative miss)
    ggml_context *        bak_ctx = nullptr;
    ggml_backend_buffer_t bak_buf = nullptr;
    std::vector<ggml_tensor *> conv_bak, ssm_bak;

    // recurrent / KV state (persistent across decode steps)
    ggml_context *        st_ctx = nullptr;
    // KV / recurrent state, one buffer per GPU: a layer's state must sit on the
    // device that computes the layer. Single-GPU leaves exactly one entry.
    std::vector<ggml_backend_buffer_t> st_bufs;
    std::vector<ggml_tensor *> k_cache;     // [n_embd_gqa, n_ctx]  (attention layers)
    std::vector<ggml_tensor *> v_cache;     // [n_embd_gqa, n_ctx]  (attention layers, non-transposed)
    std::vector<ggml_tensor *> conv_state;  // [d_conv-1, conv_ch]  (GDN layers)
    std::vector<ggml_tensor *> ssm_state;   // [S, S, H_v]          (GDN layers)
    // qwen4exp PLE: the module's own dilated-conv history, kept per PLE layer
    // and shaped like conv_state -- tokens on ne[0] so a batch concatenates.
    std::vector<ggml_tensor *> ple_conv_state;  // [(K-1)*ngram, hc_dim]
    // qwen4exp QSA: raw (unnormed, unroped) indexer keys, one per cached token.
    // Pooling happens over these, so they are stored before either transform --
    // and in F32, because the pooled scores decide which cells are attended to
    // at all and an F16 round trip can reorder a near tie.
    std::vector<ggml_tensor *> idx_k_cache;     // [indexer_head_dim, n_ctx]

    // The n-gram table lives outside every backend buffer (see ngram_table.h).
    // ngram_hist carries the tokens preceding the current batch so a chunked
    // prefill and a token-at-a-time decode hash the same context a single-shot
    // pass would; it is cleared by reset() alongside the KV.
    std::unique_ptr<NgramTable> ngram;
    std::vector<int32_t> ngram_hist;
    std::vector<int32_t> ngram_rows;    // scratch: n_heads row indices per token
    std::vector<float>   ngram_embd;    // scratch: the gathered rows, F32

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
            if (getenv("QWEN_LAYER_STATS")) for (auto * c : all_ecaches()) c->dump_layer_stats("decode");
            // What quota would the generation's own routing have asked for?
            if (getenv("QWEN_LAYER_QUOTA_DRIFT")) for (auto * c : all_ecaches()) c->dump_quota_plan("decode-fit");
            const auto & s = ec_stats_sum();
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
            // Each device profiles only its own layers, so a split writes one
            // file per device (".gpu<N>") rather than having them clobber each
            // other; a single GPU keeps the plain filename it always had.
            if (cfg.cache_profile_save && !cfg.cache_profile.empty()) {
                for (size_t d = 0; d < ecaches.size(); ++d) {
                    if (!ecaches[d]) continue;
                    const std::string path = multi_gpu()
                        ? cfg.cache_profile + ".gpu" + std::to_string(d)
                        : cfg.cache_profile;
                    if (ecaches[d]->save_profile(path))
                        fprintf(stderr, "expert cache: saved profile to '%s'\n", path.c_str());
                }
            }
        }
        ecaches.clear();
        ecache = nullptr;
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
        if (f_sched)        ggml_backend_sched_free(f_sched);
        if (f_ctx)          ggml_free(f_ctx);
        if (bak_buf)        ggml_backend_buffer_free(bak_buf);
        if (bak_ctx)        ggml_free(bak_ctx);
        for (auto g : cache_gallocs) if (g) ggml_gallocr_free(g);
        cache_galloc = nullptr;   // aliases cache_gallocs[0], already freed
        for (auto & c : carries) {
            if (c.buf) ggml_backend_buffer_free(c.buf);
            if (c.ctx) ggml_free(c.ctx);
        }
        carries.clear();
        if (cbuf)           ggml_backend_buffer_free(cbuf);
        if (cctx)           ggml_free(cctx);
        if (sched)          ggml_backend_sched_free(sched);
        if (galloc)         ggml_gallocr_free(galloc);
        if (dgalloc)        ggml_gallocr_free(dgalloc);
        if (dctx)           ggml_free(dctx);
        for (auto b : st_bufs) if (b) ggml_backend_buffer_free(b);
        if (st_ctx)         ggml_free(st_ctx);
        // expert_cpu_bufs and weights_bufs are owned here (not by Model) in
        // split/ssd mode; in single-backend mode Model owns and frees the weights.
        for (auto b : expert_cpu_bufs) if (b) ggml_backend_buffer_free(b);
        if (cpu_backend)    ggml_backend_free(cpu_backend);
        if (weights_buf_owned)
            for (auto b : weights_bufs) if (b) ggml_backend_buffer_free(b);
        // gpus[0] aliases `backend`; free the secondaries first, then it.
        for (size_t i = 1; i < gpus.size(); ++i)
            if (gpus[i]) ggml_backend_free(gpus[i]);
        if (backend)        ggml_backend_free(backend);
    }

    ggml_tensor * W(const char * fmt, int il) {
        char name[256];
        snprintf(name, sizeof(name), fmt, il);
        ggml_tensor * t = model.tensor(name);
        if (!t) throw std::runtime_error(std::string("missing tensor: ") + name);
        return t;
    }
    // Is the PLE module part of the graph? False for a model without one and
    // for --ngram off, which is what makes that flag a graph-level switch
    // rather than just a storage choice.
    bool use_ple() const {
        return ngram && ngram->mode() != NgramTable::Mode::OFF &&
               model.hparams().has_ple();
    }

    ggml_tensor * Wopt(const char * fmt, int il) {
        char name[256];
        snprintf(name, sizeof(name), fmt, il);
        return model.tensor(name);
    }

    void init();
    void plan_layers();   // fill layer_dev from the --gpu-split shares
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
    // kv_pos: KV-cache write position for this batch (-1 = n_past). Sub-chunked
    // prefill passes n_past+offset so a layer's chunks land consecutively.
    ggml_tensor * build_attn(ggml_context * ctx, ggml_cgraph * gf, int il,
                             ggml_tensor * Q, ggml_tensor * K, ggml_tensor * V,
                             ggml_tensor * mask, int n_tokens, int n_kv, int kv_pos = -1);
    ggml_tensor * build_gdn(ggml_context * ctx, ggml_cgraph * gf, int il,
                            ggml_tensor * x, int n_tokens);
    // ---- qwen4exp hyper-connections ----
    // The residual is hc_count streams wide. Each block reads one n_embd-wide
    // view of it (build_hc_mix) and its output is scattered back over all the
    // streams (build_hc_combine). The mixer's grouped RMSNorm is what a plain
    // architecture would call attn_norm / ffn_norm -- qwen4exp carries no
    // separate norm tensors, and the final mixer doubles as the output norm.
    //   res_hc is [n_embd, hc, n_tokens]; the mix returns [n_embd, n_tokens].
    // `inject`, when non-null, receives the [hc, n_tokens] scatter weights that
    // the matching build_hc_combine needs.
    ggml_tensor * build_hc_mix(ggml_context * ctx, ggml_tensor * res_hc,
                               ggml_tensor * w_norm, ggml_tensor * w_down,
                               ggml_tensor * w_up, ggml_tensor * w_inject,
                               ggml_tensor ** inject);
    ggml_tensor * build_hc_combine(ggml_context * ctx, ggml_tensor * res_hc,
                                   ggml_tensor * block_out, ggml_tensor * inject);
    // qwen4exp PLE. `emb` is the host-gathered n-gram rows [n_embd, n_tokens].
    ggml_tensor * build_ple(ggml_context * ctx, ggml_cgraph * gf, int il,
                            ggml_tensor * res_hc, ggml_tensor * emb, int n_tokens);
    void fill_ple_input(ggml_cgraph * gf, const int32_t * tokens, int n_tokens);

    // qwen4exp QSA. The host-side inputs depend on the cache layout and the
    // batch's positions but not on the layer, so one set serves every QSA layer
    // in a graph; build_qsa_mask fills this on first use.
    struct QsaShared {
        ggml_tensor * cell_blk = nullptr, * blk_cells = nullptr;
        ggml_tensor * blk_pos  = nullptr, * bias      = nullptr;
        int n_blocks = 0, width = 0;
    };
    ggml_tensor * build_qsa_mask(ggml_context * ctx, ggml_cgraph * gf, int il,
                                 ggml_tensor * cur, ggml_tensor * inp_pos,
                                 ggml_tensor * mask, int n_tokens, int n_kv,
                                 int kv_pos, QsaShared & shared);
    void set_qsa_inputs(ggml_cgraph * gf, int n_tokens, int n_kv, int kv_pos);
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
    // Prefer a GPU device (CUDA/Metal/etc.) when requested and available.
    if (cfg.use_cuda) {
        const std::vector<ggml_backend_dev_t> devs = gpu_devices();
        // Without an explicit plan, take the first device that initializes --
        // the historical behavior. With one, honor the requested order and
        // treat a device that fails to init as fatal: silently dropping it
        // would shift every layer's placement away from what was measured.
        if (cfg.gpus.empty()) {
            for (ggml_backend_dev_t dev : devs) {
                backend = ggml_backend_dev_init(dev, nullptr);
                if (backend) {
                    fprintf(stderr, "backend: GPU [%s] %s\n",
                            ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
                    gpus.push_back(backend);
                    dev_budget.push_back(cfg.vram_budget_mb * 1024ull * 1024ull);
                    break;
                }
            }
        } else {
            for (const GpuPlan & p : cfg.gpus) {
                if (p.device < 0 || (size_t) p.device >= devs.size())
                    throw std::runtime_error("--gpus: no GPU device with index "
                                             + std::to_string(p.device) + " (found "
                                             + std::to_string(devs.size()) + ")");
                ggml_backend_t be = ggml_backend_dev_init(devs[(size_t) p.device], nullptr);
                if (!be)
                    throw std::runtime_error("failed to init GPU device "
                                             + std::to_string(p.device));
                // With no --vram-budget there is still a ratio to pick: use what
                // the device actually has free, so a resident (non-offload)
                // split lands layers in proportion to real capacity instead of
                // piling them all onto the primary.
                const bool from_free = (p.budget_mb == VRAM_BUDGET_AUTO);
                size_t budget;
                if (from_free) {
                    size_t dev_free = 0, dev_total = 0;
                    ggml_backend_dev_memory(devs[(size_t) p.device], &dev_free, &dev_total);
                    budget = dev_free;
                } else {
                    budget = p.budget_mb * 1024ull * 1024ull;
                }
                fprintf(stderr, "backend: GPU%d [%s] %s (%s %zu MB, split %.3g)\n",
                        p.device, ggml_backend_dev_name(devs[(size_t) p.device]),
                        ggml_backend_dev_description(devs[(size_t) p.device]),
                        from_free ? "free" : "budget", budget >> 20, p.split);
                gpus.push_back(be);
                dev_budget.push_back(budget);
            }
            backend = gpus[0];
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
        gpus.assign(1, backend);          // "device 0" is the CPU here
        dev_budget.assign(1, 0);
    }

    // Decide which device computes which layer before any weight is placed.
    plan_layers();

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
        DevicePlan ssd_plan = device_plan();
        model.load_weights_ssd(backend, weights_bufs, &ssd_plan, &dev_weight_bytes);
        reuse_graph = false;   // every token goes through the per-token cache path
        // The SSD tier has no CPU backend to fall back to, so it normally runs
        // without a scheduler at all. A layer split still needs one to route
        // each op to the device holding that layer's weights. ggml_backend_sched
        // asserts that the LAST backend is a CPU device (it is the fallback for
        // ops no accelerator claims), so one has to be created here even though
        // this tier never runs expert matmuls on it.
        if (multi_gpu()) {
            cpu_backend = ggml_backend_cpu_init();
            if (!cpu_backend) throw std::runtime_error("failed to init CPU backend (ssd tier)");
            int nth = cfg.n_threads > 0 ? cfg.n_threads : (int) std::thread::hardware_concurrency();
            if (nth <= 0) nth = 4;
            ggml_backend_cpu_set_n_threads(cpu_backend, nth);

            std::vector<ggml_backend_t> sb(gpus.begin(), gpus.end());
            std::vector<ggml_backend_buffer_type_t> st;
            for (ggml_backend_t be : gpus) st.push_back(ggml_backend_get_default_buffer_type(be));
            sb.push_back(cpu_backend);
            st.push_back(ggml_backend_get_default_buffer_type(cpu_backend));
            sched = ggml_backend_sched_new(sb.data(), st.data(), (int) sb.size(),
                                           GRAPH_SIZE, false, false);
            if (!sched) throw std::runtime_error("failed to create backend scheduler (ssd tier)");
        }
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

        // Load weights: non-expert → GPU (its layer's GPU under a split),
        // expert → CPU (pinned).
        weights_buf_owned = true;   // set first: a throw mid-load still frees what was allocated
        DevicePlan plan = device_plan();
        model.load_weights_split(backend, cpu_buft, weights_bufs, expert_cpu_bufs,
                                 &plan, &dev_weight_bytes);

        // Create backend scheduler: GPUs first (higher priority, in plan order),
        // CPU last. The sched routes each op to the backend that owns its
        // weights and inserts the cross-device copies at the layer boundaries.
        std::vector<ggml_backend_t> sched_be(gpus.begin(), gpus.end());
        std::vector<ggml_backend_buffer_type_t> sched_bt;
        for (ggml_backend_t be : gpus) sched_bt.push_back(ggml_backend_get_default_buffer_type(be));
        sched_be.push_back(cpu_backend);
        sched_bt.push_back(ggml_backend_get_default_buffer_type(cpu_backend));
        sched = ggml_backend_sched_new(sched_be.data(), sched_bt.data(),
                                       (int) sched_be.size(), GRAPH_SIZE, false, false);
        if (!sched) throw std::runtime_error("failed to create backend scheduler");

        // Disable persistent reuse graph — sched alloc is incompatible with it.
        reuse_graph = false;

        fprintf(stderr, "expert offload: ON (experts stream into the VRAM cache;"
                        " QWEN_CPU_PREFILL=1 runs prefill experts on CPU instead)\n");
    } else if (multi_gpu()) {
        // Plain resident weights spread over the GPUs: no offload, the model
        // simply does not fit one card. The graph then spans devices, so it
        // needs a scheduler (and the persistent reuse graph, which assumes a
        // single backend and its own buffer, has to go).
        weights_buf_owned = true;   // set first: a throw mid-load still frees what was allocated
        DevicePlan plan = device_plan();
        model.load_weights_multi(backend, plan, weights_bufs, &dev_weight_bytes);

        cpu_backend = ggml_backend_cpu_init();
        if (!cpu_backend) throw std::runtime_error("failed to init CPU backend for multi-GPU");
        int nth = cfg.n_threads > 0 ? cfg.n_threads : (int) std::thread::hardware_concurrency();
        if (nth <= 0) nth = 4;
        ggml_backend_cpu_set_n_threads(cpu_backend, nth);
        sched = make_sched();
        reuse_graph = false;
        fprintf(stderr, "multi-GPU: model resident across %zu GPUs (no expert offload)\n",
                gpus.size());
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

    const int conv_ch = hp.ssm_d_inner + 2 * hp.ssm_n_group * hp.ssm_d_state;
    const int S       = hp.ssm_d_state;
    const int H_v     = hp.ssm_dt_rank;

    // The n-gram table is deliberately built after the weights: it is not one of
    // them. No backend buffer holds it in any mode.
    {
        NgramTable::Mode m = NgramTable::Mode::DISK;
        if (!NgramTable::parse_mode(cfg.ngram_mode, m)) {
            throw std::runtime_error("--ngram: expected off, disk or ram, got '"
                                     + cfg.ngram_mode + "'");
        }
        ngram.reset(new NgramTable(model, m, cfg.ngram_cache_mb));
        if (hp.has_ple()) {
            fprintf(stderr, "ngram: %s", NgramTable::mode_name(ngram->mode()));
            if (ngram->mode() == NgramTable::Mode::OFF) {
                fprintf(stderr, " (n-gram embedding disabled)\n");
            } else {
                fprintf(stderr, " (%zu MB resident)\n", ngram->resident_bytes() >> 20);
            }
        }
    }

    ggml_init_params kp{};
    kp.mem_size   = (size_t) ggml_tensor_overhead() * n_layer * 6 + 4096;
    kp.no_alloc   = true;
    st_ctx = ggml_init(kp);

    k_cache.assign(n_layer, nullptr);
    v_cache.assign(n_layer, nullptr);
    conv_state.assign(n_layer, nullptr);
    ssm_state.assign(n_layer, nullptr);
    ple_conv_state.assign(n_layer, nullptr);
    idx_k_cache.assign(n_layer, nullptr);

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
            k_cache[il] = ggml_new_tensor_2d(st_ctx, GGML_TYPE_F16, n_embd_gqa, n_ctx);
            v_cache[il] = ggml_new_tensor_2d(st_ctx, GGML_TYPE_F16, n_embd_gqa, n_ctx);
            ggml_set_name(k_cache[il], ("k_" + std::to_string(il)).c_str());
            ggml_set_name(v_cache[il], ("v_" + std::to_string(il)).c_str());
            if (hp.has_qsa() && hp.compress_ratio(il) > 0) {
                idx_k_cache[il] = ggml_new_tensor_2d(st_ctx, GGML_TYPE_F32,
                        hp.indexer_head_dim, n_ctx);
                ggml_set_name(idx_k_cache[il], ("idxk_" + std::to_string(il)).c_str());
            }
        }
        // independent of the attention/GDN split: a PLE layer can be either
        if (use_ple() && hp.is_ple(il)) {
            ple_conv_state[il] = ggml_new_tensor_2d(st_ctx, GGML_TYPE_F32,
                    hp.ple_conv_state(), hp.n_embd_hc());
            ggml_set_name(ple_conv_state[il], ("ple_conv_" + std::to_string(il)).c_str());
        }
    }
    // A layer's KV / recurrent state must live on the device that computes the
    // layer, so with a split the one state buffer becomes one per device.
    st_bufs.assign(std::max<size_t>(gpus.size(), 1), nullptr);
    if (!multi_gpu()) {
        st_bufs[0] = ggml_backend_alloc_ctx_tensors(st_ctx, backend);
        if (!st_bufs[0]) throw std::runtime_error("failed to alloc state buffer");
    } else {
        for (size_t d = 0; d < gpus.size(); ++d) {
            std::vector<ggml_tensor *> ts;
            for (int il = 0; il < n_layer; ++il) {
                if ((size_t) dev_of(il) != d) continue;
                if (k_cache[il])    { ts.push_back(k_cache[il]);    ts.push_back(v_cache[il]); }
                if (idx_k_cache[il]) ts.push_back(idx_k_cache[il]);
                if (conv_state[il]) { ts.push_back(conv_state[il]); ts.push_back(ssm_state[il]); }
                if (ple_conv_state[il]) ts.push_back(ple_conv_state[il]);
            }
            st_bufs[d] = alloc_tensor_list(ggml_backend_get_default_buffer_type(gpus[d]),
                                           ts, "state buffer");
        }
    }
    for (size_t d = 0; d < st_bufs.size() && d < dev_kv_bytes.size(); ++d)
        dev_kv_bytes[d] = st_bufs[d] ? ggml_backend_buffer_get_size(st_bufs[d]) : 0;

    if (!sched) {
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    // A sched no longer implies offload: a multi-GPU resident model has one too,
    // and it has no expert cache to build.
    if (use_expert_offload || ssd_mode) {
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

// Assign each transformer layer to a GPU. Devices get contiguous ranges of the
// main stack in plan order, sized by their --gpu-split share (or, with no
// explicit split, by their VRAM budget). A device whose share is 0 gets no
// layers at all -- its budget then goes entirely to the expert pool, which is
// how a pure "expert pool" device falls out of the same knob as a layer split.
void Runtime::Impl::plan_layers() {
    const auto & hp = model.hparams();
    const int n_layer = (int) hp.n_layer;
    const int n_main  = (int) hp.n_main();

    layer_dev.assign((size_t) n_layer, 0);
    dev_weight_bytes.assign(gpus.size(), 0);
    dev_kv_bytes.assign(gpus.size(), 0);
    if (gpus.size() <= 1) return;

    // An explicit --gpu-split on any device switches the whole plan to explicit
    // shares; otherwise split proportionally to each device's VRAM budget.
    bool auto_split = true;
    for (const GpuPlan & p : cfg.gpus) if (p.split >= 0.0f) auto_split = false;

    std::vector<double> share(gpus.size(), 0.0);
    double total = 0.0;
    for (size_t i = 0; i < gpus.size(); ++i) {
        share[i] = auto_split ? (double) dev_budget[i]
                              : (i < cfg.gpus.size() && cfg.gpus[i].split >= 0.0f
                                     ? (double) cfg.gpus[i].split : 0.0);
        total += share[i];
    }
    if (total <= 0.0) { share[0] = 1.0; total = 1.0; }   // nothing to go on: all on primary

    // Cumulative rounding, so the ranges tile [0, n_main) exactly. Shares are
    // relative, which means values that happen to add up to the layer count come
    // back out as themselves: "36,4" on a 40-layer model gives exactly 36 and 4,
    // with no special case for it.
    int start = 0;
    double cum = 0.0;
    for (size_t i = 0; i < gpus.size(); ++i) {
        cum += share[i] / total;
        int end = (i + 1 == gpus.size()) ? n_main : (int) llround(cum * n_main);
        if (share[i] <= 0.0) end = start;            // 0 share: expert pool only
        end = std::max(start, std::min(end, n_main));
        for (int il = start; il < end; ++il) layer_dev[(size_t) il] = (int) i;
        start = end;
    }
    for (int il = start; il < n_main; ++il) layer_dev[(size_t) il] = 0;   // rounding tail

    // The trailing MTP (nextn) blocks stay on the primary device. Their graphs
    // are built by the single-backend MTP path, and one hidden-state hop per
    // drafted token is far cheaper than making that path span devices.
    for (int il = n_main; il < n_layer; ++il) layer_dev[(size_t) il] = 0;

    fprintf(stderr, "gpu layer split (%d main layers over %zu devices):\n", n_main, gpus.size());
    for (size_t i = 0; i < gpus.size(); ++i) {
        int lo = -1, hi = -1, cnt = 0;
        for (int il = 0; il < n_main; ++il)
            if (layer_dev[(size_t) il] == (int) i) { if (lo < 0) lo = il; hi = il; ++cnt; }
        const int devidx = i < cfg.gpus.size() ? cfg.gpus[i].device : (int) i;
        if (cnt == 0)
            fprintf(stderr, "  GPU%d: no layers (expert pool only, %zu MB budget)\n",
                    devidx, dev_budget[i] >> 20);
        else
            fprintf(stderr, "  GPU%d: layers %d-%d (%d, %.0f%%), %zu MB budget\n",
                    devidx, lo, hi, cnt, 100.0 * cnt / n_main, dev_budget[i] >> 20);
    }

    // Expert pools are placed here too, because weight placement follows them:
    // a layer's shared-expert weights have to sit with its expert matmul.
    plan_pools();
}

// Point the p_* carry members at device `d`'s set, so the segment graphs built
// after this call read and write carries in that device's memory.
void Runtime::Impl::use_carry(int d) {
    if (carries.empty()) return;
    const CarrySet & c = carries[(size_t) std::min<size_t>((size_t) d, carries.size() - 1)];
    p_h        = c.h;
    p_ffn_in   = c.ffn_in;
    p_resid    = c.resid;
    p_weights  = c.weights;
    p_ffn_in2  = c.ffn_in2;
    p_resid2   = c.resid2;
    p_weights2 = c.weights2;
    p_slot_g   = c.slot_g;
    p_slot_u   = c.slot_u;
    p_slot_d   = c.slot_d;
    p_inject   = c.inject;
    p_inject2  = c.inject2;
    p_ple      = c.ple;
}

// Move the running hidden state across a device boundary. This is the only
// value that crosses -- n_embd floats (~8 KB), once per boundary per token,
// which is the whole point of assigning layers in contiguous ranges.
void Runtime::Impl::hop_carry(int from, int to) {
    if (from == to || carries.size() <= 1) return;
    const size_t n = ggml_nbytes(carries[(size_t) from].h);
    hop_buf.resize(n / sizeof(float));
    ggml_backend_tensor_get(carries[(size_t) from].h, hop_buf.data(), 0, n);
    ggml_backend_tensor_set(carries[(size_t) to].h,   hop_buf.data(), 0, n);
}

// Copy layer `il`'s seg A -> seg B carries across a device boundary. Needed
// when the layer's expert pool sits on a different device than its attention:
// the normed FFN input, the residual base and the router weights all have to
// follow the expert matmul. ~2*n_embd + n_used floats, against expert weights
// that are three orders of magnitude larger -- which is why the matmul is moved
// to the pool rather than the pool to the matmul.
void Runtime::Impl::hop_carry_ab(int from, int to, int il) {
    if (from == to || carries.size() <= 1) return;
    const CarrySet & a = carries[(size_t) from];
    const CarrySet & b = carries[(size_t) to];
    auto cp = [&](ggml_tensor * s, ggml_tensor * d) {
        const size_t n = ggml_nbytes(s);
        hop_buf.resize((n + sizeof(float) - 1) / sizeof(float));
        ggml_backend_tensor_get(s, hop_buf.data(), 0, n);
        ggml_backend_tensor_set(d, hop_buf.data(), 0, n);
    };
    if (il & 1) { cp(a.ffn_in2, b.ffn_in2); cp(a.resid2, b.resid2); cp(a.weights2, b.weights2); }
    else        { cp(a.ffn_in,  b.ffn_in ); cp(a.resid,  b.resid ); cp(a.weights,  b.weights ); }
    if (a.inject) cp(il & 1 ? a.inject2 : a.inject, il & 1 ? b.inject2 : b.inject);
}

// Decide where each layer's expert pool lives, from how much VRAM each device
// has left once its own weights and KV are placed. A device with no compute
// layers contributes its whole budget here, which is what turns --gpu-split 0
// into a pure expert pool. Ranges are contiguous so that when the split ratios
// happen to match the surplus ratios, pool_dev == layer_dev and nothing hops.
void Runtime::Impl::plan_pools() {
    const auto & hp = model.hparams();
    const int n_main = (int) hp.n_main();
    pool_dev.assign((size_t) n_main, 0);
    for (int il = 0; il < n_main; ++il) pool_dev[(size_t) il] = dev_of(il);
    if (gpus.size() <= 1) return;
    // Without expert offload there are no pools to place: the weights are all
    // resident and a layer's experts are wherever its other weights are.
    if (cfg.vram_budget_mb == 0 || !model.has_expert_tensors()) return;

    // Runs before the weights are placed (their placement depends on the answer),
    // so the surplus is budget minus the KV cache and the compute headroom, with
    // the non-expert weights left out. They are roughly proportional to a
    // device's layer count, so leaving them out skews the ratio a little but not
    // the shape; the placement line below makes the outcome visible.
    const size_t compute = 1024ull * 1024ull * 1024ull;
    const int n_embd_gqa = hp.n_head_kv * hp.n_embd_head;
    const int conv_ch    = hp.ssm_d_inner + 2 * hp.ssm_n_group * hp.ssm_d_state;
    std::vector<double> surplus(gpus.size(), 0.0);
    double total = 0.0;
    for (size_t d = 0; d < gpus.size(); ++d) {
        size_t kv = 0;
        for (int il = 0; il < (int) hp.n_layer; ++il) {
            if ((size_t) dev_of(il) != d) continue;
            if (hp.is_recurrent(il))
                kv += (size_t) (hp.ssm_d_conv - 1) * conv_ch * 4
                    + (size_t) hp.ssm_d_state * hp.ssm_d_state * hp.ssm_dt_rank * 4;
            else
                kv += (size_t) 2 * n_embd_gqa * cfg.n_ctx * 2;   // F16 K and V
        }
        const size_t reserve = kv + compute;
        surplus[d] = dev_budget[d] > reserve ? (double) (dev_budget[d] - reserve) : 0.0;
        total += surplus[d];
    }
    if (total <= 0.0) return;   // nothing to distribute: pools stay with their layers

    int start = 0;
    double cum = 0.0;
    for (size_t d = 0; d < gpus.size(); ++d) {
        cum += surplus[d] / total;
        int end = (d + 1 == gpus.size()) ? n_main : (int) llround(cum * n_main);
        if (surplus[d] <= 0.0) end = start;
        end = std::max(start, std::min(end, n_main));
        for (int il = start; il < end; ++il) pool_dev[(size_t) il] = (int) d;
        start = end;
    }
    for (int il = start; il < n_main; ++il) pool_dev[(size_t) il] = (int) gpus.size() - 1;

    int hops = 0;
    for (int il = 0; il < n_main; ++il) if (pool_dev[(size_t) il] != dev_of(il)) ++hops;
    fprintf(stderr, "expert pool placement:");
    for (size_t d = 0; d < gpus.size(); ++d) {
        int cnt = 0;
        for (int il = 0; il < n_main; ++il) if (pool_dev[(size_t) il] == (int) d) ++cnt;
        const int devidx = d < cfg.gpus.size() ? cfg.gpus[d].device : (int) d;
        fprintf(stderr, " GPU%d=%d layers (%.0f MB surplus)", devidx, cnt, surplus[d] / 1048576.0);
    }
    fprintf(stderr, "; %d layer(s) compute experts off their attention device\n", hops);
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
    const size_t compute = 1024ull * 1024ull * 1024ull;   // gallocr graph buffers

    // Only the main stack's experts are offloaded; the trailing MTP (nextn) block
    // stays fully VRAM-resident, so the cache covers n_main() layers (not n_layer).
    const int n_main = (int) hp.n_main();

    if (!multi_gpu()) {
        const size_t budget = cfg.vram_budget_mb * 1024ull * 1024ull;
        size_t gpu_w = 0;
        for (auto b : weights_bufs) gpu_w += ggml_backend_buffer_get_size(b);
        size_t kv_bytes = 0;
        for (auto b : st_bufs) if (b) kv_bytes += ggml_backend_buffer_get_size(b);
        const size_t reserve = gpu_w + kv_bytes + compute;
        const size_t avail   = budget > reserve ? budget - reserve : 0;
        fprintf(stderr, "VRAM budget %zu MB = weights %zu + KV %zu + compute %zu + expert pool %zu MB\n",
                cfg.vram_budget_mb, gpu_w >> 20, kv_bytes >> 20, compute >> 20, avail >> 20);
        ecaches.push_back(std::make_unique<ExpertCache>(
            backend, model, n_main, hp.n_expert, n_used, avail, ssd_mode));
    } else {
        // One cache per device, each sized against that device's own budget and
        // covering the layers whose expert pool it holds. That is not always the
        // set of layers it computes: a device given --gpu-split 0 has no layers
        // but still takes pools, which is the expert-pool layout (plan_pools()
        // ran back in plan_layers(), since weight placement follows it).
        for (size_t d = 0; d < gpus.size(); ++d) {
            std::vector<bool> mask((size_t) n_main, false);
            int n_owned = 0;
            for (int il = 0; il < n_main; ++il)
                if ((size_t) pdev_of(il) == d) { mask[(size_t) il] = true; ++n_owned; }

            const size_t gpu_w    = d < dev_weight_bytes.size() ? dev_weight_bytes[d] : 0;
            const size_t kv_bytes = d < dev_kv_bytes.size()     ? dev_kv_bytes[d]     : 0;
            const size_t budget   = dev_budget[d];
            // A device that computes nothing still needs a graph buffer only for
            // the splits sched routes to it, but reserving the full compute
            // headroom on every device is the safe side of the estimate.
            const size_t reserve = gpu_w + kv_bytes + compute;
            const size_t avail   = budget > reserve ? budget - reserve : 0;
            const int devidx = d < cfg.gpus.size() ? cfg.gpus[d].device : (int) d;
            fprintf(stderr, "GPU%d budget %zu MB = weights %zu + KV %zu + compute %zu"
                            " + expert pool %zu MB (%d layers)\n",
                    devidx, budget >> 20, gpu_w >> 20, kv_bytes >> 20, compute >> 20,
                    avail >> 20, n_owned);
            if (n_owned == 0) { ecaches.push_back(nullptr); continue; }
            ecaches.push_back(std::make_unique<ExpertCache>(
                gpus[d], model, n_main, hp.n_expert, n_used, avail, ssd_mode, &mask));
        }
    }
    ecache = ecaches[0].get();

    // persistent carry tensors (bridge per-layer graph segments) + fast-path
    // in-graph remap table (g2s_all) and selection readback (sel_all).
    // One carry set per GPU (exactly one without a split, which reproduces the
    // original single-buffer layout). See the CarrySet comment in Impl.
    carries.resize(std::max<size_t>(gpus.size(), 1));
    for (size_t d = 0; d < carries.size(); ++d) {
        CarrySet & c = carries[d];
        ggml_init_params ccp{};
        ccp.mem_size = ggml_tensor_overhead() * 20 + 256;
        ccp.no_alloc = true;
        c.ctx      = ggml_init(ccp);
        // hc widens the running hidden state; everything else is per-token
        c.h        = hp.has_hc()
                ? ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, n_embd, hp.hc_count)
                : ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n_embd);
        c.ffn_in   = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n_embd);
        c.resid    = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n_embd);
        c.weights  = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n_used);
        c.ffn_in2  = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n_embd);
        c.resid2   = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n_embd);
        c.weights2 = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, n_used);
        c.slot_g   = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, n_used, 1);
        c.slot_u   = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, n_used, 1);
        c.slot_d   = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, n_used, 1);
        if (hp.has_hc()) {
            c.inject  = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, hp.hc_count);
            c.inject2 = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, hp.hc_count);
        }
        if (use_ple()) {
            c.ple = ggml_new_tensor_2d(c.ctx, GGML_TYPE_F32, n_embd, 1);
        }
        char nm[64];
        const char * base[10] = { "h", "ffn_in", "resid", "weights", "ffn_in2",
                                  "resid2", "weights2", "slot_g", "slot_u", "slot_d" };
        ggml_tensor * ts[10] = { c.h, c.ffn_in, c.resid, c.weights, c.ffn_in2,
                                 c.resid2, c.weights2, c.slot_g, c.slot_u, c.slot_d };
        for (int k = 0; k < 10; ++k) {
            snprintf(nm, sizeof(nm), "carry%zu.%s", d, base[k]);
            ggml_set_name(ts[k], nm);
        }
        const char * opt_base[3] = { "inject", "inject2", "ple" };
        ggml_tensor * opt[3] = { c.inject, c.inject2, c.ple };
        for (int k = 0; k < 3; ++k) {
            if (!opt[k]) continue;
            snprintf(nm, sizeof(nm), "carry%zu.%s", d, opt_base[k]);
            ggml_set_name(opt[k], nm);
        }
        c.buf = ggml_backend_alloc_ctx_tensors(c.ctx, gpus.empty() ? backend : gpus[d]);
        if (!c.buf) throw std::runtime_error("init_cache: failed to alloc carry buffer");
    }
    use_carry(0);

    // Model-wide tables for the fused single-graph path. That graph is planned
    // by a scheduler, which copies these to whichever device needs them, so one
    // copy on the primary is enough.
    ggml_init_params cp{};
    cp.mem_size = ggml_tensor_overhead() * 8 + 256;
    cp.no_alloc = true;
    cctx = ggml_init(cp);
    g2s_all   = ggml_new_tensor_3d(cctx, GGML_TYPE_I32, 1, hp.n_expert, 3 * hp.n_layer);
    sel_all   = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, n_used, hp.n_layer);
    resmask_all = ggml_new_tensor_2d(cctx, GGML_TYPE_F32, hp.n_expert, hp.n_layer);
    ggml_set_name(resmask_all, "carry.resmask");
    want_all = ggml_new_tensor_2d(cctx, GGML_TYPE_I32, n_used, hp.n_layer);
    ggml_set_name(want_all, "carry.want");
    ggml_set_name(g2s_all, "carry.g2s");
    ggml_set_name(sel_all, "carry.sel");
    cbuf = ggml_backend_alloc_ctx_tensors(cctx, backend);
    if (!cbuf) throw std::runtime_error("init_cache: failed to alloc table buffer");

    cache_gallocs.clear();
    for (size_t d = 0; d < std::max<size_t>(gpus.size(), 1); ++d)
        cache_gallocs.push_back(ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(gpus.empty() ? backend : gpus[d])));
    cache_galloc = cache_gallocs[0];   // alias: the primary's, used by batched prefill
    f_galloc     = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    g2s_host.assign((size_t) 3 * hp.n_layer * hp.n_expert, 0);
    sel_host.assign((size_t) n_used * hp.n_layer, 0);
    resmask_host.assign((size_t) hp.n_layer * hp.n_expert, 0.0f);
    want_host.assign((size_t) n_used * hp.n_layer, 0);

    if (getenv("QWEN_FASTCACHE")) cache_fast_enabled = true;   // experimental single-graph path
    if (getenv("QWEN_RESIDENT_DECODE")) { cache_fast_enabled = true; resident_decode = true; }


    init_state_backup();   // GDN rollback buffers (speculative miss / MTP reject)

    // warm restart: pre-fill VRAM slots from a saved hot-expert profile. Under a
    // split each device reads its own ".gpu<N>" file, falling back to a
    // single-GPU profile (load_prefetch keeps only the layers this device owns),
    // so a profile recorded before the split still warms both devices.
    if (!cfg.cache_profile.empty()) {
        for (size_t d = 0; d < ecaches.size(); ++d) {
            if (!ecaches[d]) continue;
            std::string path = multi_gpu() ? cfg.cache_profile + ".gpu" + std::to_string(d)
                                           : cfg.cache_profile;
            size_t n = ecaches[d]->load_prefetch(path);
            if (n == 0 && multi_gpu()) {
                path = cfg.cache_profile;
                n = ecaches[d]->load_prefetch(path);
            }
            if (n > 0)
                fprintf(stderr, "expert cache: prefetched %zu experts from profile '%s'\n",
                        n, path.c_str());
        }
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
    for (auto * t : conv_state)     zero(t);
    for (auto * t : ssm_state)      zero(t);
    for (auto * t : ple_conv_state) zero(t);
    for (auto * t : idx_k_cache)    zero(t);
    ngram_hist.clear();
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
        ggml_tensor * mask, int n_tokens, int n_kv, int kv_pos) {
    if (kv_pos < 0) kv_pos = n_past;
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

    // gated RMSNorm with z: rms_norm(output)*ssm_norm * gate(z)
    output = ggml_cont(ctx, output);
    output = ggml_rms_norm(ctx, output, eps);
    output = ggml_mul(ctx, output, W("blk.%d.ssm_norm.weight", il));   // broadcast [S]
    ggml_tensor * zr = ggml_reshape_4d(ctx, z, S, H_v, n_tokens, 1);
    // qwen4exp is Qwen3.5's GDN with one numerical difference: the output gate
    // is a sigmoid rather than a silu.
    output = ggml_mul(ctx, output, hp.arch == Arch::QWEN4EXP ? ggml_sigmoid(ctx, zr)
                                                             : ggml_silu(ctx, zr));

    output = ggml_reshape_2d(ctx, output, S * H_v, n_tokens);
    ggml_tensor * cur = ggml_mul_mat(ctx, W("blk.%d.ssm_out.weight", il), output);  // [n_embd, n_tokens]
    return cur;
}

// ---- qwen4exp hyper-connections ----
//
// The residual stream is `hc` parallel copies of the model width. A block does
// not read it directly: build_hc_mix normalizes each stream, gates them with a
// low-rank projection of the whole thing, and averages them into one n_embd-wide
// block input. build_hc_combine then adds the block's output back into every
// stream, weighted per stream by the same mixer's `inject` head.
//
// The 2*sigmoid in combine centres the scatter weights on 1, so a zero
// injection matrix degenerates to the ordinary residual add -- which is the
// clearest way to read what the mechanism is doing.
ggml_tensor * Runtime::Impl::build_hc_mix(ggml_context * ctx, ggml_tensor * res_hc,
        ggml_tensor * w_norm, ggml_tensor * w_down, ggml_tensor * w_up,
        ggml_tensor * w_inject, ggml_tensor ** inject) {
    const auto & hp   = model.hparams();
    const int n_embd  = hp.n_embd;
    const int hc      = hp.hc_count;
    const int hc_dim  = hc * n_embd;
    const int64_t T   = res_hc->ne[2];

    // grouped RMSNorm: rms_norm reduces over ne[0] = one stream, then the
    // [hc_dim] gamma scales all of them. The converter folded the gammas to
    // (1 + w), so this is a plain multiply.
    ggml_tensor * xn = ggml_rms_norm(ctx, res_hc, hp.rms_eps);
    xn = ggml_reshape_2d(ctx, xn, hc_dim, T);
    xn = ggml_mul(ctx, xn, w_norm);

    ggml_tensor * lo = ggml_mul_mat(ctx, w_down, xn);                  // [low_rank, T]
    lo = ggml_silu(ctx, ggml_scale(ctx, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, w_up, lo)); // [hc_dim, T]

    ggml_tensor * gated = ggml_reshape_3d(ctx, ggml_mul(ctx, xn, gate), n_embd, hc, T);

    // collapse the streams by their mean. hc is 4, so an explicit sum of views
    // is both shorter and cheaper than a reduction op.
    const size_t stride = ggml_row_size(gated->type, n_embd) * hc;
    ggml_tensor * mixed = ggml_cont(ctx,
            ggml_view_2d(ctx, gated, n_embd, T, stride, 0));
    for (int c = 1; c < hc; ++c) {
        mixed = ggml_add(ctx, mixed,
                ggml_view_2d(ctx, gated, n_embd, T, stride,
                             ggml_row_size(gated->type, n_embd) * c));
    }
    mixed = ggml_scale(ctx, mixed, 1.0f / (float) hc);

    if (inject) {
        *inject = ggml_mul_mat(ctx, w_inject, xn);                     // [hc, T]
    }
    return mixed;
}

ggml_tensor * Runtime::Impl::build_hc_combine(ggml_context * ctx, ggml_tensor * res_hc,
        ggml_tensor * block_out, ggml_tensor * inject) {
    const auto & hp  = model.hparams();
    const int n_embd = hp.n_embd;
    const int hc     = hp.hc_count;
    const int64_t T  = res_hc->ne[2];

    ggml_tensor * w = ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float) hc));
    w = ggml_reshape_3d(ctx, ggml_scale(ctx, w, 2.0f), 1, hc, T);

    ggml_tensor * b = ggml_repeat_4d(ctx,
            ggml_reshape_3d(ctx, block_out, n_embd, 1, T), n_embd, hc, T, 1);

    return ggml_add(ctx, res_hc, ggml_mul(ctx, b, w));
}

// Hash this batch's tokens to n-gram table rows, fetch them and upload the
// result into the graph's "inp_ple" input. No-op when the graph has none
// (a model without PLE, or --ngram off).
void Runtime::Impl::fill_ple_input(ggml_cgraph * gf, const int32_t * tokens, int n_tokens) {
    if (!use_ple()) return;
    ggml_tensor * t = ggml_graph_get_tensor(gf, "inp_ple");
    if (!t) return;

    const auto & hp = model.hparams();
    ngram_rows.clear();
    ngram->hash_rows(tokens, n_tokens, ngram_hist, ngram_rows);

    ngram_embd.resize(ngram_rows.size() * hp.ple_head_dim);
    ngram->gather(ngram_rows.data(), ngram_rows.size(), ngram_embd.data());

    GGML_ASSERT(ggml_nbytes(t) == ngram_embd.size() * sizeof(float));
    ggml_backend_tensor_set(t, ngram_embd.data(), 0, ggml_nbytes(t));
}

// ---- qwen4exp QSA (query-sparse attention) ----
//
// A full-attention layer does not attend to every cached token. The cache is cut
// into blocks of `compress_ratio` tokens; each block is scored once, by a small
// indexer head against the mean of its members' indexer keys; and a query attends
// to a budget of the best blocks plus the incomplete tail it sits in.
//
// Only the mask changes -- the attention itself is the ordinary dense one over
// the same K/V. Below indexer_top_k + ratio - 1 cached tokens the budget covers
// everything, so this returns the causal mask untouched and the layer is exactly
// dense. That is not an optimisation: it is why a short-context run can be
// validated against the reference before any of this exists.
//
// `shared` caches the host-side inputs, which depend on the cache layout and the
// batch's positions but not on the layer, so every QSA layer in one graph fills
// them once.
ggml_tensor * Runtime::Impl::build_qsa_mask(ggml_context * ctx, ggml_cgraph * gf, int il,
        ggml_tensor * cur, ggml_tensor * inp_pos, ggml_tensor * mask,
        int n_tokens, int n_kv, int kv_pos, QsaShared & shared) {
    const auto & hp = model.hparams();
    const int r = (int) hp.compress_ratio(il);
    if (!hp.has_qsa() || r <= 0 || !idx_k_cache[il]) return mask;

    // QWEN_QSA_DEBUG=1 exposes the indexer's intermediates so they can be diffed
    // against llama-debug --tensor-filter '.*indexer_.*-<il>$'. Everything from
    // the raw keys to the finished mask is reachable this way, which is how the
    // one real difference that remains -- see set_qsa_inputs -- was pinned down.
    const bool dbg_on = getenv("QWEN_QSA_DEBUG") != nullptr;
    auto dbg = [&](const char * tag, ggml_tensor * t) {
        if (!dbg_on) return;
        ggml_tensor * c = ggml_cont(ctx, t);
        ggml_set_name(c, (std::string("qsa_") + tag + "_" + std::to_string(il)).c_str());
        ggml_set_output(c);
        ggml_build_forward_expand(gf, c);
    };

    const int idx_dim = (int) hp.indexer_head_dim;
    const int n_idx_h = (int) hp.indexer_n_head;
    const int width   = std::min(n_kv, (int) hp.indexer_top_k + r - 1);
    // Blocks cover the graph's whole cache width, not just what is cached right
    // now. A persistent decode graph is built once per KV bucket and reused for
    // up to KV_BUCKET more tokens, so a count frozen at build time would leave
    // every later cell in no block at all -- including the token being decoded,
    // whose own row would then be selected from stale blocks only. Cells that
    // hold nothing yet are ruled out by the host-side bias instead, and their
    // keys read as zero because the indexer cache is cleared with the KV cache.
    // For a one-shot graph n_kv is exactly what is cached, so this is the same
    // count as before.
    const int n_blocks = n_kv / r;              // whole blocks only; the rest is tail

    // this token's raw indexer key, straight into the cache
    ggml_tensor * k_raw = ggml_mul_mat(ctx, W("blk.%d.indexer.k_proj.weight", il), cur);
    if (persistent) {
        // Same reason build_attn writes K/V through an index input: this graph is
        // built once per KV bucket and replayed, so a write offset baked in at
        // build time would send every token of the bucket to the first token's
        // cell -- losing their keys and corrupting the block they pool into.
        ggml_build_forward_expand(gf, ggml_set_rows(ctx, idx_k_cache[il],
                ggml_reshape_2d(ctx, k_raw, idx_dim, n_tokens), d_kvidx));
    } else {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, k_raw,
                ggml_view_2d(ctx, idx_k_cache[il], idx_dim, n_tokens,
                             idx_k_cache[il]->nb[1],
                             (size_t) kv_pos * idx_k_cache[il]->nb[1])));
    }

    if (width >= n_kv || n_blocks == 0) return mask;   // the budget covers everything

    // ---- host-side inputs, shared by every QSA layer in this graph ----
    if (!shared.cell_blk) {
        shared.n_blocks = n_blocks;
        shared.width    = width;
        shared.cell_blk  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);
        shared.blk_cells = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, r * n_blocks);
        shared.blk_pos   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4 * n_blocks);
        shared.bias      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_tokens);
        ggml_set_input(shared.cell_blk);  ggml_set_name(shared.cell_blk,  "inp_qsa_cell_blk");
        ggml_set_input(shared.blk_cells); ggml_set_name(shared.blk_cells, "inp_qsa_blk_cells");
        ggml_set_input(shared.blk_pos);   ggml_set_name(shared.blk_pos,   "inp_qsa_blk_pos");
        ggml_set_input(shared.bias);      ggml_set_name(shared.bias,      "inp_qsa_bias");
    }
    GGML_ASSERT(shared.n_blocks == n_blocks && shared.width == width &&
                "QSA layers in one graph must share a compress ratio");

    // ---- pool each block's member keys, then norm and rotate the result ----
    ggml_tensor * k_all = ggml_view_2d(ctx, idx_k_cache[il], idx_dim, n_kv,
                                       idx_k_cache[il]->nb[1], 0);
    ggml_tensor * members = ggml_get_rows(ctx, k_all, shared.blk_cells);
    members = ggml_reshape_3d(ctx, members, idx_dim, r, n_blocks);

    // r is small (4 in the released model), so summing slices beats transposing
    // the tensor to reach ggml_sum_rows
    ggml_tensor * pooled = nullptr;
    for (int i = 0; i < r; ++i) {
        ggml_tensor * slice = ggml_cont(ctx,
                ggml_view_2d(ctx, members, idx_dim, n_blocks,
                             members->nb[2], (size_t) i * members->nb[1]));
        pooled = pooled ? ggml_add(ctx, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx, pooled, 1.0f / (float) r);
    dbg("kraw", k_raw);
    dbg("pool0", pooled);
    pooled = ggml_reshape_3d(ctx, pooled, idx_dim, 1, n_blocks);
    pooled = ggml_mul(ctx, ggml_rms_norm(ctx, pooled, hp.rms_eps),
                      W("blk.%d.indexer.k_norm.weight", il));
    pooled = apply_rope(ctx, pooled, shared.blk_pos);
    pooled = ggml_reshape_2d(ctx, pooled, idx_dim, n_blocks);

    ggml_tensor * q = ggml_mul_mat(ctx, W("blk.%d.indexer.q_proj.weight", il), cur);
    q = ggml_reshape_3d(ctx, q, idx_dim, n_idx_h, n_tokens);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp.rms_eps),
                 W("blk.%d.indexer.q_norm.weight", il));
    q = apply_rope(ctx, q, inp_pos);

    // Each head's dot product is rectified before the heads are summed, as in
    // DeepSeek's lightning indexer: there is no per-head weight, so nothing can
    // reorder the sum.
    ggml_tensor * score = ggml_mul_mat(ctx, pooled,
            ggml_reshape_2d(ctx, ggml_cont(ctx, q), idx_dim, n_idx_h * n_tokens));
    score = ggml_reshape_3d(ctx, score, n_blocks, n_idx_h, n_tokens);
    score = ggml_relu(ctx, score);
    score = ggml_cont(ctx, ggml_permute(ctx, score, 1, 0, 2, 3));   // [n_idx_h, n_blocks, T]
    score = ggml_sum_rows(ctx, score);
    score = ggml_reshape_2d(ctx, score, n_blocks, n_tokens);

    // Give every cell its block's score rather than expanding block indices,
    // which would need an integer multiply-add ggml has no op for. get_rows
    // gathers rows, so the scores are transposed on the way in and back out.
    ggml_tensor * expanded = ggml_get_rows(ctx,
            ggml_cont(ctx, ggml_transpose(ctx, score)), shared.cell_blk);
    expanded = ggml_cont(ctx, ggml_transpose(ctx,
            ggml_reshape_2d(ctx, expanded, n_tokens, n_kv)));       // [n_kv, T]
    expanded = ggml_add(ctx, expanded, shared.bias);

    dbg("pooled", pooled);
    dbg("q", q);
    dbg("bias", expanded);
    ggml_tensor * top_k = ggml_cont(ctx, ggml_top_k(ctx, expanded, width));  // [width, T]
    dbg("topk", top_k);

    // Unmask exactly the selected cells: start from all -inf and scatter zeros
    // into the chosen rows. The rows are size 1, so set_rows writes one cell each.
    ggml_tensor * um = ggml_fill(ctx,
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_kv, n_tokens), -INFINITY);
    ggml_tensor * zeros = ggml_fill(ctx,
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, width, n_tokens), 0.0f);
    um = ggml_set_rows(ctx, um, zeros,
            ggml_reshape_3d(ctx, top_k, width, n_tokens, 1));
    um = ggml_reshape_2d(ctx, um, n_kv, n_tokens);

    // Re-apply the causal mask on top: when fewer than `width` cells are
    // eligible, top_k still returns `width` indices and some of them name cells
    // the bias had already ruled out.
    ggml_tensor * combined = ggml_add(ctx, um, ggml_cast(ctx, mask, GGML_TYPE_F32));
    dbg("mask", combined);
    return ggml_cast(ctx, combined, mask->type);
}

// Fill the QSA host inputs of a graph, for a batch whose first token sits at
// `kv_pos`. Found by name so every call site can use it without threading the
// QsaShared through; a no-op on a graph that has no QSA layer.
void Runtime::Impl::set_qsa_inputs(ggml_cgraph * gf, int n_tokens, int n_kv, int kv_pos) {
    QsaShared shared;
    shared.cell_blk  = ggml_graph_get_tensor(gf, "inp_qsa_cell_blk");
    if (!shared.cell_blk) return;
    shared.blk_cells = ggml_graph_get_tensor(gf, "inp_qsa_blk_cells");
    shared.blk_pos   = ggml_graph_get_tensor(gf, "inp_qsa_blk_pos");
    shared.bias      = ggml_graph_get_tensor(gf, "inp_qsa_bias");

    const int n_blocks = (int) (shared.blk_pos->ne[0] / 4);
    const int r        = (int) (shared.blk_cells->ne[0] / n_blocks);
    const int n_valid  = kv_pos + n_tokens;

    // block b covers [b*r, (b+1)*r), so it is rotated at the position of its
    // first token. All four mrope sections carry the same value: exact for text.
    std::vector<int32_t> blk_pos((size_t) 4 * n_blocks);
    for (int sec = 0; sec < 4; ++sec)
        for (int b = 0; b < n_blocks; ++b) blk_pos[(size_t) sec * n_blocks + b] = b * r;
    ggml_backend_tensor_set(shared.blk_pos, blk_pos.data(), 0, blk_pos.size() * sizeof(int32_t));

    std::vector<int32_t> blk_cells((size_t) r * n_blocks);
    for (int b = 0; b < n_blocks; ++b)
        for (int m = 0; m < r; ++m) blk_cells[(size_t) b * r + m] = b * r + m;
    ggml_backend_tensor_set(shared.blk_cells, blk_cells.data(), 0,
                            blk_cells.size() * sizeof(int32_t));

    // cell -> block, or 0 for a cell no complete block covers (the gather has to
    // stay in range; the bias below is what actually rules those cells in or out)
    std::vector<int32_t> cell_blk((size_t) n_kv, 0);
    for (int j = 0; j < n_kv && j < n_valid; ++j) {
        const int b = j / r;
        if (b < n_blocks) cell_blk[(size_t) j] = b;
    }
    ggml_backend_tensor_set(shared.cell_blk, cell_blk.data(), 0, cell_blk.size() * sizeof(int32_t));

    std::vector<float> bias((size_t) n_kv * n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        const int q = kv_pos + i;
        // whatever follows the last complete block is always attended to, which
        // is what lands the selection on block boundaries like the reference
        const int tail_start = (q + 1) / r * r;
        float * row = bias.data() + (size_t) i * n_kv;
        for (int j = 0; j < n_kv; ++j) {
            float v = -INFINITY;
            if (j < n_valid && j <= q) {
                // finite, so it can never meet a -inf and produce a nan
                v = (j >= tail_start) ? 1e9f : (j / r < n_blocks ? 0.0f : -INFINITY);
            }
            row[j] = v;
        }
    }
    ggml_backend_tensor_set(shared.bias, bias.data(), 0, bias.size() * sizeof(float));
}

// ---- qwen4exp PLE: n-gram hash embedding (one designated layer) ----
//
// `emb` is the gathered table rows, [n_embd, n_tokens], already dequantised on
// the host: which rows to fetch is a pure function of the token ids, so the
// gather never enters the graph and the 26.8 GiB table is not a tensor here.
// See ngram_table.h for why that matters.
//
// The module is a gated additive branch onto the wide residual:
//   res += sigmoid(<key(emb), query(res)>) * value(emb) + conv(that)
// which is why dropping it (--ngram off) leaves a stack that still runs.
ggml_tensor * Runtime::Impl::build_ple(ggml_context * ctx, ggml_cgraph * gf, int il,
        ggml_tensor * res_hc, ggml_tensor * emb, int n_tokens) {
    const auto & hp  = model.hparams();
    const int n_embd = hp.n_embd;
    const int hc     = hp.hc_count;
    const int hc_dim = hc * n_embd;
    const float eps  = hp.rms_eps;

    // grouped RMSNorm over one residual stream, with a gamma spanning all of
    // them -- the same shape build_hc_mix uses
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_rms_norm(ctx, ggml_reshape_3d(ctx, x, n_embd, hc, n_tokens), eps);
        t = ggml_mul(ctx, ggml_reshape_2d(ctx, t, hc_dim, n_tokens), w);
        return ggml_reshape_3d(ctx, t, n_embd, hc, n_tokens);
    };

    ggml_tensor * key   = ggml_mul_mat(ctx, W("blk.%d.ple_key.weight",   il), emb);
    ggml_tensor * value = ggml_mul_mat(ctx, W("blk.%d.ple_value.weight", il), emb);

    key = grouped_norm(key, W("blk.%d.ple_norm_key.weight", il));
    ggml_tensor * query = grouped_norm(res_hc, W("blk.%d.ple_norm_query.weight", il));

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * sc = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, key, query)),
                                  1.0f / sqrtf((float) n_embd));
    ggml_tensor * mag  = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, sc), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx, ggml_mul(ctx, ggml_sgn(ctx, sc), mag));

    // one n_embd-wide value broadcast across the streams, scaled by the gate
    ggml_tensor * v3 = ggml_repeat_4d(ctx,
            ggml_reshape_3d(ctx, value, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    ggml_tensor * gated = ggml_mul(ctx, v3, gate);

    ggml_tensor * normalized = grouped_norm(gated, W("blk.%d.ple_norm_conv.weight", il));
    normalized = ggml_reshape_2d(ctx, normalized, hc_dim, n_tokens);

    // Depthwise causal conv, dilated by the n-gram size. Written as a sum of
    // shifted per-channel-scaled copies rather than via ggml_conv_1d_dw, which
    // upstream flags as suspect; on a tensor this small it is a handful of ops.
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    const int kern = (int) hp.ple_conv_kernel;
    const int dil  = (int) hp.ple_ngram_size;
    const int hist = (int) hp.ple_conv_state();

    // [hist + n_tokens, hc_dim] with tokens on ne[0], so history concatenates
    ggml_tensor * padded = ggml_concat(ctx, ple_conv_state[il],
            ggml_cont(ctx, ggml_transpose(ctx, normalized)), 0);
    ggml_tensor * tail = ggml_view_2d(ctx, padded, hist, hc_dim, padded->nb[1],
            ggml_row_size(padded->type, n_tokens));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_cont(ctx, tail), ple_conv_state[il]));

    ggml_tensor * conv_out = nullptr;
    ggml_tensor * kernel   = W("blk.%d.ple_conv1d.weight", il);   // [kern, hc_dim]
    for (int k = 0; k < kern; ++k) {
        const int start = hist - (kern - 1 - k) * dil;
        ggml_tensor * shifted = ggml_cont(ctx, ggml_transpose(ctx,
                ggml_view_2d(ctx, padded, n_tokens, hc_dim, padded->nb[1],
                             ggml_row_size(padded->type, start))));

        // column k of the kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx, ggml_view_2d(ctx, kernel, 1, hc_dim,
                kernel->nb[1], (size_t) k * kernel->nb[0]));
        wk = ggml_reshape_1d(ctx, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) wk = ggml_cast(ctx, wk, GGML_TYPE_F32);

        ggml_tensor * term = ggml_mul(ctx, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx, conv_out, term) : term;
    }
    conv_out = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_silu(ctx, conv_out)),
                               n_embd, hc, n_tokens);

    return ggml_add(ctx, res_hc, ggml_add(ctx, gated, conv_out));
}

// ---- MoE FFN (qwen3moe / qwen35moe): softmax gating, top-k, normalized weights ----
ggml_tensor * Runtime::Impl::build_moe(ggml_context * ctx, ggml_cgraph * gf, int il,
        ggml_tensor * x, int n_tokens) {
    const auto & hp = model.hparams();
    const int n_embd  = hp.n_embd;
    const int n_exp   = hp.n_expert;
    const int n_used  = hp.n_expert_used;

    ggml_tensor * logits = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp.weight", il), x); // [n_exp, n_tokens]
    if (cache_fast_build && resident_decode) {
        // record the unmasked router preference (top-k of the raw logits) so
        // the host can refill wanted-but-absent experts in the background
        ggml_tensor * want = ggml_argsort_top_k(ctx, logits, n_used);      // [n_used, 1]
        ggml_tensor * want_col = ggml_view_2d(ctx, want_all, n_used, 1,
                                              want_all->nb[1], (size_t) il * want_all->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, want, want_col));
        // resident-only routing: bias non-resident experts to -inf before the
        // softmax/top-k so the fused graph can never select a cache miss
        ggml_tensor * rm = ggml_view_2d(ctx, resmask_all, n_exp, 1,
                                        resmask_all->nb[1], (size_t) il * resmask_all->nb[1]);
        logits = ggml_add(ctx, logits, rm);
    }
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
        ggml_tensor * up   = ggml_mul_mat_id(ctx, ec_of(il)->up(il),   x3,  slot_u);
        ggml_tensor * gate = ggml_mul_mat_id(ctx, ec_of(il)->gate(il), x3,  slot_g);
        ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, ec_of(il)->down(il), act, slot_d);
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

    // qwen4exp: the residual is hc identical copies of the embedding. There is
    // no attn_norm / ffn_norm / output_norm anywhere in the file -- the hc
    // mixers carry every normalization the stack has.
    const bool hc_on = hp.has_hc();
    const int  n_hc  = hc_on ? (int) hp.hc_count : 1;
    // The PLE rows are gathered on the host and arrive as one plain input; the
    // graph never touches the table. Sized n_embd because ple_head_dim times
    // ple_n_heads is exactly the model width.
    QsaShared qsa;

    ggml_tensor * inp_ple = nullptr;
    if (use_ple()) {
        inp_ple = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp_ple);
        ggml_set_name(inp_ple, "inp_ple");
    }

    ggml_tensor * res_hc = nullptr;
    if (hc_on) {
        res_hc = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, inpL, n_embd, 1, n_tokens),
                                n_embd, n_hc, n_tokens, 1);
    }

    for (int il = 0; il < (int) hp.n_main(); ++il) {
        ggml_tensor * inpSA = inpL;
        ggml_tensor * inject = nullptr;

        if (hc_on && use_ple() && hp.is_ple(il)) {
            res_hc = build_ple(ctx, gf, il, res_hc, inp_ple, n_tokens);
        }

        if (hc_on) {
            cur = build_hc_mix(ctx, res_hc,
                    W("blk.%d.hc_attn_norm.weight",   il),
                    W("blk.%d.hc_attn_down.weight",   il),
                    W("blk.%d.hc_attn_up.weight",     il),
                    W("blk.%d.hc_attn_inject.weight", il), &inject);
        } else {
            cur = ggml_rms_norm(ctx, inpL, eps);
            cur = ggml_mul(ctx, cur, W("blk.%d.attn_norm.weight", il));
        }

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

            ggml_tensor * m = build_qsa_mask(ctx, gf, il, cur, inp_pos, inp_mask,
                                             n_tokens, n_kv, n_past, qsa);
            ggml_tensor * att = build_attn(ctx, gf, il, Q, K, V, m, n_tokens, n_kv);
            if (gated) att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate));
            cur = ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", il), att);
        }

        ggml_tensor * ffn_res = nullptr;
        ggml_tensor * ffn_in;
        if (hc_on) {
            res_hc = build_hc_combine(ctx, res_hc, cur, inject);
            ffn_in = build_hc_mix(ctx, res_hc,
                    W("blk.%d.hc_ffn_norm.weight",   il),
                    W("blk.%d.hc_ffn_down.weight",   il),
                    W("blk.%d.hc_ffn_up.weight",     il),
                    W("blk.%d.hc_ffn_inject.weight", il), &inject);
        } else {
            cur = ggml_add(ctx, cur, inpSA);

            // FFN with (qwen35) post-attention norm placement
            ffn_res = cur;
            if (gated) {
                ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), W(post_norm_name, il));
            } else {
                ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), W("blk.%d.ffn_norm.weight", il));
            }
        }

        ggml_tensor * ff;
        if (hp.is_moe()) {
            ff = build_moe(ctx, gf, il, ffn_in, n_tokens);
        } else {
            ggml_tensor * gt = ggml_mul_mat(ctx, W("blk.%d.ffn_gate.weight", il), ffn_in);
            ggml_tensor * up = ggml_mul_mat(ctx, W("blk.%d.ffn_up.weight",   il), ffn_in);
            ff = ggml_mul_mat(ctx, W("blk.%d.ffn_down.weight", il), ggml_mul(ctx, ggml_silu(ctx, gt), up));
        }

        if (hc_on) {
            res_hc = build_hc_combine(ctx, res_hc, ff, inject);
        } else {
            inpL = ggml_add(ctx, ff, ffn_res);
        }
    }

    // expose the main stack's last hidden (pre-output-norm) for the MTP module
    // (hc has no single n_embd-wide "last hidden", but no hc model carries an
    //  MTP block either, so the two never meet)
    GGML_ASSERT(!(capture_hidden && hc_on) && "MTP capture is not defined for a hyper-connection residual");
    if (capture_hidden) {
        ggml_set_name(inpL, "main_hidden");
        ggml_set_output(inpL);
        ggml_build_forward_expand(gf, inpL);
    }

    // Narrow to the last token before the output norm when the caller only wants
    // that row: everything downstream then runs once instead of n_tokens times.
    if (hc_on) {
        // Narrow the wide residual before the final mixer, not after: the mixer
        // is two hc_dim matmuls per token and a prefill throws all but the last
        // row away.
        ggml_tensor * r = res_hc;
        if (!logits_all && n_tokens > 1) {
            r = ggml_cont(ctx, ggml_view_3d(ctx, res_hc, n_embd, n_hc, 1,
                    res_hc->nb[1], res_hc->nb[2],
                    (size_t) (n_tokens - 1) * res_hc->nb[2]));
        }
        // the final mixer IS the output norm; the file carries no separate one
        cur = build_hc_mix(ctx, r,
                model.tensor("output_hc_norm.weight"),
                model.tensor("output_hc_down.weight"),
                model.tensor("output_hc_up.weight"), nullptr, nullptr);
    } else {
        ggml_tensor * head_in = inpL;
        if (!logits_all && n_tokens > 1) {
            head_in = ggml_cont(ctx, ggml_view_2d(ctx, inpL, n_embd, 1, inpL->nb[1],
                                                  (size_t) (n_tokens - 1) * inpL->nb[1]));
        }

        cur = ggml_rms_norm(ctx, head_in, eps);
        cur = ggml_mul(ctx, cur, model.tensor("output_norm.weight"));
    }

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

    if (!alloc_graph(mtp_galloc, gf))
        throw std::runtime_error("mtp_draft: gallocr alloc failed");

    ggml_backend_tensor_set(h_in, mtp_hidden.data(), 0, n_embd * sizeof(float));
    ggml_backend_tensor_set(t_in, &token, 0, sizeof(int32_t));
    int32_t pos = mtp_past; ggml_backend_tensor_set(p_in, &pos, 0, sizeof(int32_t));
    std::vector<ggml_fp16_t> mask(n_kv, ggml_fp32_to_fp16(0.0f));   // all past positions visible
    ggml_backend_tensor_set(m_in, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));

    if (compute_graph(gf) != GGML_STATUS_SUCCESS)
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
        if (!alloc_graph(m_galloc, m_gf))
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
        if (!alloc_graph(r_galloc, r_gf))
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
    if (!alloc_graph(ga, gf)) {
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

    const ggml_status st = compute_graph(gf);
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
    if (ec_min_slots() >= T * n_used) {
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
        if (!alloc_graph(v_galloc, v_gf))
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
    // No released model has both an MTP block and QSA, so this graph has never
    // had QSA nodes in it -- but leaving the inputs unfilled is what made the
    // fast decode graph read a garbage cell -> block table and crash, so fill
    // them here too rather than leave the same hole open.
    set_qsa_inputs(v_gf, n_tokens, v_nkv, n_past);

    if (compute_graph(v_gf) != GGML_STATUS_SUCCESS)
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
        if (!alloc_graph(dgalloc, dgf))
            throw std::runtime_error("persistent gallocr alloc failed");
    }
    auto pt_build = pnow();

    ggml_tensor * inp_tokens = ggml_graph_get_tensor(dgf, "inp_tokens");
    ggml_tensor * inp_pos    = ggml_graph_get_tensor(dgf, "inp_pos");
    ggml_tensor * inp_mask   = ggml_graph_get_tensor(dgf, "inp_mask");
    ggml_tensor * inp_kvidx  = ggml_graph_get_tensor(dgf, "inp_kvidx");

    ggml_backend_tensor_set(inp_tokens, &token, 0, sizeof(int32_t));
    fill_ple_input(dgf, &token, 1);
    std::vector<int32_t> posv;
    fill_rope_pos(posv, 1, mrope_next);   // generation token: text, sequential
    ggml_backend_tensor_set(inp_pos, posv.data(), 0, posv.size() * sizeof(int32_t));
    int64_t kvidx = n_past;
    ggml_backend_tensor_set(inp_kvidx, &kvidx, 0, sizeof(int64_t));

    std::vector<ggml_fp16_t> mask(d_nkv);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int j = 0; j < d_nkv; ++j) mask[j] = (j <= n_past) ? z : ninf;
    ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    set_qsa_inputs(dgf, 1, d_nkv, n_past);
    auto pt_input = pnow();

    if (compute_graph(dgf) != GGML_STATUS_SUCCESS)
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
    ggml_tensor * up   = ggml_mul_mat_id(ctx, ec_of(il)->up(il),   x3,  slot_u);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, ec_of(il)->gate(il), x3,  slot_g);
    ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);              // [ff_exp,n_used,1]
    ggml_tensor * experts = ggml_mul_mat_id(ctx, ec_of(il)->down(il), act, slot_d); // [n_embd,n_used,1]

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

    // qwen4exp: the residual carried between segments is hc streams wide, and
    // the per-layer scatter weights ("inject") produced by seg A's second mixer
    // have to reach seg B, so they join the double-buffered carries.
    const bool hc_on = hp.has_hc();
    const int  n_hc  = hc_on ? (int) hp.hc_count : 1;
    if (hc_on && n_ovr > 0) {
        throw std::runtime_error("image embeddings are not wired into the "
                                 "hyper-connection residual yet");
    }

    // temp carry tensors sized for this batch (bridge seg A->B and layer->layer).
    // ffn_in/resid/weights are double-buffered (parity by layer) so the fused
    // segB(L)+segA(L+1) graph has no write-after-read hazard (mirrors decode_cached).
    ggml_init_params tp{};
    tp.mem_size = ggml_tensor_overhead() * 16 + 256;
    tp.no_alloc = true;
    ggml_context * tctx = ggml_init(tp);
    ggml_tensor * h_b       = hc_on
            ? ggml_new_tensor_3d(tctx, GGML_TYPE_F32, n_embd, n_hc, T)
            : ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * inject_b  = hc_on ? ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_hc, T) : nullptr;
    ggml_tensor * inject_b2 = hc_on ? ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_hc, T) : nullptr;
    // the whole chunk's gathered n-gram rows, filled once on the host below
    ggml_tensor * ple_b     = use_ple() ? ggml_new_tensor_2d(tctx, GGML_TYPE_F32, n_embd, T) : nullptr;
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
        if (!alloc_graph(cache_galloc, gf))
            throw std::runtime_error("decode_cached_batch: gallocr alloc failed");
    };

    // ---- seg 0: token embeddings -> h_b ----
    {
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_set_input(inp); ggml_set_name(inp, "inp_tok");
        ggml_tensor * emb = ggml_get_rows(ctx, model.tok_embd_rows(), inp);  // [n_embd, T]
        if (hc_on) {
            // the wide residual starts as hc identical copies of the embedding
            emb = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, emb, n_embd, 1, T),
                                 n_embd, n_hc, T, 1);
        }
        ggml_build_forward_expand(gf, ggml_cpy(ctx, emb, h_b));
        run(ctx, gf);
        ggml_backend_tensor_set(inp, toks, 0, T * sizeof(int32_t));
        if (compute_graph(gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("decode_cached_batch: embed compute failed");
        ggml_free(ctx);
    }

    // The n-gram rows for the whole chunk in one host pass: every index is known
    // before any graph runs, so the misses go to disk in file order (see
    // NgramTable::gather) instead of one scattered read per seg A slice.
    if (ple_b) {
        ngram_rows.clear();
        ngram->hash_rows(toks, T, ngram_hist, ngram_rows);
        ngram_embd.resize(ngram_rows.size() * hp.ple_head_dim);
        ngram->gather(ngram_rows.data(), ngram_rows.size(), ngram_embd.data());
        ggml_backend_tensor_set(ple_b, ngram_embd.data(), 0, ggml_nbytes(ple_b));
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
    ggml_tensor * carry_inj[2] = { inject_b,  inject_b2  };

    // Append seg A (attention/GDN + router) for layer `il` over the token slice
    // [tc0, tc0+tlen) of the chunk; writes that slice of the layer's parity carry
    // and exposes `selected` ([n_used, tlen]) for host readback. Attention KV
    // lands at n_past+tc0 and the causal mask covers n_kv_c columns, so a layer's
    // sub-chunks processed in order are equivalent to one full-chunk pass (GDN
    // states likewise chain across sub-chunks).
    // one per seg A call: each builds its own graph, so nothing is shared across them
    auto build_segA = [&](ggml_context * ctx, ggml_cgraph * gf, int il, int tc0, int tlen) -> ggml_tensor * {
        QsaShared qsa;
        auto tslice = [&](ggml_tensor * t) {   // token-dim slice view [ne0, tlen]
            return ggml_view_2d(ctx, t, t->ne[0], tlen, t->nb[1], (size_t) tc0 * t->nb[1]);
        };
        // the wide residual has tokens on ne[2], so it slices one dimension over
        auto hslice = [&](ggml_tensor * t) {
            return hc_on ? ggml_view_3d(ctx, t, n_embd, n_hc, tlen,
                                        t->nb[1], t->nb[2], (size_t) tc0 * t->nb[2])
                         : tslice(t);
        };
        const int n_kv_c = n_past + tc0 + tlen;
        const bool recurrent = hp.is_recurrent(il);
        ggml_tensor * inp_pos = nullptr, * inp_mask = nullptr;
        if (!recurrent) {
            inp_pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rope_dim(tlen));
            ggml_set_input(inp_pos);  ggml_set_name(inp_pos, "inp_pos");
            inp_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_c, tlen);
            ggml_set_input(inp_mask); ggml_set_name(inp_mask, "inp_mask");
        }

        ggml_tensor * h_c = hslice(h_b);
        ggml_tensor * res_hc = nullptr, * inject = nullptr;
        ggml_tensor * cur;
        if (hc_on) {
            res_hc = h_c;
            if (ple_b && hp.is_ple(il)) {
                res_hc = build_ple(ctx, gf, il, res_hc, tslice(ple_b), tlen);
            }
            cur = build_hc_mix(ctx, res_hc,
                    W("blk.%d.hc_attn_norm.weight",   il),
                    W("blk.%d.hc_attn_down.weight",   il),
                    W("blk.%d.hc_attn_up.weight",     il),
                    W("blk.%d.hc_attn_inject.weight", il), &inject);
        } else {
            cur = ggml_rms_norm(ctx, h_c, eps);
            cur = ggml_mul(ctx, cur, W("blk.%d.attn_norm.weight", il));
        }

        if (recurrent) {
            cur = build_gdn(ctx, gf, il, cur, tlen);
        } else {
            ggml_tensor * Q, * K, * V, * gate_t = nullptr;
            if (gated) {
                ggml_tensor * Qf = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur);
                const size_t es = ggml_element_size(Qf);
                Q = ggml_view_3d(ctx, Qf, n_embd_head, n_head, tlen,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head, 0);
                gate_t = ggml_view_3d(ctx, Qf, n_embd_head, n_head, tlen,
                        es * n_embd_head * 2, es * n_embd_head * 2 * n_head, es * n_embd_head);
                gate_t = ggml_cont_2d(ctx, gate_t, n_embd_head * n_head, tlen);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, tlen);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, tlen);
            } else {
                Q = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", il), cur);
                K = ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", il), cur);
                V = ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", il), cur);
                Q = ggml_reshape_3d(ctx, Q, n_embd_head, n_head,    tlen);
                K = ggml_reshape_3d(ctx, K, n_embd_head, n_head_kv, tlen);
                V = ggml_reshape_3d(ctx, V, n_embd_head, n_head_kv, tlen);
            }
            Q = ggml_mul(ctx, ggml_rms_norm(ctx, Q, eps), W("blk.%d.attn_q_norm.weight", il));
            K = ggml_mul(ctx, ggml_rms_norm(ctx, K, eps), W("blk.%d.attn_k_norm.weight", il));
            Q = apply_rope(ctx, Q, inp_pos);
            K = apply_rope(ctx, K, inp_pos);
            ggml_tensor * m = build_qsa_mask(ctx, gf, il, cur, inp_pos, inp_mask,
                                             tlen, n_kv_c, n_past + tc0, qsa);
            ggml_tensor * att = build_attn(ctx, gf, il, Q, K, V, m, tlen, n_kv_c, n_past + tc0);
            if (gated) att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate_t));
            cur = ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", il), att);
        }

        ggml_tensor * attn_resid = nullptr;
        ggml_tensor * ffn_in;
        if (hc_on) {
            res_hc = build_hc_combine(ctx, res_hc, cur, inject);
            ffn_in = build_hc_mix(ctx, res_hc,
                    W("blk.%d.hc_ffn_norm.weight",   il),
                    W("blk.%d.hc_ffn_down.weight",   il),
                    W("blk.%d.hc_ffn_up.weight",     il),
                    W("blk.%d.hc_ffn_inject.weight", il), &inject);
            // seg B resumes from the residual as it stands after attention
            ggml_build_forward_expand(gf, ggml_cpy(ctx, res_hc, hslice(h_b)));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, inject, tslice(carry_inj[il & 1])));
        } else {
            attn_resid = ggml_add(ctx, cur, h_c);          // [n_embd, tlen]
            if (gated) ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.post_attention_norm.weight", il));
            else       ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.ffn_norm.weight", il));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, attn_resid, tslice(carry_res[il & 1])));
        }

        // multi-token router
        ggml_tensor * logits = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp.weight", il), ffn_in);  // [n_exp, tlen]
        ggml_tensor * probs   = ggml_soft_max(ctx, logits);
        // argsort_top_k returns a STRIDED view (nb[1] = n_exp*4); make it contiguous
        // so the [n_used, tlen] host readback (ggml_backend_tensor_get) is not corrupted
        // for columns >= 1 (it ignores strides). Single-token path is T=1 so unaffected.
        ggml_tensor * selected = ggml_cont(ctx, ggml_argsort_top_k(ctx, probs, n_used));  // [n_used, tlen]
        ggml_tensor * probs3  = ggml_reshape_3d(ctx, probs, 1, n_exp, tlen);
        ggml_tensor * weights = ggml_get_rows(ctx, probs3, selected);             // [1, n_used, tlen]
        weights = ggml_reshape_2d(ctx, weights, n_used, tlen);
        ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
        wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
        weights = ggml_div(ctx, weights, wsum);
        if (hp.expert_weights_scale != 0.0f && hp.expert_weights_scale != 1.0f)
            weights = ggml_scale(ctx, weights, hp.expert_weights_scale);
        ggml_set_output(selected);
        ggml_build_forward_expand(gf, selected);

        ggml_build_forward_expand(gf, ggml_cpy(ctx, ffn_in, tslice(carry_ffn[il & 1])));
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
        ggml_tensor * up   = ggml_mul_mat_id(ctx, ec_of(il)->up(il),   x3, tslice(slot_u_b));
        ggml_tensor * gate = ggml_mul_mat_id(ctx, ec_of(il)->gate(il), x3, tslice(slot_g_b));
        ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
        ggml_tensor * experts = ggml_mul_mat_id(ctx, ec_of(il)->down(il), act, tslice(slot_d_b)); // [n_embd, n_used, len]
        experts = ggml_mul(ctx, experts, ggml_reshape_3d(ctx, tslice(carry_wgt[il & 1]), 1, n_used, len));
        ggml_tensor * moe_out = nullptr;
        for (int i = 0; i < n_used; ++i) {
            ggml_tensor * v = ggml_view_2d(ctx, experts, n_embd, len, experts->nb[2], (size_t) i * experts->nb[1]);
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
        if (hc_on) {
            ggml_tensor * res_hc = ggml_view_3d(ctx, h_b, n_embd, n_hc, len,
                    h_b->nb[1], h_b->nb[2], (size_t) t0 * h_b->nb[2]);
            ggml_tensor * h_new = build_hc_combine(ctx, res_hc, moe_out,
                                                   tslice(carry_inj[il & 1]));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, h_new, res_hc));
        } else {
            ggml_tensor * h_new = ggml_add(ctx, moe_out, tslice(carry_res[il & 1]));  // [n_embd, len]
            ggml_build_forward_expand(gf, ggml_cpy(ctx, h_new, tslice(h_b)));
        }
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
        if (ggml_tensor * im = ggml_graph_get_tensor(gf, "inp_mask")) {
            const int n_kv_c = n_past + tc0 + tlen;
            std::vector<ggml_fp16_t> mask((size_t) n_kv_c * tlen);
            const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
            for (int i = 0; i < tlen; ++i) {
                const int abs_i = n_past + tc0 + i;
                for (int j = 0; j < n_kv_c; ++j)
                    mask[(size_t) i * n_kv_c + j] = (j <= abs_i) ? z : ninf;
            }
            ggml_backend_tensor_set(im, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
        }
        set_qsa_inputs(gf, tlen, n_past + tc0 + tlen, n_past + tc0);
    };

    // QWEN_PREFILL_STATS=1 logs, per (chunk,layer), the distinct-expert union and
    // the fetch cost of ensure() — diagnoses "traffic = n_chunks x full sweep".
    // plan_slices() below flips quota enforcement per layer; restore whatever
    // the caller had on the way out so a batch cannot leave the mode changed.
    struct QuotaScope {
        std::vector<ExpertCache *> cs;
        std::vector<char>          prev;
        explicit QuotaScope(std::vector<ExpertCache *> v) : cs(std::move(v)) {
            for (auto * c : cs) prev.push_back((char) c->quotas_active());
        }
        ~QuotaScope() { for (size_t i = 0; i < cs.size(); ++i) cs[i]->set_quotas(prev[i] != 0); }
    } quota_scope(all_ecaches());

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
        ec_of(il)->set_quotas(plan_union <= ec_of(il)->quota_of(il) - 8);

        // Pass 2: cut the chunk so no single ensure() references more distinct
        // experts than the capacity now in force (it would evict its own slots).
        slices.clear();
        const int cap = std::max(n_used, ec_of(il)->capacity(il) - 8);
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
        ec_of(il)->ensure(il, sel.data() + off, n_used * len,
                       sg.data() + off, su.data() + off, sd.data() + off);
        const size_t ob = off * sizeof(int32_t), nb = (size_t) n_used * len * sizeof(int32_t);
        ggml_backend_tensor_set(slot_g_b, sg.data() + off, ob, nb);
        ggml_backend_tensor_set(slot_u_b, su.data() + off, ob, nb);
        ggml_backend_tensor_set(slot_d_b, sd.data() + off, ob, nb);
    };

    const int N = (int) hp.n_main();

    auto log_layer = [&](int il, int nsl, const ExpertCache::Stats & s0) {
        if (!pf_stats) return;
        const auto s1 = ec_stats_sum();
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
    static const int sega_chunk = []{ const char * c = getenv("QWEN_SEGA_CHUNK");
                                      int v = c ? atoi(c) : 256; return v < 1 ? 256 : v; }();

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
            const bool res = ec_of(il)->resident(ExpertCache::GATE, il, e) &&
                             ec_of(il)->resident(ExpertCache::UP,   il, e) &&
                             ec_of(il)->resident(ExpertCache::DOWN, il, e);
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

    // QWEN_NAN_CHECK=1: read the running hidden state back after every layer and
    // report the first one that is not finite. A NaN anywhere in a 48-layer
    // stack reaches the logits as a uniform NaN, which says nothing about where
    // it started; this says which layer.
    static const bool nan_check = getenv("QWEN_NAN_CHECK") != nullptr;
    bool nan_seen = false;
    std::vector<float> nan_buf;
    auto check_nan = [&](int il, const char * where) {
        if (!nan_check) return;
        nan_buf.resize(ggml_nelements(h_b));
        ggml_backend_tensor_get(h_b, nan_buf.data(), 0, ggml_nbytes(h_b));
        float mx = 0.0f;
        size_t bad = (size_t) -1;
        for (size_t i = 0; i < nan_buf.size(); ++i) {
            const float v = nan_buf[i];
            if (!std::isfinite(v)) { if (bad == (size_t) -1) bad = i; continue; }
            mx = std::max(mx, std::fabs(v));
        }
        fprintf(stderr, "nan-check: layer %2d after %s  max|h|=%.4g  recr=%d",
                il, where, mx, (int) hp.is_recurrent(il));
        if (bad != (size_t) -1) {
            fprintf(stderr, "  FIRST NON-FINITE at [%zu] = %g", bad, nan_buf[bad]);
        }
        fprintf(stderr, "\n");
        if (bad != (size_t) -1) nan_seen = true;
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
                if (compute_graph(gf) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("decode_cached_batch: seg A compute failed");
                ggml_backend_tensor_get(selected, sel.data() + (size_t) tc0 * n_used, 0,
                                        (size_t) n_used * tlen * sizeof(int32_t));
                ggml_free(ctx);
            }
            check_nan(il, "seg A");
            prune_layer(il);
            plan_slices(il);
            const auto s0 = ec_stats_sum();
            const int nsl = (int) slices.size();
            // Prefetch the layer's whole union with one ensure() when it fits
            // the pool: the per-slice ensures below then only hit, so the SSD
            // sweep happens once per layer as one dense coalesced read instead
            // of sparse per-slice residual fetches (which re-read the tensor).
            if (nsl > 1 && plan_union <= ec_of(il)->capacity(il) - 8) {
                pre_g.resize(plan_list.size());
                pre_u.resize(plan_list.size());
                pre_d.resize(plan_list.size());
                ec_of(il)->ensure(il, plan_list.data(), (int) plan_list.size(),
                               pre_g.data(), pre_u.data(), pre_d.data());
            }
            for (int k = 0; k < nsl; ++k) {
                const auto [st0, slen] = slices[k];
                ensure_slice(il, st0, slen);
                ggml_context * ctx = new_ctx();
                ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
                build_segB(ctx, gf, il, st0, slen);
                run(ctx, gf);
                if (compute_graph(gf) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("decode_cached_batch: seg B compute failed");
                ggml_free(ctx);
            }
            log_layer(il, nsl, s0);
            check_nan(il, "seg B");
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
        if (compute_graph(gf) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("decode_cached_batch: seg A0 compute failed");
        ggml_backend_tensor_get(selected, sel.data(), 0, (size_t) n_used * T * sizeof(int32_t));
        ggml_free(ctx);
    }
    for (int il = 0; il < N; ++il) {
        plan_slices(il);
        const auto s0 = ec_stats_sum();
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
            if (compute_graph(gf) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("decode_cached_batch: fused segB/segA compute failed");
            if (nsel)
                ggml_backend_tensor_get(nsel, sel.data(), 0, (size_t) n_used * T * sizeof(int32_t));
            ggml_free(ctx);
        }
        log_layer(il, nsl, s0);
        check_nan(il, "seg B");
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
        ggml_tensor * cur;
        if (hc_on) {
            // the final mixer is the output norm; qwen4exp carries no other one
            ggml_tensor * last = ggml_view_3d(ctx, h_b, n_embd, n_hc, 1,
                    h_b->nb[1], h_b->nb[2], (size_t) (T - 1) * h_b->nb[2]);
            cur = build_hc_mix(ctx, last,
                    model.tensor("output_hc_norm.weight"),
                    model.tensor("output_hc_down.weight"),
                    model.tensor("output_hc_up.weight"), nullptr, nullptr);
        } else {
            ggml_tensor * last = ggml_view_2d(ctx, h_b, n_embd, 1, h_b->nb[1],
                                              (size_t) (T - 1) * h_b->nb[1]);
            cur = ggml_rms_norm(ctx, last, eps);
            cur = ggml_mul(ctx, cur, model.tensor("output_norm.weight"));
        }
        ggml_tensor * output_w = model.tensor("output.weight");
        if (!output_w) output_w = model.tensor("token_embd.weight");
        cur = ggml_mul_mat(ctx, output_w, cur);
        ggml_set_name(cur, "logits");
        ggml_build_forward_expand(gf, cur);
        run(ctx, gf);
        if (compute_graph(gf) != GGML_STATUS_SUCCESS)
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
        if (compute_graph(gf) != GGML_STATUS_SUCCESS)
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
    const bool  hc_on     = hp.has_hc();
    const int   n_hc      = hc_on ? (int) hp.hc_count : 1;

    auto new_ctx = [&]() {
        ggml_init_params gp{};
        gp.mem_size = ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false);
        gp.no_alloc = true;
        return ggml_init(gp);
    };
    // Every segment graph here touches exactly one device -- the layer's weights,
    // KV rows, expert pool and carries all live together -- so it is allocated
    // and run directly on that device. Going through the scheduler instead would
    // re-plan the graph and synchronize per segment, which at one token per
    // graph costs far more than the segment computes.
    auto run = [&](ggml_context * ctx, ggml_cgraph * gf, int dev) {
        if (!ggml_gallocr_alloc_graph(cache_gallocs[(size_t) dev], gf))
            throw std::runtime_error("decode_cached: gallocr alloc failed");
    };
    const bool prof_dc = getenv("QWEN_PROF_DC") != nullptr;
    auto wall0 = std::chrono::steady_clock::now();
    auto compute = [&](ggml_cgraph * gf, int dev, const char * msg) {
        auto t = std::chrono::steady_clock::now();
        ggml_backend_t be = gpus.empty() ? backend : gpus[(size_t) dev];
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS)
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
        emb = hp.has_hc()
                ? ggml_repeat_4d(ctx, ggml_reshape_2d(ctx, emb, n_embd, 1), n_embd, hp.hc_count, 1, 1)
                : ggml_reshape_1d(ctx, emb, n_embd);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, emb, p_h));
        run(ctx, gf, 0);
        ggml_backend_tensor_set(inp, &token, 0, sizeof(int32_t));
        compute(gf, 0, "embed compute failed");
        ggml_free(ctx);
    }

    if (p_ple) {
        ngram_rows.clear();
        ngram->hash_rows(&token, 1, ngram_hist, ngram_rows);
        ngram_embd.resize(ngram_rows.size() * hp.ple_head_dim);
        ngram->gather(ngram_rows.data(), ngram_rows.size(), ngram_embd.data());
        ggml_backend_tensor_set(p_ple, ngram_embd.data(), 0, ggml_nbytes(p_ple));
    }

    std::vector<int32_t> sel(n_used), slot_g(n_used), slot_u(n_used), slot_d(n_used);

    // Double-buffered carry tensors (parity by layer) so a fused segB(L)+segA(L+1)
    // graph has no write-after-read hazard on the carry buffers. These read the
    // p_* members at call time, so they follow use_carry() to the right device.
    auto carry_ffn = [&](int il) { return (il & 1) ? p_ffn_in2  : p_ffn_in;  };
    auto carry_res = [&](int il) { return (il & 1) ? p_resid2   : p_resid;   };
    auto carry_wgt = [&](int il) { return (il & 1) ? p_weights2 : p_weights; };
    auto carry_inj = [&](int il) { return (il & 1) ? p_inject2  : p_inject;  };

    // Append seg A (attention/GDN + router) for layer `il` to (ctx,gf). Writes the
    // normed FFN input / residual / router weights into the layer's parity carry,
    // and exposes `selected` (router top-k) for host readback. Returns selected.
    auto build_segA = [&](ggml_context * ctx, ggml_cgraph * gf, int il) -> ggml_tensor * {
        QsaShared qsa;
        const bool recurrent = hp.is_recurrent(il);
        ggml_tensor * inp_pos = nullptr, * inp_mask = nullptr;
        if (!recurrent) {
            inp_pos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, rope_dim(1));
            ggml_set_input(inp_pos);  ggml_set_name(inp_pos, "inp_pos");
            inp_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, 1);
            ggml_set_input(inp_mask); ggml_set_name(inp_mask, "inp_mask");
        }
        ggml_tensor * res_hc = nullptr, * inject = nullptr;
        ggml_tensor * cur;
        if (hc_on) {
            res_hc = ggml_reshape_3d(ctx, p_h, n_embd, n_hc, 1);
            if (p_ple && hp.is_ple(il)) {
                res_hc = build_ple(ctx, gf, il, res_hc, p_ple, 1);
            }
            cur = build_hc_mix(ctx, res_hc,
                    W("blk.%d.hc_attn_norm.weight",   il),
                    W("blk.%d.hc_attn_down.weight",   il),
                    W("blk.%d.hc_attn_up.weight",     il),
                    W("blk.%d.hc_attn_inject.weight", il), &inject);
        } else {
            cur = ggml_rms_norm(ctx, p_h, eps);
            cur = ggml_mul(ctx, cur, W("blk.%d.attn_norm.weight", il));
        }
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
            ggml_tensor * m = build_qsa_mask(ctx, gf, il, cur, inp_pos, inp_mask,
                                             1, n_kv, n_past, qsa);
            ggml_tensor * att = build_attn(ctx, gf, il, Q, K, V, m, 1, n_kv);
            if (gated) att = ggml_mul(ctx, att, ggml_sigmoid(ctx, gate_t));
            cur = ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", il), att);
        }
        ggml_tensor * ffn_in;
        if (hc_on) {
            res_hc = build_hc_combine(ctx, res_hc, cur, inject);
            ffn_in = build_hc_mix(ctx, res_hc,
                    W("blk.%d.hc_ffn_norm.weight",   il),
                    W("blk.%d.hc_ffn_down.weight",   il),
                    W("blk.%d.hc_ffn_up.weight",     il),
                    W("blk.%d.hc_ffn_inject.weight", il), &inject);
            ggml_build_forward_expand(gf, ggml_cpy(ctx,
                    ggml_reshape_2d(ctx, res_hc, n_embd, n_hc), p_h));
            ggml_build_forward_expand(gf, ggml_cpy(ctx,
                    ggml_reshape_1d(ctx, inject, n_hc), carry_inj(il)));
        } else {
            ggml_tensor * attn_resid = ggml_add(ctx, cur, p_h);
            if (gated) ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.post_attention_norm.weight", il));
            else       ffn_in = ggml_mul(ctx, ggml_rms_norm(ctx, attn_resid, eps), W("blk.%d.ffn_norm.weight", il));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, attn_resid, carry_res(il)));
        }
        ggml_tensor * weights = nullptr;
        ggml_tensor * selected = build_router(ctx, gf, il, ffn_in, weights);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ffn_in, carry_ffn(il)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_1d(ctx, weights, n_used), carry_wgt(il)));
        return selected;
    };
    // Append seg B (cached expert matmul + residual) for layer `il`; writes p_h.
    auto build_segB = [&](ggml_context * ctx, ggml_cgraph * gf, int il) {
        ggml_tensor * moe_out = build_moe_cached(ctx, gf, il, carry_ffn(il),
                                                 p_slot_g, p_slot_u, p_slot_d, carry_wgt(il));
        if (hc_on) {
            ggml_tensor * res_hc = ggml_reshape_3d(ctx, p_h, n_embd, n_hc, 1);
            ggml_tensor * h_new = build_hc_combine(ctx, res_hc,
                    ggml_reshape_2d(ctx, moe_out, n_embd, 1),
                    ggml_reshape_2d(ctx, carry_inj(il), n_hc, 1));
            ggml_build_forward_expand(gf, ggml_cpy(ctx,
                    ggml_reshape_2d(ctx, h_new, n_embd, n_hc), p_h));
        } else {
            ggml_tensor * h_new = ggml_add(ctx, moe_out, carry_res(il));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_1d(ctx, h_new, n_embd), p_h));
        }
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
        set_qsa_inputs(gf, 1, n_kv, n_past);
    };
    // Read back layer `il`'s router selection and make those experts resident.
    auto ensure_layer = [&](ggml_tensor * selected, int il) {
        ggml_backend_tensor_get(selected, sel.data(), 0, n_used * sizeof(int32_t));
        ec_of(il)->ensure(il, sel.data(), n_used, slot_g.data(), slot_u.data(), slot_d.data());
        ggml_backend_tensor_set(p_slot_g, slot_g.data(), 0, n_used * sizeof(int32_t));
        ggml_backend_tensor_set(p_slot_u, slot_u.data(), 0, n_used * sizeof(int32_t));
        ggml_backend_tensor_set(p_slot_d, slot_d.data(), 0, n_used * sizeof(int32_t));
    };

    const int N = (int) hp.n_main();

    // seg A(0) on its own, then fuse segB(L)+segA(L+1) per step so each layer
    // boundary is a single GPU submit instead of two (~halves the dispatches).
    // Under a layer split the fusion is broken at the one device boundary,
    // where the two halves belong to different devices: segB finishes on the
    // old device, the hidden state hops, and segA starts on the new one.
    // Run seg A for `il` on its attention device, then move the A->B carries to
    // that layer's pool device and resolve residency there, so the following
    // seg B finds its inputs and slot ids local. Leaves the carry members
    // pointing at the pool device.
    auto do_segA = [&](int il) {
        const int a = dev_of(il);
        use_carry(a);
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * selected = build_segA(ctx, gf, il);
        run(ctx, gf, a);
        set_attn_inputs(gf);
        compute(gf, a, "seg A compute failed");
        const int p = pdev_of(il);
        hop_carry_ab(a, p, il);
        use_carry(p);
        ensure_layer(selected, il);
        ggml_free(ctx);
    };

    do_segA(0);
    for (int il = 0; il < N; ++il) {
        const int p    = pdev_of(il);                          // seg B runs at the pool
        const int anxt = (il + 1 < N) ? dev_of(il + 1) : -1;    // next attention device
        // Keep the segB(L)+segA(L+1) fusion (one submit per layer instead of
        // two) whenever the next layer's attention runs where this seg B does.
        const bool fuse = anxt >= 0 && anxt == p;

        use_carry(p);
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        build_segB(ctx, gf, il);
        ggml_tensor * nsel = fuse ? build_segA(ctx, gf, il + 1) : nullptr;
        run(ctx, gf, p);
        set_attn_inputs(gf);
        compute(gf, p, "fused segB/segA compute failed");
        if (fuse) {
            const int p1 = pdev_of(il + 1);
            hop_carry_ab(p, p1, il + 1);
            use_carry(p1);
            ensure_layer(nsel, il + 1);
        }
        ggml_free(ctx);

        if (!fuse && anxt >= 0) {
            hop_carry(p, anxt);   // only the hidden state crosses
            do_segA(il + 1);
        }
    }
    // the final norm and output head are model-global, so they live on the primary
    hop_carry(pdev_of(N - 1), 0);
    use_carry(0);

    // ---- final norm + output projection ----
    std::vector<float> out;
    {
        ggml_context * ctx = new_ctx();
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);
        ggml_tensor * cur;
        if (hc_on) {
            // the final mixer is the output norm; qwen4exp carries no other one
            cur = build_hc_mix(ctx, ggml_reshape_3d(ctx, p_h, n_embd, n_hc, 1),
                    model.tensor("output_hc_norm.weight"),
                    model.tensor("output_hc_down.weight"),
                    model.tensor("output_hc_up.weight"), nullptr, nullptr);
        } else {
            cur = ggml_rms_norm(ctx, p_h, eps);
            cur = ggml_mul(ctx, cur, model.tensor("output_norm.weight"));
        }
        ggml_tensor * output_w = model.tensor("output.weight");
        if (!output_w) output_w = model.tensor("token_embd.weight");
        cur = ggml_mul_mat(ctx, output_w, cur);
        ggml_set_name(cur, "logits");
        ggml_build_forward_expand(gf, cur);
        run(ctx, gf, 0);
        compute(gf, 0, "output compute failed");
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
        // This graph is allocated ONCE and reused every token -- that is what
        // makes it CUDA-graph friendly. With one GPU a gallocr owns its own
        // buffer and nothing disturbs it. Across devices it needs a scheduler,
        // but it must NOT be the shared one: any other graph going through that
        // sched (the fallback decode_cached after a speculative miss, or a
        // prefill batch) calls ggml_backend_sched_reset and invalidates this
        // allocation, so the next compute runs on reused memory. Re-allocating
        // per token is not an alternative -- it changes the tensor addresses the
        // input uploads below already captured. Hence a dedicated scheduler.
        if (multi_gpu()) {
            if (!f_sched) f_sched = make_sched();
            ggml_backend_sched_reset(f_sched);
            if (!ggml_backend_sched_alloc_graph(f_sched, f_gf))
                throw std::runtime_error("decode_cached_fast: sched alloc failed");
        } else if (!ggml_gallocr_alloc_graph(f_galloc, f_gf)) {
            throw std::runtime_error("decode_cached_fast: gallocr alloc failed");
        }
    }

    // graph inputs
    ggml_tensor * inp_tokens = ggml_graph_get_tensor(f_gf, "inp_tokens");
    ggml_tensor * inp_pos    = ggml_graph_get_tensor(f_gf, "inp_pos");
    ggml_tensor * inp_mask   = ggml_graph_get_tensor(f_gf, "inp_mask");
    ggml_tensor * inp_kvidx  = ggml_graph_get_tensor(f_gf, "inp_kvidx");
    ggml_backend_tensor_set(inp_tokens, &token, 0, sizeof(int32_t));
    fill_ple_input(f_gf, &token, 1);
    std::vector<int32_t> posv;
    fill_rope_pos(posv, 1, mrope_next);   // generation token: text, sequential
    ggml_backend_tensor_set(inp_pos, posv.data(), 0, posv.size() * sizeof(int32_t));
    int64_t kvidx = n_past; ggml_backend_tensor_set(inp_kvidx, &kvidx, 0, sizeof(int64_t));
    std::vector<ggml_fp16_t> mask(f_nkv);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int j = 0; j < f_nkv; ++j) mask[j] = (j <= n_past) ? z : ninf;
    ggml_backend_tensor_set(inp_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    // QSA selection inputs. Not optional: once the cache passes
    // indexer_top_k + ratio - 1 the graph grows the indexer nodes, and they read
    // whatever the buffer held -- a garbage cell -> block table indexes the
    // gather out of bounds and the run dies in ggml_get_rows.
    set_qsa_inputs(f_gf, 1, f_nkv, n_past);

    // refresh the in-graph remap table from current residency
    auto fill_g2s = [&]() {
        for (int il = 0; il < n_layer; ++il)
            for (int r = 0; r < 3; ++r) {
                const int32_t * row = ec_of(il)->slot_of_row((ExpertCache::Role) r, il);
                memcpy(&g2s_host[(size_t) (il * 3 + r) * n_exp], row, n_exp * sizeof(int32_t));
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
            float * row = &resmask_host[(size_t) il * n_exp];
            int n_res = 0;
            for (int e = 0; e < n_exp; ++e) {
                const bool res = ec_of(il)->resident(ExpertCache::GATE, il, e) &&
                                 ec_of(il)->resident(ExpertCache::UP,   il, e) &&
                                 ec_of(il)->resident(ExpertCache::DOWN, il, e);
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
        for (int il = 0; il < n_layer; ++il)
            for (int k = 0; k < n_used; ++k) {
                const int e = sel_host[(size_t) il * n_used + k];
                if (!ec_of(il)->resident(ExpertCache::GATE, il, e) ||
                    !ec_of(il)->resident(ExpertCache::UP,   il, e) ||
                    !ec_of(il)->resident(ExpertCache::DOWN, il, e)) ok = false;
            }
        return ok;
    };

    // Residency only changes through fetches/evictions, so the mask and remap
    // uploads are versioned by the miss+eviction count (and the warmup floor)
    // and skipped while stale-free — steady-state resident decode uploads
    // nothing but token/pos/mask.
    ++fast_warm_tokens;
    const uint64_t stamp = ec_stats_sum().misses + ec_stats_sum().evictions;
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
    // must run on the graph's own scheduler (see the allocation above)
    const ggml_status f_st = multi_gpu() ? ggml_backend_sched_graph_compute(f_sched, f_gf)
                                         : ggml_backend_graph_compute(backend, f_gf);
    if (f_st != GGML_STATUS_SUCCESS)
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
                ec_of(il)->touch(ExpertCache::GATE, il, e);
                ec_of(il)->touch(ExpertCache::UP,   il, e);
                ec_of(il)->touch(ExpertCache::DOWN, il, e);
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
            for (int il = 0; il < n_layer; ++il)
                for (int k = 0; k < n_used; ++k) {
                    const int e = sel_host[(size_t) il * n_used + k];
                    ec_of(il)->touch(ExpertCache::GATE, il, e);
                    ec_of(il)->touch(ExpertCache::UP,   il, e);
                    ec_of(il)->touch(ExpertCache::DOWN, il, e);
                }
            // Instrumentation: how often the layer's frozen palette failed to
            // hold what the *unmasked* router wanted. This is the quality cost
            // of the layer's slot share (masked decode never stalls on a miss).
            static const bool lstats = getenv("QWEN_LAYER_STATS") != nullptr;
            if (lstats) {
                for (int il = 0; il < n_layer; ++il)
                    for (int k = 0; k < n_used; ++k) {
                        const int e = want_host[(size_t) il * n_used + k];
                        const bool res = ec_of(il)->resident(ExpertCache::GATE, il, e) &&
                                         ec_of(il)->resident(ExpertCache::UP,   il, e) &&
                                         ec_of(il)->resident(ExpertCache::DOWN, il, e);
                        ec_of(il)->note_want(il, !res);
                    }
            }
            int budget = refill_budget;
            for (int step = 0; step < n_layer && budget > 0; ++step) {
                const int il = (refill_cursor + step) % n_layer;
                for (int k = 0; k < n_used && budget > 0; ++k) {
                    const int e = want_host[(size_t) il * n_used + k];
                    if (ec_of(il)->resident(ExpertCache::GATE, il, e) &&
                        ec_of(il)->resident(ExpertCache::UP,   il, e) &&
                        ec_of(il)->resident(ExpertCache::DOWN, il, e)) continue;
                    ec_of(il)->ensure_resident(il, e);   // async H2D (RAM tier); joins mask next token
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
            for (auto * c : all_ecaches()) c->dump_layer_stats("prefill");
            for (auto * c : all_ecaches()) c->reset_layer_stats();   // the destructor dump is decode-only
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
            // Each device rebalances its own pools over its own layers.
            for (auto * c : all_ecaches()) c->rebalance(v > 0 ? (size_t) v : (size_t) -1);
            // Forget the prompt's history so the exit dump measures how far the
            // ideal shape drifts once generation takes over.
            if (getenv("QWEN_LAYER_QUOTA_DRIFT")) for (auto * c : all_ecaches()) c->clear_counts();
            if (getenv("QWEN_LAYER_STATS")) {
                for (auto * c : all_ecaches()) c->dump_layer_stats("rebalanced");
                for (auto * c : all_ecaches()) c->reset_layer_stats();
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
    ggml_tensor * inp_mask_t   = ggml_graph_get_tensor(gf, "inp_mask");

    ggml_backend_tensor_set(inp_tokens_t, tokens.data(), 0, n_tokens * sizeof(int32_t));
    fill_ple_input(gf, tokens.data(), n_tokens);

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
    set_qsa_inputs(gf, n_tokens, n_kv, n_past);

    const bool qsa_dbg = getenv("QWEN_QSA_DEBUG") != nullptr;

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
        status = compute_graph(gf);
    }
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("graph compute failed");
    }

    ggml_tensor * logits_t = ggml_graph_get_tensor(gf, "logits");
    const int n_vocab = (int) logits_t->ne[0];
    logits.resize(n_vocab);
    // one row when the head was narrowed, n_tokens rows otherwise
    const int row = logits_t->ne[1] == 1 ? 0 : n_tokens - 1;
    const size_t off = (size_t) row * logits_t->nb[1];
    ggml_backend_tensor_get(logits_t, logits.data(), off, n_vocab * sizeof(float));
    if (qsa_dbg) {
        for (int il = 0; il < (int) model.hparams().n_layer; ++il) {
            for (const char * tag : { "qsa_kraw_", "qsa_pool0_", "qsa_pooled_", "qsa_q_", "qsa_bias_", "qsa_topk_", "qsa_mask_" }) {
                ggml_tensor * t = ggml_graph_get_tensor(gf, (std::string(tag) + std::to_string(il)).c_str());
                if (!t) continue;
                const int64_t w = t->ne[0];
                std::vector<int32_t> row((size_t) w);
                for (int64_t r = std::max<int64_t>(0, t->ne[1] - 3); r < t->ne[1]; ++r) {
                    ggml_backend_tensor_get(t, row.data(), (size_t) r * w * 4, (size_t) w * 4);
                    fprintf(stderr, "%s%d[%lld]:", tag, il, (long long) r);
                    for (int64_t k = 0; k < std::min<int64_t>(22, w); ++k) {
                        if (t->type == GGML_TYPE_I32) fprintf(stderr, " %d", row[(size_t) k]);
                        else fprintf(stderr, " %.4g", *(float *) &row[(size_t) k]);
                    }
                    fprintf(stderr, "\n");
                }
            }
        }
    }

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
        const auto & s = impl_->ec_stats_sum();
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

// Split "a,b,c" into its fields, trimming surrounding blanks. An empty field is
// kept so the caller's per-field parser can reject it.
static std::vector<std::string> split_csv(const std::string & arg) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : arg) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else          { cur += c; }
    }
    out.push_back(cur);
    for (auto & s : out) {
        size_t b = s.find_first_not_of(" \t");
        size_t e = s.find_last_not_of(" \t");
        s = (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
    }
    return out;
}

bool parse_vram_budget_list(const std::string & arg, std::vector<size_t> & out, bool * legacy_mb) {
    if (legacy_mb) *legacy_mb = false;
    std::vector<size_t> v;
    for (const auto & f : split_csv(arg)) {
        std::string lower;
        for (char c : f) lower += (char) tolower((unsigned char) c);
        // "auto" = all the VRAM this device has free.
        if (lower == "auto") { v.push_back(VRAM_BUDGET_AUTO); continue; }
        // An explicit 0 turns the device off; build_gpu_plan drops it.
        {
            char * end = nullptr;
            const double d = strtod(f.c_str(), &end);
            if (end != f.c_str() && d == 0.0) {
                std::string suf;
                for (const char * p = end; *p; ++p)
                    if (!isspace((unsigned char) *p)) suf += (char) tolower((unsigned char) *p);
                if (suf.empty() || suf == "m" || suf == "mb" || suf == "g" || suf == "gb") {
                    v.push_back(0);
                    continue;
                }
                return false;
            }
        }
        bool legacy = false;
        const size_t mb = parse_vram_budget_mb(f, &legacy);
        if (mb == 0) return false;
        if (legacy && legacy_mb) *legacy_mb = true;
        v.push_back(mb);
    }
    out.swap(v);
    return true;
}

bool parse_gpu_list(const std::string & arg, std::vector<int> & out) {
    std::vector<int> v;
    for (const auto & f : split_csv(arg)) {
        if (f.empty()) return false;
        char * end = nullptr;
        const long d = strtol(f.c_str(), &end, 10);
        if (end == f.c_str() || *end || d < 0) return false;
        for (int prev : v) if (prev == (int) d) return false;   // a device twice is a typo
        v.push_back((int) d);
    }
    out.swap(v);
    return true;
}

bool parse_gpu_split(const std::string & arg, std::vector<float> & out) {
    std::string lower;
    for (char c : arg) lower += (char) tolower((unsigned char) c);
    if (lower == "auto") { out.clear(); return true; }   // ratio decided from budgets / free VRAM
    std::vector<float> v;
    double sum = 0.0;
    for (const auto & f : split_csv(arg)) {
        if (f.empty()) return false;
        char * end = nullptr;
        const double d = strtod(f.c_str(), &end);
        if (end == f.c_str() || *end || d < 0.0) return false;
        sum += d;
        v.push_back((float) d);
    }
    if (sum <= 0.0) return false;   // every device an expert pool leaves nothing to compute on
    out.swap(v);
    return true;
}

bool build_gpu_plan(const std::vector<int> & gpu_ids,
                    const std::vector<float> & gpu_split,
                    bool split_given,
                    const std::vector<size_t> & vram_mb,
                    std::vector<GpuPlan> & out)
{
    out.clear();

    // Nothing that names a device: the historical "first device that works".
    // A lone --vram-budget belongs here too -- one budget has always meant one GPU.
    if (gpu_ids.empty() && !split_given && vram_mb.size() <= 1) return true;

    // How many GPUs is whatever the user actually named. Naming two devices, or
    // two budgets, IS the request for two GPUs; --gpu-split only says how to
    // divide them, and defaults to the budgets (or free VRAM) when omitted.
    size_t n = std::max(gpu_ids.size(), std::max(gpu_split.size(), vram_mb.size()));
    // "--gpu-split auto" on its own means every GPU present.
    if (n <= 1 && split_given && gpu_split.empty() && gpu_ids.empty() && vram_mb.empty())
        n = gpu_devices().size();
    if (n == 0) return true;   // no GPU at all: caller falls back to the CPU

    // A single budget cannot be shared across GPUs -- each device has its own
    // VRAM, so splitting one number between them would be a guess. Likewise a
    // split must name every device, or the unnamed ones silently get nothing.
    if (!vram_mb.empty() && vram_mb.size() != n) {
        fprintf(stderr, "error: --vram-budget has %zu value(s) but the plan has %zu GPU(s); "
                        "give one budget per GPU (e.g. --vram-budget 13.5,5)\n",
                vram_mb.size(), n);
        return false;
    }
    if (!gpu_split.empty() && gpu_split.size() != n) {
        fprintf(stderr, "error: --gpu-split has %zu value(s) but the plan has %zu GPU(s); "
                        "give one share per GPU (e.g. --gpu-split 0.8,0)\n",
                gpu_split.size(), n);
        return false;
    }
    if (!gpu_ids.empty() && gpu_ids.size() != n) {
        fprintf(stderr, "error: --gpus has %zu value(s) but the plan has %zu GPU(s)\n",
                gpu_ids.size(), n);
        return false;
    }

    out.clear();
    for (size_t i = 0; i < n; ++i) {
        GpuPlan p;
        p.device    = gpu_ids.empty()   ? (int) i : gpu_ids[i];
        p.budget_mb = vram_mb.empty()   ? VRAM_BUDGET_AUTO : vram_mb[i];
        p.split     = gpu_split.empty() ? -1.0f   : gpu_split[i];
        // A budget of 0 turns the device off. Asking --gpu-split to put layers
        // on it anyway is a contradiction, not something to silently resolve.
        if (p.budget_mb == 0) {
            if (p.split > 0.0f) {
                fprintf(stderr, "error: --vram-budget 0 turns GPU%d off, but --gpu-split gives it "
                                "a share of %g; use 0 in both, or leave the device out of --gpus\n",
                        p.device, p.split);
                return false;
            }
            continue;   // dropped: not part of the plan at all
        }
        out.push_back(p);
    }
    if (out.empty()) {
        fprintf(stderr, "error: --vram-budget turns every GPU off; leave at least one non-zero\n");
        return false;
    }
    return true;
}

} // namespace questwend
