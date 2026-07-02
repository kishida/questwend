#pragma once

// Inference runtime: backend setup, KV cache, forward graph, generation.
// Phase A first slice: Qwen3 dense (GQA + RoPE + SwiGLU). GDN/MoE added later.
// Phase B: expert weight CPU offload via ggml_backend_sched (--vram-budget).

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace questwend {

class Model;

struct RuntimeConfig {
    int    n_ctx          = 4096;
    int    n_threads      = 0;       // 0 = auto
    bool   use_cuda       = true;    // try a GPU device first, fall back to CPU
    size_t vram_budget_mb = 0;       // >0: offload expert weights to CPU,
                                     //     enables running large MoE models with limited VRAM
    std::string cache_profile;       // file to persist/prefetch the hot-expert profile
    bool   cache_profile_save = true; // false: load/prefetch only, never overwrite the profile
    bool   experts_ssd    = false;   // stream experts from the GGUF on SSD (no RAM copy)
    bool   use_mtp        = false;   // MTP self-speculative decode (keeps nextn block VRAM-resident)
    bool   embd_q8        = false;   // use Q8_0 (not F16) for token embedding fallback (saves VRAM)
};

class Runtime {
public:
    Runtime(Model & model, const RuntimeConfig & cfg);
    ~Runtime();

    // Run a forward pass over `tokens` placed at the current position, advancing
    // the KV cache. Returns the logits (size n_vocab) for the LAST token.
    const std::vector<float> & decode(const std::vector<int32_t> & tokens);

    // Vision: override the token embeddings of a span of the NEXT batched
    // decode() call (e.g. the <|image_pad|> run) with precomputed embeddings.
    // `data` must stay valid until that decode() returns; n_embd floats per
    // token, `count` tokens starting at prompt index `first`. Cleared after use.
    // grid_h/grid_w: the image's post-merge patch grid (e.g. 24x24 -> 576).
    // Used for M-RoPE 2D positions; 0 means "square, derived from count".
    struct EmbdOverride { int first = 0; int count = 0; const float * data = nullptr;
                          int grid_h = 0; int grid_w = 0; };
    void set_embd_overrides(std::vector<EmbdOverride> ovr);

    // MTP (multi-token prediction): draft the token *after* `token`, using the
    // main hidden captured by the most recent decode(); advances the MTP KV.
    // Only valid when has_mtp(). Returns draft logits for the next-next token.
    const std::vector<float> & mtp_draft(int32_t token);
    bool has_mtp() const;

    // Advance the KV cache (and, when mtp_kv, the nextn KV + draft context)
    // over `tokens` without sampling: one chunk of a preemptible prefill.
    // Embedding overrides apply with indices relative to `tokens`. Finish
    // with decode() / generate_mtp() on the remaining tail.
    void prefill(const std::vector<int32_t> & tokens, bool mtp_kv = false);

    // MTP self-speculative greedy decode. Calls on_token for each accepted token
    // (return false to stop, e.g. on EOS). has_mtp() must be true. Continues
    // from the current state when n_past > 0 (prompt = the new tail tokens).
    // When generation stops because on_token returned false, *out_pending (if
    // given) receives the next confirmed token that was NOT decoded yet; pass
    // it as the prompt of a follow-up call to resume seamlessly. Confirmed
    // tokens that were never offered to on_token can additionally sit at the
    // tail of kv_tokens() (a mid-cycle stop); the caller is responsible for
    // delivering those before resuming.
    // ckpt_after_prefill: take a prompt checkpoint (see snapshot_ckpt) once the
    // prompt is fully in KV, before any token is generated.
    void generate_mtp(const std::vector<int32_t> & prompt, int max_new, int n_draft,
                      const std::function<bool(int32_t)> & on_token,
                      int32_t * out_pending = nullptr,
                      bool ckpt_after_prefill = false);

    void reset();                 // clear KV cache / position
    int  n_past() const;

    // Expert-cache stats (cumulative since load; all zero when there is no
    // offload cache). The server snapshots before/after a request for hit rate.
    struct CacheStats { uint64_t hits = 0, misses = 0; double fetch_ms = 0; uint64_t fetch_bytes = 0; };
    CacheStats cache_stats() const;
    bool has_expert_cache() const;

    // Progress callback invoked once per prefill chunk (offload path only;
    // resident prefill is a single fast pass): (tokens_done, tokens_total).
    void set_progress_cb(std::function<void(int, int)> cb);

    // Tokens currently represented in the KV cache / recurrent state, in order.
    // Maintained across decode() / generate_mtp(); cleared by reset(). Enables
    // prompt prefix reuse: if a new prompt extends exactly these tokens, the
    // caller may skip reset() and decode only the tail. When the new prompt
    // diverges mid-way, rewind_to() can resume from a prompt checkpoint at or
    // before the divergence instead of a full reset.
    const std::vector<int32_t> & kv_tokens() const;

    // ---- prompt checkpoints (mid-prefix rewind) ----
    // Attention KV rows survive a rewind in place; the GDN recurrent state
    // cannot be truncated, so snapshot_ckpt() copies it (plus the MTP bridge
    // state) to host RAM tagged with the current n_past (~2 MB per GDN layer;
    // ring of QWEN_PROMPT_CKPTS, default 8, geometric thinning). Call it at
    // prefill chunk boundaries / prompt end. rewind_to(n) restores the newest
    // checkpoint at pos <= n, truncates kv_tokens()/n_past to it and drops
    // later checkpoints; returns the new position, or -1 (caller resets).
    // best_ckpt(n) previews that position without touching state. On a pure
    // attention model rewind_to(n) is exact (no snapshots needed). All
    // checkpoints are dropped on reset() and load_state().
    void snapshot_ckpt();
    int  best_ckpt(int n) const;
    int  rewind_to(int n);

    // ---- prompt-cache slots: save / restore the full inference state ----
    // The stream covers the KV prefix for the current n_past (and the MTP KV
    // for mtp_past), the GDN recurrent states, mtp_hidden and the kv_tokens
    // bookkeeping. Opaque format: feed load_state exactly the bytes produced
    // by save_state on the same model + n_ctx (validated by a header; throws
    // std::runtime_error on mismatch or truncation).
    size_t state_bytes() const;   // size save_state() will produce right now
    void save_state(const std::function<void(const void *, size_t)> & sink) const;
    void load_state(const std::function<void(void *, size_t)> & src);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace questwend
