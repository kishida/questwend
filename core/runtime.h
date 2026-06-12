#pragma once

// Inference runtime: backend setup, KV cache, forward graph, generation.
// Phase A first slice: Qwen3 dense (GQA + RoPE + SwiGLU). GDN/MoE added later.
// Phase B: expert weight CPU offload via ggml_backend_sched (--vram-budget).

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace qwencpp {

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
    struct EmbdOverride { int first = 0; int count = 0; const float * data = nullptr; };
    void set_embd_overrides(std::vector<EmbdOverride> ovr);

    // MTP (multi-token prediction): draft the token *after* `token`, using the
    // main hidden captured by the most recent decode(); advances the MTP KV.
    // Only valid when has_mtp(). Returns draft logits for the next-next token.
    const std::vector<float> & mtp_draft(int32_t token);
    bool has_mtp() const;

    // MTP self-speculative greedy decode. Calls on_token for each accepted token
    // (return false to stop, e.g. on EOS). has_mtp() must be true. Continues
    // from the current state when n_past > 0 (prompt = the new tail tokens).
    // When generation stops because on_token returned false, *out_pending (if
    // given) receives the next confirmed token that was NOT decoded yet; pass
    // it as the prompt of a follow-up call to resume seamlessly. Confirmed
    // tokens that were never offered to on_token can additionally sit at the
    // tail of kv_tokens() (a mid-cycle stop); the caller is responsible for
    // delivering those before resuming.
    void generate_mtp(const std::vector<int32_t> & prompt, int max_new, int n_draft,
                      const std::function<bool(int32_t)> & on_token,
                      int32_t * out_pending = nullptr);

    void reset();                 // clear KV cache / position
    int  n_past() const;

    // Tokens currently represented in the KV cache / recurrent state, in order.
    // Maintained across decode() / generate_mtp(); cleared by reset(). Enables
    // prompt prefix reuse: if a new prompt extends exactly these tokens, the
    // caller may skip reset() and decode only the tail (GDN state cannot be
    // rewound, so reuse requires the full token list to be a prompt prefix).
    const std::vector<int32_t> & kv_tokens() const;

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

} // namespace qwencpp
