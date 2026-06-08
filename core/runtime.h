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
};

class Runtime {
public:
    Runtime(Model & model, const RuntimeConfig & cfg);
    ~Runtime();

    // Run a forward pass over `tokens` placed at the current position, advancing
    // the KV cache. Returns the logits (size n_vocab) for the LAST token.
    const std::vector<float> & decode(const std::vector<int32_t> & tokens);

    void reset();                 // clear KV cache / position
    int  n_past() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qwencpp
