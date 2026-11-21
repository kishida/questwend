#pragma once

// Inkling 2 (arch "inkling2", incl. the looped Inkling-2-Mini) on its own
// engine. The architecture shares almost nothing with the Qwen-shaped graph in
// runtime.cpp -- attention residuals over depth, relative-position bias with
// per-query sinks, short convolutions on K / V / both block outputs, SiTU
// experts with per-expert scales, and a core of layers that runs several
// times -- so it gets its own graph builder instead of a branch at every step
// of the existing ones. Runtime forwards to it when the model is inkling2.

#include "runtime.h"

#include <memory>
#include <vector>

namespace questwend {

class Model;

class InklingEngine {
public:
    InklingEngine(Model & model, const RuntimeConfig & cfg);
    ~InklingEngine();

    // Forward `tokens` at the current position; logits (n_vocab) of the last one.
    const std::vector<float> & decode(const std::vector<int32_t> & tokens);
    void prefill(const std::vector<int32_t> & tokens);
    void reset();
    int  n_past() const;
    const std::vector<int32_t> & kv_tokens() const;

    bool has_expert_cache() const;
    Runtime::CacheStats cache_stats() const;
    void set_progress_cb(std::function<void(int, int)> cb);

    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};

} // namespace questwend
