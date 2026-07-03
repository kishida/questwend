#pragma once

// Token sampling: greedy / temperature / top-k / top-p (nucleus), with
// repetition control (repeat / presence / frequency penalties) applied to
// the logits first — also in greedy mode, so greedy decoding gets loop
// protection without becoming stochastic.

#include <cstdint>
#include <deque>
#include <random>
#include <unordered_map>
#include <vector>

namespace questwend {

struct SamplerConfig {
    float    temperature = 0.8f;   // <= 0 => greedy
    float    top_p       = 0.95f;
    int      top_k       = 40;     // <= 0 => disabled
    uint32_t seed        = 0;      // 0 => random device

    // Repetition control over the last `repeat_last_n` tokens seen via
    // prime() (prompt tail) and accept() (generated tokens).
    float repeat_penalty    = 1.0f; // llama.cpp-style: logit > 0 => /p, < 0 => *p (1 = off)
    float presence_penalty  = 0.0f; // OpenAI-style: flat subtraction if present (0 = off)
    float frequency_penalty = 0.0f; // OpenAI-style: subtraction * occurrence count (0 = off)
    int   repeat_last_n     = 64;   // penalty window (<= 0 disables all three)
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig & cfg);

    // Sample a token id from raw logits (modifies internal scratch).
    int32_t sample(const std::vector<float> & logits);

    // Repetition-penalty bookkeeping: seed the window with the prompt tail /
    // record a generated token. No-ops when all penalties are off.
    void prime(const std::vector<int32_t> & prompt);
    void accept(int32_t id);

private:
    bool penalized() const;

    SamplerConfig cfg_;
    std::mt19937  rng_;
    std::vector<std::pair<float, int32_t>> work_;  // (logit, id)
    std::vector<float> scratch_;                   // penalty-adjusted logits
    std::deque<int32_t> recent_;                   // penalty window, newest at back
    std::unordered_map<int32_t, int> counts_;      // id -> occurrences in recent_
};

} // namespace questwend
