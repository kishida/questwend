#pragma once

// Token sampling: greedy / temperature / top-k / top-p (nucleus).

#include <cstdint>
#include <random>
#include <vector>

namespace questwend {

struct SamplerConfig {
    float    temperature = 0.8f;   // <= 0 => greedy
    float    top_p       = 0.95f;
    int      top_k       = 40;     // <= 0 => disabled
    uint32_t seed        = 0;      // 0 => random device
};

class Sampler {
public:
    explicit Sampler(const SamplerConfig & cfg);

    // Sample a token id from raw logits (modifies internal scratch).
    int32_t sample(const std::vector<float> & logits);

private:
    SamplerConfig cfg_;
    std::mt19937  rng_;
    std::vector<std::pair<float, int32_t>> work_;  // (logit, id)
};

} // namespace questwend
