#include "sampler.h"

#include <algorithm>
#include <cmath>

namespace questwend {

Sampler::Sampler(const SamplerConfig & cfg) : cfg_(cfg) {
    uint32_t seed = cfg_.seed;
    if (seed == 0) seed = std::random_device{}();
    rng_.seed(seed);
}

bool Sampler::penalized() const {
    return cfg_.repeat_last_n > 0 &&
           (cfg_.repeat_penalty != 1.0f || cfg_.presence_penalty != 0.0f ||
            cfg_.frequency_penalty != 0.0f);
}

void Sampler::accept(int32_t id) {
    if (!penalized()) return;
    recent_.push_back(id);
    ++counts_[id];
    while ((int) recent_.size() > cfg_.repeat_last_n) {
        const int32_t old = recent_.front();
        recent_.pop_front();
        auto it = counts_.find(old);
        if (it != counts_.end() && --it->second == 0) counts_.erase(it);
    }
}

void Sampler::prime(const std::vector<int32_t> & prompt) {
    if (!penalized()) return;
    const size_t n = std::min(prompt.size(), (size_t) cfg_.repeat_last_n);
    for (size_t i = prompt.size() - n; i < prompt.size(); ++i) accept(prompt[i]);
}

int32_t Sampler::sample(const std::vector<float> & logits) {
    const int n = (int) logits.size();

    // repetition control: adjust the logits of recently seen tokens
    const std::vector<float> * src = &logits;
    if (penalized() && !counts_.empty()) {
        scratch_ = logits;
        for (const auto & kv : counts_) {
            const int32_t id = kv.first;
            if (id < 0 || id >= n) continue;
            float & l = scratch_[id];
            if (cfg_.repeat_penalty != 1.0f)
                l = l > 0.0f ? l / cfg_.repeat_penalty : l * cfg_.repeat_penalty;
            l -= cfg_.presence_penalty + cfg_.frequency_penalty * kv.second;
        }
        src = &scratch_;
    }
    const std::vector<float> & lg = *src;

    // greedy
    if (cfg_.temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n; ++i) if (lg[i] > lg[best]) best = i;
        return best;
    }

    work_.clear();
    work_.reserve(n);
    for (int i = 0; i < n; ++i) work_.emplace_back(lg[i], i);

    // top-k: keep highest-k by logit
    int k = cfg_.top_k > 0 ? std::min(cfg_.top_k, n) : n;
    if (k < n) {
        std::partial_sort(work_.begin(), work_.begin() + k, work_.end(),
                          [](auto & a, auto & b){ return a.first > b.first; });
        work_.resize(k);
    } else {
        std::sort(work_.begin(), work_.end(),
                  [](auto & a, auto & b){ return a.first > b.first; });
    }

    // softmax with temperature (subtract max for stability)
    const float maxl = work_.front().first;
    double sum = 0.0;
    for (auto & p : work_) {
        float prob = std::exp((p.first - maxl) / cfg_.temperature);
        p.first = prob;
        sum += prob;
    }

    // top-p (nucleus): keep smallest prefix whose cumulative prob >= top_p
    if (cfg_.top_p < 1.0f) {
        double cum = 0.0;
        size_t cut = work_.size();
        for (size_t i = 0; i < work_.size(); ++i) {
            cum += work_[i].first / sum;
            if (cum >= cfg_.top_p) { cut = i + 1; break; }
        }
        work_.resize(cut);
        sum = 0.0;
        for (auto & p : work_) sum += p.first;
    }

    // sample
    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng_);
    for (auto & p : work_) {
        r -= p.first;
        if (r <= 0.0) return p.second;
    }
    return work_.back().second;
}

} // namespace questwend
