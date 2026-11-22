#pragma once

// Inkling 2 vision tower (mmproj projector_type "inkling2v"). It shares little
// with the Qwen3-VL tower in vision.cpp: the image keeps its size (padded up to
// 40-pixel cells, shrunk only above 4096 tokens), a CLS token and LayerScale,
// separate Q/K/V, axis-split 2D RoPE -- so the token count varies per image.
//
// The weights (640 MB) stay in RAM and are on the GPU only while an image is
// being encoded: a looped 27B with a 60000 context and its DSpark draft already
// fills a 16GB card, and text requests should not pay for the tower.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

typedef struct ggml_backend * ggml_backend_t;

namespace questwend {

class InklingVision {
public:
    static constexpr const char * PROJECTOR = "inkling2v";

    InklingVision(const std::string & mmproj_path, ggml_backend_t backend);
    ~InklingVision();

    int    n_embd() const;
    // [n_tokens * n_embd] floats; n_tokens follows from the image size.
    std::vector<float> encode_rgb(const uint8_t * rgb, int w, int h);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace questwend
