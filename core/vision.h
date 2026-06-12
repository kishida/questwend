#pragma once

// Qwen3-VL vision tower (mmproj GGUF): image file -> image token embeddings
// in the LLM's embedding space, computed as a ggml graph on the GPU backend.
//
// Architecture (verified against the Java reference implementation and
// llama.cpp's clip.cpp): 768x768 fixed input, 16x16 patches -> ViT
// (LayerNorm+bias, biased QKV, vision M-RoPE, GELU 2-layer FFN) -> post-LN ->
// 2x2 spatial merge -> 2-layer GELU MLP projector -> n_image_tokens (576)
// embeddings of size projection_dim (= LLM n_embd).

#include <memory>
#include <string>
#include <vector>

typedef struct ggml_backend * ggml_backend_t;

namespace qwencpp {

class VisionEncoder {
public:
    // Load an mmproj GGUF (arch=clip, projector_type=qwen3vl_merger) and
    // upload its weights to `backend`. Throws std::runtime_error on failure.
    static std::unique_ptr<VisionEncoder> load(const std::string & mmproj_path,
                                               ggml_backend_t backend);
    ~VisionEncoder();

    int n_image_tokens() const;   // tokens per image after spatial merge (e.g. 576)
    int n_embd() const;           // output embedding dim (= LLM n_embd)

    // Total backend memory held by the tower (weights + persistent compute
    // buffer). Callers running a maxed-out --vram-budget should subtract this
    // from the budget before sizing the expert cache.
    size_t gpu_bytes() const;

    // Decode (JPEG/PNG/...), resize to the trained resolution, normalize, and
    // run the vision tower. Returns [n_image_tokens * n_embd] floats.
    std::vector<float> encode_image(const std::string & image_path);

    // Same, from an in-memory encoded image (JPEG/PNG/... bytes).
    std::vector<float> encode_bytes(const uint8_t * data, size_t size);

    // Same, from raw RGB8 pixels (h x w x 3, row-major).
    std::vector<float> encode_rgb(const uint8_t * rgb, int w, int h);

private:
    VisionEncoder();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qwencpp
