#pragma once

// Thin helpers over ggml's gguf API for reading metadata in a typed,
// convenient way. Self-contained: depends only on vendored ggml.

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace questwend {

// Read a single-valued metadata key. Returns `def` if the key is missing.
std::string  gguf_str (const gguf_context * ctx, const std::string & key, const std::string & def = "");
uint32_t     gguf_u32 (const gguf_context * ctx, const std::string & key, uint32_t def = 0);
int32_t      gguf_i32 (const gguf_context * ctx, const std::string & key, int32_t  def = 0);
float        gguf_f32 (const gguf_context * ctx, const std::string & key, float    def = 0.0f);
bool         gguf_has (const gguf_context * ctx, const std::string & key);

// Read a string array (e.g. tokenizer.ggml.tokens).
std::vector<std::string> gguf_str_array(const gguf_context * ctx, const std::string & key);

// Read a float32 array (e.g. tokenizer.ggml.scores).
std::vector<float> gguf_f32_array(const gguf_context * ctx, const std::string & key);

// Read an int32 array (e.g. tokenizer.ggml.token_type).
std::vector<int32_t> gguf_i32_array(const gguf_context * ctx, const std::string & key);

} // namespace questwend
