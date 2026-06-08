#include "gguf_util.h"

#include "gguf.h"

#include <cstring>

namespace qwencpp {

bool gguf_has(const gguf_context * ctx, const std::string & key) {
    return gguf_find_key(ctx, key.c_str()) >= 0;
}

std::string gguf_str(const gguf_context * ctx, const std::string & key, const std::string & def) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return def;
    }
    return gguf_get_val_str(ctx, id);
}

uint32_t gguf_u32(const gguf_context * ctx, const std::string & key, uint32_t def) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (uint32_t) gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return (uint32_t) gguf_get_val_i64(ctx, id);
        default:               return def;
    }
}

int32_t gguf_i32(const gguf_context * ctx, const std::string & key, int32_t def) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT64:  return (int32_t) gguf_get_val_i64(ctx, id);
        default:               return def;
    }
}

float gguf_f32(const gguf_context * ctx, const std::string & key, float def) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) return def;
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, id);
        case GGUF_TYPE_FLOAT64: return (float) gguf_get_val_f64(ctx, id);
        default:                return def;
    }
}

std::vector<std::string> gguf_str_array(const gguf_context * ctx, const std::string & key) {
    std::vector<std::string> out;
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) return out;
    if (gguf_get_arr_type(ctx, id) != GGUF_TYPE_STRING) return out;
    const size_t n = gguf_get_arr_n(ctx, id);
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(ctx, id, i));
    }
    return out;
}

std::vector<float> gguf_f32_array(const gguf_context * ctx, const std::string & key) {
    std::vector<float> out;
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) return out;
    if (gguf_get_arr_type(ctx, id) != GGUF_TYPE_FLOAT32) return out;
    const size_t n = gguf_get_arr_n(ctx, id);
    out.resize(n);
    std::memcpy(out.data(), gguf_get_arr_data(ctx, id), n * sizeof(float));
    return out;
}

std::vector<int32_t> gguf_i32_array(const gguf_context * ctx, const std::string & key) {
    std::vector<int32_t> out;
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0 || gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) return out;
    if (gguf_get_arr_type(ctx, id) != GGUF_TYPE_INT32) return out;
    const size_t n = gguf_get_arr_n(ctx, id);
    out.resize(n);
    std::memcpy(out.data(), gguf_get_arr_data(ctx, id), n * sizeof(int32_t));
    return out;
}

} // namespace qwencpp
