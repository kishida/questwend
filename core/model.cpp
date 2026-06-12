#include "model.h"
#include "gguf_util.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace qwencpp {

const char * arch_name(Arch a) {
    switch (a) {
        case Arch::QWEN3:     return "qwen3";
        case Arch::QWEN3MOE:  return "qwen3moe";
        case Arch::QWEN35:    return "qwen35";
        case Arch::QWEN35MOE: return "qwen35moe";
        case Arch::QWEN3NEXT: return "qwen3next";
        default:              return "unknown";
    }
}

static Arch arch_from_string(const std::string & s) {
    if (s == "qwen3")     return Arch::QWEN3;
    if (s == "qwen3moe")  return Arch::QWEN3MOE;
    if (s == "qwen35")    return Arch::QWEN35;
    if (s == "qwen35moe") return Arch::QWEN35MOE;
    if (s == "qwen3next") return Arch::QWEN3NEXT;
    return Arch::UNKNOWN;
}

// Returns true for routed MoE expert weight tensors (gate/up/down _exps).
// Shared expert (_shexp) tensors are kept on GPU — they run once per token.
/*static*/ bool Model::is_expert_tensor(const std::string & name) {
    return name.find("ffn_gate_exps") != std::string::npos ||
           name.find("ffn_up_exps")   != std::string::npos ||
           name.find("ffn_down_exps") != std::string::npos;
}

bool Model::has_expert_tensors() const {
    for (auto & kv : tensors_) {
        if (is_expert_tensor(kv.first)) return true;
    }
    return false;
}

bool Model::is_offloaded_expert(const std::string & name) const {
    if (!is_expert_tensor(name)) return false;
    // The trailing MTP (nextn) block (block index >= n_main) is kept VRAM-resident
    // only when MTP is in use; otherwise it is offloaded like the main stack.
    int blk = -1;
    if (keep_nextn_resident_ &&
        std::sscanf(name.c_str(), "blk.%d.", &blk) == 1 && blk >= (int) hp_.n_main())
        return false;
    return true;
}

Model::~Model() {
    if (embd_buf_)    ggml_backend_buffer_free(embd_buf_);
    if (embd_ctx_)    ggml_free(embd_ctx_);
    if (weights_buf_) ggml_backend_buffer_free(weights_buf_);
    if (gguf_) gguf_free(gguf_);
    if (meta_) ggml_free(meta_);
}

// Whether `backend` can run ggml_get_rows directly on a tensor of te's type.
// CUDA lacks K-quant/IQ get_rows kernels (needs the F16/Q8_0 fallback copy);
// Metal and CPU support them natively, so the copy would just waste memory.
static bool backend_supports_get_rows(ggml_backend_t backend, const ggml_tensor * te) {
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * 4 + 256;
    p.no_alloc = true;
    ggml_context * c = ggml_init(p);
    ggml_tensor * src = ggml_new_tensor_2d(c, te->type, te->ne[0], te->ne[1]);
    ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
    ggml_tensor * op  = ggml_get_rows(c, src, ids);
    const bool ok = ggml_backend_supports_op(backend, op);
    ggml_free(c);
    return ok;
}

// Quantized token embedding needs a get_rows-friendly copy only when the
// backend has no native kernel for the stored type.
static bool need_embd_fallback(ggml_backend_t backend, const ggml_tensor * te) {
    if (!te || te->type == GGML_TYPE_F32 || te->type == GGML_TYPE_F16) return false;
    if (backend_supports_get_rows(backend, te)) {
        fprintf(stderr, "token_embd: %s natively supported by backend get_rows (no fallback copy)\n",
                ggml_type_name(te->type));
        return false;
    }
    return true;
}

ggml_backend_buffer * Model::load_weights(ggml_backend_t backend) {
    // Allocate one backend buffer holding all weight tensors.
    weights_buf_ = ggml_backend_alloc_ctx_tensors(meta_, backend);
    if (!weights_buf_) {
        throw std::runtime_error("failed to allocate weights buffer");
    }

    // token embedding for get_rows: dequantize if the stored type has no native
    // get_rows kernel on this backend (K-quants / IQ types on CUDA).
    ggml_tensor * te = tensor("token_embd.weight");
    tok_embd_rows_ = te;
    const bool need_f32_embd = need_embd_fallback(backend, te);
    if (need_f32_embd) {
        const ggml_type dst_type = embd_q8_ ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        const double dst_mb = embd_q8_
            ? te->ne[0] * te->ne[1] * 1.0625 / 1048576.0
            : te->ne[0] * te->ne[1] * 2.0    / 1048576.0;
        fprintf(stderr, "token_embd: %s is not GPU get_rows compatible, re-quantizing to %s "
                "(%.0f MB -> %.0f MB)\n",
                ggml_type_name(te->type), ggml_type_name(dst_type),
                ggml_nbytes(te) / 1048576.0, dst_mb);
        ggml_init_params ep{};
        ep.mem_size = ggml_tensor_overhead() + 256;
        ep.no_alloc = true;
        embd_ctx_ = ggml_init(ep);
        tok_embd_rows_ = ggml_new_tensor_2d(embd_ctx_, dst_type, te->ne[0], te->ne[1]);
        ggml_set_name(tok_embd_rows_, embd_q8_ ? "token_embd.q8_0" : "token_embd.f16");
        embd_buf_ = ggml_backend_alloc_ctx_tensors(embd_ctx_, backend);
        if (!embd_buf_) throw std::runtime_error("failed to alloc token embedding fallback");
    }

    std::map<std::string, void *> files;
    std::vector<uint8_t> buf;
    std::vector<float>   f32buf;

    for (auto & kv : tensors_) {
        ggml_tensor * t = kv.second;
        const size_t nb = ggml_nbytes(t);
        buf.resize(nb);
        read_tensor_bytes(kv.first, buf.data(), nb, files);
        ggml_backend_tensor_set(t, buf.data(), 0, nb);

        // also populate the F16/Q8_0 token-embedding copy
        if (need_f32_embd && t == te) {
            const int64_t ne = ggml_nelements(te);
            f32buf.resize(ne);
            ggml_get_type_traits(te->type)->to_float(buf.data(), f32buf.data(), ne);
            std::vector<uint8_t> qbuf(ggml_nbytes(tok_embd_rows_));
            if (embd_q8_) {
                ggml_quantize_chunk(GGML_TYPE_Q8_0, f32buf.data(), qbuf.data(), 0, te->ne[1], te->ne[0], nullptr);
            } else {
                ggml_fp32_to_fp16_row(f32buf.data(), (ggml_fp16_t *) qbuf.data(), ne);
            }
            ggml_backend_tensor_set(tok_embd_rows_, qbuf.data(), 0, qbuf.size());
        }
    }
    for (auto & kv : files) if (kv.second) fclose((FILE *) kv.second);
    return weights_buf_;
}

void Model::load_weights_split(
    ggml_backend_t          gpu_backend,
    ggml_backend_buffer_type_t cpu_buft,
    ggml_backend_buffer_t & out_gpu_buf,
    std::vector<ggml_backend_buffer_t> & out_cpu_bufs)
{
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);

    // ---- Prep embedding fallback copy (same as single-backend path) ----
    ggml_tensor * te = tensor("token_embd.weight");
    tok_embd_rows_ = te;
    const bool need_f32_embd = need_embd_fallback(gpu_backend, te);
    if (need_f32_embd) {
        const ggml_type dst_type = embd_q8_ ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        const double dst_mb = embd_q8_
            ? te->ne[0] * te->ne[1] * 1.0625 / 1048576.0
            : te->ne[0] * te->ne[1] * 2.0    / 1048576.0;
        fprintf(stderr, "token_embd: %s is not GPU get_rows compatible, re-quantizing to %s "
                "(%.0f MB -> %.0f MB)\n",
                ggml_type_name(te->type), ggml_type_name(dst_type),
                ggml_nbytes(te) / 1048576.0, dst_mb);
        ggml_init_params ep{};
        ep.mem_size = ggml_tensor_overhead() + 256;
        ep.no_alloc = true;
        embd_ctx_ = ggml_init(ep);
        tok_embd_rows_ = ggml_new_tensor_2d(embd_ctx_, dst_type, te->ne[0], te->ne[1]);
        ggml_set_name(tok_embd_rows_, embd_q8_ ? "token_embd.q8_0" : "token_embd.f16");
        // will be allocated into out_gpu_buf below
    }

    // ---- GPU (non-expert) buffer: size, allocate, assign ----
    size_t gpu_size = 0;
    std::vector<ggml_tensor *> exp_tensors;     // offloaded experts (kept ordered)
    for (auto & kv : tensors_) {
        if (is_offloaded_expert(kv.first)) exp_tensors.push_back(kv.second);
        else gpu_size += ggml_backend_buft_get_alloc_size(gpu_buft, kv.second);
    }
    if (need_f32_embd) gpu_size += ggml_backend_buft_get_alloc_size(gpu_buft, tok_embd_rows_);

    out_gpu_buf = ggml_backend_buft_alloc_buffer(gpu_buft, gpu_size);
    if (!out_gpu_buf)
        throw std::runtime_error("failed to allocate GPU weight buffer (split mode)");
    ggml_backend_buffer_set_usage(out_gpu_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    struct ggml_tallocr gpu_talloc = ggml_tallocr_new(out_gpu_buf);
    for (auto & kv : tensors_) {
        if (is_offloaded_expert(kv.first)) continue;
        if (ggml_tallocr_alloc(&gpu_talloc, kv.second) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("tallocr alloc failed (gpu): " + kv.first);
    }
    if (need_f32_embd) {
        if (ggml_tallocr_alloc(&gpu_talloc, tok_embd_rows_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("tallocr alloc failed (embd.f32)");
        // lives inside out_gpu_buf (caller-owned); embd_buf_ stays null here
    }

    // ---- CPU (expert) buffers: chunked so each stays under the single
    // cudaHostAlloc cap (~15.5 GB on WDDM) and the whole set can be page-locked ----
    const size_t CHUNK = 8ull * 1024 * 1024 * 1024;   // 8 GB per pinned buffer
    size_t i = 0;
    while (i < exp_tensors.size()) {
        size_t group_sz = 0, j = i;
        for (; j < exp_tensors.size(); ++j) {
            const size_t sz = ggml_backend_buft_get_alloc_size(cpu_buft, exp_tensors[j]);
            if (j > i && group_sz + sz > CHUNK) break;   // at least one tensor per buffer
            group_sz += sz;
        }
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(cpu_buft, group_sz > 0 ? group_sz : 1);
        if (!b) throw std::runtime_error("failed to allocate CPU expert buffer (chunk)");
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        struct ggml_tallocr ta = ggml_tallocr_new(b);
        for (size_t k = i; k < j; ++k)
            if (ggml_tallocr_alloc(&ta, exp_tensors[k]) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("tallocr alloc failed (cpu expert chunk)");
        out_cpu_bufs.push_back(b);
        i = j;
    }

    // ---- Load tensor data (shard-aware) ----
    std::map<std::string, void *> files;
    std::vector<uint8_t> buf;
    std::vector<float>   f32buf;

    size_t gpu_bytes = 0, cpu_bytes = 0;

    for (auto & kv : tensors_) {
        ggml_tensor * t = kv.second;
        const size_t nb = ggml_nbytes(t);
        buf.resize(nb);
        read_tensor_bytes(kv.first, buf.data(), nb, files);
        ggml_backend_tensor_set(t, buf.data(), 0, nb);

        if (is_offloaded_expert(kv.first)) cpu_bytes += nb;
        else                               gpu_bytes += nb;

        // also populate the F16/Q8_0 token-embedding copy
        if (need_f32_embd && t == te) {
            const int64_t ne = ggml_nelements(te);
            f32buf.resize(ne);
            ggml_get_type_traits(te->type)->to_float(buf.data(), f32buf.data(), ne);
            std::vector<uint8_t> qbuf(ggml_nbytes(tok_embd_rows_));
            if (embd_q8_) {
                ggml_quantize_chunk(GGML_TYPE_Q8_0, f32buf.data(), qbuf.data(), 0, te->ne[1], te->ne[0], nullptr);
            } else {
                ggml_fp32_to_fp16_row(f32buf.data(), (ggml_fp16_t *) qbuf.data(), ne);
            }
            ggml_backend_tensor_set(tok_embd_rows_, qbuf.data(), 0, qbuf.size());
        }
    }
    for (auto & kv : files) if (kv.second) fclose((FILE *) kv.second);

    fprintf(stderr, "expert cache: GPU %.1f MB | CPU (experts) %.1f MB in %zu pinned chunk(s)\n",
            gpu_bytes / 1024.0 / 1024.0, cpu_bytes / 1024.0 / 1024.0, out_cpu_bufs.size());
}

size_t Model::tensor_file_offset(const std::string & name) const {
    auto it = src_.find(name);
    if (it == src_.end()) throw std::runtime_error("no source for tensor: " + name);
    return it->second.off;
}

const std::string & Model::tensor_file(const std::string & name) const {
    auto it = src_.find(name);
    if (it == src_.end()) throw std::runtime_error("no source for tensor: " + name);
    return it->second.path;
}

void Model::read_tensor_bytes(const std::string & name, void * dst, size_t nb,
                              std::map<std::string, void *> & files) const {
    auto it = src_.find(name);
    if (it == src_.end()) throw std::runtime_error("no source for tensor: " + name);
    void *& fp = files[it->second.path];
    if (!fp) {
        fp = (void *) fopen(it->second.path.c_str(), "rb");
        if (!fp) throw std::runtime_error("failed to open shard file: " + it->second.path);
    }
    FILE * f = (FILE *) fp;
#ifdef _WIN32
    if (_fseeki64(f, (long long) it->second.off, SEEK_SET) != 0 ||
#else
    if (fseeko(f, (off_t) it->second.off, SEEK_SET) != 0 ||
#endif
        fread(dst, 1, nb, f) != nb)
        throw std::runtime_error("failed to read tensor data: " + name);
}

// SSD tier: load only non-expert weights to the GPU; routed experts stay on disk
// (their meta tensors keep ne/nb but have no backing buffer — ExpertCache streams
// them via pread). Mirrors load_weights for the non-expert subset.
void Model::load_weights_ssd(ggml_backend_t gpu_backend,
                             ggml_backend_buffer_t & out_gpu_buf) {
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);

    // token-embedding fallback (same as load_weights)
    ggml_tensor * te = tensor("token_embd.weight");
    tok_embd_rows_ = te;
    const bool need_f32_embd = need_embd_fallback(gpu_backend, te);
    if (need_f32_embd) {
        const ggml_type dst_type = embd_q8_ ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        const double dst_mb = embd_q8_
            ? te->ne[0] * te->ne[1] * 1.0625 / 1048576.0
            : te->ne[0] * te->ne[1] * 2.0    / 1048576.0;
        fprintf(stderr, "token_embd: %s is not GPU get_rows compatible, re-quantizing to %s "
                "(%.0f MB -> %.0f MB)\n",
                ggml_type_name(te->type), ggml_type_name(dst_type),
                ggml_nbytes(te) / 1048576.0, dst_mb);
        ggml_init_params ep{};
        ep.mem_size = ggml_tensor_overhead() + 256;
        ep.no_alloc = true;
        embd_ctx_ = ggml_init(ep);
        tok_embd_rows_ = ggml_new_tensor_2d(embd_ctx_, dst_type, te->ne[0], te->ne[1]);
        ggml_set_name(tok_embd_rows_, embd_q8_ ? "token_embd.q8_0" : "token_embd.f16");
    }

    size_t gpu_size = 0;
    for (auto & kv : tensors_) {
        if (is_offloaded_expert(kv.first)) continue;   // stays on SSD
        gpu_size += ggml_backend_buft_get_alloc_size(gpu_buft, kv.second);
    }
    if (need_f32_embd) gpu_size += ggml_backend_buft_get_alloc_size(gpu_buft, tok_embd_rows_);

    out_gpu_buf = ggml_backend_buft_alloc_buffer(gpu_buft, gpu_size);
    if (!out_gpu_buf) throw std::runtime_error("failed to allocate GPU weight buffer (ssd mode)");
    ggml_backend_buffer_set_usage(out_gpu_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    struct ggml_tallocr talloc = ggml_tallocr_new(out_gpu_buf);
    for (auto & kv : tensors_) {
        if (is_offloaded_expert(kv.first)) continue;
        if (ggml_tallocr_alloc(&talloc, kv.second) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("tallocr alloc failed (ssd gpu): " + kv.first);
    }
    if (need_f32_embd) {
        if (ggml_tallocr_alloc(&talloc, tok_embd_rows_) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("tallocr alloc failed (ssd embd.f32)");
        // lives inside out_gpu_buf (caller-owned); embd_buf_ stays null here
    }

    std::map<std::string, void *> files;
    std::vector<uint8_t> buf;
    std::vector<float>   f32buf;
    size_t gpu_bytes = 0, ssd_bytes = 0;

    for (auto & kv : tensors_) {
        ggml_tensor * t = kv.second;
        const size_t nb = ggml_nbytes(t);
        if (is_offloaded_expert(kv.first)) { ssd_bytes += nb; continue; }   // not loaded

        buf.resize(nb);
        read_tensor_bytes(kv.first, buf.data(), nb, files);
        ggml_backend_tensor_set(t, buf.data(), 0, nb);
        gpu_bytes += nb;

        if (need_f32_embd && t == te) {
            const int64_t ne = ggml_nelements(te);
            f32buf.resize(ne);
            ggml_get_type_traits(te->type)->to_float(buf.data(), f32buf.data(), ne);
            std::vector<uint8_t> qbuf(ggml_nbytes(tok_embd_rows_));
            if (embd_q8_) {
                ggml_quantize_chunk(GGML_TYPE_Q8_0, f32buf.data(), qbuf.data(), 0, te->ne[1], te->ne[0], nullptr);
            } else {
                ggml_fp32_to_fp16_row(f32buf.data(), (ggml_fp16_t *) qbuf.data(), ne);
            }
            ggml_backend_tensor_set(tok_embd_rows_, qbuf.data(), 0, qbuf.size());
        }
    }
    for (auto & kv : files) if (kv.second) fclose((FILE *) kv.second);

    fprintf(stderr, "expert tier: SSD (experts %.1f MB on disk) | GPU %.1f MB resident\n",
            ssd_bytes / 1024.0 / 1024.0, gpu_bytes / 1024.0 / 1024.0);
}

// Expand a sharded model path (`<prefix>-NNNNN-of-MMMMM.gguf`) to the full list
// of shard files. Returns just {path} when not sharded.
static std::vector<std::string> discover_shards(const std::string & path) {
    auto all_digits = [&](size_t pos, int n) {
        for (int i = 0; i < n; ++i)
            if (pos + i >= path.size() || !isdigit((unsigned char) path[pos + i])) return false;
        return true;
    };
    const size_t op = path.rfind("-of-");
    if (op != std::string::npos && op >= 6 && path[op - 6] == '-' &&
        all_digits(op - 5, 5) && all_digits(op + 4, 5) &&
        path.substr(op + 9) == ".gguf") {
        const int total = std::stoi(path.substr(op + 4, 5));
        const std::string prefix = path.substr(0, op - 5);   // ends with '-'
        if (total >= 1) {
            std::vector<std::string> out;
            char buf[32];
            for (int i = 1; i <= total; ++i) {
                snprintf(buf, sizeof(buf), "%05d-of-%05d.gguf", i, total);
                out.push_back(prefix + buf);
            }
            return out;
        }
    }
    return { path };
}

// ggml log filter: the default callback prints every level. ggml's DEBUG/INFO
// output is internal chatter (Metal device/init/free banners, a DEBUG line per
// buffer allocation, ...) that drowns qwencpp's own status lines, so drop both
// and keep WARN/ERROR (collapsing a WARN repeated per-allocation to one line).
// QWEN_GGML_DEBUG=1 restores the full firehose.
static void qwencpp_ggml_log(ggml_log_level level, const char * text, void * /*user*/) {
    static bool debug = getenv("QWEN_GGML_DEBUG") != nullptr;
    static ggml_log_level last_level = GGML_LOG_LEVEL_INFO;
    static std::string last_warn;
    if (debug) { fputs(text, stderr); return; }
    if (level != GGML_LOG_LEVEL_CONT) last_level = level;
    if (last_level == GGML_LOG_LEVEL_DEBUG || last_level == GGML_LOG_LEVEL_INFO) return;
    if (last_level == GGML_LOG_LEVEL_WARN && level != GGML_LOG_LEVEL_CONT) {
        if (text == last_warn) return;   // same warning repeating (e.g. per-alloc)
        last_warn = text;
    }
    fputs(text, stderr);
}

std::unique_ptr<Model> Model::load(const std::string & path) {
    ggml_log_set(qwencpp_ggml_log, nullptr);

    auto m = std::unique_ptr<Model>(new Model());
    m->path_ = path;

    const std::vector<std::string> shards = discover_shards(path);

    // Open each shard's metadata (KV lives in the first shard; tensor data is
    // spread across all shards).
    struct ShardTmp { gguf_context * g; ggml_context * meta; std::string path; size_t data_off; };
    std::vector<ShardTmp> tmp;
    size_t total_tensors = 0;
    for (const auto & sp : shards) {
        ggml_context * sm = nullptr;
        gguf_init_params params{};
        params.no_alloc = true;
        params.ctx      = &sm;
        gguf_context * sg = gguf_init_from_file(sp.c_str(), params);
        if (!sg) {
            for (auto & x : tmp) { gguf_free(x.g); ggml_free(x.meta); }
            throw std::runtime_error("failed to open GGUF: " + sp);
        }
        for (ggml_tensor * t = ggml_get_first_tensor(sm); t; t = ggml_get_next_tensor(sm, t))
            ++total_tensors;
        tmp.push_back({ sg, sm, sp, gguf_get_data_offset(sg) });
    }

    // Build one unified metadata context holding every tensor, with a per-tensor
    // source map (shard file + absolute offset) for data loading / SSD streaming.
    ggml_init_params mp{};
    mp.mem_size = ggml_tensor_overhead() * (total_tensors + 1) + 1024;
    mp.no_alloc = true;
    m->meta_ = ggml_init(mp);

    for (const auto & sh : tmp) {
        for (ggml_tensor * t = ggml_get_first_tensor(sh.meta); t; t = ggml_get_next_tensor(sh.meta, t)) {
            const char * nm = ggml_get_name(t);
            ggml_tensor * ut = ggml_new_tensor(m->meta_, t->type, ggml_n_dims(t), t->ne);
            ggml_set_name(ut, nm);
            m->tensors_[nm] = ut;
            const int64_t tid = gguf_find_tensor(sh.g, nm);
            m->src_[nm] = { sh.path, sh.data_off + gguf_get_tensor_offset(sh.g, tid) };
        }
    }

    // Keep the first shard's gguf for KV metadata; free the rest.
    m->gguf_ = tmp[0].g;
    for (size_t i = 0; i < tmp.size(); ++i) {
        if (i != 0) gguf_free(tmp[i].g);
        ggml_free(tmp[i].meta);
    }
    if (shards.size() > 1)
        fprintf(stderr, "gguf: %zu shards, %zu tensors total\n", shards.size(), total_tensors);

    m->load_hparams();
    m->load_vocab();
    return m;
}

void Model::load_hparams() {
    hp_.general_name = gguf_str(gguf_, "general.name", "");
    hp_.file_type    = gguf_u32(gguf_, "general.file_type", 0);

    const std::string arch = gguf_str(gguf_, "general.architecture", "unknown");
    hp_.arch = arch_from_string(arch);
    if (hp_.arch == Arch::UNKNOWN) {
        throw std::runtime_error("unsupported architecture: " + arch);
    }

    auto k = [&](const char * suffix) { return arch + "." + suffix; };

    hp_.n_layer     = gguf_u32(gguf_, k("block_count"));
    hp_.n_embd      = gguf_u32(gguf_, k("embedding_length"));
    hp_.n_ff        = gguf_u32(gguf_, k("feed_forward_length"));
    hp_.n_head      = gguf_u32(gguf_, k("attention.head_count"));
    hp_.n_head_kv   = gguf_u32(gguf_, k("attention.head_count_kv"), hp_.n_head);
    hp_.n_ctx_train = gguf_u32(gguf_, k("context_length"), 32768);
    hp_.rope_freq_base = gguf_f32(gguf_, k("rope.freq_base"), 1000000.0f);
    hp_.rms_eps        = gguf_f32(gguf_, k("attention.layer_norm_rms_epsilon"), 1e-6f);

    // head dim: Qwen3 stores it explicitly (key_length); fall back to n_embd/n_head
    hp_.n_embd_head = gguf_u32(gguf_, k("attention.key_length"),
                               hp_.n_head ? hp_.n_embd / hp_.n_head : 0);
    hp_.n_rot = gguf_u32(gguf_, k("rope.dimension_count"), hp_.n_embd_head);

    // MoE
    hp_.n_expert      = gguf_u32(gguf_, k("expert_count"), 0);
    hp_.n_expert_used = gguf_u32(gguf_, k("expert_used_count"), 0);
    hp_.n_ff_exp      = gguf_u32(gguf_, k("expert_feed_forward_length"), 0);
    hp_.expert_weights_scale = gguf_f32(gguf_, k("expert_weights_scale"), 1.0f);

    // Gated DeltaNet
    hp_.ssm_d_inner = gguf_u32(gguf_, k("ssm.inner_size"), 0);
    hp_.ssm_n_group = gguf_u32(gguf_, k("ssm.group_count"), 0);
    hp_.ssm_d_state = gguf_u32(gguf_, k("ssm.state_size"), 0);
    hp_.ssm_d_conv  = gguf_u32(gguf_, k("ssm.conv_kernel"), 0);
    hp_.ssm_dt_rank = gguf_u32(gguf_, k("ssm.time_step_rank"), 0);
    hp_.full_attn_interval   = gguf_u32(gguf_, k("full_attention_interval"), 4);
    hp_.nextn_predict_layers = gguf_u32(gguf_, k("nextn_predict_layers"), 0);
    hp_.has_gdn = (hp_.arch == Arch::QWEN35 || hp_.arch == Arch::QWEN35MOE ||
                   hp_.arch == Arch::QWEN3NEXT) && hp_.ssm_d_inner > 0;

    // vocab size: prefer output.weight rows, else token_embd
    if (auto * t = tensor("output.weight")) {
        hp_.n_vocab = (uint32_t) t->ne[1];
    } else if (auto * t = tensor("token_embd.weight")) {
        hp_.n_vocab = (uint32_t) t->ne[1];
    }
}

void Model::load_vocab() {
    vocab_.model         = gguf_str(gguf_, "tokenizer.ggml.model", "gpt2");
    vocab_.tokens        = gguf_str_array(gguf_, "tokenizer.ggml.tokens");
    vocab_.token_types   = gguf_i32_array(gguf_, "tokenizer.ggml.token_type");
    vocab_.merges        = gguf_str_array(gguf_, "tokenizer.ggml.merges");
    vocab_.bos_id        = gguf_i32(gguf_, "tokenizer.ggml.bos_token_id", 151643);
    vocab_.eos_id        = gguf_i32(gguf_, "tokenizer.ggml.eos_token_id", 151645);
    vocab_.pad_id        = gguf_i32(gguf_, "tokenizer.ggml.padding_token_id", -1);
    vocab_.chat_template = gguf_str(gguf_, "tokenizer.chat_template", "");
    if (hp_.n_vocab == 0) hp_.n_vocab = (uint32_t) vocab_.tokens.size();
}

ggml_tensor * Model::tensor(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : it->second;
}

std::string Model::debug_dump() const {
    std::ostringstream os;
    const std::string arch = arch_name(hp_.arch);
    // rope-related metadata
    os << "rope.dimension_count = " << gguf_u32(gguf_, arch + ".rope.dimension_count", 0) << "\n";
    os << "attention.key_length = " << gguf_u32(gguf_, arch + ".attention.key_length", 0) << "\n";
    os << "attention.value_length = " << gguf_u32(gguf_, arch + ".attention.value_length", 0) << "\n";
    os << "ssm.time_step_rank = " << gguf_u32(gguf_, arch + ".ssm.time_step_rank", 0) << "\n";
    auto secs = gguf_i32_array(gguf_, arch + ".rope.dimension_sections");
    os << "rope.dimension_sections = [";
    for (size_t i = 0; i < secs.size(); ++i) os << (i?",":"") << secs[i];
    os << "]\n";
    os << "full_attention_interval = " << gguf_u32(gguf_, arch + ".full_attention_interval", 0) << "\n";
    os << "nextn_predict_layers = " << gguf_u32(gguf_, arch + ".nextn_predict_layers", 0) << "\n";
    // tensors for blk.0 (GDN) and blk.3 (attn)
    for (int il : {0, 3}) {
        os << "--- blk." << il << " ---\n";
        for (auto & kv : tensors_) {
            const std::string pfx = "blk." + std::to_string(il) + ".";
            if (kv.first.rfind(pfx, 0) == 0) {
                ggml_tensor * t = kv.second;
                os << "  " << kv.first << "  [";
                for (int d = 0; d < ggml_n_dims(t); ++d) os << (d?",":"") << t->ne[d];
                os << "] " << ggml_type_name(t->type) << "\n";
            }
        }
    }
    return os.str();
}

std::string Model::summary() const {
    std::ostringstream os;
    os << "arch          = " << arch_name(hp_.arch) << "\n"
       << "n_layer       = " << hp_.n_layer << "\n"
       << "n_embd        = " << hp_.n_embd << "\n"
       << "n_ff          = " << hp_.n_ff << "\n"
       << "n_head        = " << hp_.n_head << " (kv=" << hp_.n_head_kv << ")\n"
       << "n_embd_head   = " << hp_.n_embd_head << "\n"
       << "n_ctx_train   = " << hp_.n_ctx_train << "\n"
       << "n_vocab       = " << hp_.n_vocab << "\n"
       << "rope_freq_base= " << hp_.rope_freq_base << "\n"
       << "rms_eps       = " << hp_.rms_eps << "\n";
    if (hp_.is_moe()) {
        os << "n_expert      = " << hp_.n_expert << " (used=" << hp_.n_expert_used
           << ", ff_exp=" << hp_.n_ff_exp << ")\n";
    }
    if (hp_.has_gdn) {
        os << "GDN: d_inner=" << hp_.ssm_d_inner << " n_group=" << hp_.ssm_n_group
           << " d_state=" << hp_.ssm_d_state << " d_conv=" << hp_.ssm_d_conv << "\n";
    }
    os << "tensors       = " << tensors_.size() << "\n"
       << "vocab tokens  = " << vocab_.tokens.size() << "\n"
       << "bos/eos       = " << vocab_.bos_id << "/" << vocab_.eos_id << "\n";
    return os.str();
}

} // namespace qwencpp
