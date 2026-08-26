#include "model.h"
#include "gguf_util.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace questwend {

const char * arch_name(Arch a) {
    switch (a) {
        case Arch::QWEN3:     return "qwen3";
        case Arch::QWEN3MOE:  return "qwen3moe";
        case Arch::QWEN35:    return "qwen35";
        case Arch::QWEN35MOE: return "qwen35moe";
        case Arch::QWEN3NEXT: return "qwen3next";
        case Arch::QWEN4EXP:  return "qwen4exp";
        default:              return "unknown";
    }
}

static Arch arch_from_string(const std::string & s) {
    if (s == "qwen3")     return Arch::QWEN3;
    if (s == "qwen3moe")  return Arch::QWEN3MOE;
    if (s == "qwen35")    return Arch::QWEN35;
    if (s == "qwen35moe") return Arch::QWEN35MOE;
    if (s == "qwen3next") return Arch::QWEN3NEXT;
    if (s == "qwen4exp")  return Arch::QWEN4EXP;
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

// Allocate `ts` into backend buffers, starting a new buffer whenever the next
// tensor would push the current one past `cap`. A backend's single-buffer limit
// is not always generous: CUDA reports none and Metal's runs to tens of GB, but
// Vulkan caps one buffer at 1 GiB by default, and a driver that refuses a larger
// allocation fails the load outright however much VRAM is free. Pinned host
// memory has a cap of its own (see the call site). ggml_backend_alloc_ctx_tensors_from_buft
// does the same walk for a whole context; it cannot be used here because the
// offloaded experts share that context and must stay unallocated.
static void alloc_tensors_chunked(ggml_backend_buffer_type_t buft,
                                  const std::vector<ggml_tensor *> & ts,
                                  size_t cap, const char * what,
                                  std::vector<ggml_backend_buffer_t> & out_bufs) {
    const size_t alignment = ggml_backend_buft_get_alignment(buft);
    // Lets the split path be exercised on a backend that reports no limit
    // (CUDA); a Vulkan driver already imposes one of its own.
    if (const char * e = getenv("QWEN_MAX_BUFFER_MB")) {
        const size_t mb = strtoull(e, nullptr, 10);
        if (mb > 0) cap = std::min(cap, mb * 1024 * 1024);
    }
    const size_t first_buf = out_bufs.size();   // this call's share of a shared vector
    size_t i = 0;
    while (i < ts.size()) {
        size_t group_sz = 0, j = i;
        for (; j < ts.size(); ++j) {
            const size_t sz = GGML_PAD(ggml_backend_buft_get_alloc_size(buft, ts[j]), alignment);
            if (j > i && group_sz + sz > cap) break;   // at least one tensor per buffer
            group_sz += sz;
        }
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft, group_sz > 0 ? group_sz : 1);
        if (!b) throw std::runtime_error(std::string("failed to allocate buffer for ") + what);
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        out_bufs.push_back(b);
        struct ggml_tallocr ta = ggml_tallocr_new(b);
        for (size_t k = i; k < j; ++k)
            if (ggml_tallocr_alloc(&ta, ts[k]) != GGML_STATUS_SUCCESS)
                throw std::runtime_error(std::string("tallocr alloc failed for ") + what + ": "
                                         + ggml_get_name(ts[k]));
        i = j;
    }
    if (out_bufs.size() - first_buf > 1) {
        size_t total = 0;
        for (size_t k = first_buf; k < out_bufs.size(); ++k)
            total += ggml_backend_buffer_get_size(out_bufs[k]);
        fprintf(stderr, "%s: %zu MB split over %zu buffers (backend caps one at %zu MB)\n",
                what, total / 1048576, out_bufs.size() - first_buf, cap / 1048576);
    }
}

int DevicePlan::dev_of_name(const std::string & name) const {
    // "blk.<N>.rest"; anything else is model-global and belongs to the primary.
    if (name.compare(0, 4, "blk.") != 0) return 0;
    size_t p = 4;
    int il = 0;
    if (p >= name.size() || !isdigit((unsigned char) name[p])) return 0;
    for (; p < name.size() && isdigit((unsigned char) name[p]); ++p)
        il = il * 10 + (name[p] - '0');
    // Shared experts run with the routed experts, on the pool's device.
    const bool shexp = name.find("_shexp") != std::string::npos;
    const std::vector<int> & map = (shexp && !pool_dev.empty()) ? pool_dev : layer_dev;
    if (il < 0 || (size_t) il >= map.size()) return 0;
    const int d = map[(size_t) il];
    return (d >= 0 && (size_t) d < bufts.size()) ? d : 0;
}

void Model::load_weights_multi(ggml_backend_t primary, const DevicePlan & plan,
                               std::vector<ggml_backend_buffer_t> & out_bufs,
                               std::vector<size_t> * out_dev_bytes)
{
    const size_t n_dev = plan.n_dev();
    if (n_dev == 0) throw std::runtime_error("load_weights_multi: empty device plan");

    // token embedding for get_rows: same fallback rule as load_weights, but the
    // copy is allocated into out_bufs on the primary rather than into embd_buf_.
    ggml_tensor * te = tensor("token_embd.weight");
    tok_embd_rows_ = te;
    const bool need_f32_embd = need_embd_fallback(primary, te);
    if (need_f32_embd) {
        const ggml_type dst_type = embd_q8_ ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        fprintf(stderr, "token_embd: %s is not GPU get_rows compatible, re-quantizing to %s\n",
                ggml_type_name(te->type), ggml_type_name(dst_type));
        ggml_init_params ep{};
        ep.mem_size = ggml_tensor_overhead() + 256;
        ep.no_alloc = true;
        embd_ctx_ = ggml_init(ep);
        tok_embd_rows_ = ggml_new_tensor_2d(embd_ctx_, dst_type, te->ne[0], te->ne[1]);
        ggml_set_name(tok_embd_rows_, embd_q8_ ? "token_embd.q8_0" : "token_embd.f16");
    }

    std::vector<std::vector<ggml_tensor *>> per_dev(n_dev);
    for (auto & kv : tensors_) per_dev[(size_t) plan.dev_of_name(kv.first)].push_back(kv.second);
    if (need_f32_embd) per_dev[0].push_back(tok_embd_rows_);

    if (out_dev_bytes) out_dev_bytes->assign(n_dev, 0);
    for (size_t d = 0; d < n_dev; ++d) {
        const size_t before = out_bufs.size();
        char what[64];
        snprintf(what, sizeof(what), "gpu%zu weights", d);
        alloc_tensors_chunked(plan.bufts[d], per_dev[d],
                              ggml_backend_buft_get_max_size(plan.bufts[d]), what, out_bufs);
        if (out_dev_bytes)
            for (size_t k = before; k < out_bufs.size(); ++k)
                (*out_dev_bytes)[d] += ggml_backend_buffer_get_size(out_bufs[k]);
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

    size_t total = 0;
    for (auto b : out_bufs) total += ggml_backend_buffer_get_size(b);
    fprintf(stderr, "weights: %.1f MB resident across %zu GPU(s)", total / 1048576.0, n_dev);
    if (out_dev_bytes)
        for (size_t d = 0; d < n_dev; ++d)
            fprintf(stderr, "%s GPU%zu %.1f MB", d ? "," : ":", d, (*out_dev_bytes)[d] / 1048576.0);
    fprintf(stderr, "\n");
}

void Model::load_weights_split(
    ggml_backend_t          gpu_backend,
    ggml_backend_buffer_type_t cpu_buft,
    std::vector<ggml_backend_buffer_t> & out_gpu_bufs,
    std::vector<ggml_backend_buffer_t> & out_cpu_bufs,
    const DevicePlan *      plan,
    std::vector<size_t> *   out_dev_bytes)
{
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);
    const size_t n_dev = (plan && plan->n_dev() > 1) ? plan->n_dev() : 1;

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
        // will be allocated into out_gpu_bufs below
    }

    // ---- GPU (non-expert) buffers: split by the backend's single-buffer limit,
    // and, under a multi-device plan, by the device that computes the layer ----
    std::vector<std::vector<ggml_tensor *>> gpu_tensors(n_dev);
    std::vector<ggml_tensor *> exp_tensors;     // offloaded experts (kept ordered)
    for (auto & kv : tensors_) {
        if (is_offloaded_expert(kv.first)) exp_tensors.push_back(kv.second);
        else gpu_tensors[n_dev > 1 ? (size_t) plan->dev_of_name(kv.first) : 0].push_back(kv.second);
    }
    // lives in out_gpu_bufs (caller-owned); embd_buf_ stays null here
    if (need_f32_embd) gpu_tensors[0].push_back(tok_embd_rows_);   // primary: no layer of its own

    if (out_dev_bytes) out_dev_bytes->assign(n_dev, 0);
    for (size_t d = 0; d < n_dev; ++d) {
        ggml_backend_buffer_type_t bt = (n_dev > 1) ? plan->bufts[d] : gpu_buft;
        const size_t before = out_gpu_bufs.size();
        char what[64];
        if (n_dev > 1) snprintf(what, sizeof(what), "gpu%zu weights (split mode)", d);
        else           snprintf(what, sizeof(what), "gpu weights (split mode)");
        alloc_tensors_chunked(bt, gpu_tensors[d], ggml_backend_buft_get_max_size(bt),
                              what, out_gpu_bufs);
        if (out_dev_bytes)
            for (size_t k = before; k < out_gpu_bufs.size(); ++k)
                (*out_dev_bytes)[d] += ggml_backend_buffer_get_size(out_gpu_bufs[k]);
    }

    // ---- CPU (expert) buffers: chunked so each stays under the single
    // cudaHostAlloc cap (~15.5 GB on WDDM) and the whole set can be page-locked ----
    const size_t CHUNK = 8ull * 1024 * 1024 * 1024;   // 8 GB per pinned buffer
    alloc_tensors_chunked(cpu_buft, exp_tensors,
                          std::min(CHUNK, ggml_backend_buft_get_max_size(cpu_buft)),
                          "cpu expert weights", out_cpu_bufs);

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
                             std::vector<ggml_backend_buffer_t> & out_gpu_bufs,
                             const DevicePlan * plan,
                             std::vector<size_t> * out_dev_bytes) {
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);
    const size_t n_dev = (plan && plan->n_dev() > 1) ? plan->n_dev() : 1;

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

    std::vector<std::vector<ggml_tensor *>> gpu_tensors(n_dev);
    for (auto & kv : tensors_) {
        if (is_offloaded_expert(kv.first)) continue;   // stays on SSD
        gpu_tensors[n_dev > 1 ? (size_t) plan->dev_of_name(kv.first) : 0].push_back(kv.second);
    }
    // lives in out_gpu_bufs (caller-owned); embd_buf_ stays null here
    if (need_f32_embd) gpu_tensors[0].push_back(tok_embd_rows_);   // primary: no layer of its own

    if (out_dev_bytes) out_dev_bytes->assign(n_dev, 0);
    for (size_t d = 0; d < n_dev; ++d) {
        ggml_backend_buffer_type_t bt = (n_dev > 1) ? plan->bufts[d] : gpu_buft;
        const size_t before = out_gpu_bufs.size();
        char what[64];
        if (n_dev > 1) snprintf(what, sizeof(what), "gpu%zu weights (ssd mode)", d);
        else           snprintf(what, sizeof(what), "gpu weights (ssd mode)");
        alloc_tensors_chunked(bt, gpu_tensors[d], ggml_backend_buft_get_max_size(bt),
                              what, out_gpu_bufs);
        if (out_dev_bytes)
            for (size_t k = before; k < out_gpu_bufs.size(); ++k)
                (*out_dev_bytes)[d] += ggml_backend_buffer_get_size(out_gpu_bufs[k]);
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
// buffer allocation, ...) that drowns questwend's own status lines, so drop both
// and keep WARN/ERROR (collapsing a WARN repeated per-allocation to one line).
// QWEN_GGML_DEBUG=1 restores the full firehose.
static void questwend_ggml_log(ggml_log_level level, const char * text, void * /*user*/) {
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
    ggml_log_set(questwend_ggml_log, nullptr);

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

    // M-RoPE sections (Qwen-VL): [t, h, w, 0]. Present only on VL text models;
    // when their sum*2 == n_rot, decode applies multi-axis RoPE so image tokens
    // get 2D grid positions (text tokens use t=h=w=pos -> identical to 1D RoPE).
    {
        auto secs = gguf_i32_array(gguf_, k("rope.dimension_sections"));
        int sum = 0;
        for (size_t i = 0; i < secs.size() && i < 4; ++i) {
            hp_.rope_sections[i] = secs[i];
            sum += secs[i];
        }
        hp_.use_mrope = secs.size() >= 3 && sum > 0 && (uint32_t) (sum * 2) == hp_.n_rot
                        && getenv("QWEN_NO_MROPE") == nullptr;   // A/B escape hatch
    }

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
                   hp_.arch == Arch::QWEN3NEXT || hp_.arch == Arch::QWEN4EXP) &&
                  hp_.ssm_d_inner > 0;

    if (hp_.arch == Arch::QWEN4EXP) load_qwen4exp_hparams(arch);

    // vocab size: prefer output.weight rows, else token_embd
    if (auto * t = tensor("output.weight")) {
        hp_.n_vocab = (uint32_t) t->ne[1];
    } else if (auto * t = tensor("token_embd.weight")) {
        hp_.n_vocab = (uint32_t) t->ne[1];
    }
}

// qwen4exp-only hyper-parameters: hyper-connections, QSA and the PLE n-gram table.
void Model::load_qwen4exp_hparams(const std::string & arch) {
    auto k = [&](const char * suffix) { return arch + "." + suffix; };

    hp_.hc_count    = gguf_u32(gguf_, k("hyper_connection.count"), 0);
    hp_.hc_low_rank = gguf_u32(gguf_, k("hyper_connection.low_rank"), 0);
    if (hp_.hc_count == 0 || hp_.hc_low_rank == 0) {
        throw std::runtime_error("qwen4exp: missing hyper_connection.count / .low_rank");
    }

    hp_.indexer_n_head   = gguf_u32(gguf_, k("attention.indexer.head_count"), 0);
    hp_.indexer_head_dim = gguf_u32(gguf_, k("attention.indexer.key_length"), 0);
    hp_.indexer_top_k    = gguf_u32(gguf_, k("attention.indexer.top_k"), 0);
    for (int32_t r : gguf_i32_array(gguf_, k("attention.compress_ratios"))) {
        hp_.compress_ratios.push_back((uint32_t) (r < 0 ? 0 : r));
    }

    // PLE. Absent key group -> every field stays 0 and has_ple() is false.
    auto ple_layers = gguf_i32_array(gguf_, k("ple.layers"));
    if (ple_layers.empty()) return;

    hp_.ple_layers.assign(hp_.n_layer, 0);
    for (int32_t il : ple_layers) {
        if (il < 0 || (uint32_t) il >= hp_.n_layer) {
            throw std::runtime_error("qwen4exp: ple.layers out of range");
        }
        hp_.ple_layers[il] = 1;
    }

    hp_.ple_ngram_size      = gguf_u32(gguf_, k("ple.ngram_size"));
    hp_.ple_heads_per_ngram = gguf_u32(gguf_, k("ple.heads_per_ngram"));
    hp_.ple_conv_kernel     = gguf_u32(gguf_, k("ple.conv_kernel"));
    hp_.ple_eos_token_id    = gguf_u32(gguf_, k("ple.eos_token_id"));
    hp_.ple_image_token_id  = gguf_u32(gguf_, k("ple.image_token_id"), 0);
    hp_.ple_head_dim        = gguf_u32(gguf_, k("embedding_length_per_layer_input"));
    hp_.ple_n_heads         = (hp_.ple_ngram_size - 1) * hp_.ple_heads_per_ngram;

    hp_.ple_layer_multipliers = gguf_u64_array(gguf_, k("ple.layer_multipliers"));
    hp_.ple_head_offsets      = gguf_u64_array(gguf_, k("ple.head_offsets"));
    hp_.ple_head_vocab_sizes  = gguf_u64_array(gguf_, k("ple.head_vocab_sizes"));

    if (hp_.ple_ngram_size < 2 || hp_.ple_n_heads == 0 || hp_.ple_head_dim == 0 ||
        hp_.ple_layer_multipliers.size() != hp_.ple_ngram_size ||
        hp_.ple_head_offsets.size()      != hp_.ple_n_heads ||
        hp_.ple_head_vocab_sizes.size()  != hp_.ple_n_heads) {
        throw std::runtime_error("qwen4exp: inconsistent ple.* metadata");
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
    if (hp_.has_hc()) {
        os << "hyper-conn    = " << hp_.hc_count << " streams (low_rank="
           << hp_.hc_low_rank << ", residual width=" << hp_.n_embd_hc() << ")\n";
    }
    if (hp_.has_qsa()) {
        os << "QSA           = top_k " << hp_.indexer_top_k << ", indexer "
           << hp_.indexer_n_head << "x" << hp_.indexer_head_dim
           << ", dense below " << (hp_.indexer_top_k + hp_.compress_ratio(3) - 1)
           << " cached tokens\n";
    }
    if (hp_.has_ple()) {
        uint32_t ple_il = 0;
        for (uint32_t i = 0; i < hp_.ple_layers.size(); ++i) {
            if (hp_.ple_layers[i]) { ple_il = i; break; }
        }
        uint64_t rows = 0;
        for (uint64_t v : hp_.ple_head_vocab_sizes) rows += v;
        os << "PLE n-gram    = layer " << ple_il << ", " << hp_.ple_ngram_size
           << "-gram, " << hp_.ple_n_heads << " heads x " << hp_.ple_head_dim
           << " dim, " << rows << " table rows\n";
    }
    os << "tensors       = " << tensors_.size() << "\n"
       << "vocab tokens  = " << vocab_.tokens.size() << "\n"
       << "bos/eos       = " << vocab_.bos_id << "/" << vocab_.eos_id << "\n";
    return os.str();
}

} // namespace questwend
