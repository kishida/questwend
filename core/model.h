#pragma once

// Qwen3 / Qwen3.5 / Qwen3-Next model loading on top of vendored ggml.
// Phase 0: load GGUF metadata + tensors, expose hyper-parameters and info.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct gguf_context;
struct ggml_backend_buffer;
struct ggml_backend_buffer_type;
typedef struct ggml_backend *             ggml_backend_t;
typedef struct ggml_backend_buffer *      ggml_backend_buffer_t;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;

namespace qwencpp {

enum class Arch {
    UNKNOWN,
    QWEN3,        // dense transformer, no GDN
    QWEN3MOE,     // MoE transformer, no GDN
    QWEN35,       // dense + Gated DeltaNet hybrid
    QWEN35MOE,    // MoE  + Gated DeltaNet hybrid
    QWEN3NEXT,    // GDN + MoE hybrid
};

const char * arch_name(Arch a);

struct HParams {
    Arch     arch = Arch::UNKNOWN;
    uint32_t n_layer        = 0;
    uint32_t n_embd         = 0;
    uint32_t n_ff           = 0;     // dense FFN hidden size
    uint32_t n_head         = 0;
    uint32_t n_head_kv      = 0;
    uint32_t n_embd_head    = 0;     // head dim (qkv)
    uint32_t n_rot          = 0;     // rotary dims (may be < n_embd_head: partial rope)
    uint32_t n_ctx_train    = 0;
    uint32_t n_vocab        = 0;
    float    rope_freq_base = 1000000.0f;
    float    rms_eps        = 1e-6f;

    // MoE (qwen3moe / qwen35moe / qwen3next)
    uint32_t n_expert       = 0;     // total experts (0 = dense)
    uint32_t n_expert_used  = 0;     // top-k
    uint32_t n_ff_exp       = 0;     // per-expert FFN size
    float    expert_weights_scale = 1.0f;

    // Gated DeltaNet (qwen35 / qwen35moe / qwen3next)
    bool     has_gdn        = false;
    uint32_t ssm_d_inner    = 0;     // key_dim = value_dim
    uint32_t ssm_n_group    = 0;     // H_k (number of GDN key heads/groups)
    uint32_t ssm_d_state    = 0;     // D (state size per group = head dim)
    uint32_t ssm_d_conv     = 0;     // conv kernel (=4)
    uint32_t ssm_dt_rank    = 0;     // H_v (number of GDN value heads)
    uint32_t full_attn_interval = 4; // attention every `interval` layers ((il+1)%interval==0)
    uint32_t nextn_predict_layers = 0;

    bool is_moe() const { return n_expert > 0; }
    // GDN (recurrent) layer iff hybrid, within main stack, and not a full-attn slot.
    bool is_recurrent(uint32_t il) const {
        if (!has_gdn) return false;
        const uint32_t n_main = n_layer - nextn_predict_layers;
        return il < n_main && ((il + 1) % full_attn_interval != 0);
    }
};

struct TensorInfo {
    std::string  name;
    ggml_tensor * tensor = nullptr;  // meta tensor (data lives in mmap-backed buffer)
};

struct Vocab {
    std::vector<std::string> tokens;
    std::vector<int32_t>     token_types;
    std::vector<std::string> merges;
    std::string              model;          // "gpt2" for Qwen
    int32_t bos_id = -1, eos_id = -1, pad_id = -1;
    std::string              chat_template;
};

class Model {
public:
    ~Model();

    // Load a GGUF model. Throws std::runtime_error on failure.
    static std::unique_ptr<Model> load(const std::string & path);

    const HParams & hparams() const { return hp_; }
    const Vocab   & vocab()   const { return vocab_; }

    // Tensor lookup by exact GGUF name (e.g. "blk.0.attn_q.weight").
    ggml_tensor * tensor(const std::string & name) const;

    // Token embedding tensor suitable for ggml_get_rows on any backend.
    // (CUDA get_rows does not support K-quant/IQ types, so a dequantized F32
    //  copy is provided for those; otherwise the original tensor is returned.)
    ggml_tensor * tok_embd_rows() const { return tok_embd_rows_; }

    // Allocate a backend buffer for all weights and upload their data from the
    // GGUF file. After this call, tensor() pointers are backed by real data.
    // Returns the owning buffer (freed by the caller / engine).
    ggml_backend_buffer * load_weights(ggml_backend_t backend);

    // Split variant: expert weight tensors (ffn_*_exps) go to cpu_buft,
    // everything else goes to gpu_backend. Both output buffers are caller-owned.
    // Enables running large MoE models when GPU VRAM is limited.
    void load_weights_split(ggml_backend_t gpu_backend,
                            ggml_backend_buffer_type_t cpu_buft,
                            ggml_backend_buffer_t & out_gpu_buf,
                            ggml_backend_buffer_t & out_cpu_buf);

    // SSD-tier variant: routed expert weights are NOT loaded into memory at all
    // (they stay on disk and are streamed on demand by ExpertCache). Everything
    // else (incl. shared experts) goes to gpu_backend. Saves the experts' RAM.
    void load_weights_ssd(ggml_backend_t gpu_backend,
                          ggml_backend_buffer_t & out_gpu_buf);

    // Source file + absolute byte offset of a tensor's data (for pread).
    // For sharded models these vary per tensor (different shard files).
    size_t tensor_file_offset(const std::string & name) const;
    const std::string & tensor_file(const std::string & name) const;
    const std::string & path() const { return path_; }

    // Returns true if this model has any routed expert tensors (MoE layers).
    bool has_expert_tensors() const;

    // Returns true if the tensor name belongs to a routed expert (not shared expert).
    static bool is_expert_tensor(const std::string & name);

    std::string summary() const;
    std::string debug_dump() const;

private:
    Model() = default;

    HParams hp_;
    Vocab   vocab_;

    std::string    path_;
    gguf_context * gguf_   = nullptr;        // KV metadata (first shard)
    ggml_context * meta_   = nullptr;        // unified tensor metadata (+ data after load_weights)
    ggml_backend_buffer * weights_buf_ = nullptr;
    std::map<std::string, ggml_tensor *> tensors_;

    // Per-tensor data source (shard file + absolute offset). One entry per tensor;
    // for a single-file model all entries share path_.
    struct Src { std::string path; size_t off = 0; };
    std::map<std::string, Src> src_;

    // F32 token-embedding fallback for ggml_get_rows (see tok_embd_rows()).
    ggml_context *        embd_ctx_      = nullptr;
    ggml_backend_buffer * embd_buf_      = nullptr;
    ggml_tensor *         tok_embd_rows_ = nullptr;

    void load_hparams();
    void load_vocab();

    // Read nb bytes of tensor `name` from its (possibly sharded) source file into
    // dst, reusing open file handles cached in `files`.
    void read_tensor_bytes(const std::string & name, void * dst, size_t nb,
                           std::map<std::string, void *> & files) const;
};

} // namespace qwencpp
