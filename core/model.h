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

namespace questwend {

// Report a backend allocation failure with the numbers that explain it: what was
// being allocated, how much it asked for, and what the device says it has. ggml's
// allocators only return null, and "failed to allocate buffer" on its own leaves
// exactly the question the reader has unanswered. Always throws.
[[noreturn]] void throw_alloc_failure(ggml_backend_buffer_type_t buft, size_t bytes,
                                      const std::string & what);

enum class Arch {
    UNKNOWN,
    QWEN3,        // dense transformer, no GDN
    QWEN3MOE,     // MoE transformer, no GDN
    QWEN35,       // dense + Gated DeltaNet hybrid
    QWEN35MOE,    // MoE  + Gated DeltaNet hybrid
    QWEN3NEXT,    // GDN + MoE hybrid
    QWEN4EXP,     // Qwen3.8-Flash-Next: GDN + MoE + hyper-connections + QSA + PLE
    LFM2,         // LiquidAI LFM2 / LFM2.5: short-conv + GQA hybrid, dense
    LFM2MOE,      // LiquidAI LFM2-MoE: the same hybrid with sigmoid-routed experts
    INKLING2,     // Thinking Machines Inkling 2: depth-attention residuals, relative bias,
                  // short convs on K/V/outputs, looped core (see core/inkling.cpp).
                  // Both of its GGUF dialects: "inkling2" and "inkling2_loop".
};

// Inkling 2 hyper-parameters. Kept apart from HParams' shared fields because
// almost none of them mean anything to the other architectures, and the
// looped core changes what "a layer" is: the GGUF stores n_phys physical
// blocks, the forward pass makes n_visit layer visits (the core
// [core_start, core_end) runs n_pass times), and every visit has its own KV
// cache and conv state while the weights stay per physical layer.
//
// A looped core comes in two GGUF dialects, read into the same fields:
//   "inkling2" (unsloth)           block_count and the per-layer arrays count
//                                  visits; num_loops, loop.core_start / _end
//   "inkling2_loop" (llama.cpp)    they count physical layers; loop.count,
//                                  loop.start / .end, loop.varying
// The loop's tensors are named and grouped differently as well (per_layer_msg).
struct InklingParams {
    uint32_t n_visit      = 0;
    uint32_t core_start   = 0, core_end = 0;
    uint32_t n_pass       = 1;   // < 2: no loop
    uint32_t n_pass_file  = 1;   // what the GGUF says; n_pass differs under --loops or a request's "loops"
    bool     varying      = false;    // loop.varying: trained for a varying pass count
    uint32_t gate_rank    = 0;   // loop message controllers' rank
    bool     per_layer_msg = false;   // blk.N.loop_msg_query [q, k, v, ffn] and loop_*_ctrl_* (llama.cpp);
                                      // else loop_res_query / loop_kv_res_query and loop_{attn,k,v,mlp}_*
    std::vector<uint8_t> is_swa; // per visit
    uint32_t n_swa        = 0;
    uint32_t n_rel        = 0;   // relative-position bias rank
    uint32_t rel_extent   = 0;   // distances >= this share one (zero) bias row
    float    rel_clip     = 0.0f;
    float    qk_clip      = 0.0f;
    uint32_t log_floor    = 0;
    float    log_alpha    = 0.0f;
    uint32_t block_size   = 0;   // attention-residual block size (layers)
    uint32_t n_conv       = 0;   // short-conv taps
    float    situ_beta    = 0.0f, situ_lin_beta = 0.0f;
    float    logit_cap    = 0.0f;
    uint32_t n_vocab_pad  = 0, n_vocab = 0;   // padded rows in embd/output vs real tokens

    bool     looped() const { return n_pass > 1; }
    uint32_t n_core() const { return core_end - core_start; }
    bool     has_core() const { return core_end > core_start && gate_rank > 0; }   // loops, at any pass count
    // physical layer of visit `iv`
    uint32_t phys(uint32_t iv) const {
        if (!looped() || iv < core_start) return iv;
        if (iv < core_start + n_pass * n_core()) return core_start + (iv - core_start) % n_core();
        return iv - (n_pass - 1) * n_core();
    }
    // loop pass of visit `iv` (0 outside the core)
    uint32_t pass(uint32_t iv) const {
        if (!looped() || iv < core_start || iv >= core_start + n_pass * n_core()) return 0;
        return (iv - core_start) / n_core();
    }
    bool in_core(uint32_t ip) const { return looped() && ip >= core_start && ip < core_end; }
};

const char * arch_name(Arch a);

struct HParams {
    Arch     arch = Arch::UNKNOWN;
    uint32_t n_layer        = 0;
    uint32_t n_embd         = 0;
    uint32_t n_ff           = 0;     // dense FFN hidden size
    uint32_t n_head         = 0;     // max over layers when the count varies
    uint32_t n_head_kv      = 0;     // max over layers when the count varies
    uint32_t n_embd_head    = 0;     // head dim (qkv)
    uint32_t n_rot          = 0;     // rotary dims (may be < n_embd_head: partial rope)
    uint32_t n_ctx_train    = 0;
    uint32_t n_vocab        = 0;
    float    rope_freq_base = 1000000.0f;
    float    rms_eps        = 1e-6f;

    // M-RoPE (Qwen-VL): multi-axis RoPE sections [t, h, w, 0]. use_mrope set
    // when present and consistent with n_rot; otherwise plain 1D NEOX RoPE.
    int      rope_sections[4] = {0, 0, 0, 0};
    bool     use_mrope        = false;

    // Per-layer head counts. head_count / head_count_kv are scalars on every
    // Qwen arch but per-layer arrays on step35 (64 vs 96 heads) and lfm2moe
    // (kv = 0 on its convolution layers). Empty = uniform; always go through
    // head_count(il) / head_count_kv(il).
    std::vector<uint32_t> n_head_l;
    std::vector<uint32_t> n_head_kv_l;

    // MoE (qwen3moe / qwen35moe / qwen3next / qwen4exp / step35 / lfm2moe)
    uint32_t n_expert       = 0;     // total experts (0 = dense)
    uint32_t n_expert_used  = 0;     // top-k
    uint32_t n_ff_exp       = 0;     // per-expert FFN size
    float    expert_weights_scale = 1.0f;
    // Routing shape. Qwen ranks experts on softmax(logits); step35 and lfm2moe
    // squash each logit with a sigmoid and rank on that plus a per-expert
    // selection bias (blk.N.exp_probs_b.bias), mixing with the unbiased value.
    // Values follow llama.cpp's llama_expert_gating_func_type.
    uint32_t expert_gating_func  = 1;     // 1 = softmax, 2 = sigmoid
    bool     expert_weights_norm = true;  // renormalise the top-k weights to sum 1
    // Leading blocks with a dense FFN before the first MoE block (step35: 3,
    // lfm2moe: 2). They have no routed experts at all.
    uint32_t n_dense_lead   = 0;

    // Gated DeltaNet (qwen35 / qwen35moe / qwen3next)
    bool     has_gdn        = false;
    uint32_t ssm_d_inner    = 0;     // key_dim = value_dim
    uint32_t ssm_n_group    = 0;     // H_k (number of GDN key heads/groups)
    uint32_t ssm_d_state    = 0;     // D (state size per group = head dim)
    uint32_t ssm_d_conv     = 0;     // conv kernel (=4)
    uint32_t ssm_dt_rank    = 0;     // H_v (number of GDN value heads)
    uint32_t full_attn_interval = 4; // attention every `interval` layers ((il+1)%interval==0)

    // Short convolution (lfm2 / lfm2moe): the recurrent layers are the ones with
    // no kv heads. Their only state is the last l_cache-1 inputs of the conv.
    bool     has_shortconv     = false;
    uint32_t shortconv_l_cache = 0;
    uint32_t nextn_predict_layers = 0;

    // ---- qwen4exp: hyper-connections ----
    // The residual stream carries hc_count identical-width copies ("streams")
    // instead of one, so it is hc_count * n_embd wide everywhere between layers.
    // A low-rank gate mixes them down to n_embd for each block and scatters the
    // block output back across the streams. 0 = ordinary single-stream residual.
    uint32_t hc_count    = 0;
    uint32_t hc_low_rank = 0;
    bool     has_hc() const { return hc_count > 0; }
    uint32_t n_embd_hc() const { return has_hc() ? hc_count * n_embd : n_embd; }

    // ---- qwen4exp: QSA (query-sparse attention) ----
    // Full-attention layers score whole blocks of compress_ratio tokens with a
    // small indexer head and attend to the best indexer_top_k of them (plus the
    // incomplete tail). Below indexer_top_k + compress_ratio - 1 cached tokens
    // this is exactly dense attention, so a short-context run needs none of it.
    uint32_t indexer_n_head   = 0;
    uint32_t indexer_head_dim = 0;
    uint32_t indexer_top_k    = 0;
    std::vector<uint32_t> compress_ratios;   // per layer; 0 = plain dense attention
    uint32_t compress_ratio(uint32_t il) const {
        return il < compress_ratios.size() ? compress_ratios[il] : 0;
    }
    bool has_qsa() const { return indexer_top_k > 0; }

    // ---- qwen4exp: PLE (per-layer n-gram hash embedding) ----
    // One designated layer gathers ple_n_heads rows of a shared hash table,
    // indexed by a 64-bit hash of the token and its predecessors. The table is
    // enormous (26.8 GiB in the IQ1_S quant) and is never a resident weight --
    // see NgramTable. Every field is 0 when the model has no PLE.
    uint32_t ple_ngram_size      = 0;
    uint32_t ple_heads_per_ngram = 0;
    uint32_t ple_conv_kernel     = 0;
    uint32_t ple_n_heads         = 0;  // (ngram_size - 1) * heads_per_ngram
    uint32_t ple_head_dim        = 0;  // embedding_length_per_layer_input
    uint32_t ple_eos_token_id    = 0;  // segment boundary -- NOT the chat EOS
    uint32_t ple_image_token_id  = 0;  // 0 = file predates the key, falls back to EOS
    std::vector<uint8_t>  ple_layers;            // per-layer flag
    std::vector<uint64_t> ple_layer_multipliers; // [ngram_size]
    std::vector<uint64_t> ple_head_offsets;      // [ple_n_heads]
    std::vector<uint64_t> ple_head_vocab_sizes;  // [ple_n_heads]
    bool has_ple() const { return ple_n_heads > 0; }
    bool is_ple(uint32_t il) const {
        return il < ple_layers.size() && ple_layers[il] != 0;
    }
    // Rows of conv history the PLE module adds to the recurrent state row.
    uint32_t ple_conv_state() const {
        return has_ple() ? (ple_conv_kernel - 1) * ple_ngram_size : 0;
    }

    InklingParams ink;   // arch == INKLING2 only; n_layer is then the PHYSICAL count

    // GGUF metadata (display / identification)
    std::string  general_name;  // general.name
    std::string  arch_file;     // general.architecture as written: the prefix of the file's keys
    // general.file_type. The value is a llama_ftype, not the ggml_ftype the
    // name suggests -- the two agree up to Q5_1 and diverge from there.
    uint32_t     file_type = 0;

    bool is_moe() const { return n_expert > 0; }
    // Does any layer carry recurrent state (GDN or short conv)? That state
    // cannot be truncated, so rewinds and speculative rollback need snapshots.
    bool has_recurrent() const { return has_gdn || has_shortconv; }
    bool is_moe_layer(uint32_t il) const { return is_moe() && il >= n_dense_lead; }
    bool gating_sigmoid() const { return expert_gating_func == 2; }

    // Per-layer shape accessors; uniform architectures fall back to the scalar,
    // so these are safe to call unconditionally.
    uint32_t head_count   (uint32_t il) const { return il < n_head_l.size()    ? n_head_l[il]    : n_head; }
    uint32_t head_count_kv(uint32_t il) const { return il < n_head_kv_l.size() ? n_head_kv_l[il] : n_head_kv; }
    // Main transformer stack excludes the trailing MTP (next-token-prediction) blocks.
    uint32_t n_main()  const { return n_layer - nextn_predict_layers; }
    bool     has_mtp() const { return nextn_predict_layers > 0; }
    // GDN (recurrent) layer iff hybrid, within main stack, and not a full-attn slot.
    bool is_recurrent(uint32_t il) const {
        if (has_shortconv) return il < n_layer && head_count_kv(il) == 0;
        if (!has_gdn) return false;
        return il < n_main() && ((il + 1) % full_attn_interval != 0);
    }
};

// Multi-GPU weight placement. `bufts[i]` is device i's buffer type and
// `layer_dev[il]` names the device that computes layer il, so a tensor named
// "blk.<il>.*" is allocated on that device and its matmuls run there. Tensors
// with no layer prefix (token embedding, output head, final norm) go to device
// 0, the primary. A plan with one device is the ordinary single-GPU case.
struct DevicePlan {
    std::vector<ggml_backend_buffer_type_t> bufts;
    std::vector<int>                        layer_dev;
    // Device holding each layer's expert pool. The shared-expert weights
    // (ffn_*_shexp) are computed alongside the routed experts, so they follow
    // this rather than layer_dev; everything else in the layer -- attention,
    // norms, the router -- stays with layer_dev. Empty means "same as layer_dev".
    std::vector<int>                        pool_dev;

    size_t n_dev() const { return bufts.size(); }
    // Device index for a GGUF tensor name; 0 when it has no "blk.<N>." prefix
    // or the layer is out of range.
    int dev_of_name(const std::string & name) const;
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
    // tokenizer.ggml.add_bos_token: the Qwen models say false, step35 says true.
    bool                     add_bos = false;
    std::string              pre;            // tokenizer.ggml.pre ("qwen2", "deepseek-v3", "lfm2", ...)
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

    // The PLE n-gram table's metadata (type and shape), or null when the model
    // has none. It is not in tensor(): no backend buffer ever holds it, and
    // NgramTable streams its rows straight from tensor_file()/_offset().
    ggml_tensor * ple_table() const { return ple_table_; }

    // Token embedding tensor suitable for ggml_get_rows on any backend.
    // (CUDA get_rows does not support K-quant/IQ types, so a dequantized F16
    //  copy is provided for those; Q8_0 is used instead when set_embd_q8(true).)
    ggml_tensor * tok_embd_rows() const { return tok_embd_rows_; }

    // Allocate a backend buffer for all weights and upload their data from the
    // GGUF file. After this call, tensor() pointers are backed by real data.
    // Returns the owning buffer (freed by the caller / engine).
    ggml_backend_buffer * load_weights(ggml_backend_t backend);

    // Multi-GPU variant of load_weights: no offload, every weight stays on a
    // GPU, but each layer's tensors go to the device that computes the layer.
    // This is what lets a model that does not fit one card run on two. All
    // buffers are caller-owned (unlike load_weights, where Model owns the one
    // buffer it returns).
    void load_weights_multi(ggml_backend_t primary, const DevicePlan & plan,
                            std::vector<ggml_backend_buffer_t> & out_bufs,
                            std::vector<size_t> * out_dev_bytes = nullptr);

    // Split variant: expert weight tensors (ffn_*_exps) go to cpu_buft,
    // everything else goes to gpu_backend. All output buffers are caller-owned.
    // Expert weights are spread over several cpu buffers (each below the single
    // cudaHostAlloc cap) so the whole set can be page-locked. Enables running
    // large MoE models when GPU VRAM is limited.
    // Both sides are lists because a backend can cap the size of one buffer
    // (Vulkan does, at 1 GiB by default) well below the weights' total.
    // `plan`, when given with more than one device, places each layer's tensors
    // on that layer's device instead of all of them on gpu_backend; out_gpu_bufs
    // still collects every GPU buffer (the caller owns them all) and
    // out_dev_bytes, if given, reports the bytes landed on each device so the
    // caller can size that device's expert pool against its own budget.
    void load_weights_split(ggml_backend_t gpu_backend,
                            ggml_backend_buffer_type_t cpu_buft,
                            std::vector<ggml_backend_buffer_t> & out_gpu_bufs,
                            std::vector<ggml_backend_buffer_t> & out_cpu_bufs,
                            const DevicePlan * plan = nullptr,
                            std::vector<size_t> * out_dev_bytes = nullptr);

    // SSD-tier variant: routed expert weights are NOT loaded into memory at all
    // (they stay on disk and are streamed on demand by ExpertCache). Everything
    // else (incl. shared experts) goes to gpu_backend. Saves the experts' RAM.
    void load_weights_ssd(ggml_backend_t gpu_backend,
                          std::vector<ggml_backend_buffer_t> & out_gpu_bufs,
                          const DevicePlan * plan = nullptr,
                          std::vector<size_t> * out_dev_bytes = nullptr);

    // Source file + absolute byte offset of a tensor's data (for pread).
    // For sharded models these vary per tensor (different shard files).
    size_t tensor_file_offset(const std::string & name) const;
    const std::string & tensor_file(const std::string & name) const;
    const std::string & path() const { return path_; }

    // Returns true if this model has any routed expert tensors (MoE layers).
    bool has_expert_tensors() const;

    // Bytes the weights occupy once loaded onto `backend`: every tensor, plus
    // the F16/Q8_0 copy of the token embedding that load_weights_* makes when
    // the backend has no get_rows kernel for the stored type. Computable from
    // the metadata alone, so the runtime can ask whether the model fits before
    // it commits a single byte of it. Depends on set_embd_q8() and, for
    // offloaded_expert_bytes(), on set_keep_nextn_resident().
    size_t weight_bytes(ggml_backend_t backend) const;
    // The part of that total held by routed experts, i.e. what the SSD and RAM
    // tiers take out of VRAM.
    size_t offloaded_expert_bytes() const;

    // Returns true if the tensor name belongs to a routed expert (not shared expert).
    static bool is_expert_tensor(const std::string & name);

    // Returns true if the tensor is a routed expert that should be offloaded to the
    // SSD/CPU tier. When MTP is in use the trailing nextn block is kept fully
    // VRAM-resident (its experts are NOT offloaded), since it runs once per drafted
    // token and benefits from staying on the GPU. When MTP is not used, the nextn
    // experts are offloaded like the rest to avoid wasting VRAM.
    bool is_offloaded_expert(const std::string & name) const;

    // Keep the MTP (nextn) block's experts VRAM-resident (set before load_weights_*).
    void set_keep_nextn_resident(bool v) { keep_nextn_resident_ = v; }

    // Use Q8_0 instead of F16 for the get_rows embedding fallback (saves ~45% VRAM
    // vs F16, but introduces additional quantization error). Set before load_weights_*.
    void set_embd_q8(bool v) { embd_q8_ = v; }
    // Run the looped core n times instead of the file's count. Call it before
    // anything is sized by the layer visits. Throws on a model without a loop.
    void set_loop_passes(int n);

    std::string summary() const;
    std::string debug_dump() const;

private:
    Model() = default;

    HParams hp_;
    Vocab   vocab_;
    bool    keep_nextn_resident_ = false;  // keep MTP nextn experts in VRAM (set by Runtime)
    bool    embd_q8_             = false;  // use Q8_0 (vs F16) for get_rows embedding fallback

    std::string    path_;
    gguf_context * gguf_   = nullptr;        // KV metadata (first shard)
    ggml_context * meta_   = nullptr;        // unified tensor metadata (+ data after load_weights)
    ggml_context * ple_ctx_ = nullptr;       // metadata for the PLE table alone (never allocated)
    ggml_tensor  * ple_table_ = nullptr;
    ggml_backend_buffer * weights_buf_ = nullptr;
    std::map<std::string, ggml_tensor *> tensors_;

    // Per-tensor data source (shard file + absolute offset). One entry per tensor;
    // for a single-file model all entries share path_.
    // half >= 0: one half of a fused per-expert tensor (Inkling's ffn_gate_up_exps
    // split into gate = half 0, up = half 1). Each expert's slab of `stride`
    // bytes holds both halves back to back, so the data is n_exp strided reads.
    struct Src { std::string path; size_t off = 0; int half = -1; size_t stride = 0; int64_t n_exp = 0; };
    std::map<std::string, Src> src_;

    // F32 token-embedding fallback for ggml_get_rows (see tok_embd_rows()).
    ggml_context *        embd_ctx_      = nullptr;
    ggml_backend_buffer * embd_buf_      = nullptr;
    ggml_tensor *         tok_embd_rows_ = nullptr;

    void load_hparams();
    void load_qwen4exp_hparams(const std::string & arch);
    void load_inkling2_hparams(const std::string & arch);
    void load_vocab();

    // Read nb bytes of tensor `name` from its (possibly sharded) source file into
    // dst, reusing open file handles cached in `files`.
    void read_tensor_bytes(const std::string & name, void * dst, size_t nb,
                           std::map<std::string, void *> & files) const;
};

} // namespace questwend
