#pragma once

// PLE n-gram hash embedding table (qwen4exp / Qwen3.8-Flash-Next).
//
// The table is one tensor, `per_layer_token_embd.weight`, and in the IQ1_S
// quant of Qwen3.8-Flash-Next it is 26.82 GiB -- 320,001,536 rows of 160
// values. It dwarfs everything else in the file: the routed experts are
// 37.11 GiB and all the remaining weights together are 3.62 GiB.
//
// It is also nothing like a weight in how it is used. Each token gathers
// ple_n_heads rows (16 in the real model, 1440 bytes), and *which* rows is a
// pure function of the token ids -- a 64-bit hash of the token and its
// predecessors, computed on the host before the graph runs. Nothing about the
// gather depends on model state.
//
// So this does not make it a ggml tensor at all. llama.cpp keeps the whole
// thing resident and uses ggml_get_rows; here the rows are pread out of the
// GGUF, dequantised on the host, and handed to the graph as one small F32
// input. The resident cost is then the row cache, not the table, and the
// question of whether a 26.8 GiB tensor fits in VRAM never comes up.
//
// The cost that remains is IOPS: ple_n_heads random reads per token, scattered
// over the whole table. On an NVMe that is nothing; on a spinning disk it is
// the dominant per-token cost, which is what Mode::OFF exists for.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_tensor;

namespace questwend {

class Model;
struct HParams;

class NgramTable {
public:
    enum class Mode {
        OFF,    // skip the PLE module entirely; the table is never opened
        DISK,   // stream rows from the GGUF, keeping a bounded row cache
        RAM,    // hold the whole table in host memory
    };

    static const char * mode_name(Mode m);
    // "off" / "disk" / "ram"; returns false (leaving `out` alone) on anything else.
    static bool parse_mode(const std::string & s, Mode & out);

    // Throws if the model has PLE metadata but no table tensor. `cache_mb`
    // bounds the DISK-mode row cache; it is ignored in the other modes.
    NgramTable(const Model & model, Mode mode, size_t cache_mb);
    ~NgramTable();

    Mode mode() const { return mode_; }

    // Row indices for `n` tokens, ple_n_heads per token, appended to `out`.
    // `hist` carries the (ngram_size - 1) tokens preceding this batch and is
    // updated to the batch's own tail, so consecutive calls see the context a
    // single-shot pass would. Pass an empty `hist` at the start of a sequence.
    void hash_rows(const int32_t * tokens, int n, std::vector<int32_t> & hist,
                   std::vector<int32_t> & out) const;

    // Gather the rows named by hash_rows into `dst`, which must hold
    // n_rows * ple_head_dim floats. Rows are dequantised to F32.
    void gather(const int32_t * rows, size_t n_rows, float * dst);

    struct Stats { uint64_t rows = 0, hits = 0, reads = 0; double read_ms = 0; };
    const Stats & stats() const { return stats_; }
    // Bytes this instance holds resident (the RAM copy, or the row cache).
    size_t resident_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    Mode   mode_;
    Stats  stats_;
};

} // namespace questwend
