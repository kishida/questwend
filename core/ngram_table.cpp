#include "ngram_table.h"

#include "model.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <list>
#include <stdexcept>

namespace questwend {

static const char * PLE_TENSOR = "per_layer_token_embd.weight";

// The table is past 2^31 bytes, so the seek has to be 64-bit on every platform.
static int seek64(FILE * f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, (long long) off, SEEK_SET);
#else
    return fseeko(f, (off_t) off, SEEK_SET);
#endif
}

const char * NgramTable::mode_name(Mode m) {
    switch (m) {
        case Mode::OFF:  return "off";
        case Mode::DISK: return "disk";
        case Mode::RAM:  return "ram";
    }
    return "?";
}

bool NgramTable::parse_mode(const std::string & s, Mode & out) {
    if (s == "off"  || s == "0" || s == "none") { out = Mode::OFF;  return true; }
    if (s == "disk" || s == "ssd")              { out = Mode::DISK; return true; }
    if (s == "ram"  || s == "mem")              { out = Mode::RAM;  return true; }
    return false;
}

struct NgramTable::Impl {
    // geometry
    uint32_t n_heads   = 0;
    uint32_t head_dim  = 0;
    uint32_t ngram     = 0;
    uint32_t per_gram  = 0;
    int32_t  eos       = 0;
    std::vector<uint64_t> mult, offset, vocab;

    // storage
    ggml_type type      = GGML_TYPE_F32;
    size_t    row_bytes = 0;
    int64_t   n_rows    = 0;
    size_t    file_off  = 0;
    std::string path;
    FILE *    fp = nullptr;

    std::vector<uint8_t> ram;          // RAM mode: the whole table, still quantised

    // DISK mode row cache. Rows are kept in their file representation (90 bytes
    // for the real model's IQ4_NL) rather than dequantised (640): 7x the rows
    // for the same memory, against one cheap dequantise per hit.
    size_t cache_cap = 0;              // rows
    std::vector<uint8_t> cache_data;
    std::list<int32_t>   lru;          // front = most recent
    struct Slot { size_t idx; std::list<int32_t>::iterator it; };
    std::unordered_map<int32_t, Slot> cache_index;
    std::vector<int32_t> free_slots;

    // scratch reused across gathers so a prefill does not churn allocations
    std::vector<std::pair<int32_t, size_t>> order;   // (row, first output slot)

    ~Impl() {
        if (fp) fclose(fp);
    }

    // ggml leaves to_float null for F32 (there is nothing to convert), so a
    // plain unquantised table would call through a null pointer.
    ggml_to_float_t to_float = nullptr;

    void row_to_float(const uint8_t * src, float * dst) const {
        if (to_float) to_float(src, dst, head_dim);
        else          memcpy(dst, src, (size_t) head_dim * sizeof(float));
    }

    const uint8_t * cached(int32_t row) {
        auto it = cache_index.find(row);
        if (it == cache_index.end()) return nullptr;
        lru.splice(lru.begin(), lru, it->second.it);
        return cache_data.data() + it->second.idx * row_bytes;
    }

    void insert(int32_t row, const uint8_t * src) {
        if (cache_cap == 0) return;
        size_t idx;
        if (!free_slots.empty()) {
            idx = (size_t) free_slots.back();
            free_slots.pop_back();
        } else if (cache_index.size() >= cache_cap) {
            const int32_t victim = lru.back();
            lru.pop_back();
            auto v = cache_index.find(victim);
            idx = v->second.idx;
            cache_index.erase(v);
        } else {
            idx = cache_index.size();
        }
        memcpy(cache_data.data() + idx * row_bytes, src, row_bytes);
        lru.push_front(row);
        cache_index[row] = { idx, lru.begin() };
    }

    void read_row(int32_t row, uint8_t * dst) {
        if (seek64(fp, file_off + (uint64_t) row * row_bytes) != 0 ||
            fread(dst, 1, row_bytes, fp) != row_bytes) {
            throw std::runtime_error("ngram table: read failed at row " + std::to_string(row));
        }
    }
};

NgramTable::NgramTable(const Model & model, Mode mode, size_t cache_mb)
    : impl_(new Impl), mode_(mode) {
    const HParams & hp = model.hparams();
    if (!hp.has_ple()) {
        mode_ = Mode::OFF;
        return;
    }

    auto & im = *impl_;
    im.n_heads  = hp.ple_n_heads;
    im.head_dim = hp.ple_head_dim;
    im.ngram    = hp.ple_ngram_size;
    im.per_gram = hp.ple_heads_per_ngram;
    im.eos      = (int32_t) hp.ple_eos_token_id;
    im.mult     = hp.ple_layer_multipliers;
    im.offset   = hp.ple_head_offsets;
    im.vocab    = hp.ple_head_vocab_sizes;

    if (mode_ == Mode::OFF) return;

    ggml_tensor * t = model.ple_table();
    if (!t) {
        throw std::runtime_error(
                std::string("model declares PLE but has no ") + PLE_TENSOR +
                " (use --ngram off to run without the n-gram embedding)");
    }
    if (t->ne[0] != (int64_t) im.head_dim) {
        throw std::runtime_error("ngram table: row width does not match ple head dim");
    }

    im.type      = t->type;
    im.to_float  = ggml_get_type_traits(t->type)->to_float;
    if (!im.to_float && t->type != GGML_TYPE_F32) {
        throw std::runtime_error(std::string("ngram table: no dequantiser for ") +
                                 ggml_type_name(t->type));
    }
    im.n_rows    = t->ne[1];
    im.row_bytes = ggml_row_size(t->type, t->ne[0]);
    im.path      = model.tensor_file(PLE_TENSOR);
    im.file_off  = model.tensor_file_offset(PLE_TENSOR);

    im.fp = fopen(im.path.c_str(), "rb");
    if (!im.fp) throw std::runtime_error("ngram table: cannot open " + im.path);

    if (mode_ == Mode::RAM) {
        const size_t bytes = (size_t) im.n_rows * im.row_bytes;
        im.ram.resize(bytes);
        if (seek64(im.fp, im.file_off) != 0) {
            throw std::runtime_error("ngram table: seek failed");
        }
        // one big sequential read, reported so a slow disk does not look like a hang
        size_t done = 0;
        while (done < bytes) {
            const size_t chunk = std::min<size_t>(64u << 20, bytes - done);
            if (fread(im.ram.data() + done, 1, chunk, im.fp) != chunk) {
                throw std::runtime_error("ngram table: short read loading the table");
            }
            done += chunk;
            fprintf(stderr, "\rngram table: %.1f / %.1f GiB",
                    done / 1073741824.0, bytes / 1073741824.0);
        }
        fprintf(stderr, "\n");
    } else {
        const size_t rows = cache_mb ? (cache_mb << 20) / im.row_bytes : 0;
        im.cache_cap = std::min<size_t>(rows, (size_t) im.n_rows);
        im.cache_data.resize(im.cache_cap * im.row_bytes);
        im.cache_index.reserve(im.cache_cap * 2);
    }
}

NgramTable::~NgramTable() = default;

void NgramTable::hash_rows(const int32_t * tokens, int n, std::vector<int32_t> & hist,
                           std::vector<int32_t> & out) const {
    const auto & im = *impl_;
    const int  ngram = (int) im.ngram;
    const int  heads = (int) im.n_heads;
    const int  eos   = im.eos;

    // A fresh sequence reads as if every predecessor were the segment boundary,
    // which is what the reference's zero-padded context comes out as.
    if ((int) hist.size() < ngram - 1) {
        hist.insert(hist.begin(), (size_t) (ngram - 1) - hist.size(), eos);
    }

    // Snapshot before updating: read and write in one pass and a token near the
    // start of the batch would pick up an earlier token of this same batch out
    // of the history as if it were prior context.
    const std::vector<int32_t> prev_hist = hist;

    const size_t base_out = out.size();
    out.resize(base_out + (size_t) n * heads);

    std::vector<int64_t> ctx((size_t) ngram);
    for (int i = 0; i < n; ++i) {
        // predecessor s (1-based): inside this batch, else the carried history
        auto prev = [&](int s) -> int64_t {
            const int j = i - s;
            if (j >= 0) return tokens[j];
            const int k = (int) prev_hist.size() - (s - i);
            return (k >= 0 && k < (int) prev_hist.size()) ? prev_hist[k] : eos;
        };

        // An EOS in the window hides everything at or before it. The token's own
        // EOS does not cut its context: the reference takes the last EOS strictly
        // before this position, so a boundary only hides it from what follows.
        ctx[0] = tokens[i];
        bool cut = false;
        for (int s = 1; s < ngram; ++s) {
            ctx[s] = cut ? eos : prev(s);
            if (ctx[s] == eos) cut = true;
        }

        for (int g = 2; g <= ngram; ++g) {
            uint64_t mixed = (uint64_t) ctx[0] * im.mult[0];
            for (int j = 1; j < g; ++j) {
                mixed ^= (uint64_t) ctx[j] * im.mult[j];
            }
            const int base = (g - 2) * (int) im.per_gram;
            for (int h = 0; h < (int) im.per_gram; ++h) {
                const int hi = base + h;
                out[base_out + (size_t) i * heads + hi] =
                        (int32_t) (mixed % im.vocab[hi] + im.offset[hi]);
            }
        }
    }

    // carry the last ngram-1 tokens forward: this batch's tail, topped up from
    // the previous history when the batch is shorter than the window
    hist = prev_hist;
    hist.insert(hist.end(), tokens, tokens + n);
    hist.erase(hist.begin(), hist.end() - (ngram - 1));
}

void NgramTable::gather(const int32_t * rows, size_t n_rows, float * dst) {
    auto & im = *impl_;
    const int64_t dim = im.head_dim;

    stats_.rows += n_rows;

    if (mode_ == Mode::RAM) {
        for (size_t i = 0; i < n_rows; ++i) {
            im.row_to_float(im.ram.data() + (size_t) rows[i] * im.row_bytes,
                            dst + (size_t) i * dim);
        }
        stats_.hits += n_rows;
        return;
    }

    // Sort the misses by file offset before reading them. Every index is known
    // before the graph runs, so a prefill's reads can go out in ascending order
    // -- on a spinning disk that is the difference between thousands of full
    // seeks and one forward sweep.
    im.order.clear();
    std::vector<uint8_t> row_buf(im.row_bytes);
    for (size_t i = 0; i < n_rows; ++i) {
        if (const uint8_t * hit = im.cached(rows[i])) {
            im.row_to_float(hit, dst + (size_t) i * dim);
            stats_.hits++;
        } else {
            im.order.emplace_back(rows[i], i);
        }
    }
    std::sort(im.order.begin(), im.order.end());

    const auto t0 = std::chrono::steady_clock::now();
    int32_t last_row = -1;
    for (const auto & e : im.order) {
        if (e.first != last_row) {
            im.read_row(e.first, row_buf.data());
            im.insert(e.first, row_buf.data());
            last_row = e.first;
            stats_.reads++;
        }
        im.row_to_float(row_buf.data(), dst + (size_t) e.second * dim);
    }
    stats_.read_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
}

size_t NgramTable::resident_bytes() const {
    const auto & im = *impl_;
    return mode_ == Mode::RAM ? im.ram.size() : im.cache_data.size();
}

} // namespace questwend
