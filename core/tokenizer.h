#pragma once

// GPT-2 / Qwen byte-level BPE tokenizer, built from GGUF vocab + merges.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace questwend {

struct Vocab;

class Tokenizer {
public:
    explicit Tokenizer(const Vocab & vocab);

    std::vector<int32_t> encode(const std::string & text, bool add_bos = false) const;
    std::string          decode(int32_t token) const;
    std::string          decode(const std::vector<int32_t> & tokens) const;

    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    int32_t token_to_id(const std::string & tok) const;

private:
    std::unordered_map<std::string, int32_t> token_ids_;     // byte-level token -> id
    std::vector<std::string>                 id_to_token_;
    std::unordered_map<std::string, int>     merge_rank_;    // "A B" -> rank
    int32_t bos_ = -1, eos_ = -1;

    // byte <-> unicode (GPT-2 mapping)
    std::unordered_map<uint8_t, uint32_t> byte_to_uni_;
    std::unordered_map<uint32_t, uint8_t> uni_to_byte_;

    std::vector<std::string> pretokenize(const std::string & text) const;
    std::vector<int32_t>     bpe_encode_piece(const std::string & piece) const;
};

} // namespace questwend
