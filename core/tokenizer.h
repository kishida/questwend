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
    // Which pre-tokenizer splits the text before BPE. The GGUF names it in
    // tokenizer.ggml.pre; the two differ enough that using the wrong one gives
    // subtly different ids for digits, CJK and punctuation runs.
    enum class Pre { QWEN, DEEPSEEK3 };

    explicit Tokenizer(const Vocab & vocab);

    // parse_special: recognise control / user-defined tokens written literally in
    // the text. True matches llama.cpp for a raw prompt. Chat message bodies pass
    // false -- otherwise a user who types "<|im_end|>" ends their own turn.
    std::vector<int32_t> encode(const std::string & text, bool add_bos = false,
                                bool parse_special = true) const;
    std::string          decode(int32_t token) const;
    std::string          decode(const std::vector<int32_t> & tokens) const;

    int32_t bos() const { return bos_; }
    // Does the model want a BOS in front of a raw prompt? (tokenizer.ggml.add_bos_token)
    bool    wants_bos() const { return add_bos_; }
    int32_t eos() const { return eos_; }
    int32_t token_to_id(const std::string & tok) const;

private:
    std::unordered_map<std::string, int32_t> token_ids_;     // byte-level token -> id
    std::vector<std::string>                 id_to_token_;
    std::unordered_map<std::string, int>     merge_rank_;    // "A B" -> rank
    int32_t bos_ = -1, eos_ = -1;
    bool    add_bos_ = false;

    // byte <-> unicode (GPT-2 mapping)
    std::unordered_map<uint8_t, uint32_t> byte_to_uni_;
    std::unordered_map<uint32_t, uint8_t> uni_to_byte_;

    Pre     pre_ = Pre::QWEN;

    // Control / user-defined tokens, longest first. Text is split on these
    // before the pre-tokenizer runs, so "<|im_start|>" in a raw prompt becomes
    // the single control id and not the six pieces its characters would BPE to.
    std::vector<std::pair<std::string, int32_t>> special_;
    bool special_first_[256] = { false };   // first byte of some special token

    // Longest special token starting at text[pos], or -1.
    int32_t match_special(const std::string & text, size_t pos, size_t * len) const;

    std::vector<std::string> pretokenize(const std::string & text) const;
    std::vector<std::string> pretokenize_qwen(const std::string & text) const;
    std::vector<std::string> pretokenize_deepseek3(const std::string & text) const;
    std::vector<int32_t>     bpe_encode_piece(const std::string & piece) const;
};

} // namespace questwend
