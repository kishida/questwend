#include "tokenizer.h"
#include "model.h"   // Vocab

#include <algorithm>
#include <limits>

namespace questwend {

// ---- UTF-8 helpers ----
static void utf8_append(std::string & s, uint32_t cp) {
    if (cp < 0x80) {
        s.push_back((char) cp);
    } else if (cp < 0x800) {
        s.push_back((char) (0xC0 | (cp >> 6)));
        s.push_back((char) (0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back((char) (0xE0 | (cp >> 12)));
        s.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        s.push_back((char) (0x80 | (cp & 0x3F)));
    } else {
        s.push_back((char) (0xF0 | (cp >> 18)));
        s.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
        s.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        s.push_back((char) (0x80 | (cp & 0x3F)));
    }
}

struct CP { uint32_t cp; size_t off; size_t len; };

static std::vector<CP> utf8_decode(const std::string & s) {
    std::vector<CP> out;
    size_t i = 0;
    while (i < s.size()) {
        const uint8_t c = (uint8_t) s[i];
        uint32_t cp; size_t len;
        if (c < 0x80)        { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E){ cp = c & 0x07; len = 4; }
        else                 { cp = c; len = 1; }
        for (size_t k = 1; k < len && i + k < s.size(); ++k)
            cp = (cp << 6) | ((uint8_t) s[i + k] & 0x3F);
        out.push_back({cp, i, len});
        i += len;
    }
    return out;
}

// ---- GPT-2 byte<->unicode ----
static void build_byte_unicode(std::unordered_map<uint8_t, uint32_t> & b2u,
                               std::unordered_map<uint32_t, uint8_t> & u2b) {
    auto in_printable = [](int b) {
        return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    };
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp;
        if (in_printable(b)) {
            cp = (uint32_t) b;
        } else {
            cp = (uint32_t) (256 + n);
            ++n;
        }
        b2u[(uint8_t) b] = cp;
        u2b[cp] = (uint8_t) b;
    }
}

Tokenizer::Tokenizer(const Vocab & vocab) {
    bos_ = vocab.bos_id;
    eos_ = vocab.eos_id;

    id_to_token_ = vocab.tokens;
    token_ids_.reserve(vocab.tokens.size());
    for (int32_t i = 0; i < (int32_t) vocab.tokens.size(); ++i) {
        token_ids_[vocab.tokens[i]] = i;
    }
    for (int r = 0; r < (int) vocab.merges.size(); ++r) {
        merge_rank_[vocab.merges[r]] = r;
    }
    build_byte_unicode(byte_to_uni_, uni_to_byte_);
}

int32_t Tokenizer::token_to_id(const std::string & tok) const {
    auto it = token_ids_.find(tok);
    return it == token_ids_.end() ? -1 : it->second;
}

// Qwen3.5 pretokenization (tokenizer.json / llama.cpp pre_type "qwen35"):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
// | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
// | \p{N}
// |  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
// | \s*[\r\n]+
// | \s+(?!\S)
// | \s+
// Alternatives are tried in order at each position, like the regex engine.
// The details matter for staying on the model's canonical tokenization:
// a word absorbs ONE preceding symbol (".spring" is a single piece, which
// is how "org.springframework" tokenizes), an indentation run leaves its
// last space to the following word (" return"), digits split one at a
// time, and a symbol run absorbs trailing newlines (";\n").
// \p{L}\p{M} is approximated as "ASCII letters or any codepoint > 0x7F"
// (CJK punctuation counts as letters; acceptable for chat text).
std::vector<std::string> Tokenizer::pretokenize(const std::string & text) const {
    auto cps = utf8_decode(text);
    std::vector<std::string> pieces;

    auto is_space  = [](uint32_t c){ return c==' '||c=='\t'||c=='\n'||c=='\r'||c==0x0b||c==0x0c; };
    auto is_letter = [](uint32_t c){ return (c>='A'&&c<='Z')||(c>='a'&&c<='z')|| c>0x7F; };
    auto is_digit  = [](uint32_t c){ return c>='0'&&c<='9'; };
    auto is_punct  = [&](uint32_t c){ return !is_space(c) && !is_letter(c) && !is_digit(c); };
    auto lower     = [](uint32_t c){ return c>='A'&&c<='Z' ? c+32 : c; };

    const size_t n = cps.size();
    size_t i = 0;
    auto emit = [&](size_t a, size_t b){
        if (b <= a) return;
        pieces.emplace_back(text.substr(cps[a].off, cps[b-1].off + cps[b-1].len - cps[a].off));
    };

    while (i < n) {
        const uint32_t c = cps[i].cp;
        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c == '\'' && i + 1 < n) {
            const uint32_t d = lower(cps[i+1].cp);
            if (d=='s'||d=='t'||d=='m'||d=='d') { emit(i, i+2); i += 2; continue; }
            if (i + 2 < n) {
                const uint32_t e = lower(cps[i+2].cp);
                if ((d=='r'&&e=='e')||(d=='v'&&e=='e')||(d=='l'&&e=='l')) { emit(i, i+3); i += 3; continue; }
            }
        }
        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (is_letter(c)) {
            size_t j = i + 1; while (j < n && is_letter(cps[j].cp)) ++j;
            emit(i, j); i = j; continue;
        }
        if (c != '\r' && c != '\n' && !is_digit(c) && i + 1 < n && is_letter(cps[i+1].cp)) {
            size_t j = i + 2; while (j < n && is_letter(cps[j].cp)) ++j;
            emit(i, j); i = j; continue;
        }
        // \p{N}  (one digit per piece)
        if (is_digit(c)) { emit(i, i + 1); ++i; continue; }
        //  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        if (is_punct(c) || (c == ' ' && i + 1 < n && is_punct(cps[i+1].cp))) {
            size_t j = c == ' ' ? i + 1 : i;
            while (j < n && is_punct(cps[j].cp)) ++j;
            while (j < n && (cps[j].cp=='\n' || cps[j].cp=='\r')) ++j;
            emit(i, j); i = j; continue;
        }
        // \s*[\r\n]+ | \s+(?!\S) | \s+
        size_t j = i; while (j < n && is_space(cps[j].cp)) ++j;
        size_t nl_end = i;   // one past the last \r\n in the run (i: none)
        for (size_t k = i; k < j; ++k) if (cps[k].cp=='\n' || cps[k].cp=='\r') nl_end = k + 1;
        if (nl_end > i)         { emit(i, nl_end); i = nl_end; continue; }  // \s*[\r\n]+
        if (j < n && j - i > 1) { emit(i, j - 1); i = j - 1; continue; }    // \s+(?!\S)
        emit(i, j); i = j;                                                  // \s+
    }
    return pieces;
}

std::vector<int32_t> Tokenizer::bpe_encode_piece(const std::string & piece) const {
    // map each raw byte to its unicode symbol (as a UTF-8 string)
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (unsigned char b : piece) {
        std::string s;
        utf8_append(s, byte_to_uni_.at(b));
        symbols.push_back(std::move(s));
    }

    // iteratively merge the lowest-rank adjacent pair
    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        int best_i = -1;
        for (int i = 0; i + 1 < (int) symbols.size(); ++i) {
            auto it = merge_rank_.find(symbols[i] + " " + symbols[i+1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }

    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (auto & s : symbols) {
        auto it = token_ids_.find(s);
        if (it != token_ids_.end()) {
            ids.push_back(it->second);
        } else {
            // fall back to per-symbol (byte) tokens
            for (auto & cp : utf8_decode(s)) {
                std::string one; utf8_append(one, cp.cp);
                auto jt = token_ids_.find(one);
                if (jt != token_ids_.end()) ids.push_back(jt->second);
            }
        }
    }
    return ids;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text, bool add_bos) const {
    std::vector<int32_t> ids;
    if (add_bos && bos_ >= 0) ids.push_back(bos_);
    for (auto & piece : pretokenize(text)) {
        auto pids = bpe_encode_piece(piece);
        ids.insert(ids.end(), pids.begin(), pids.end());
    }
    return ids;
}

std::string Tokenizer::decode(int32_t token) const {
    if (token < 0 || token >= (int32_t) id_to_token_.size()) return "";
    const std::string & t = id_to_token_[token];
    std::string out;
    for (auto & cp : utf8_decode(t)) {
        auto it = uni_to_byte_.find(cp.cp);
        if (it != uni_to_byte_.end()) out.push_back((char) it->second);
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t> & tokens) const {
    std::string out;
    for (int32_t t : tokens) out += decode(t);
    return out;
}

} // namespace questwend
