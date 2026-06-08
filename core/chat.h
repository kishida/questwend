#pragma once

// Minimal ChatML (Qwen) prompt construction.
//   <|im_start|>{role}\n{content}<|im_end|>\n ... <|im_start|>assistant\n

#include "tokenizer.h"

#include <string>
#include <vector>

namespace qwencpp {

struct ChatMessage {
    std::string role;     // "system" / "user" / "assistant"
    std::string content;
};

inline std::vector<int32_t> build_chatml_tokens(
        const Tokenizer & tok,
        const std::vector<ChatMessage> & messages,
        bool add_generation_prompt = true) {
    const int32_t im_start = tok.token_to_id("<|im_start|>");
    const int32_t im_end   = tok.token_to_id("<|im_end|>");

    std::vector<int32_t> ids;
    auto append = [&](const std::vector<int32_t> & v) { ids.insert(ids.end(), v.begin(), v.end()); };

    const bool have_specials = (im_start >= 0 && im_end >= 0);

    for (const auto & m : messages) {
        if (have_specials) {
            ids.push_back(im_start);
            append(tok.encode(m.role + "\n" + m.content, false));
            ids.push_back(im_end);
            append(tok.encode("\n", false));
        } else {
            append(tok.encode(m.role + ": " + m.content + "\n", false));
        }
    }
    if (add_generation_prompt) {
        if (have_specials) {
            ids.push_back(im_start);
            append(tok.encode("assistant\n", false));
        } else {
            append(tok.encode("assistant: ", false));
        }
    }
    return ids;
}

} // namespace qwencpp
