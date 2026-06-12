#pragma once

// Qwen3.5/3.6 chat prompt construction.
//
// Hand-coded equivalent of the official Qwen3.6 chat_template.jinja (no jinja
// engine): ChatML framing, tools block, <think> handling, multimodal content
// parts with <|vision_start|><|image_pad|>...<|vision_end|> expansion.
// Video content is not supported (out of scope for now).

#include "tokenizer.h"

#include <string>
#include <utility>
#include <vector>

namespace questwend {

struct ContentPart {
    enum class Kind { Text, Image };
    Kind        kind = Kind::Text;
    std::string text;             // Kind::Text
    int         image_index = -1; // Kind::Image: index into the caller's image list

    static ContentPart make_text(std::string t) {
        ContentPart p; p.kind = Kind::Text; p.text = std::move(t); return p;
    }
    static ContentPart make_image(int index) {
        ContentPart p; p.kind = Kind::Image; p.image_index = index; return p;
    }
};

// One tool call on an assistant message (rendered in Qwen3.6's
// <tool_call><function=...> XML format). `arguments` values must be
// pre-serialized: raw text for string values, JSON for everything else.
struct ToolCall {
    std::string name;
    std::vector<std::pair<std::string, std::string>> arguments;
};

struct ChatMessage {
    std::string role;       // "system" / "user" / "assistant" / "tool"
    std::string content;    // plain-text content (used when `parts` is empty)
    std::vector<ContentPart> parts;       // multimodal content (overrides `content`)
    std::string reasoning_content;        // explicit reasoning (assistant)
    bool        has_reasoning = false;    // reasoning_content is present (may be "")
    std::vector<ToolCall> tool_calls;     // assistant tool calls
};

struct ChatPromptOptions {
    bool add_generation_prompt = true;
    bool reasoning             = true;   // enable_thinking (generation prompt only)
    bool preserve_thinking     = false;  // keep <think> blocks of all past assistant turns
    bool add_vision_id         = false;  // prefix images with "Picture N: "
    int  n_image_tokens        = 1;      // <|image_pad|> repetitions per image (e.g. 576)
    std::vector<std::string> tools_json; // pre-serialized JSON, one per tool
};

struct ChatPrompt {
    std::vector<int32_t> ids;
    // Per image, in order of appearance: (first token index, token count) of its
    // <|image_pad|> run, plus the ContentPart::image_index it refers to.
    struct ImageSpan { int first = 0, count = 0, image_index = -1; };
    std::vector<ImageSpan> image_spans;
};

// Build the full prompt token sequence (Qwen3.6 template semantics).
// Throws std::runtime_error on unsupported content (e.g. images on a model
// without vision tokens, video parts, system message with images).
ChatPrompt build_qwen_prompt(const Tokenizer & tok,
                             const std::vector<ChatMessage> & messages,
                             const ChatPromptOptions & opts = {});

// Back-compat helper used by existing CLI/server code paths (text-only).
inline std::vector<int32_t> build_chatml_tokens(
        const Tokenizer & tok,
        const std::vector<ChatMessage> & messages,
        bool add_generation_prompt = true,
        bool reasoning = true) {
    ChatPromptOptions o;
    o.add_generation_prompt = add_generation_prompt;
    o.reasoning = reasoning;
    return build_qwen_prompt(tok, messages, o).ids;
}

} // namespace questwend
