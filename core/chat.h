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
    // The same arguments as Python literals ('x', 3, True, [1, 2]), which is
    // how lfm2 writes a call. Parallel to `arguments`; empty = not supplied.
    std::vector<std::string> arg_literals;
    // The whole call as compact JSON, {"name":...,"args":{...}}, which is how
    // inkling2 writes one. Empty = not supplied.
    std::string json_body;
};

struct ChatMessage {
    std::string role;       // "system" / "user" / "assistant" / "tool"
    std::string content;    // plain-text content (used when `parts` is empty)
    std::vector<ContentPart> parts;       // multimodal content (overrides `content`)
    std::string reasoning_content;        // explicit reasoning (assistant)
    bool        has_reasoning = false;    // reasoning_content is present (may be "")
    std::vector<ToolCall> tool_calls;     // assistant tool calls
    std::string name;                     // inkling2: header name (a tool result carries the tool's)
    bool        no_content = false;       // content was null / absent, which is not the same as ""
};

struct ChatPromptOptions {
    bool add_generation_prompt = true;
    bool reasoning             = true;   // enable_thinking (generation prompt only)
    bool preserve_thinking     = false;  // keep <think> blocks of all past assistant turns
    bool add_vision_id         = false;  // prefix images with "Picture N: "
    int  n_image_tokens        = 1;      // <|image_pad|> repetitions per image (e.g. 576)
    std::vector<int> image_tokens;       // per image_index, for a variable-size tower (overrides the above)
    std::vector<std::string> tools_json; // pre-serialized JSON, one per tool
};

struct ChatPrompt {
    std::vector<int32_t> ids;
    // Per image, in order of appearance: (first token index, token count) of its
    // <|image_pad|> run, plus the ContentPart::image_index it refers to.
    struct ImageSpan { int first = 0, count = 0, image_index = -1; };
    std::vector<ImageSpan> image_spans;
    // Index in `ids` where the generation prompt (<|im_start|>assistant ...)
    // starts; -1 without add_generation_prompt. A future request that rewrites
    // the assistant turn (e.g. drops its reasoning) re-tokenizes identically up
    // to exactly this point, so it is a natural prompt-checkpoint position.
    int gen_prompt_begin = -1;
};

// How a model's chat output is marked up, so the server can route it.
struct ChatFormat {
    bool        lfm2       = false;          // lfm2 template (see build_lfm2_prompt)
    bool        inkling    = false;          // Inkling 2 template (see build_inkling_prompt)
    std::string tool_open  = "<tool_call>";  // start of a tool-call block
    std::string tool_close = "</tool_call>";
};
ChatFormat chat_format(const Tokenizer & tok);

// One Inkling 2 reply, token by token. It is a list of blocks,
//   <|message_model|><|content_thinking|>R<|end_message|>
//   <|message_model|><|content_text|>C<|end_message|>
//   <|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>
// and only the tokens tell a block's kind: the text of any of them may spell
// "</think>" or a marker.
class ReplyDecoder {
public:
    explicit ReplyDecoder(const Tokenizer & tok);
    // Appends what the token adds to the visible reply.
    void add(int32_t token, std::string & reasoning_out, std::string & content_out);
    const std::string & reasoning() const { return reasoning_; }
    const std::string & content() const   { return content_; }
    const std::vector<std::string> & calls() const { return calls_; }   // JSON bodies
private:
    enum class Block { Header, Thinking, Text, Call };
    const Tokenizer & tok_;
    int32_t model_ = -1, think_ = -1, text_ = -1, tool_ = -1, end_ = -1;
    Block block_ = Block::Text;
    std::string reasoning_, content_;
    std::vector<std::string> calls_;
};

// Build the full prompt token sequence (Qwen3.6 template semantics; an lfm2
// vocabulary gets the lfm2 template instead).
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
