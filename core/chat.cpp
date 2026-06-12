#include "chat.h"

#include <stdexcept>

namespace questwend {

namespace {

std::string ltrim_ws(std::string s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    return s.substr(i);
}
std::string rtrim_ws(std::string s) {
    size_t e = s.size();
    while (e > 0 && (s[e-1] == ' ' || s[e-1] == '\t' || s[e-1] == '\n' || s[e-1] == '\r')) --e;
    return s.substr(0, e);
}
std::string trim_ws(std::string s) { return rtrim_ws(ltrim_ws(std::move(s))); }

bool starts_with(const std::string & s, const char * p) { return s.rfind(p, 0) == 0; }
bool ends_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// Fixed tool-calling instruction block (verbatim from the Qwen3.6 template).
const char * TOOL_INSTRUCTIONS =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n</IMPORTANT>";

// Plain text of a message (text parts concatenated; image parts ignored).
std::string flat_text(const ChatMessage & m) {
    if (m.parts.empty()) return m.content;
    std::string s;
    for (const auto & p : m.parts)
        if (p.kind == ContentPart::Kind::Text) s += p.text;
    return s;
}

bool has_image(const ChatMessage & m) {
    for (const auto & p : m.parts)
        if (p.kind == ContentPart::Kind::Image) return true;
    return false;
}

} // namespace

ChatPrompt build_qwen_prompt(const Tokenizer & tok,
                             const std::vector<ChatMessage> & messages,
                             const ChatPromptOptions & opts) {
    const int32_t im_start    = tok.token_to_id("<|im_start|>");
    const int32_t im_end      = tok.token_to_id("<|im_end|>");
    const int32_t think_open  = tok.token_to_id("<think>");
    const int32_t think_close = tok.token_to_id("</think>");
    const int32_t vis_start   = tok.token_to_id("<|vision_start|>");
    const int32_t vis_end     = tok.token_to_id("<|vision_end|>");
    const int32_t image_pad   = tok.token_to_id("<|image_pad|>");

    ChatPrompt out;
    auto & ids = out.ids;
    auto emit_text = [&](const std::string & s) {
        if (s.empty()) return;
        const auto v = tok.encode(s, false);
        ids.insert(ids.end(), v.begin(), v.end());
    };
    auto emit_tok = [&](int32_t t) { ids.push_back(t); };

    // ---- fallback: base model without ChatML specials (text only) ----
    if (im_start < 0 || im_end < 0) {
        for (const auto & m : messages) {
            if (has_image(m)) throw std::runtime_error("chat: model has no ChatML/vision tokens");
            emit_text(m.role + ": " + flat_text(m) + "\n");
        }
        if (opts.add_generation_prompt) emit_text("assistant: ");
        return out;
    }

    int image_counter = 0;

    // Render a user/system message's content (text + image parts). Applies the
    // template's |trim at the string edges (leading edge of the first text part,
    // trailing edge of the last).
    auto emit_content = [&](const ChatMessage & m, bool is_system) {
        struct Seg { bool is_text; std::string s; int img = -1; };
        std::vector<Seg> segs;
        if (m.parts.empty()) {
            segs.push_back({ true, m.content, -1 });
        } else {
            for (const auto & p : m.parts) {
                if (p.kind == ContentPart::Kind::Text) {
                    segs.push_back({ true, p.text, -1 });
                } else {
                    if (is_system) throw std::runtime_error("chat: system message cannot contain images");
                    if (vis_start < 0 || image_pad < 0 || vis_end < 0)
                        throw std::runtime_error("chat: model has no vision tokens (<|image_pad|>)");
                    ++image_counter;
                    if (opts.add_vision_id)
                        segs.push_back({ true, "Picture " + std::to_string(image_counter) + ": ", -1 });
                    segs.push_back({ false, "", p.image_index });
                }
            }
        }
        if (!segs.empty() && segs.front().is_text) segs.front().s = ltrim_ws(segs.front().s);
        if (!segs.empty() && segs.back().is_text)  segs.back().s  = rtrim_ws(segs.back().s);
        for (const auto & sg : segs) {
            if (sg.is_text) {
                emit_text(sg.s);
            } else {
                emit_tok(vis_start);
                ChatPrompt::ImageSpan span;
                span.first = (int) ids.size();
                span.count = opts.n_image_tokens > 0 ? opts.n_image_tokens : 1;
                span.image_index = sg.img;
                for (int r = 0; r < span.count; ++r) emit_tok(image_pad);
                out.image_spans.push_back(span);
                emit_tok(vis_end);
            }
        }
    };

    // ---- system header (with optional tools block) ----
    size_t msg_begin = 0;
    const bool has_system = !messages.empty() && messages[0].role == "system";
    if (has_system && has_image(messages[0]))
        throw std::runtime_error("chat: system message cannot contain images");
    if (!opts.tools_json.empty()) {
        emit_tok(im_start);
        emit_text("system\n");
        emit_text("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (const auto & t : opts.tools_json) emit_text("\n" + t);
        emit_text("\n</tools>");
        emit_text(TOOL_INSTRUCTIONS);
        if (has_system) {
            const std::string c = trim_ws(flat_text(messages[0]));
            if (!c.empty()) emit_text("\n\n" + c);
            msg_begin = 1;
        }
        emit_tok(im_end);
        emit_text("\n");
    } else if (has_system) {
        emit_tok(im_start);
        emit_text("system\n" + trim_ws(flat_text(messages[0])));
        emit_tok(im_end);
        emit_text("\n");
        msg_begin = 1;
    }

    // ---- last "real" user query index (tool responses don't count) ----
    // Past assistant turns keep their <think> block only after this index.
    int last_query_index = (int) messages.size() - 1;
    for (int i = (int) messages.size() - 1; i >= 0; --i) {
        if (messages[i].role != "user") continue;
        const std::string c = trim_ws(flat_text(messages[i]));
        if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) {
            last_query_index = i;
            break;
        }
    }

    // ---- main message loop ----
    for (size_t i = msg_begin; i < messages.size(); ++i) {
        const auto & m = messages[i];
        if (m.role == "system") {
            throw std::runtime_error("chat: system message must be at the beginning");
        } else if (m.role == "user") {
            emit_tok(im_start);
            emit_text("user\n");
            emit_content(m, false);
            emit_tok(im_end);
            emit_text("\n");
        } else if (m.role == "assistant") {
            if (has_image(m)) throw std::runtime_error("chat: assistant message cannot contain images");
            std::string content = trim_ws(flat_text(m));
            std::string reasoning;
            if (m.has_reasoning) {
                reasoning = m.reasoning_content;
            } else if (content.find("</think>") != std::string::npos) {
                // reasoning = before first </think> (inside the last <think>),
                // content = after the last </think>
                std::string head = content.substr(0, content.find("</think>"));
                while (!head.empty() && head.back() == '\n') head.pop_back();
                const size_t o = head.rfind("<think>");
                if (o != std::string::npos) head = head.substr(o + 7);
                while (!head.empty() && head.front() == '\n') head.erase(0, 1);
                reasoning = head;
                content = content.substr(content.rfind("</think>") + 8);
                while (!content.empty() && content.front() == '\n') content.erase(0, 1);
            }
            reasoning = trim_ws(reasoning);

            emit_tok(im_start);
            const bool keep_think =
                (opts.preserve_thinking || (int) i > last_query_index) &&
                think_open >= 0 && think_close >= 0;
            if (keep_think) {
                emit_text("assistant\n");
                emit_tok(think_open);
                emit_text("\n" + reasoning + "\n");
                emit_tok(think_close);
                emit_text("\n\n" + content);
            } else {
                emit_text("assistant\n" + content);
            }
            for (size_t k = 0; k < m.tool_calls.size(); ++k) {
                const auto & tc = m.tool_calls[k];
                if (k == 0 && content.empty()) emit_text("<tool_call>\n<function=" + tc.name + ">\n");
                else if (k == 0)               emit_text("\n\n<tool_call>\n<function=" + tc.name + ">\n");
                else                           emit_text("\n<tool_call>\n<function=" + tc.name + ">\n");
                for (const auto & arg : tc.arguments)
                    emit_text("<parameter=" + arg.first + ">\n" + arg.second + "\n</parameter>\n");
                emit_text("</function>\n</tool_call>");
            }
            emit_tok(im_end);
            emit_text("\n");
        } else if (m.role == "tool") {
            const bool prev_tool = i > msg_begin && messages[i-1].role == "tool";
            const bool next_tool = i + 1 < messages.size() && messages[i+1].role == "tool";
            if (!prev_tool) {
                emit_tok(im_start);
                emit_text("user");
            }
            emit_text("\n<tool_response>\n" + trim_ws(flat_text(m)) + "\n</tool_response>");
            if (!next_tool) {
                emit_tok(im_end);
                emit_text("\n");
            }
        } else {
            throw std::runtime_error("chat: unexpected message role '" + m.role + "'");
        }
    }

    // ---- generation prompt ----
    if (opts.add_generation_prompt) {
        emit_tok(im_start);
        emit_text("assistant\n");
        if (think_open >= 0) {
            if (!opts.reasoning && think_close >= 0) {
                emit_tok(think_open);
                emit_text("\n\n");
                emit_tok(think_close);
                emit_text("\n\n");
            } else {
                emit_tok(think_open);
                emit_text("\n");
            }
        }
    }
    return out;
}

} // namespace questwend
