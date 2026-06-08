// OpenAI-compatible inference server.
//   POST /v1/chat/completions  (streaming SSE + non-streaming)
//   GET  /v1/models
//   GET  /health

#include "model.h"
#include "tokenizer.h"
#include "runtime.h"
#include "sampler.h"
#include "chat.h"

#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>

using json = nlohmann::json;
using namespace qwencpp;

static std::string now_id(const char * prefix) {
    auto t = std::chrono::system_clock::now().time_since_epoch().count();
    return std::string(prefix) + std::to_string(t);
}

int main(int argc, char ** argv) {
    std::string model_path;
    int  port  = 8080;
    int  n_ctx = 4096;
    bool force_cpu = false;
    std::string host = "127.0.0.1";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--n-ctx" && i + 1 < argc) n_ctx = std::stoi(argv[++i]);
        else if (a == "--cpu")              force_cpu = true;
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m <model.gguf> [--port 8080] [--host 127.0.0.1]\n", argv[0]);
        return 1;
    }

    std::unique_ptr<Model> model;
    std::unique_ptr<Tokenizer> tok;
    std::unique_ptr<Runtime> rt;
    try {
        model = Model::load(model_path);
        tok   = std::make_unique<Tokenizer>(model->vocab());
        RuntimeConfig cfg; cfg.n_ctx = n_ctx; cfg.use_cuda = !force_cpu;
        rt = std::make_unique<Runtime>(*model, cfg);
    } catch (const std::exception & e) {
        fprintf(stderr, "load error: %s\n", e.what());
        return 1;
    }

    const std::string model_id = "qwencpp:" + std::string(arch_name(model->hparams().arch));
    const int32_t eos    = model->vocab().eos_id;
    const int32_t im_end = tok->token_to_id("<|im_end|>");
    auto is_stop = [&](int32_t t){ return t == eos || (im_end >= 0 && t == im_end); };

    std::mutex infer_mtx;   // runtime is single-threaded + stateful

    auto parse_messages = [](const json & body) {
        std::vector<ChatMessage> msgs;
        if (body.contains("messages")) {
            for (auto & m : body["messages"]) {
                msgs.push_back({ m.value("role", "user"), m.value("content", "") });
            }
        }
        return msgs;
    };
    auto make_sampler = [](const json & body) {
        SamplerConfig sc;
        sc.temperature = body.value("temperature", 0.7f);
        sc.top_p       = body.value("top_p", 0.95f);
        sc.top_k       = body.value("top_k", 40);
        sc.seed        = body.value("seed", 0);
        return sc;
    };

    httplib::Server srv;

    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    });

    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json j = {{"object", "list"}, {"data", json::array({
            {{"id", model_id}, {"object", "model"}, {"owned_by", "qwencpp"}}
        })}};
        res.set_content(j.dump(), "application/json");
    });

    srv.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) { res.status = 400; res.set_content(R"({"error":"invalid json"})", "application/json"); return; }

        const auto messages   = parse_messages(body);
        const int  max_tokens = body.value("max_tokens", 512);
        const bool stream      = body.value("stream", false);
        const SamplerConfig sc = make_sampler(body);
        const std::string id   = now_id("chatcmpl-");

        auto prompt = build_chatml_tokens(*tok, messages, true);

        if (stream) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");

            struct State {
                std::unique_lock<std::mutex> lock;
                Sampler smp;
                std::vector<int32_t> prompt;
                std::vector<float> logits;
                int generated = 0, max_tokens;
                bool started = false, finished = false;
                State(std::mutex & m, const SamplerConfig & sc) : lock(m), smp(sc) {}
            };
            auto st = std::make_shared<State>(infer_mtx, sc);
            st->prompt = prompt;
            st->max_tokens = max_tokens;

            res.set_chunked_content_provider("text/event-stream",
                [&, st, id](size_t, httplib::DataSink & sink) -> bool {
                    if (st->finished) { return false; }
                    if (!st->started) {
                        st->started = true;
                        rt->reset();
                        st->logits = rt->decode(st->prompt);
                    }
                    int next = st->smp.sample(st->logits);
                    if (is_stop(next) || st->generated >= st->max_tokens) {
                        json fin = {{"id", id}, {"object", "chat.completion.chunk"},
                            {"model", model_id}, {"choices", json::array({
                                {{"index", 0}, {"delta", json::object()}, {"finish_reason", "stop"}}})}};
                        std::string ev = "data: " + fin.dump() + "\n\ndata: [DONE]\n\n";
                        sink.write(ev.data(), ev.size());
                        st->finished = true;
                        sink.done();
                        return false;
                    }
                    std::string piece = tok->decode(next);
                    json chunk = {{"id", id}, {"object", "chat.completion.chunk"},
                        {"model", model_id}, {"choices", json::array({
                            {{"index", 0}, {"delta", {{"content", piece}}}, {"finish_reason", nullptr}}})}};
                    std::string ev = "data: " + chunk.dump() + "\n\n";
                    sink.write(ev.data(), ev.size());
                    st->generated++;
                    st->logits = rt->decode({ next });
                    return true;
                });
            return;
        }

        // non-streaming
        std::string text;
        int generated = 0;
        {
            std::lock_guard<std::mutex> lk(infer_mtx);
            Sampler smp(sc);
            rt->reset();
            auto logits = rt->decode(prompt);
            for (int t = 0; t < max_tokens; ++t) {
                int next = smp.sample(logits);
                if (is_stop(next)) break;
                text += tok->decode(next);
                ++generated;
                logits = rt->decode({ next });
            }
        }
        json resp = {
            {"id", id}, {"object", "chat.completion"}, {"model", model_id},
            {"choices", json::array({
                {{"index", 0}, {"message", {{"role", "assistant"}, {"content", text}}},
                 {"finish_reason", "stop"}}})},
            {"usage", {{"prompt_tokens", (int) prompt.size()},
                       {"completion_tokens", generated},
                       {"total_tokens", (int) prompt.size() + generated}}}
        };
        res.set_content(resp.dump(), "application/json");
    });

    fprintf(stderr, "qwencpp server: http://%s:%d  (model: %s)\n", host.c_str(), port, model_id.c_str());
    if (!srv.listen(host.c_str(), port)) {
        fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
