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

// Minimal self-contained chat UI (served at GET /). Talks to the server's own
// OpenAI-compatible /v1/chat/completions endpoint with streaming.
static const char * CHAT_HTML = R"HTML(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>qwencpp chat</title>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#262b36; --txt:#e6e8ee; --muted:#8b93a7; --accent:#5b8cff; }
  * { box-sizing:border-box; }
  body { margin:0; height:100vh; display:flex; flex-direction:column; background:var(--bg); color:var(--txt);
         font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; }
  header b { font-weight:600; }
  header .model { color:var(--muted); font-size:13px; }
  header .spacer { flex:1; }
  header button, .ctl { background:var(--panel); color:var(--txt); border:1px solid var(--line);
         border-radius:8px; padding:6px 10px; font-size:13px; cursor:pointer; }
  header label { color:var(--muted); font-size:12px; display:flex; align-items:center; gap:6px; }
  header input[type=number] { width:64px; background:var(--bg); color:var(--txt); border:1px solid var(--line);
         border-radius:6px; padding:4px 6px; }
  #chat { flex:1; overflow-y:auto; padding:20px 0; }
  .wrap { max-width:780px; margin:0 auto; padding:0 16px; }
  .msg { display:flex; gap:12px; margin:14px 0; }
  .msg .who { width:30px; height:30px; border-radius:7px; flex:none; display:flex; align-items:center;
         justify-content:center; font-size:13px; font-weight:600; }
  .msg.user .who { background:#2a3550; color:#bcd0ff; }
  .msg.assistant .who { background:#23314a; color:#9fd6c0; }
  .msg .body { white-space:pre-wrap; word-wrap:break-word; padding-top:3px; }
  .msg .body.think { color:var(--muted); font-style:italic; }
  footer { border-top:1px solid var(--line); padding:12px 0 18px; }
  .inrow { max-width:780px; margin:0 auto; padding:0 16px; display:flex; gap:10px; align-items:flex-end; }
  textarea { flex:1; resize:none; background:var(--panel); color:var(--txt); border:1px solid var(--line);
         border-radius:10px; padding:10px 12px; font:inherit; max-height:200px; }
  .send { background:var(--accent); color:#fff; border:none; border-radius:10px; padding:10px 18px;
         font-size:15px; cursor:pointer; }
  .send:disabled { opacity:.5; cursor:default; }
  .hint { color:var(--muted); font-size:12px; text-align:center; margin-top:8px; }
</style>
</head>
<body>
<header>
  <b>qwencpp</b><span class="model" id="model">…</span>
  <span class="spacer"></span>
  <label>temp <input type="number" id="temp" value="0.7" step="0.1" min="0" max="2"></label>
  <label>max <input type="number" id="maxtok" value="512" step="64" min="1"></label>
  <label><input type="checkbox" id="think" checked> think</label>
  <button id="clear">Clear</button>
</header>
<div id="chat"><div class="wrap" id="list"></div></div>
<footer>
  <div class="inrow">
    <textarea id="input" rows="1" placeholder="Message…  (Enter to send, Shift+Enter for newline)"></textarea>
    <button class="send" id="send">Send</button>
  </div>
  <div class="hint">streams from /v1/chat/completions</div>
</footer>
<script>
const list = document.getElementById('list');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
let history = [];
let busy = false;

fetch('/v1/models').then(r=>r.json()).then(j=>{
  document.getElementById('model').textContent = (j.data && j.data[0] && j.data[0].id) || '';
}).catch(()=>{});

function addMsg(role, text){
  const m = document.createElement('div'); m.className = 'msg ' + role;
  const who = document.createElement('div'); who.className = 'who'; who.textContent = role==='user'?'You':'AI';
  const body = document.createElement('div'); body.className = 'body'; body.textContent = text;
  m.appendChild(who); m.appendChild(body); list.appendChild(m);
  document.getElementById('chat').scrollTop = 1e9;
  return body;
}
function autosize(){ input.style.height='auto'; input.style.height = Math.min(input.scrollHeight,200)+'px'; }
input.addEventListener('input', autosize);
input.addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
sendBtn.addEventListener('click', send);
document.getElementById('clear').addEventListener('click', ()=>{ history=[]; list.innerHTML=''; });

async function send(){
  const text = input.value.trim();
  if(!text || busy) return;
  busy = true; sendBtn.disabled = true;
  history.push({role:'user', content:text});
  addMsg('user', text);
  input.value=''; autosize();
  const body = addMsg('assistant', '');
  let acc = '';
  try {
    const resp = await fetch('/v1/chat/completions', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        messages: history, stream: true,
        temperature: parseFloat(document.getElementById('temp').value)||0,
        max_tokens: parseInt(document.getElementById('maxtok').value)||512,
        reasoning: document.getElementById('think').checked
      })
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      buf += dec.decode(value, {stream:true});
      let idx;
      while((idx = buf.indexOf('\n\n')) >= 0){
        const ev = buf.slice(0, idx); buf = buf.slice(idx+2);
        const line = ev.replace(/^data:\s*/, '');
        if(line === '[DONE]') continue;
        try {
          const j = JSON.parse(line);
          const piece = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
          if(piece){ acc += piece; renderAssistant(body, acc); document.getElementById('chat').scrollTop = 1e9; }
        } catch(e){}
      }
    }
  } catch(e){ acc += '\n[error: '+e+']'; renderAssistant(body, acc); }
  history.push({role:'assistant', content:acc});
  busy = false; sendBtn.disabled = false; input.focus();
}
// dim the reasoning span. With thinking on, the prompt pre-opens <think>, so the
// model's output is "reasoning…</think>\n\nanswer" (no leading <think> tag).
function renderAssistant(el, txt){
  el.innerHTML = '';
  let think = null, answer = txt;
  const close = txt.indexOf('</think>');
  if(close >= 0){
    think  = txt.slice(0, close).replace(/^\s*<think>/, '');
    answer = txt.slice(close + 8);
  } else if(/^\s*<think>/.test(txt)){
    think = txt.replace(/^\s*<think>/, ''); answer = '';
  }
  if(think !== null && think.trim().length){
    const t = document.createElement('span'); t.className = 'body think';
    t.textContent = think.replace(/^\n+/, ''); el.appendChild(t);
  }
  if(answer.length){
    el.appendChild(document.createTextNode(answer.replace(/^\n+/, '')));
  }
}
</script>
</body>
</html>
)HTML";

int main(int argc, char ** argv) {
    std::string model_path;
    int  port  = 8080;
    int  n_ctx = 4096;
    bool force_cpu = false;
    std::string host = "127.0.0.1";
    size_t vram_budget_mb = 0;
    std::string cache_profile;
    bool experts_ssd = false;
    bool reasoning_default = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--n-ctx" && i + 1 < argc) n_ctx = std::stoi(argv[++i]);
        else if (a == "--vram-budget" && i + 1 < argc) vram_budget_mb = (size_t) std::stoul(argv[++i]);
        else if (a == "--cache-profile" && i + 1 < argc) cache_profile = argv[++i];
        else if (a == "--experts-ssd")      experts_ssd = true;
        else if (a == "--reasoning" && i + 1 < argc) { std::string v = argv[++i]; reasoning_default = (v != "off" && v != "0" && v != "false"); }
        else if (a == "--cpu")              force_cpu = true;
    }
    if (model_path.empty()) {
        fprintf(stderr,
            "usage: %s -m <model.gguf> [options]\n"
            "  --port <N>          listen port (default 8080)\n"
            "  --host <addr>       bind address (default 127.0.0.1)\n"
            "  --n-ctx <N>         context length (default 4096)\n"
            "  --vram-budget <MB>  offload expert weights; keep non-expert on GPU\n"
            "  --cache-profile <f> prefetch hot-expert profile (read-only on the server)\n"
            "  --experts-ssd       stream experts from the GGUF on SSD (no RAM copy)\n"
            "  --reasoning <on|off> default thinking mode (per-request override: \"reasoning\")\n"
            "  --cpu               force CPU backend\n", argv[0]);
        return 1;
    }

    std::unique_ptr<Model> model;
    std::unique_ptr<Tokenizer> tok;
    std::unique_ptr<Runtime> rt;
    try {
        model = Model::load(model_path);
        tok   = std::make_unique<Tokenizer>(model->vocab());
        RuntimeConfig cfg;
        cfg.n_ctx              = n_ctx;
        cfg.use_cuda           = !force_cpu;
        cfg.vram_budget_mb     = vram_budget_mb;
        cfg.cache_profile      = cache_profile;
        cfg.cache_profile_save = false;   // server only reads the profile, never overwrites it
        cfg.experts_ssd        = experts_ssd;
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

    srv.Get("/", [reasoning_default](const httplib::Request &, httplib::Response & res) {
        std::string html = CHAT_HTML;
        // initialise the UI "think" checkbox from the server default
        if (!reasoning_default) {
            const std::string from = "id=\"think\" checked";
            auto p = html.find(from);
            if (p != std::string::npos) html.replace(p, from.size(), "id=\"think\"");
        }
        res.set_content(html, "text/html; charset=utf-8");
    });

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

        // per-request thinking control: body "reasoning" overrides the server default
        bool reasoning = reasoning_default;
        if (body.contains("reasoning")) reasoning = body["reasoning"].get<bool>();
        else if (body.contains("enable_thinking")) reasoning = body["enable_thinking"].get<bool>();
        auto prompt = build_chatml_tokens(*tok, messages, true, reasoning);

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
                    if (st->finished) return false;
                    auto finish = [&]() -> bool {
                        json fin = {{"id", id}, {"object", "chat.completion.chunk"},
                            {"model", model_id}, {"choices", json::array({
                                {{"index", 0}, {"delta", json::object()}, {"finish_reason", "stop"}}})}};
                        std::string ev = "data: " + fin.dump() + "\n\ndata: [DONE]\n\n";
                        sink.write(ev.data(), ev.size());
                        st->finished = true;
                        sink.done();
                        return false;
                    };
                    try {
                        if (!st->started) {
                            st->started = true;
                            rt->reset();
                            st->logits = rt->decode(st->prompt);
                        }
                        int next = st->smp.sample(st->logits);
                        // stop on EOS, token budget, or context limit (avoids overflow crash)
                        if (is_stop(next) || st->generated >= st->max_tokens || rt->n_past() + 1 >= n_ctx)
                            return finish();
                        std::string piece = tok->decode(next);
                        json chunk = {{"id", id}, {"object", "chat.completion.chunk"},
                            {"model", model_id}, {"choices", json::array({
                                {{"index", 0}, {"delta", {{"content", piece}}}, {"finish_reason", nullptr}}})}};
                        std::string ev = "data: " + chunk.dump() + "\n\n";
                        if (!sink.write(ev.data(), ev.size())) { st->finished = true; return false; }  // client gone
                        st->generated++;
                        st->logits = rt->decode({ next });
                        return true;
                    } catch (const std::exception & e) {
                        fprintf(stderr, "stream error: %s\n", e.what());
                        return finish();
                    }
                });
            return;
        }

        // non-streaming
        std::string text;
        int generated = 0;
        try {
            std::lock_guard<std::mutex> lk(infer_mtx);
            Sampler smp(sc);
            rt->reset();
            auto logits = rt->decode(prompt);
            for (int t = 0; t < max_tokens; ++t) {
                int next = smp.sample(logits);
                if (is_stop(next) || rt->n_past() + 1 >= n_ctx) break;
                text += tok->decode(next);
                ++generated;
                logits = rt->decode({ next });
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "generation error: %s\n", e.what());  // return what we have
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

    // generous timeouts: a single token (esp. SSD/offload prefill) can take a while
    srv.set_read_timeout(600, 0);
    srv.set_write_timeout(600, 0);
    srv.set_keep_alive_timeout(600);

    fprintf(stderr, "qwencpp server: http://%s:%d  (chat UI at /, model: %s)\n",
            host.c_str(), port, model_id.c_str());
    if (!srv.listen(host.c_str(), port)) {
        fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
