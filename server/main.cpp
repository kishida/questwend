// OpenAI-compatible inference server.
//   POST /v1/chat/completions  (streaming SSE + non-streaming)
//   GET  /v1/models
//   GET  /health

#include "model.h"
#include "tokenizer.h"
#include "runtime.h"
#include "sampler.h"
#include "chat.h"
#include "vision.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

using json = nlohmann::json;
using namespace qwencpp;

static std::string now_id(const char * prefix) {
    auto t = std::chrono::system_clock::now().time_since_epoch().count();
    return std::string(prefix) + std::to_string(t);
}

static std::vector<uint8_t> base64_decode(const std::string & in) {
    static int8_t tbl[256];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 256; ++i) tbl[i] = -1;
        const char * cs = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) tbl[(uint8_t) cs[i]] = (int8_t) i;
        init = true;
    }
    std::vector<uint8_t> out;
    out.reserve(in.size() * 3 / 4);
    int acc = 0, bits = 0;
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        const int8_t v = tbl[(uint8_t) c];
        if (v < 0) continue;
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t) ((acc >> bits) & 0xFF));
        }
    }
    return out;
}

// Parse Qwen3.6-format tool calls out of generated text:
//   <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>...\n</function>\n</tool_call>
// Returns the text with the tool-call blocks removed; parsed calls go to `calls`
// as (name, arguments-json-string). Reasoning (<think>...</think>) is skipped.
struct ParsedToolCall { std::string name; std::string arguments; };
static std::string parse_tool_calls(const std::string & text, std::vector<ParsedToolCall> & calls) {
    size_t scan = 0;
    const size_t think_end = text.rfind("</think>");
    if (think_end != std::string::npos) scan = think_end + 8;

    std::string out = text.substr(0, scan);
    size_t pos = scan;
    while (true) {
        const size_t tc = text.find("<tool_call>", pos);
        if (tc == std::string::npos) { out += text.substr(pos); break; }
        const size_t tce = text.find("</tool_call>", tc);
        if (tce == std::string::npos) { out += text.substr(pos); break; }
        out += text.substr(pos, tc - pos);
        const std::string block = text.substr(tc, tce - tc);

        ParsedToolCall call;
        const size_t fn = block.find("<function=");
        if (fn != std::string::npos) {
            const size_t fe = block.find('>', fn);
            if (fe != std::string::npos) {
                call.name = block.substr(fn + 10, fe - fn - 10);
                json args = json::object();
                size_t pp = fe;
                while (true) {
                    const size_t ps = block.find("<parameter=", pp);
                    if (ps == std::string::npos) break;
                    const size_t pe = block.find('>', ps);
                    const size_t pc = block.find("</parameter>", ps);
                    if (pe == std::string::npos || pc == std::string::npos) break;
                    const std::string key = block.substr(ps + 11, pe - ps - 11);
                    std::string val = block.substr(pe + 1, pc - pe - 1);
                    // values are wrapped in newlines by the format
                    if (!val.empty() && val.front() == '\n') val.erase(0, 1);
                    if (!val.empty() && val.back() == '\n')  val.pop_back();
                    // non-string JSON values were serialized as JSON; try to recover
                    json jv;
                    try {
                        jv = json::parse(val);
                        if (jv.is_string()) jv = val;   // quoted string stays raw
                    } catch (...) { jv = val; }
                    args[key] = jv;
                    pp = pc + 12;
                }
                call.arguments = args.dump();
                calls.push_back(std::move(call));
            }
        }
        pos = tce + 12;
    }
    // trim trailing whitespace left by block removal
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out;
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
  .stats { color:var(--muted); font-size:12px; margin-top:8px; font-variant-numeric:tabular-nums; }
  footer { border-top:1px solid var(--line); padding:12px 0 18px; }
  .inrow { max-width:780px; margin:0 auto; padding:0 16px; display:flex; gap:10px; align-items:flex-end; }
  textarea { flex:1; resize:none; background:var(--panel); color:var(--txt); border:1px solid var(--line);
         border-radius:10px; padding:10px 12px; font:inherit; max-height:200px; }
  .send { background:var(--accent); color:#fff; border:none; border-radius:10px; padding:10px 18px;
         font-size:15px; cursor:pointer; }
  .send:disabled { opacity:.5; cursor:default; }
  .attach { background:var(--panel); color:var(--txt); border:1px solid var(--line); border-radius:10px;
         padding:10px 12px; font-size:15px; cursor:pointer; }
  .hint { color:var(--muted); font-size:12px; text-align:center; margin-top:8px; }
  #thumbs { max-width:780px; margin:0 auto 6px; padding:0 16px; display:flex; gap:8px; }
  #thumbs img { height:56px; border-radius:8px; border:1px solid var(--line); cursor:pointer; }
  .msg .body img { max-height:160px; border-radius:8px; display:block; margin:4px 0; }
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
  <div id="thumbs"></div>
  <div class="inrow">
    <button class="attach" id="attach" title="attach image">&#128206;</button>
    <input type="file" id="file" accept="image/*" multiple style="display:none">
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

function addMsg(role, text, imgs){
  const m = document.createElement('div'); m.className = 'msg ' + role;
  const who = document.createElement('div'); who.className = 'who'; who.textContent = role==='user'?'You':'AI';
  const body = document.createElement('div'); body.className = 'body';
  if(imgs) for(const u of imgs){ const im = document.createElement('img'); im.src = u; body.appendChild(im); }
  if(text) body.appendChild(document.createTextNode(text));
  m.appendChild(who); m.appendChild(body); list.appendChild(m);
  document.getElementById('chat').scrollTop = 1e9;
  return body;
}

// pending image attachments (data URIs)
let pendingImgs = [];
const thumbs = document.getElementById('thumbs');
const fileInput = document.getElementById('file');
document.getElementById('attach').addEventListener('click', ()=> fileInput.click());
fileInput.addEventListener('change', ()=>{
  for(const f of fileInput.files){
    const r = new FileReader();
    r.onload = ()=>{ pendingImgs.push(r.result); renderThumbs(); };
    r.readAsDataURL(f);
  }
  fileInput.value = '';
});
function renderThumbs(){
  thumbs.innerHTML = '';
  pendingImgs.forEach((u, i)=>{
    const im = document.createElement('img'); im.src = u; im.title = 'click to remove';
    im.addEventListener('click', ()=>{ pendingImgs.splice(i,1); renderThumbs(); });
    thumbs.appendChild(im);
  });
}
function autosize(){ input.style.height='auto'; input.style.height = Math.min(input.scrollHeight,200)+'px'; }
input.addEventListener('input', autosize);
input.addEventListener('keydown', e=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
sendBtn.addEventListener('click', send);
document.getElementById('clear').addEventListener('click', ()=>{ history=[]; list.innerHTML=''; });

async function send(){
  const text = input.value.trim();
  if((!text && !pendingImgs.length) || busy) return;
  busy = true; sendBtn.disabled = true;
  let content = text;
  if(pendingImgs.length){
    content = pendingImgs.map(u=>({type:'image_url', image_url:{url:u}}));
    if(text) content.push({type:'text', text});
  }
  history.push({role:'user', content});
  addMsg('user', text, pendingImgs);
  pendingImgs = []; renderThumbs();
  input.value=''; autosize();
  const body = addMsg('assistant', '');
  let acc = '';
  let timings = null;
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
          if(j.timings) timings = j.timings;
          const piece = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
          if(piece){ acc += piece; renderAssistant(body, acc); document.getElementById('chat').scrollTop = 1e9; }
        } catch(e){}
      }
    }
  } catch(e){ acc += '\n[error: '+e+']'; renderAssistant(body, acc); }
  if(timings) showStats(body, timings);
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
function showStats(el, t){
  const s = document.createElement('div'); s.className = 'stats';
  const f = (x)=> (x||0).toFixed(1);
  s.textContent = `TTFT ${Math.round(t.ttft_ms||0)} ms · prefill ${f(t.prefill_tps)} tok/s `
    + `(${t.prompt_tokens} in) · gen ${f(t.gen_tps)} tok/s · ${t.completion_tokens} out`;
  el.appendChild(s);
  document.getElementById('chat').scrollTop = 1e9;
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
    bool use_mtp = false;
    bool embd_q8 = false;
    int  n_draft = 1;
    std::string mmproj_path;
    bool no_mmproj = false;

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
        else if (a == "--mtp")              use_mtp = true;
        else if (a == "--draft" && i + 1 < argc) n_draft = std::stoi(argv[++i]);
        else if (a == "--embd-q8")          embd_q8 = true;
        else if (a == "--mmproj" && i + 1 < argc) mmproj_path = argv[++i];
        else if (a == "--no-mmproj")        no_mmproj = true;
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
            "  --mtp               MTP self-speculative decode (models with a nextn block)\n"
            "  --draft <N>         MTP draft length (default 1)\n"
            "  --embd-q8           use Q8_0 (not F16) for embedding fallback (saves ~45%% VRAM)\n"
            "  --mmproj <gguf>     vision tower for image input (default: mmproj-*.gguf next to the model)\n"
            "  --no-mmproj         disable image input even if an mmproj file is present\n"
            "  --cpu               force CPU backend\n", argv[0]);
        return 1;
    }

    std::unique_ptr<Model> model;
    std::unique_ptr<Tokenizer> tok;
    std::unique_ptr<Runtime> rt;
    ggml_backend_t vis_backend = nullptr;
    std::unique_ptr<VisionEncoder> venc;
    std::mutex vis_mtx;   // the encoder graph is single-threaded
    try {
        model = Model::load(model_path);
        tok   = std::make_unique<Tokenizer>(model->vocab());

        // ---- vision tower (image input): explicit --mmproj or auto-discovery ----
        // Loaded BEFORE the runtime so its GPU footprint can be subtracted from
        // the expert-cache budget: a maxed-out --vram-budget sized for text-only
        // would otherwise exceed the backend memory limit once the tower is
        // added on top (Metal: kIOGPUCommandBufferCallbackErrorOutOfMemory).
        if (!no_mmproj) {
            if (mmproj_path.empty()) {
                std::string dir = model_path;
                const size_t sl = dir.find_last_of("/\\");
                dir = sl == std::string::npos ? "." : dir.substr(0, sl);
                try {
                    for (const auto & e : std::filesystem::directory_iterator(dir)) {
                        const std::string fn = e.path().filename().string();
                        if (fn.rfind("mmproj", 0) == 0 && fn.size() > 5 &&
                            fn.substr(fn.size() - 5) == ".gguf") {
                            mmproj_path = e.path().string();
                            break;
                        }
                    }
                } catch (...) {}
            }
            if (!mmproj_path.empty()) {
                try {
                    if (!force_cpu)
                        if (ggml_backend_dev_t d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))
                            vis_backend = ggml_backend_dev_init(d, nullptr);
                    if (!vis_backend) vis_backend = ggml_backend_cpu_init();
                    venc = VisionEncoder::load(mmproj_path, vis_backend);
                    if (venc->n_embd() != (int) model->hparams().n_embd) {
                        fprintf(stderr, "mmproj: projection dim mismatch, disabling image input\n");
                        venc.reset();
                    } else {
                        fprintf(stderr, "image input: ON (%s)\n", mmproj_path.c_str());
                    }
                } catch (const std::exception & e) {
                    fprintf(stderr, "mmproj load failed (%s), image input disabled\n", e.what());
                    venc.reset();
                }
            }
            if (venc && vram_budget_mb > 0) {
                const size_t vmb = (venc->gpu_bytes() + 1024 * 1024 - 1) / (1024 * 1024);
                const size_t cut = std::min(vram_budget_mb, vmb);
                fprintf(stderr, "vision tower: %zu MB GPU; expert cache budget %zu -> %zu MB"
                                " (--no-mmproj reclaims it for text-only runs)\n",
                        vmb, vram_budget_mb, vram_budget_mb - cut);
                vram_budget_mb -= cut;
            }
        }

        RuntimeConfig cfg;
        cfg.n_ctx              = n_ctx;
        cfg.use_cuda           = !force_cpu;
        cfg.vram_budget_mb     = vram_budget_mb;
        cfg.cache_profile      = cache_profile;
        cfg.cache_profile_save = false;   // server only reads the profile, never overwrites it
        cfg.experts_ssd        = experts_ssd;
        cfg.use_mtp            = use_mtp;  // keeps the nextn block VRAM-resident
        cfg.embd_q8            = embd_q8;
        rt = std::make_unique<Runtime>(*model, cfg);
    } catch (const std::exception & e) {
        fprintf(stderr, "load error: %s\n", e.what());
        return 1;
    }

    // MTP self-speculative decode is active only if requested AND the model has a
    // nextn block; otherwise fall back to plain decoding.
    const bool mtp = use_mtp && rt->has_mtp();
    if (use_mtp && !rt->has_mtp())
        fprintf(stderr, "warning: --mtp requested but model has no nextn block; using plain decode\n");
    if (mtp) fprintf(stderr, "MTP self-speculative decode ON (draft=%d)\n", n_draft);

    const std::string model_id = "qwencpp:" + std::string(arch_name(model->hparams().arch));
    const int32_t eos    = model->vocab().eos_id;
    const int32_t im_end = tok->token_to_id("<|im_end|>");
    auto is_stop = [&](int32_t t){ return t == eos || (im_end >= 0 && t == im_end); };

    std::mutex infer_mtx;   // runtime is single-threaded + stateful

    // Parse OpenAI-style messages: plain string content, multimodal content
    // arrays (text / image_url with base64 data URIs), assistant tool_calls,
    // reasoning_content, and tool-role messages. Image bytes go to `images`.
    auto parse_messages = [](const json & body, std::vector<std::vector<uint8_t>> & images) {
        std::vector<ChatMessage> msgs;
        if (!body.contains("messages")) return msgs;
        for (auto & m : body["messages"]) {
            ChatMessage cm;
            cm.role = m.value("role", "user");
            const auto & c = m.contains("content") ? m["content"] : json();
            if (c.is_string()) {
                cm.content = c.get<std::string>();
            } else if (c.is_array()) {
                for (auto & item : c) {
                    const std::string type = item.value("type", "");
                    if (type == "text" || item.contains("text")) {
                        cm.parts.push_back(ContentPart::make_text(item.value("text", "")));
                    } else if (type == "image_url" || item.contains("image_url")) {
                        std::string url = item.contains("image_url") && item["image_url"].is_object()
                            ? item["image_url"].value("url", "")
                            : item.value("image_url", "");
                        const size_t comma = url.find(',');
                        if (url.rfind("data:", 0) != 0 || comma == std::string::npos)
                            throw std::runtime_error("image_url must be a base64 data URI");
                        images.push_back(base64_decode(url.substr(comma + 1)));
                        cm.parts.push_back(ContentPart::make_image((int) images.size() - 1));
                    } else {
                        throw std::runtime_error("unsupported content part type: " + type);
                    }
                }
            }
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                cm.reasoning_content = m["reasoning_content"].get<std::string>();
                cm.has_reasoning = true;
            }
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                for (auto & tc : m["tool_calls"]) {
                    const auto & fn = tc.contains("function") ? tc["function"] : tc;
                    ToolCall call;
                    call.name = fn.value("name", "");
                    json args = json::object();
                    if (fn.contains("arguments")) {
                        if (fn["arguments"].is_string()) {
                            try { args = json::parse(fn["arguments"].get<std::string>()); } catch (...) {}
                        } else if (fn["arguments"].is_object()) {
                            args = fn["arguments"];
                        }
                    }
                    for (auto & kv : args.items())
                        call.arguments.push_back({ kv.key(),
                            kv.value().is_string() ? kv.value().get<std::string>() : kv.value().dump() });
                    cm.tool_calls.push_back(std::move(call));
                }
            }
            msgs.push_back(std::move(cm));
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

        const int  max_tokens = body.value("max_tokens", 512);
        const bool stream      = body.value("stream", false);
        const SamplerConfig sc = make_sampler(body);
        const std::string id   = now_id("chatcmpl-");

        // per-request thinking control: body "reasoning" overrides the server default
        bool reasoning = reasoning_default;
        if (body.contains("reasoning")) reasoning = body["reasoning"].get<bool>();
        else if (body.contains("enable_thinking")) reasoning = body["enable_thinking"].get<bool>();

        // ---- parse messages (text / images / tool calls) and build the prompt ----
        std::vector<std::vector<uint8_t>> images;
        std::vector<ChatMessage> messages;
        ChatPromptOptions copts;
        copts.reasoning = reasoning;
        ChatPrompt cp;
        auto vembs = std::make_shared<std::vector<std::vector<float>>>();
        std::vector<Runtime::EmbdOverride> ovr;
        try {
            messages = parse_messages(body, images);
            if (body.contains("tools") && body["tools"].is_array())
                for (auto & t : body["tools"]) copts.tools_json.push_back(t.dump());
            if (!images.empty()) {
                if (!venc) throw std::runtime_error("image input not available (no mmproj loaded)");
                copts.n_image_tokens = venc->n_image_tokens();
                copts.add_vision_id  = images.size() > 1;
            }
            cp = build_qwen_prompt(*tok, messages, copts);
            if (!images.empty()) {
                std::lock_guard<std::mutex> vl(vis_mtx);
                for (const auto & ib : images)
                    vembs->push_back(venc->encode_bytes(ib.data(), ib.size()));
            }
            for (const auto & sp : cp.image_spans)
                ovr.push_back({ sp.first, sp.count, (*vembs)[sp.image_index].data() });
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }
        const auto & prompt = cp.ids;
        const bool req_mtp = mtp;

        if (stream) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");

            using clk = std::chrono::steady_clock;
            struct State {
                std::unique_lock<std::mutex> lock;
                Sampler smp;
                std::vector<int32_t> prompt;
                std::vector<float> logits;
                std::string text;   // accumulated output (for tool-call parsing)
                std::shared_ptr<std::vector<std::vector<float>>> vembs;   // keep image embds alive
                std::vector<Runtime::EmbdOverride> ovr;
                int generated = 0, max_tokens = 0, prompt_tokens = 0;
                bool started = false, finished = false;
                clk::time_point t0, t_prefill, t_first;
                State(std::mutex & m, const SamplerConfig & sc) : lock(m), smp(sc) {}
            };
            auto st = std::make_shared<State>(infer_mtx, sc);
            st->prompt = prompt;
            st->max_tokens = max_tokens;
            st->prompt_tokens = (int) prompt.size();
            st->vembs = vembs;
            st->ovr   = std::move(ovr);

            res.set_chunked_content_provider("text/event-stream",
                [&, st, id, req_mtp, n_draft](size_t, httplib::DataSink & sink) -> bool {
                    if (st->finished) return false;
                    auto finish = [&]() -> bool {
                        const auto t_end = clk::now();
                        auto ms = [](clk::time_point a, clk::time_point b) {
                            return std::chrono::duration<double, std::milli>(b - a).count();
                        };
                        const double prefill_ms = ms(st->t0, st->t_prefill);
                        const double gen_ms     = ms(st->t_prefill, t_end);
                        json timings = {
                            {"prompt_tokens",     st->prompt_tokens},
                            {"completion_tokens", st->generated},
                            {"ttft_ms",     ms(st->t0, st->t_first)},
                            {"prefill_tps", prefill_ms > 0 ? st->prompt_tokens * 1000.0 / prefill_ms : 0.0},
                            {"gen_tps",     gen_ms     > 0 ? st->generated     * 1000.0 / gen_ms     : 0.0},
                        };
                        // tool calls: parse the accumulated text; if any, emit them as a
                        // final delta and finish with "tool_calls"
                        std::vector<ParsedToolCall> calls;
                        parse_tool_calls(st->text, calls);
                        std::string ev;
                        const char * fr = "stop";
                        if (!calls.empty()) {
                            json tcs = json::array();
                            for (size_t k = 0; k < calls.size(); ++k)
                                tcs.push_back({{"index", (int) k}, {"id", id + "-tc" + std::to_string(k)},
                                               {"type", "function"},
                                               {"function", {{"name", calls[k].name},
                                                             {"arguments", calls[k].arguments}}}});
                            json tchunk = {{"id", id}, {"object", "chat.completion.chunk"},
                                {"model", model_id}, {"choices", json::array({
                                    {{"index", 0}, {"delta", {{"tool_calls", tcs}}}, {"finish_reason", nullptr}}})}};
                            ev += "data: " + tchunk.dump() + "\n\n";
                            fr = "tool_calls";
                        }
                        json fin = {{"id", id}, {"object", "chat.completion.chunk"},
                            {"model", model_id}, {"timings", timings}, {"choices", json::array({
                                {{"index", 0}, {"delta", json::object()}, {"finish_reason", fr}}})}};
                        ev += "data: " + fin.dump() + "\n\ndata: [DONE]\n\n";
                        sink.write(ev.data(), ev.size());
                        st->finished = true;
                        sink.done();
                        return false;
                    };
                    try {
                        // ---- MTP self-speculative decode: stream the whole run in one
                        // blocking provider call, emitting an SSE chunk per token ----
                        if (req_mtp) {
                            if (st->started) return false;
                            st->started = true;
                            st->t0 = clk::now();
                            rt->reset();
                            if (!st->ovr.empty()) rt->set_embd_overrides(st->ovr);   // vision
                            st->t_prefill = st->t_first = st->t0;   // defaults if 0 tokens
                            bool first = true;
                            rt->generate_mtp(st->prompt, st->max_tokens, n_draft, [&](int32_t t) -> bool {
                                if (is_stop(t) || rt->n_past() + 1 >= n_ctx) return false;
                                if (first) { st->t_prefill = st->t_first = clk::now(); first = false; }
                                std::string piece = tok->decode(t);
                                st->text += piece;
                                json chunk = {{"id", id}, {"object", "chat.completion.chunk"},
                                    {"model", model_id}, {"choices", json::array({
                                        {{"index", 0}, {"delta", {{"content", piece}}}, {"finish_reason", nullptr}}})}};
                                std::string ev = "data: " + chunk.dump() + "\n\n";
                                if (!sink.write(ev.data(), ev.size())) return false;  // client gone
                                st->generated++;
                                return st->generated < st->max_tokens;
                            });
                            return finish();
                        }
                        if (!st->started) {
                            st->started = true;
                            st->t0 = clk::now();
                            rt->reset();
                            if (!st->ovr.empty()) rt->set_embd_overrides(st->ovr);   // vision
                            st->logits = rt->decode(st->prompt);
                            st->t_prefill = clk::now();
                            st->t_first   = st->t_prefill;   // overwritten on the first emitted token
                        }
                        int next = st->smp.sample(st->logits);
                        // stop on EOS, token budget, or context limit (avoids overflow crash)
                        if (is_stop(next) || st->generated >= st->max_tokens || rt->n_past() + 1 >= n_ctx)
                            return finish();
                        if (st->generated == 0) st->t_first = clk::now();
                        std::string piece = tok->decode(next);
                        st->text += piece;
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
        using clk = std::chrono::steady_clock;
        clk::time_point t0, t_prefill = {}, t_end = {};
        try {
            std::lock_guard<std::mutex> lk(infer_mtx);
            rt->reset();
            t0 = clk::now();
            if (req_mtp) {
                t_prefill = t0;
                if (!ovr.empty()) rt->set_embd_overrides(ovr);   // vision
                bool first = true;
                rt->generate_mtp(prompt, max_tokens, n_draft, [&](int32_t t) -> bool {
                    if (is_stop(t) || rt->n_past() + 1 >= n_ctx) return false;
                    if (first) { t_prefill = clk::now(); first = false; }
                    text += tok->decode(t);
                    ++generated;
                    return generated < max_tokens;
                });
            } else {
                Sampler smp(sc);
                if (!ovr.empty()) rt->set_embd_overrides(ovr);   // vision
                auto logits = rt->decode(prompt);
                t_prefill = clk::now();
                for (int t = 0; t < max_tokens; ++t) {
                    int next = smp.sample(logits);
                    if (is_stop(next) || rt->n_past() + 1 >= n_ctx) break;
                    text += tok->decode(next);
                    ++generated;
                    logits = rt->decode({ next });
                }
            }
            t_end = clk::now();
        } catch (const std::exception & e) {
            fprintf(stderr, "generation error: %s\n", e.what());  // return what we have
            if (t_end == clk::time_point{}) t_end = clk::now();
        }
        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        const double prefill_ms = ms(t0, t_prefill);
        const double gen_ms     = ms(t_prefill, t_end);
        // tool calls: parse the output; strip the blocks from content if any
        std::vector<ParsedToolCall> calls;
        const std::string content = parse_tool_calls(text, calls);
        json msg = {{"role", "assistant"}};
        msg["content"] = calls.empty() ? text : content;
        if (!calls.empty()) {
            json tcs = json::array();
            for (size_t k = 0; k < calls.size(); ++k)
                tcs.push_back({{"id", id + "-tc" + std::to_string(k)}, {"type", "function"},
                               {"function", {{"name", calls[k].name},
                                             {"arguments", calls[k].arguments}}}});
            msg["tool_calls"] = tcs;
        }
        json resp = {
            {"id", id}, {"object", "chat.completion"}, {"model", model_id},
            {"choices", json::array({
                {{"index", 0}, {"message", msg},
                 {"finish_reason", calls.empty() ? "stop" : "tool_calls"}}})},
            {"usage", {{"prompt_tokens", (int) prompt.size()},
                       {"completion_tokens", generated},
                       {"total_tokens", (int) prompt.size() + generated}}},
            {"timings", {{"prompt_tokens", (int) prompt.size()},
                         {"completion_tokens", generated},
                         {"ttft_ms", prefill_ms},
                         {"prefill_tps", prefill_ms > 0 ? prompt.size() * 1000.0 / prefill_ms : 0.0},
                         {"gen_tps", gen_ms > 0 ? generated * 1000.0 / gen_ms : 0.0}}}
        };
        res.set_content(resp.dump(), "application/json");
    });

    // generous timeouts: a single token (esp. SSD/offload prefill) can take a while
    srv.set_read_timeout(600, 0);
    srv.set_write_timeout(600, 0);
    srv.set_keep_alive_timeout(600);

    fprintf(stderr, "qwencpp server: http://%s:%d  (chat UI at /, model: %s)\n",
            host.c_str(), port, model_id.c_str());
    const bool ok = srv.listen(host.c_str(), port);
    if (!ok) fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
    venc.reset();
    if (vis_backend) ggml_backend_free(vis_backend);
    return ok ? 0 : 1;
}
