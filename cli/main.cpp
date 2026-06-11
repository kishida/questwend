#include "model.h"
#include "tokenizer.h"
#include "runtime.h"
#include "sampler.h"
#include "chat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace qwencpp;

static void usage(const char * prog) {
    printf("usage: %s -m <model.gguf> [options]\n", prog);
    printf("  -p <text>      prompt (one-shot)\n");
    printf("  -i             interactive chat\n");
    printf("  --chat         wrap -p prompt in ChatML\n");
    printf("  --reasoning <on|off>  thinking mode for chat (default on)\n");
    printf("  --mtp          MTP self-speculative greedy decode (models with a nextn block)\n");
    printf("  --draft <N>    MTP draft length (tokens drafted per verify; default 1)\n");
    printf("  --embd-q8      use Q8_0 (not F16) for token embedding fallback (saves ~45%% VRAM)\n");
    printf("  -n <N>         max new tokens (default 128)\n");
    printf("  --n-ctx <N>    context length (default 4096)\n");
    printf("  --temp <f>     temperature (0 = greedy, default 0)\n");
    printf("  --top-p <f>    nucleus top-p (default 0.95)\n");
    printf("  --top-k <N>    top-k (default 40)\n");
    printf("  --seed <N>     RNG seed (0 = random)\n");
    printf("  --vram-budget <MB>  offload expert weights to CPU; keep non-expert on GPU\n");
    printf("  --cache-profile <f> persist/prefetch hot-expert profile (warm restarts)\n");
    printf("  --experts-ssd  stream experts from the GGUF on SSD (no RAM copy)\n");
    printf("  --cpu          force CPU backend\n");
    printf("  --log-tokens-per-sec   print speed\n");
    printf("  --info         print model info and exit\n");
}

int main(int argc, char ** argv) {
    std::string model_path, prompt, cache_profile;
    int  max_tokens = 128, n_ctx = 4096, n_draft = 1;
    size_t vram_budget_mb = 0;
    bool interactive = false, info_only = false, chat = false, log_speed = false, force_cpu = false;
    bool experts_ssd = false, reasoning = true, use_mtp = false, embd_q8 = false;
    SamplerConfig sc;
    sc.temperature = 0.0f;  // CLI defaults to greedy for reproducibility

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](){ return std::string(argv[++i]); };
        if      (a == "-m" && i + 1 < argc) model_path = next();
        else if (a == "-p" && i + 1 < argc) prompt = next();
        else if (a == "-i")                 interactive = true;
        else if (a == "--chat")             chat = true;
        else if (a == "-n" && i + 1 < argc) max_tokens = std::stoi(next());
        else if (a == "--n-ctx" && i + 1 < argc) n_ctx = std::stoi(next());
        else if (a == "--temp" && i + 1 < argc)  sc.temperature = std::stof(next());
        else if (a == "--top-p" && i + 1 < argc) sc.top_p = std::stof(next());
        else if (a == "--top-k" && i + 1 < argc) sc.top_k = std::stoi(next());
        else if (a == "--seed" && i + 1 < argc)  sc.seed = (uint32_t) std::stoul(next());
        else if (a == "--vram-budget" && i + 1 < argc) vram_budget_mb = (size_t) std::stoul(next());
        else if (a == "--cache-profile" && i + 1 < argc) cache_profile = next();
        else if (a == "--experts-ssd")      experts_ssd = true;
        else if (a == "--reasoning" && i + 1 < argc) { std::string v = next(); reasoning = (v != "off" && v != "0" && v != "false"); }
        else if (a == "--mtp")              use_mtp = true;
        else if (a == "--draft" && i + 1 < argc) n_draft = std::stoi(next());
        else if (a == "--embd-q8")          embd_q8 = true;
        else if (a == "--log-tokens-per-sec")    log_speed = true;
        else if (a == "--cpu")              force_cpu = true;
        else if (a == "--info")             info_only = true;
        else if (a == "-h" || a == "--help"){ usage(argv[0]); return 0; }
    }
    if (model_path.empty()) { usage(argv[0]); return 1; }

    try {
        auto model = Model::load(model_path);
        if (info_only) {
            printf("=== model ===\n%s\n%s", model->summary().c_str(), model->debug_dump().c_str());
            return 0;
        }
        fprintf(stderr, "%s", model->summary().c_str());

        Tokenizer tok(model->vocab());
        RuntimeConfig cfg;
        cfg.n_ctx          = n_ctx;
        cfg.use_cuda       = !force_cpu;
        cfg.vram_budget_mb = vram_budget_mb;
        cfg.cache_profile  = cache_profile;
        cfg.experts_ssd    = experts_ssd;
        // MTP needs the nextn block kept VRAM-resident (also the dev MTP test mode).
        cfg.use_mtp        = use_mtp || getenv("QWEN_MTP_TEST");
        cfg.embd_q8        = embd_q8;
        Runtime rt(*model, cfg);
        Sampler smp(sc);

        // GDN equivalence check: multi-token prefill vs token-by-token must give
        // the same final-token logits (build_graph multi-token GDN == single step).
        if (getenv("QWEN_GDN_TEST")) {
            std::string tp = prompt.empty()
                ? "The quick brown fox jumps over the lazy dog near the river bank."
                : prompt;
            auto ids = tok.encode(tp, false);
            auto argmax = [](const std::vector<float> & v) {
                int b = 0; for (int i = 1; i < (int) v.size(); ++i) if (v[i] > v[b]) b = i; return b;
            };
            rt.reset();
            std::vector<float> A = rt.decode(ids);          // one multi-token forward
            rt.reset();
            std::vector<float> B;
            for (int32_t t : ids) B = rt.decode({ t });     // token-by-token
            double md = 0, l2 = 0;
            for (size_t i = 0; i < A.size(); ++i) { double d = A[i] - B[i]; md = std::max(md, std::fabs(d)); l2 += d * d; }
            const int aA = argmax(A), aB = argmax(B);
            fprintf(stderr, "GDN test (%zu prompt tokens, %s):\n", ids.size(), arch_name(model->hparams().arch));
            fprintf(stderr, "  multi-token    argmax=%d '%s'\n", aA, tok.decode(aA).c_str());
            fprintf(stderr, "  token-by-token argmax=%d '%s'\n", aB, tok.decode(aB).c_str());
            fprintf(stderr, "  max|diff|=%.5g  L2=%.5g  -> %s\n",
                    md, std::sqrt(l2), aA == aB ? "ARGMAX MATCH" : "ARGMAX DIFFER");
            return 0;
        }

        // MTP draft acceptance: lockstep main + MTP draft, count how often the
        // draft (token two ahead) matches the main model's actual next token.
        if (getenv("QWEN_MTP_TEST")) {
            if (!rt.has_mtp()) { fprintf(stderr, "model has no MTP module\n"); return 1; }
            std::string tp = prompt.empty()
                ? "The quick brown fox jumps over the lazy dog near the river bank at sunset."
                : prompt;
            auto ids = tok.encode(tp, false);
            const int n_gen = max_tokens > 0 ? max_tokens : 64;
            auto argmax = [](const std::vector<float> & v) {
                int b = 0; for (int i = 1; i < (int) v.size(); ++i) if (v[i] > v[b]) b = i; return b;
            };
            rt.reset();
            std::vector<float> mlog;
            const int P = (int) ids.size();
            for (int i = 0; i < P; ++i) {
                mlog = rt.decode({ ids[i] });
                if (i + 1 < P) rt.mtp_draft(ids[i + 1]);     // build MTP KV over the prompt
            }
            int32_t x = argmax(mlog);                        // first generated token
            int matches = 0, total = 0;
            const int32_t eos = model->vocab().eos_id;
            for (int s = 0; s < n_gen; ++s) {
                int32_t d  = argmax(rt.mtp_draft(x));        // draft: token after x
                mlog = rt.decode({ x });
                int32_t nx = argmax(mlog);                   // main model's actual next token
                ++total; if (d == nx) ++matches;
                x = nx;
                if (x == eos) break;
            }
            fprintf(stderr, "MTP draft accept rate: %d/%d = %.1f%% (%d prompt tokens)\n",
                    matches, total, total ? 100.0 * matches / total : 0.0, P);
            return 0;
        }

        const int32_t eos    = model->vocab().eos_id;
        const int32_t im_end = tok.token_to_id("<|im_end|>");
        auto is_stop = [&](int32_t t){ return t == eos || (im_end >= 0 && t == im_end); };

        // Generate from a token sequence, streaming decoded text to stdout.
        using clk = std::chrono::steady_clock;
        auto secs = [](clk::duration d){ return std::chrono::duration<double>(d).count(); };
        auto run = [&](const std::vector<int32_t> & ids) {
            // template debugging: dump the rendered prompt (QWEN_DUMP_PROMPT=1)
            if (getenv("QWEN_DUMP_PROMPT")) {
                std::string s;
                for (int32_t t : ids) s += tok.decode(t);
                fprintf(stderr, "---- prompt (%zu tokens) ----\n%s\n---- end prompt ----\n",
                        ids.size(), s.c_str());
            }
            // MTP self-speculative greedy decode
            if (use_mtp && rt.has_mtp()) {
                int generated = 0;
                auto t0 = clk::now();
                rt.reset();
                rt.generate_mtp(ids, max_tokens, n_draft, [&](int32_t t) {
                    if (is_stop(t)) return false;
                    std::cout << tok.decode(t) << std::flush;
                    return ++generated < max_tokens;
                });
                std::cout << std::endl;
                if (log_speed) {
                    double s = secs(clk::now() - t0);
                    fprintf(stderr, "[mtp gen: %d tok in %.3fs = %.1f tok/s (incl. prefill)]\n",
                            generated, s, s > 0 ? generated / s : 0.0);
                }
                return;
            }

            // prefill (prompt processing)
            auto tp0 = clk::now();
            auto logits = rt.decode(ids);
            double prefill_s = secs(clk::now() - tp0);

            // token generation
            int generated = 0;
            auto tg0 = clk::now();
            for (int t = 0; t < max_tokens; ++t) {
                int next = smp.sample(logits);
                if (is_stop(next)) break;
                std::cout << tok.decode(next) << std::flush;
                ++generated;
                logits = rt.decode({ next });
            }
            double gen_s = secs(clk::now() - tg0);
            std::cout << std::endl;
            if (log_speed) {
                fprintf(stderr,
                    "[prefill: %zu tok in %.3fs = %.1f tok/s | gen: %d tok in %.3fs = %.1f tok/s]\n",
                    ids.size(), prefill_s, prefill_s > 0 ? ids.size() / prefill_s : 0.0,
                    generated, gen_s, gen_s > 0 ? generated / gen_s : 0.0);
            }
        };

        if (interactive) {
            std::vector<ChatMessage> history;
            fprintf(stderr, "interactive chat (empty line or Ctrl-C to quit)\n");
            std::string line;
            while (true) {
                std::cout << "\n> " << std::flush;
                if (!std::getline(std::cin, line) || line.empty()) break;
                history.push_back({ "user", line });
                auto ids = build_chatml_tokens(tok, history, true, reasoning);
                // reset KV so each turn re-encodes the full history (simple + correct)
                rt.reset();
                run(ids);
            }
        } else {
            std::vector<int32_t> ids;
            if (chat) {
                ids = build_chatml_tokens(tok, {{ "user", prompt }}, true, reasoning);
            } else {
                if (prompt.empty()) prompt = "Hello";
                std::cout << prompt;
                ids = tok.encode(prompt, false);
            }
            run(ids);
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
