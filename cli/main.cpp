#include "model.h"
#include "tokenizer.h"
#include "runtime.h"
#include "sampler.h"
#include "chat.h"
#include "vision.h"
#include "args.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace questwend;

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <chrono>
// Raw unbuffered read benchmark of a file (--bench-read): sequential 16MB and
// random 0.6MB passes with FILE_FLAG_NO_BUFFERING — measures what the drive
// itself delivers to the expert-streaming path (no page cache, no ggml).
static int bench_read(const std::string & path) {
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
    if (h == INVALID_HANDLE_VALUE) { fprintf(stderr, "bench-read: cannot open %s\n", path.c_str()); return 1; }
    LARGE_INTEGER fsz; GetFileSizeEx(h, &fsz);
    const uint64_t fbytes = (uint64_t) fsz.QuadPart & ~4095ull;
    auto read_at = [&](void * buf, uint64_t off, size_t len) -> bool {
        OVERLAPPED ov{}; ov.Offset = (DWORD) (off & 0xffffffffull); ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD got = 0;
        return ReadFile(h, buf, (DWORD) len, &got, &ov) && got == len;
    };
    const size_t SEQ = 16u << 20;
    void * buf = VirtualAlloc(nullptr, SEQ, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

    auto seq_pass = [&](uint64_t start, uint64_t len, const char * name) {
        start &= ~4095ull;
        auto t0 = std::chrono::steady_clock::now();
        for (uint64_t off = start; off + SEQ <= start + len; off += SEQ)
            if (!read_at(buf, off, SEQ)) { fprintf(stderr, "bench-read: sequential read failed\n"); exit(1); }
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("sequential %s : %.1f GB in %.1fs = %.2f GB/s\n",
               name, len / 1073741824.0, s, len / 1073741824.0 / s);
    };
    const uint64_t quarter = std::min(fbytes / 2, 4ull << 30);
    seq_pass(0, quarter, "head");
    seq_pass(fbytes - quarter, quarter, "tail");

    const size_t RND = 640 * 1024;   // ~one expert slab, sector-rounded
    const uint64_t rtotal = 4ull << 30;
    uint64_t seed = 0x243F6A8885A308D3ull, done = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (done < rtotal) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const uint64_t off = (seed % ((fbytes - RND) >> 12)) << 12;   // 4KB-aligned
        if (!read_at(buf, off, RND)) { fprintf(stderr, "bench-read: random read failed\n"); return 1; }
        done += RND;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("random   0.6MB : %.1f GB in %.1fs = %.2f GB/s\n",
           rtotal / 1073741824.0, s, rtotal / 1073741824.0 / s);
    CloseHandle(h);
    return 0;
}
#endif

static void usage(const char * prog) {
    printf("usage: %s -m <model.gguf> [options]\n", prog);
    printf("prompt and sampling:\n");
    printf("  -p <text>           prompt (one-shot)\n");
    printf("  -i                  interactive chat\n");
    printf("  --chat              wrap the -p prompt in ChatML\n");
    printf("  --image <path>      attach an image to the prompt (needs a VL model + mmproj)\n");
    printf("  -n <N>              max new tokens (default 128)\n");
    printf("  --temp <f>          temperature (0 = greedy, default 0)\n");
    printf("  --top-p <f>         nucleus top-p (default 0.95)\n");
    printf("  --top-k <N>         top-k (default 40)\n");
    printf("  --repeat-last-n <N> repetition-penalty window (default 64)\n");
    printf("  --seed <N>          RNG seed (0 = random)\n");
    printf("diagnostics:\n");
    printf("  --info              print model info and exit\n");
    printf("  --log-tokens-per-sec  print speed\n");
    printf("  --dump-logits <f>   write the prompt's final-token logits to <f> (verification)\n");
    printf("  --vision-test       encode --image with --mmproj and print embedding stats (no LLM)\n");
#ifdef _WIN32
    printf("  --bench-read <file> raw unbuffered read benchmark (sequential head/tail + random) and exit\n");
#endif
    print_common_help(stdout);
}

int main(int argc, char ** argv) {
    CommonOptions opt;                  // -m, --n-ctx, GPU/offload, ... (core/args.h)
    std::string prompt;
    std::vector<std::string> image_paths;
    int  max_tokens = 128;
    std::string dump_logits;            // --dump-logits <file>
    bool interactive = false, info_only = false, chat = false, log_speed = false;
    bool vision_test = false;
    SamplerConfig sc;
    sc.temperature = 0.0f;  // CLI defaults to greedy for reproducibility

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](){ return std::string(argv[++i]); };
        if      (a == "-p" && i + 1 < argc) prompt = next();
        else if (a == "-i")                 interactive = true;
        else if (a == "--chat")             chat = true;
        else if (a == "-n" && i + 1 < argc) max_tokens = std::stoi(next());
        else if (a == "--temp" && i + 1 < argc)  sc.temperature = std::stof(next());
        else if (a == "--top-p" && i + 1 < argc) sc.top_p = std::stof(next());
        else if (a == "--top-k" && i + 1 < argc) sc.top_k = std::stoi(next());
        else if (a == "--seed" && i + 1 < argc)  sc.seed = (uint32_t) std::stoul(next());
        else if (a == "--repeat-last-n" && i + 1 < argc)  sc.repeat_last_n = std::stoi(next());
        else if (a == "--image" && i + 1 < argc)  image_paths.push_back(next());
        else if (a == "--vision-test")      vision_test = true;
        else if (a == "--log-tokens-per-sec")    log_speed = true;
#ifdef _WIN32
        else if (a == "--bench-read" && i + 1 < argc) return bench_read(next());
#endif
        else if (a == "--dump-logits" && i + 1 < argc) dump_logits = next();
        else if (a == "--info")             info_only = true;
        else if (a == "-h" || a == "--help"){ usage(argv[0]); return 0; }
        else {
            const ArgResult r = parse_common_arg(argc, argv, i, opt);
            if (r == ArgResult::Error) return 1;
            if (r == ArgResult::Unknown) {
                fprintf(stderr, "error: unknown or malformed argument: %s\n\n", a.c_str());
                usage(argv[0]);
                return 1;
            }
        }
    }
    if (!validate_common_options(opt)) return 1;
    // The CLI samples in-process, so --repeat-penalty is a sampler setting here.
    sc.repeat_penalty = opt.repeat_penalty;
    const std::string & model_path = opt.model_path;
    // standalone vision-tower test: encode images and print embedding stats
    // (for numeric comparison against the Java reference implementation)
    if (vision_test) {
        if (opt.mmproj_path.empty() || image_paths.empty()) {
            fprintf(stderr, "--vision-test requires --mmproj <gguf> and --image <path>\n");
            return 1;
        }
        try {
            ggml_backend_t be = nullptr;
            if (!opt.force_cpu)
                for (ggml_backend_dev_t d : gpu_devices())
                    if ((be = ggml_backend_dev_init(d, nullptr))) break;
            if (!be) be = ggml_backend_cpu_init();
            {
                auto enc = VisionEncoder::load(opt.mmproj_path, be);
                const int D = enc->n_embd(), T = enc->n_image_tokens();
                for (const auto & ip : image_paths) {
                    auto t0 = std::chrono::steady_clock::now();
                    auto emb = enc->encode_image(ip);
                    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
                    double sum = 0, sq = 0;
                    for (float v : emb) { sum += v; sq += (double) v * v; }
                    printf("%s: %d tokens x %d dim in %.0f ms | mean %.6f rms %.6f\n",
                           ip.c_str(), T, D, ms, sum / emb.size(), std::sqrt(sq / emb.size()));
                    printf("  tok0   [0..7]:");
                    for (int i = 0; i < 8; ++i) printf(" % .5f", emb[i]);
                    printf("\n  tok%-4d[0..7]:", T - 1);
                    for (int i = 0; i < 8; ++i) printf(" % .5f", emb[(size_t)(T - 1) * D + i]);
                    printf("\n");
                }
            }
            ggml_backend_free(be);
        } catch (const std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    if (model_path.empty()) { usage(argv[0]); return 1; }

    try {
        auto model = Model::load(model_path);
        if (info_only) {
            printf("=== model ===\n%s\n%s", model->summary().c_str(), model->debug_dump().c_str());
            return 0;
        }
        fprintf(stderr, "%s", model->summary().c_str());

        // Context length defaults to what the model was trained for. The KV
        // cache is sized from it, so cap it with --n-ctx if it does not fit.
        if (opt.n_ctx <= 0) {
            opt.n_ctx = (int) model->hparams().n_ctx_train;
            fprintf(stderr, "context: %d tokens (model's trained length; --n-ctx to change)\n", opt.n_ctx);
        }

        Tokenizer tok(model->vocab());

        // ---- vision: load the mmproj tower and encode the images up front ----
        // Done BEFORE the runtime so the tower's GPU footprint can be subtracted
        // from the expert-cache budget (a maxed-out --vram-budget sized for
        // text-only would otherwise exceed the backend memory limit).
        ggml_backend_t vis_backend = nullptr;
        std::unique_ptr<VisionEncoder> venc;
        std::vector<std::vector<float>> vembs;
        if (!image_paths.empty()) {
            if (opt.mmproj_path.empty()) {
                // auto-discover mmproj-*.gguf next to the model
                std::string dir = model_path;
                const size_t sl = dir.find_last_of("/\\");
                dir = sl == std::string::npos ? "." : dir.substr(0, sl);
                for (const auto & e : std::filesystem::directory_iterator(dir)) {
                    const std::string fn = e.path().filename().string();
                    if (fn.rfind("mmproj", 0) == 0 && fn.size() > 5 &&
                        fn.substr(fn.size() - 5) == ".gguf") {
                        opt.mmproj_path = e.path().string();
                        break;
                    }
                }
                if (opt.mmproj_path.empty())
                    throw std::runtime_error("--image given but no mmproj found (use --mmproj)");
                fprintf(stderr, "mmproj: using %s\n", opt.mmproj_path.c_str());
            }
            if (!opt.force_cpu)
                for (ggml_backend_dev_t d : gpu_devices())
                    if ((vis_backend = ggml_backend_dev_init(d, nullptr))) break;
            if (!vis_backend) vis_backend = ggml_backend_cpu_init();
            venc = VisionEncoder::load(opt.mmproj_path, vis_backend);
            if (venc->n_embd() != (int) model->hparams().n_embd)
                throw std::runtime_error("mmproj projection dim does not match the model n_embd");
            for (const auto & ip : image_paths) {
                auto t0 = std::chrono::steady_clock::now();
                vembs.push_back(venc->encode_image(ip));
                fprintf(stderr, "image: %s encoded to %d tokens in %.0f ms\n", ip.c_str(),
                        venc->n_image_tokens(),
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count());
            }
            chat = true;   // images imply chat formatting
            reserve_vision_tower_vram(opt, venc->gpu_bytes());
        }

        RuntimeConfig cfg;
        if (!apply_common_options(opt, cfg)) return 1;
        // MTP needs the nextn block kept VRAM-resident (also the dev MTP test mode).
        cfg.use_mtp = opt.use_mtp || getenv("QWEN_MTP_TEST");
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
            if (opt.use_mtp && rt.has_mtp()) {
                int generated = 0;
                auto t0 = clk::now();
                rt.reset();
                rt.generate_mtp(ids, max_tokens, opt.n_draft, [&](int32_t t) {
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

            // --dump-logits: the prompt's final-token logits, one "i: value" per
            // line -- the same shape llama-debug --save-logits writes, so the two
            // can be diffed directly (see tests/qwen4/compare_logits.py).
            if (!dump_logits.empty()) {
                FILE * f = fopen(dump_logits.c_str(), "w");
                if (!f) {
                    fprintf(stderr, "error: cannot write %s\n", dump_logits.c_str());
                    return;
                }
                for (size_t i = 0; i < logits.size(); ++i) {
                    fprintf(f, "%zu: %g\n", i, logits[i]);
                }
                fclose(f);
                fprintf(stderr, "logits (%zu) -> %s\n", logits.size(), dump_logits.c_str());
            }

            // token generation
            smp.prime(ids);   // repetition-penalty window (prompt tail)
            int generated = 0;
            auto tg0 = clk::now();
            for (int t = 0; t < max_tokens; ++t) {
                int next = smp.sample(logits);
                smp.accept(next);
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
                auto ids = build_chatml_tokens(tok, history, true, opt.reasoning);
                // reset KV so each turn re-encodes the full history (simple + correct)
                rt.reset();
                run(ids);
            }
        } else {
            std::vector<int32_t> ids;
            if (chat && !image_paths.empty()) {
                // multimodal one-shot: [images..., text] in a single user message
                ChatMessage m;
                m.role = "user";
                for (int i = 0; i < (int) image_paths.size(); ++i)
                    m.parts.push_back(ContentPart::make_image(i));
                if (!prompt.empty()) m.parts.push_back(ContentPart::make_text(prompt));
                ChatPromptOptions o;
                o.reasoning      = opt.reasoning;
                o.n_image_tokens = venc->n_image_tokens();
                o.add_vision_id  = image_paths.size() > 1;
                auto cp = build_qwen_prompt(tok, { m }, o);
                std::vector<Runtime::EmbdOverride> ovr;
                for (const auto & sp : cp.image_spans)
                    ovr.push_back({ sp.first, sp.count, vembs[sp.image_index].data() });
                rt.set_embd_overrides(std::move(ovr));
                ids = cp.ids;
            } else if (chat) {
                ids = build_chatml_tokens(tok, {{ "user", prompt }}, true, opt.reasoning);
            } else {
                if (prompt.empty()) prompt = "Hello";
                std::cout << prompt;
                ids = tok.encode(prompt, false);
            }
            run(ids);
        }
        venc.reset();
        if (vis_backend) ggml_backend_free(vis_backend);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
