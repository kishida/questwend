#include "args.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace questwend {

void set_knob(const char * env, const char * val) {
#ifdef _WIN32
    _putenv_s(env, val);
#else
    setenv(env, val, /*overwrite=*/1);
#endif
}

ArgResult parse_common_arg(int argc, char ** argv, int & i, CommonOptions & o) {
    const std::string a = argv[i];
    // A flag that needs a value but has none is left Unknown, so the caller
    // reports "unknown or malformed argument: --n-ctx" rather than reading
    // past the end of argv.
    const bool has_val = i + 1 < argc;
    auto next = [&]() { return std::string(argv[++i]); };

    if      (a == "-m"        && has_val) o.model_path = next();
    else if (a == "--n-ctx"   && has_val) o.n_ctx = std::stoi(next());
    else if (a == "--cpu")                o.force_cpu = true;
    else if (a == "--mmproj"  && has_val) o.mmproj_path = next();
    else if (a == "--cache-profile" && has_val) o.cache_profile = next();
    else if (a == "--ngram"   && has_val) o.ngram_mode = next();
    else if (a == "--ngram-cache" && has_val) o.ngram_cache_mb = (size_t) std::stoul(next());
    else if (a == "--mtp")                o.use_mtp = true;
    else if (a == "--draft"   && has_val) o.n_draft = std::stoi(next());
    else if (a == "--embd-q8")            o.embd_q8 = true;
    else if (a == "--repeat-penalty" && has_val) o.repeat_penalty = std::stof(next());
    else if (a == "--reasoning" && has_val) {
        const std::string v = next();
        o.reasoning = (v != "off" && v != "0" && v != "false");
    }
    else if (a == "--experts-ssd")        o.experts_ssd_given = true;
    else if (a == "--experts-ram")        o.experts_ram_given = true;
    else if (a == "--gpus" && has_val) {
        const std::string v = next();
        if (!parse_gpu_list(v, o.gpu_ids)) {
            fprintf(stderr, "error: bad --gpus value: %s (expected e.g. 0 or 1,0 -- "
                            "distinct device indices, primary first)\n", v.c_str());
            return ArgResult::Error;
        }
    }
    else if (a == "--gpu-split" && has_val) {
        const std::string v = next();
        if (!parse_gpu_split(v, o.gpu_split)) {
            fprintf(stderr, "error: bad --gpu-split value: %s (expected auto, exact layer "
                            "counts like 36,4, or a ratio like 0.9,0.1)\n", v.c_str());
            return ArgResult::Error;
        }
        o.gpu_split_given = true;
    }
    else if (a == "--vram-budget" && has_val) {
        const std::string v = next();
        bool legacy = false;
        if (!parse_vram_budget_list(v, o.vram_list, &legacy)) {
            fprintf(stderr, "error: bad --vram-budget value: %s (expected auto, or e.g. 14, "
                            "13.5, 14G, 13500M, or one per GPU: 13.5,auto)\n", v.c_str());
            return ArgResult::Error;
        }
        if (legacy) {
            size_t mb = 0;
            for (size_t b : o.vram_list) if (b != VRAM_BUDGET_AUTO) mb += b;
            fprintf(stderr, "note: --vram-budget %s read as %zu MB (%.1f GB); the unit is now GB, "
                            "write %.1f or %sM to be explicit\n",
                    v.c_str(), mb, mb / 1024.0, mb / 1024.0, v.c_str());
        }
    }
    // ---- offload tuning: aliases for the QWEN_* env vars ----
    else if (a == "--resident-decode")    set_knob("QWEN_RESIDENT_DECODE", "1");
    else if (a == "--ssd-direct")         set_knob("QWEN_SSD_DIRECT", "1");
    else if (a == "--resident-refill" && has_val) set_knob("QWEN_RESIDENT_REFILL", next().c_str());
    else if (a == "--resident-warmup" && has_val) set_knob("QWEN_RESIDENT_WARMUP", next().c_str());
    else if (a == "--prefill-prune"   && has_val) set_knob("QWEN_PREFILL_PRUNE", next().c_str());
    else if (a == "--batch-chunk"     && has_val) set_knob("QWEN_BATCH_CHUNK", next().c_str());
    else if (a == "--expert-alloc"    && has_val) {
        const std::string v = next();
        if (v != "lru" && v != "quota" && v != "auto") {
            fprintf(stderr, "error: bad --expert-alloc value: %s (expected lru, quota or auto)\n",
                    v.c_str());
            return ArgResult::Error;
        }
        set_knob("QWEN_EXPERT_ALLOC", v.c_str());
    }
    else return ArgResult::Unknown;

    return ArgResult::Consumed;
}

bool validate_common_options(const CommonOptions & o) {
    if (o.experts_ssd_given && o.experts_ram_given) {
        fprintf(stderr, "error: --experts-ssd and --experts-ram name two different places for "
                        "the same weights; pass one (SSD is the default)\n");
        return false;
    }
    // On a lone GPU the "device off" reading of 0 has nowhere to move the model
    // to, so say so instead of silently treating it as "measure the free VRAM".
    if (o.vram_list.size() == 1 && o.vram_list[0] == 0 &&
        o.gpu_ids.size() <= 1 && !o.gpu_split_given) {
        fprintf(stderr, "error: --vram-budget 0 turns the only GPU off; use --cpu to run on the "
                        "CPU, or give the budget the GPU should keep to\n");
        return false;
    }
    return true;
}

void reserve_vision_tower_vram(CommonOptions & o, size_t tower_bytes) {
    if (o.vram_list.empty()) return;              // auto budget: measured after the tower
    size_t & primary = o.vram_list[0];
    if (primary == VRAM_BUDGET_AUTO) return;      // same, for the primary device
    const size_t vmb = (tower_bytes + 1024 * 1024 - 1) / (1024 * 1024);
    const size_t cut = std::min(primary, vmb);
    fprintf(stderr, "vision tower: %zu MB GPU; primary VRAM budget %zu -> %zu MB\n",
            vmb, primary, primary - cut);
    primary -= cut;
}

bool apply_common_options(const CommonOptions & o, RuntimeConfig & cfg) {
    cfg.n_ctx    = o.n_ctx;
    cfg.use_cuda = !o.force_cpu;
    if (!build_gpu_plan(o.gpu_ids, o.gpu_split, o.gpu_split_given, o.vram_list, cfg.gpus))
        return false;
    // Every device in a plan carries its own budget. Without a plan there is one
    // GPU and one budget, and "auto" (or no flag) leaves it 0 = measure it.
    cfg.vram_budget_mb = 0;
    if (cfg.gpus.size() <= 1 && !o.vram_list.empty() && o.vram_list[0] != VRAM_BUDGET_AUTO)
        cfg.vram_budget_mb = o.vram_list[0];
    cfg.cache_profile  = o.cache_profile;
    cfg.expert_tier    = o.tier();
    cfg.ngram_mode     = o.ngram_mode;
    cfg.ngram_cache_mb = o.ngram_cache_mb;
    cfg.use_mtp        = o.use_mtp;   // keeps the nextn block VRAM-resident
    cfg.embd_q8        = o.embd_q8;
    return true;
}

void print_common_help(FILE * out) {
    fprintf(out,
        "model and runtime:\n"
        "  -m <file>           model .gguf (required)\n"
        "  --n-ctx <N>         context length (default: the model's trained length)\n"
        "  --reasoning <on|off>  thinking mode (default on)\n"
        "  --repeat-penalty <f>  repetition penalty over the recent tokens (default 1 = off)\n"
        "  --mmproj <gguf>     vision tower for image input (default: mmproj-*.gguf next to the model)\n"
        "  --mtp               MTP self-speculative decode (models with a nextn block)\n"
        "  --draft <N>         MTP draft length (default 1)\n"
        "  --embd-q8           use Q8_0 (not F16) for the embedding fallback (saves ~45%% VRAM)\n"
        "  --cpu               force CPU backend\n"
        "GPU memory:\n"
        "  --vram-budget <GB>  VRAM the model may use (default: what the device has free)\n"
        "                      GB, fractions ok (13.5); suffix M/G to be explicit (13500M)\n"
        "                      one value per GPU to use several (e.g. 13.5,5)\n"
        "                      auto = that GPU's free VRAM (same as omitting it);\n"
        "                      0 = that GPU is not used at all (13.5,auto and 5g,0\n"
        "                      are both valid)\n"
        "  --gpus <list>       GPU device indices, primary first (e.g. 1, or 1,0).\n"
        "                      Naming two devices -- here or via --vram-budget --\n"
        "                      is what asks for two GPUs; without a --vram-budget\n"
        "                      each one uses all the VRAM it has free. --gpus and\n"
        "                      --vram-budget must name the same number of devices.\n"
        "  --gpu-split <spec>  each GPU's share of the layers (default: by\n"
        "                      --vram-budget, or by free VRAM when that is omitted)\n"
        "                      0.9,0.1 / 9,1 / 18,2 all mean the same ratio, and\n"
        "                      36,4 gives exactly 36 and 4 on a 40-layer model.\n"
        "                      A GPU given 0 holds no layers, so its whole budget\n"
        "                      becomes expert pool for the others.\n"
        "                      auto = every GPU present, share by free VRAM\n"
        "expert offload (MoE models whose weights exceed the budget; automatic):\n"
        "  --experts-ssd       stream the routed experts from the GGUF on disk. The\n"
        "                      default: it needs no RAM, and the pages the run\n"
        "                      touches end up in the OS page cache anyway\n"
        "  --experts-ram       hold them in host RAM instead. Worth it when they fit\n"
        "                      and the drive is slow; fails to allocate when they do not\n"
        "  --cache-profile <f> persist/prefetch the hot-expert profile for warm\n"
        "                      restarts (the server only ever reads the file)\n"
        "  --ngram <mode>      qwen4exp n-gram embedding: disk (default), ram or off.\n"
        "                      The table is 26.8 GB in Qwen3.8-Flash-Next and never\n"
        "                      goes on the GPU; off skips the module entirely\n"
        "  --ngram-cache <MB>  host cache for --ngram disk (default 256)\n"
        "offload tuning (equivalent to the QWEN_* env vars; flag wins):\n"
        "  --resident-decode   resident-only routing decode: fused graph, no per-token miss\n"
        "                      (lossy; auto-warmup + background refill keep quality)\n"
        "  --resident-refill <N>  refilled experts per token while masked, all layers combined (default 8; 0 = frozen)\n"
        "  --expert-alloc <m>  how decode splits the VRAM expert pool across layers:\n"
        "                      lru | quota (per-layer, from the prompt's routing) |\n"
        "                      auto = quota with --resident-decode, lru without (default)\n"
        "                      (prefill always uses the whole pool as one LRU stream)\n"
        "  --resident-warmup <N>  decode tokens before the mask locks in (default 32)\n"
        "  --prefill-prune <eps>  skip fetching low-router-mass experts in prefill (lossy; e.g. 0.05)\n"
        "  --batch-chunk <N>   prefill chunk length in tokens (default 4096)\n"
        "  --ssd-direct        unbuffered SSD reads (bypass the OS page cache; with --experts-ssd)\n");
}

} // namespace questwend
