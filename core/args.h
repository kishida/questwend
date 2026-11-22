#pragma once

// Command-line options that qw-cli and qw-server share.
//
// Both binaries load the same model with the same runtime, so every option that
// describes the model, the GPUs or the expert offload has to mean exactly the
// same thing in both. Keeping one parser, one help text and one RuntimeConfig
// fill here is what makes that true by construction rather than by review.
// A binary's own options (--port, --chat, ...) stay in its own main.

#include <cstdio>
#include <string>
#include <vector>

#include "runtime.h"

namespace questwend {

struct CommonOptions {
    std::string model_path;                 // -m
    int         n_ctx = 0;                  // --n-ctx (0 = the model's trained length)
    bool        force_cpu = false;          // --cpu
    std::string cache_profile;              // --cache-profile
    std::string mmproj_path;                // --mmproj
    std::string ngram_mode = "disk";        // --ngram off|disk|ram
    size_t      ngram_cache_mb = 256;       // --ngram-cache <MB>
    bool        reasoning = true;           // --reasoning on|off
    bool        use_mtp = false;            // --mtp
    int         n_draft = 1;                // --draft <N>
    int         n_loops = 0;                // --loops <N> (0 = the model's own)
    bool        embd_q8 = false;            // --embd-q8
    float       repeat_penalty = 1.0f;      // --repeat-penalty <f>

    // Multi-GPU plan, as typed. build_gpu_plan() combines them (see apply()).
    std::vector<int>    gpu_ids;            // --gpus
    std::vector<float>  gpu_split;          // --gpu-split ("auto" leaves this empty)
    bool                gpu_split_given = false;
    std::vector<size_t> vram_list;          // --vram-budget, one entry per GPU

    // Expert tier: --experts ram|ssd|auto, with --experts-ram / --experts-ssd
    // as spellings of the first two. Every request is recorded; one place asked
    // for (any number of times) is that place, and anything else -- auto, or
    // two different places -- is auto, which picks by free RAM.
    bool experts_ram_req  = false;
    bool experts_ssd_req  = false;
    bool experts_auto_req = false;
    bool experts_ram_legacy = false;        // --experts-ram (deprecated spelling)
    bool experts_ssd_legacy = false;        // --experts-ssd (deprecated spelling)
    ExpertTier tier() const {
        if (experts_auto_req || experts_ram_req == experts_ssd_req) return ExpertTier::Auto;
        return experts_ram_req ? ExpertTier::Ram : ExpertTier::Ssd;
    }
};

// Consume the common option at argv[i], advancing `i` past its value.
//   Consumed - handled (the caller continues its loop)
//   Unknown  - not a common option, or one whose value is missing: the caller
//              tries its own flags and otherwise reports it
//   Error    - a common option whose value does not parse; the message is
//              already printed and the caller should exit non-zero
enum class ArgResult { Consumed, Unknown, Error };
ArgResult parse_common_arg(int argc, char ** argv, int & i, CommonOptions & o);

// Whole-command-line checks that a single flag cannot make (e.g. --experts-ssd
// together with --experts-ram). Prints what is wrong and returns false.
bool validate_common_options(const CommonOptions & o);

// Take the vision tower's VRAM out of the primary device's budget: it is built
// before the runtime, on the primary device, so its bytes are not available to
// the expert cache. Only explicit budgets need this -- an "auto" budget is
// measured after the tower is allocated and already excludes it.
void reserve_vision_tower_vram(CommonOptions & o, size_t tower_bytes);

// Fill the RuntimeConfig fields these options own. Returns false (after
// printing) when the GPU plan does not add up.
bool apply_common_options(const CommonOptions & o, RuntimeConfig & cfg);

// The shared part of --help: the model/GPU/offload options, then the offload
// tuning section. Both binaries print their own options first.
void print_common_help(FILE * out);

// Set one of the QWEN_* offload knobs (the flag and the env var are equivalent,
// the flag wins because it is applied after the environment is read).
void set_knob(const char * env, const char * val);

} // namespace questwend
