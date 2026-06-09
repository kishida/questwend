#pragma once

// Dynamic per-expert VRAM cache for MoE weight offloading.
//
// MoE models keep only a few experts active per token (e.g. 8 of 256), so the
// hot working set of expert weights is far smaller than the full set. This cache
// keeps pools of expert "slots" resident in VRAM and streams the rest from a
// slower tier (CPU RAM now; SSD later) on demand.
//
// The authoritative expert weights live elsewhere (CPU pinned buffer, owned by
// Model). This class manages VRAM-resident *copies* keyed by (layer,expert) and
// remaps logical expert ids to physical slot ids for ggml_mul_mat_id.
//
// Mixed quantization (e.g. unsloth "UD" quants) gives different layers different
// expert weight *types*, and a single ggml tensor holds exactly one type — so we
// keep one slot pool per (role, type, shape) signature. gate/up/down are pooled
// independently (separate residency), each layer routed to the pool matching its
// own tensor's signature.
//
// ensure() is the single choke point where a cache miss fetches expert bytes.
// To add an SSD tier (Phase C), replace the fetch source inside fetch_slab() with
// a pread from the on-disk weight blob — nothing else changes.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
struct ggml_context;
typedef struct ggml_backend *             ggml_backend_t;
typedef struct ggml_backend_buffer *      ggml_backend_buffer_t;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;

namespace qwencpp {

class Model;

class ExpertCache {
public:
    enum Role { GATE = 0, UP = 1, DOWN = 2, N_ROLE = 3 };

    // gpu_backend: where the slot pools live.
    // vram_avail_bytes: VRAM budget for the slot pools (resident fraction is
    //   derived so all experts share a uniform residency ratio).
    // ssd: if true, cache misses are streamed from the GGUF file on disk via
    //   pread (experts are not held in RAM); otherwise from the CPU expert tensors.
    ExpertCache(ggml_backend_t gpu_backend, Model & model,
                int n_layer, int n_expert, int n_used, size_t vram_avail_bytes,
                bool ssd = false);
    ~ExpertCache();

    // Make `layer`'s selected experts resident for all three roles, copying
    // misses into VRAM. Fills the three parallel slot-id arrays (length n each).
    void ensure(int layer, const int32_t * expert_ids, int n,
                int32_t * slot_gate, int32_t * slot_up, int32_t * slot_down);

    // Slot-pool tensor for a given layer/role (mul_mat_id weight arg).
    ggml_tensor * tensor(Role role, int layer) const;
    ggml_tensor * gate(int layer) const { return tensor(GATE, layer); }
    ggml_tensor * up(int layer)   const { return tensor(UP,   layer); }
    ggml_tensor * down(int layer) const { return tensor(DOWN, layer); }

    int n_expert() const { return n_expert_; }
    int min_slots() const;   // smallest pool capacity (bounds a safe prefill batch)

    // Is (layer,expert) currently resident in the role's pool?
    bool resident(Role role, int layer, int expert) const;

    // Dense "logical expert -> slot" row for (role,layer): n_expert entries,
    // 0 (sentinel) for non-resident experts. Used to fill the in-graph remap
    // table for the optimistic single-graph decode path.
    const int32_t * slot_of_row(Role role, int layer) const;

    // Record a resident access (fast-path hit): updates LRU/frequency/stats
    // without fetching. No-op if not resident.
    void touch(Role role, int layer, int expert);

    // Make all three roles of (layer,expert) resident, fetching any that are
    // absent (counts as misses). Used to repair a speculative single-graph miss.
    void ensure_resident(int layer, int expert);

    struct Stats { uint64_t hits = 0, misses = 0, evictions = 0;
                   double fetch_ms = 0; uint64_t fetch_bytes = 0; };
    const Stats & stats() const { return stats_; }

    // Persist the access-frequency profile (hot experts) for warm restarts, and
    // pre-fill the VRAM slots from a saved profile before generation starts.
    bool   save_profile(const std::string & path) const;
    size_t load_prefetch(const std::string & path);   // returns # experts prefetched

private:
    struct Pool {
        ggml_tensor * t = nullptr;          // [ne0, ne1, n_slots], one type
        int           role = 0;
        int           n_slots = 0;
        std::unordered_map<int, int> key2slot;  // key = layer*n_expert+expert
        std::vector<int>      slot2key;         // -1 = empty
        std::vector<uint64_t> clk;              // LRU timestamps
        std::unordered_map<int, uint64_t> count;  // access frequency per key
        std::vector<int32_t>  slot_of;          // key -> slot (0 sentinel if absent)
    };

    // A reserved-but-not-yet-loaded slot fetch (used for parallel SSD reads).
    struct FetchJob { Pool * pool; int role; int expert; int slot; };

    int  install(Pool & pool, int key, int layer, int expert, Role role); // fetch into a free/LRU slot
    int  slot_for(Pool & pool, int layer, int expert, Role role);
    int  reserve_victim(Pool & pool, int key);   // evict + claim a slot, no fetch
    void fetch_slab(Role role, int layer, int expert, ggml_tensor * dst, int slot);
    void fetch_parallel(int layer, std::vector<FetchJob> & jobs);  // SSD: parallel pread + serial H2D

    Model &        model_;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buf_ = nullptr;

    bool  ssd_  = false;
    int   prefetch_threads_ = 1;              // parallel SSD read workers
    std::vector<size_t>      foff_[N_ROLE];   // file offset of expert tensor (role,layer)
    std::vector<std::string> fpath_[N_ROLE];  // shard file of expert tensor (role,layer)
    std::unordered_map<std::string, void *> files_;  // path -> FILE* (lazy open)

    int n_layer_  = 0;
    int n_expert_ = 0;

    std::vector<Pool> pools_;
    // pool index for each (role, layer)
    std::vector<int>  layer_pool_[N_ROLE];

    uint64_t clock_ = 0;
    std::vector<uint8_t> stage_;   // host staging buffer for fetch (pageable fallback)

    // Pinned (page-locked) host staging buffer: the H2D source for a slab upload.
    // Bounded to one slab, so it is always allocatable (unlike pinning all experts)
    // yet gives pinned-DMA bandwidth. Falls back to `stage_` if the backend has no
    // host buffer type (non-CUDA). Allocated lazily on first fetch.
    ggml_backend_buffer_type_t host_buft_ = nullptr;  // pinned host buffer type (or null)
    ggml_backend_buffer_t      stage_buf_ = nullptr;  // pinned staging buffer
    void *                     stage_ptr_ = nullptr;  // base of stage_buf_
    size_t                     stage_cap_ = 0;        // bytes of stage_buf_
    void * stage_host(size_t nbytes);                 // ptr to >= nbytes pinned (or pageable) staging

    Stats stats_;
};

} // namespace qwencpp
