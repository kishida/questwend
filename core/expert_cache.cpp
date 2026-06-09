#include "expert_cache.h"
#include "model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace qwencpp {

static const char * role_fmt(ExpertCache::Role r) {
    switch (r) {
        case ExpertCache::GATE: return "blk.%d.ffn_gate_exps.weight";
        case ExpertCache::UP:   return "blk.%d.ffn_up_exps.weight";
        default:                return "blk.%d.ffn_down_exps.weight";
    }
}

static ggml_tensor * role_tensor(Model & m, ExpertCache::Role r, int layer) {
    char name[256];
    snprintf(name, sizeof(name), role_fmt(r), layer);
    return m.tensor(name);
}

ExpertCache::ExpertCache(ggml_backend_t gpu_backend, Model & model,
                         int n_layer, int n_expert, int n_used, size_t vram_avail_bytes,
                         bool ssd)
    : model_(model), ssd_(ssd), n_layer_(n_layer), n_expert_(n_expert) {

    backend_ = gpu_backend;
    if (getenv("QWEN_SYNC_FETCH")) async_fetch_ = false;   // A/B: force synchronous H2D

    // Pinned host staging buffer type (CUDA): makes the slab H2D a pinned-DMA copy.
    if (ggml_backend_dev_t dev = ggml_backend_get_device(gpu_backend))
        host_buft_ = ggml_backend_dev_host_buffer_type(dev);

    // SSD tier: precompute each expert tensor's shard file + offset (files are
    // opened lazily on first fetch; sharded models spread experts across files).
    if (ssd_) {
        prefetch_threads_ = 8;
        if (const char * e = getenv("QWEN_PREFETCH_THREADS")) {
            int v = atoi(e);
            if (v >= 1 && v <= 64) prefetch_threads_ = v;
        }
        for (int r = 0; r < N_ROLE; ++r) {
            foff_[r].assign(n_layer, 0);
            fpath_[r].assign(n_layer, std::string());
            for (int il = 0; il < n_layer; ++il) {
                char name[256];
                snprintf(name, sizeof(name), role_fmt((Role) r), il);
                foff_[r][il]  = model.tensor_file_offset(name);
                fpath_[r][il] = model.tensor_file(name);
            }
        }
    }

    // ---- discover (role,type,shape) signatures and route each layer ----
    // Role MUST be part of the signature: gate/up share type & shape but are
    // distinct weights, so they must never land in the same pool.
    struct Sig { int role; int type; int64_t ne0, ne1; size_t slab; int n_layers; };
    std::vector<Sig> sigs;
    for (int r = 0; r < N_ROLE; ++r) {
        layer_pool_[r].assign(n_layer, -1);
        for (int il = 0; il < n_layer; ++il) {
            ggml_tensor * t = role_tensor(model, (Role) r, il);
            if (!t) throw std::runtime_error("ExpertCache: missing expert tensor (role " +
                                             std::to_string(r) + ", layer " + std::to_string(il) + ")");
            int idx = -1;
            for (int s = 0; s < (int) sigs.size(); ++s) {
                if (sigs[s].role == r && sigs[s].type == (int) t->type &&
                    sigs[s].ne0 == t->ne[0] && sigs[s].ne1 == t->ne[1]) {
                    idx = s; break;
                }
            }
            if (idx < 0) {
                idx = (int) sigs.size();
                sigs.push_back({ r, (int) t->type, t->ne[0], t->ne[1], t->nb[2], 0 });
            }
            sigs[idx].n_layers++;
            layer_pool_[r][il] = idx;
        }
    }

    // ---- size each pool: uniform residency fraction across all experts ----
    size_t total_bytes = 0;
    for (auto & s : sigs) total_bytes += s.slab * (size_t) s.n_layers * n_expert;
    double frac = total_bytes ? (double) vram_avail_bytes / (double) total_bytes : 1.0;
    if (frac > 1.0) frac = 1.0;

    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * sigs.size() + 256;
    p.no_alloc = true;
    ctx_ = ggml_init(p);

    pools_.resize(sigs.size());
    for (int s = 0; s < (int) sigs.size(); ++s) {
        int cap = sigs[s].n_layers * n_expert;
        int n_slots = (int) (frac * cap);
        // Correctness floor: one ensure() needs n_used distinct slots in a pool.
        // Fewer would thrash within a single layer and corrupt the matmul.
        if (n_slots < n_used) n_slots = n_used;
        if (n_slots > cap)    n_slots = cap;

        ggml_tensor * t = ggml_new_tensor_3d(ctx_, (ggml_type) sigs[s].type,
                                             sigs[s].ne0, sigs[s].ne1, n_slots);
        char nm[64]; snprintf(nm, sizeof(nm), "expcache.pool%d", s);
        ggml_set_name(t, nm);

        pools_[s].t = t;
        pools_[s].role = sigs[s].role;
        pools_[s].n_slots = n_slots;
        pools_[s].slot2key.assign(n_slots, -1);
        pools_[s].clk.assign(n_slots, 0);
        pools_[s].slot_of.assign((size_t) n_layer * n_expert, 0);
    }

    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, gpu_backend);
    if (!buf_) throw std::runtime_error("ExpertCache: failed to alloc VRAM slot pools");
    ggml_backend_buffer_set_usage(buf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    size_t resident = 0;
    for (int s = 0; s < (int) sigs.size(); ++s)
        resident += sigs[s].slab * (size_t) pools_[s].n_slots;
    fprintf(stderr,
            "expert cache (dynamic): %zu pools, %.0f%% resident = %.1f MB VRAM "
            "(of %.1f MB experts, %d layers x %d experts)\n",
            sigs.size(), frac * 100.0, resident / 1024.0 / 1024.0,
            total_bytes / 1024.0 / 1024.0, n_layer, n_expert);
}

ExpertCache::~ExpertCache() {
    for (auto & kv : files_) if (kv.second) fclose((FILE *) kv.second);
    if (stage_buf_) ggml_backend_buffer_free(stage_buf_);
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

// Return a host pointer to at least `nbytes` of staging memory. Prefers a pinned
// (page-locked) buffer for fast H2D; falls back to a pageable vector if the
// backend has no host buffer type or the pinned allocation fails.
void * ExpertCache::stage_host(size_t nbytes) {
    if (host_buft_) {
        if (stage_cap_ < nbytes) {
            if (stage_buf_) ggml_backend_buffer_free(stage_buf_);
            stage_buf_ = ggml_backend_buft_alloc_buffer(host_buft_, nbytes);
            if (stage_buf_) {
                stage_ptr_ = ggml_backend_buffer_get_base(stage_buf_);
                stage_cap_ = nbytes;
            } else {
                host_buft_ = nullptr;   // pinned alloc failed: stop trying
                stage_ptr_ = nullptr;
                stage_cap_ = 0;
            }
        }
        if (stage_ptr_) return stage_ptr_;
    }
    if (stage_.size() < nbytes) stage_.resize(nbytes);
    return stage_.data();
}

ggml_tensor * ExpertCache::tensor(Role role, int layer) const {
    return pools_[layer_pool_[role][layer]].t;
}

int ExpertCache::min_slots() const {
    int m = pools_.empty() ? 0 : pools_[0].n_slots;
    for (const auto & p : pools_) m = std::min(m, p.n_slots);
    return m;
}

// Copy one expert's slab from the slower tier into a VRAM slot.
// This is the tiering seam: RAM (tensor_get) vs SSD (pread from the GGUF file).
void ExpertCache::fetch_slab(Role role, int layer, int expert, ggml_tensor * dst, int slot) {
    const auto t0 = std::chrono::steady_clock::now();
    ggml_tensor * src = role_tensor(model_, role, layer);
    const size_t nb2 = src->nb[2];
    if (dst->nb[2] != nb2)
        throw std::runtime_error("ExpertCache: slab size mismatch (pool vs source)");

    const void * hsrc;
    if (ssd_) {
        // SSD tier: read the slab into a pinned staging buffer, then H2D from it.
        void * stage = stage_host(nb2);
        const size_t off = foff_[role][layer] + (size_t) expert * nb2;
        void *& fp = files_[fpath_[role][layer]];
        if (!fp) {
            fp = (void *) fopen(fpath_[role][layer].c_str(), "rb");
            if (!fp) throw std::runtime_error("ExpertCache: failed to open shard: " + fpath_[role][layer]);
        }
        FILE * f = (FILE *) fp;
#ifdef _WIN32
        if (_fseeki64(f, (long long) off, SEEK_SET) != 0 ||
#else
        if (fseeko(f, (off_t) off, SEEK_SET) != 0 ||
#endif
            fread(stage, 1, nb2, f) != nb2)
            throw std::runtime_error("ExpertCache: SSD pread failed");
        hsrc = stage;
    } else {
        // RAM tier: the expert weights live in a (pinned) CPU buffer -> H2D straight
        // from the source, no intermediate staging copy.
        hsrc = (const char *) src->data + (size_t) expert * nb2;
        if (async_fetch_) {
            // Async H2D on the backend's stream: the host does not block per copy,
            // and the following seg-B compute (same stream) is naturally ordered
            // after it. Source is pinned (persistent), so this is a true async DMA.
            ggml_backend_tensor_set_async(backend_, dst, hsrc, (size_t) slot * nb2, nb2);
            stats_.fetch_bytes += nb2;
            return;
        }
    }
    ggml_backend_tensor_set(dst, hsrc, (size_t) slot * nb2, nb2);
    stats_.fetch_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    stats_.fetch_bytes += nb2;
}

// Fetch (layer,expert) into a free (or LRU-evicted) slot; updates residency maps.
int ExpertCache::install(Pool & pool, int key, int layer, int expert, Role role) {
    int victim = 0;
    uint64_t best = pool.clk[0];
    for (int s = 1; s < pool.n_slots; ++s) {
        if (pool.clk[s] < best) { best = pool.clk[s]; victim = s; }
    }
    if (pool.slot2key[victim] >= 0) {
        pool.slot_of[pool.slot2key[victim]] = 0;   // evicted key -> sentinel
        pool.key2slot.erase(pool.slot2key[victim]);
        stats_.evictions++;
    }
    fetch_slab(role, layer, expert, pool.t, victim);
    pool.slot2key[victim] = key;
    pool.clk[victim] = ++clock_;
    pool.key2slot[key] = victim;
    pool.slot_of[key] = victim;
    return victim;
}

// Claim a slot for `key` (evicting the LRU occupant) without loading data yet.
int ExpertCache::reserve_victim(Pool & pool, int key) {
    int victim = 0;
    uint64_t best = pool.clk[0];
    for (int s = 1; s < pool.n_slots; ++s) {
        if (pool.clk[s] < best) { best = pool.clk[s]; victim = s; }
    }
    if (pool.slot2key[victim] >= 0) {
        pool.slot_of[pool.slot2key[victim]] = 0;
        pool.key2slot.erase(pool.slot2key[victim]);
        stats_.evictions++;
    }
    pool.slot2key[victim] = key;
    pool.clk[victim] = ++clock_;
    pool.key2slot[key] = victim;
    pool.slot_of[key] = victim;
    return victim;
}

// Read all reserved jobs' slabs from SSD in parallel (each worker uses its own
// file handles), then upload them to their VRAM slots serially (one GPU stream).
void ExpertCache::fetch_parallel(int layer, std::vector<FetchJob> & jobs) {
    const int n = (int) jobs.size();
    if (n == 0) return;

    std::vector<std::vector<uint8_t>> stage(n);
    std::atomic<bool> ok{true};
    const int nthreads = std::min(n, prefetch_threads_);

    auto worker = [&](int tid) {
        std::unordered_map<std::string, FILE *> tf;   // per-thread file cache
        for (int j = tid; j < n; j += nthreads) {
            const FetchJob & job = jobs[j];
            const size_t nb2 = job.pool->t->nb[2];
            stage[j].resize(nb2);
            const std::string & path = fpath_[job.role][layer];
            FILE *& f = tf[path];
            if (!f) { f = fopen(path.c_str(), "rb"); if (!f) { ok = false; continue; } }
            const size_t off = foff_[job.role][layer] + (size_t) job.expert * nb2;
#ifdef _WIN32
            if (_fseeki64(f, (long long) off, SEEK_SET) != 0 ||
#else
            if (fseeko(f, (off_t) off, SEEK_SET) != 0 ||
#endif
                fread(stage[j].data(), 1, nb2, f) != nb2) ok = false;
        }
        for (auto & kv : tf) if (kv.second) fclose(kv.second);
    };

    std::vector<std::thread> ts;
    ts.reserve(nthreads - 1);
    for (int t = 1; t < nthreads; ++t) ts.emplace_back(worker, t);
    worker(0);
    for (auto & t : ts) t.join();
    if (!ok) throw std::runtime_error("ExpertCache: parallel SSD read failed");

    for (int j = 0; j < n; ++j) {
        ggml_tensor * dst = jobs[j].pool->t;
        ggml_backend_tensor_set(dst, stage[j].data(), (size_t) jobs[j].slot * dst->nb[2], stage[j].size());
    }
}

bool ExpertCache::resident(Role role, int layer, int expert) const {
    const Pool & pool = pools_[layer_pool_[role][layer]];
    return pool.key2slot.find(layer * n_expert_ + expert) != pool.key2slot.end();
}

const int32_t * ExpertCache::slot_of_row(Role role, int layer) const {
    const Pool & pool = pools_[layer_pool_[role][layer]];
    return &pool.slot_of[(size_t) layer * n_expert_];
}

void ExpertCache::touch(Role role, int layer, int expert) {
    Pool & pool = pools_[layer_pool_[role][layer]];
    const int key = layer * n_expert_ + expert;
    auto it = pool.key2slot.find(key);
    if (it == pool.key2slot.end()) return;
    pool.count[key]++;
    pool.clk[it->second] = ++clock_;
    stats_.hits++;
}

int ExpertCache::slot_for(Pool & pool, int layer, int expert, Role role) {
    const int key = layer * n_expert_ + expert;
    pool.count[key]++;

    auto it = pool.key2slot.find(key);
    if (it != pool.key2slot.end()) {
        stats_.hits++;
        pool.clk[it->second] = ++clock_;
        return it->second;
    }
    stats_.misses++;
    return install(pool, key, layer, expert, role);
}

// Profile file format (text):
//   QWENEXPCACHE 1
//   <n_layer> <n_expert>
//   <role> <layer> <expert> <count>   (one per line, hot experts)
bool ExpertCache::save_profile(const std::string & path) const {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fprintf(f, "QWENEXPCACHE 1\n%d %d\n", n_layer_, n_expert_);
    for (const auto & pool : pools_) {
        for (const auto & kv : pool.count) {
            const int layer  = kv.first / n_expert_;
            const int expert = kv.first % n_expert_;
            fprintf(f, "%d %d %d %llu\n", pool.role, layer, expert,
                    (unsigned long long) kv.second);
        }
    }
    fclose(f);
    return true;
}

size_t ExpertCache::load_prefetch(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return 0;

    int ver = 0, nl = 0, ne = 0;
    if (fscanf(f, "QWENEXPCACHE %d", &ver) != 1 || ver != 1 ||
        fscanf(f, "%d %d", &nl, &ne) != 2 || nl != n_layer_ || ne != n_expert_) {
        fclose(f);
        return 0;   // missing / incompatible profile -> cold start
    }

    struct Ent { int role, layer, expert; uint64_t count; };
    std::vector<Ent> ents;
    int r, l, e; unsigned long long c;
    while (fscanf(f, "%d %d %d %llu", &r, &l, &e, &c) == 4) {
        if (r >= 0 && r < N_ROLE && l >= 0 && l < n_layer_ && e >= 0 && e < n_expert_)
            ents.push_back({ r, l, e, c });
    }
    fclose(f);

    // Highest-frequency experts first; fill each pool up to its capacity.
    std::sort(ents.begin(), ents.end(),
              [](const Ent & a, const Ent & b) { return a.count > b.count; });

    size_t n_pref = 0;
    for (const auto & en : ents) {
        Pool & pool = pools_[layer_pool_[en.role][en.layer]];
        if ((int) pool.key2slot.size() >= pool.n_slots) continue;   // pool full
        const int key = en.layer * n_expert_ + en.expert;
        if (pool.key2slot.count(key)) continue;                     // already resident
        install(pool, key, en.layer, en.expert, (Role) en.role);
        n_pref++;
    }
    return n_pref;
}

void ExpertCache::ensure_resident(int layer, int expert) {
    for (int r = 0; r < N_ROLE; ++r) {
        Pool & pool = pools_[layer_pool_[r][layer]];
        const int key = layer * n_expert_ + expert;
        if (pool.key2slot.find(key) == pool.key2slot.end()) {
            stats_.misses++;
            install(pool, key, layer, expert, (Role) r);
        }
    }
}

void ExpertCache::ensure(int layer, const int32_t * expert_ids, int n,
                         int32_t * slot_gate, int32_t * slot_up, int32_t * slot_down) {
    Pool * pools3[N_ROLE] = {
        &pools_[layer_pool_[GATE][layer]],
        &pools_[layer_pool_[UP][layer]],
        &pools_[layer_pool_[DOWN][layer]],
    };
    int32_t * outs[N_ROLE] = { slot_gate, slot_up, slot_down };

    // RAM source: sequential is fine (memcpy is fast).
    if (!ssd_) {
        for (int i = 0; i < n; ++i) {
            const int e = (int) expert_ids[i];
            for (int r = 0; r < N_ROLE; ++r)
                outs[r][i] = slot_for(*pools3[r], layer, e, (Role) r);
        }
        return;
    }

    // SSD source: resolve hits, reserve victims for misses, then read the misses
    // in parallel (overlapping disk latency) before uploading them.
    std::vector<FetchJob> jobs;
    for (int i = 0; i < n; ++i) {
        const int e = (int) expert_ids[i];
        const int key = layer * n_expert_ + e;
        for (int r = 0; r < N_ROLE; ++r) {
            Pool & pool = *pools3[r];
            pool.count[key]++;
            auto it = pool.key2slot.find(key);
            if (it != pool.key2slot.end()) {
                stats_.hits++;
                pool.clk[it->second] = ++clock_;
                outs[r][i] = it->second;
            } else {
                stats_.misses++;
                const int slot = reserve_victim(pool, key);
                outs[r][i] = slot;
                jobs.push_back({ &pool, r, e, slot });
            }
        }
    }
    fetch_parallel(layer, jobs);
}

} // namespace qwencpp
