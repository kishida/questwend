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

#ifdef __APPLE__
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace questwend {

#ifdef __APPLE__
// Is `p` a dereferenceable host pointer (mapped in our address space)?
// Metal shared buffers (unified memory) expose real host memory at
// tensor->data; private buffers hold an unmapped virtual placeholder
// (allocated from 0x400 up, inside PAGEZERO -- never mapped).
static bool ptr_is_mapped(const void * p) {
    if (!p) return false;
    const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
    char vec = 0;
    void * al = (void *) ((uintptr_t) p & ~(uintptr_t) (pg - 1));
    return mincore(al, pg, &vec) == 0;
}
#endif

#ifdef _WIN32
// Positioned read with FILE_FLAG_NO_BUFFERING semantics: offset, length and
// buffer must be sector-aligned, so read the covering aligned range into an
// (over-allocated) aligned staging vector and copy the slab out. Bypasses the
// OS page cache — true SSD reads even when the GGUF fits in RAM.
static bool win_direct_read(void * h, void * dst, size_t nbytes, uint64_t off,
                            std::vector<uint8_t> & abuf) {
    const uint64_t SEC = 4096;
    const uint64_t a0   = off & ~(SEC - 1);
    const uint64_t a1   = (off + nbytes + SEC - 1) & ~(SEC - 1);
    const size_t   alen = (size_t) (a1 - a0);
    if (abuf.size() < alen + SEC) abuf.resize(alen + SEC);
    uint8_t * ap = (uint8_t *) (((uintptr_t) abuf.data() + SEC - 1) & ~(uintptr_t) (SEC - 1));
    OVERLAPPED ov{};
    ov.Offset     = (DWORD) (a0 & 0xffffffffull);
    ov.OffsetHigh = (DWORD) (a0 >> 32);
    DWORD got = 0;
    if (!ReadFile((HANDLE) h, ap, (DWORD) alen, &got, &ov)) return false;
    if ((uint64_t) got + a0 < off + nbytes) return false;   // short read (EOF)
    memcpy(dst, ap + (off - a0), nbytes);
    return true;
}

static void * win_direct_open(const std::string & path) {
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
    return h == INVALID_HANDLE_VALUE ? nullptr : (void *) h;
}
#endif

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
#ifdef _WIN32
        if (getenv("QWEN_SSD_DIRECT")) {
            direct_ = true;
            fprintf(stderr, "expert cache: direct (unbuffered) SSD reads — OS page cache bypassed\n");
        }
#endif
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

#ifdef __APPLE__
    // Unified memory (Metal shared buffers): the slot pools are host-writable,
    // so SSD misses can be pread() straight into the slot -- no staging buffer,
    // no memcpy, no stdio buffering.
    if (ssd_ && !getenv("QWEN_NO_DIRECT_FETCH")) {
        int n_direct = 0;
        for (auto & p : pools_)
            if (ptr_is_mapped(p.t->data)) { p.host = (uint8_t *) p.t->data; ++n_direct; }
        if (n_direct)
            fprintf(stderr, "expert cache: zero-copy SSD reads (unified memory, %d/%zu pools)\n",
                    n_direct, pools_.size());
    }
#endif

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
#ifdef __APPLE__
    for (auto & kv : fds_) if (kv.second >= 0) close(kv.second);
#endif
#ifdef _WIN32
    for (auto & kv : hfiles_) if (kv.second) CloseHandle((HANDLE) kv.second);
#endif
    if (coal_stage_buf_) ggml_backend_buffer_free(coal_stage_buf_);
    if (stage_buf_) ggml_backend_buffer_free(stage_buf_);
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

uint8_t * ExpertCache::host_of(const ggml_tensor * t) const {
    for (const auto & p : pools_) if (p.t == t) return p.host;
    return nullptr;
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

// 4KB-aligned staging for coalesced reads; pinned (page-locked) when the
// backend offers a host buffer type, so slab uploads can be true async DMA.
void * ExpertCache::coal_host(size_t nbytes) {
    if (host_buft_ && coal_stage_cap_ < nbytes) {
        if (coal_stage_buf_) ggml_backend_buffer_free(coal_stage_buf_);
        coal_stage_buf_ = ggml_backend_buft_alloc_buffer(host_buft_, nbytes);
        if (coal_stage_buf_) {
            coal_stage_ptr_ = ggml_backend_buffer_get_base(coal_stage_buf_);
            coal_stage_cap_ = nbytes;
        } else {
            coal_stage_ptr_ = nullptr;
            coal_stage_cap_ = 0;
        }
    }
    if (coal_stage_ptr_ && coal_stage_cap_ >= nbytes) { coal_pinned_ = true; return coal_stage_ptr_; }
    coal_pinned_ = false;
    if (coal_buf_.size() < nbytes + 4096) coal_buf_.resize(nbytes + 4096);
    return (void *) (((uintptr_t) coal_buf_.data() + 4095) & ~(uintptr_t) 4095);
}

ggml_tensor * ExpertCache::tensor(Role role, int layer) const {
    return pools_[layer_pool_[role][layer]].t;
}

int ExpertCache::min_slots() const {
    int m = pools_.empty() ? 0 : pools_[0].n_slots;
    for (const auto & p : pools_) m = std::min(m, p.n_slots);
    return m;
}

int ExpertCache::capacity(int layer) const {
    int m = pools_[layer_pool_[0][layer]].n_slots;
    for (int r = 1; r < N_ROLE; ++r)
        m = std::min(m, pools_[layer_pool_[r][layer]].n_slots);
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
        const size_t off = foff_[role][layer] + (size_t) expert * nb2;
#ifdef __APPLE__
        // unified memory: pread straight into the slot (no staging, no memcpy)
        if (uint8_t * hp = host_of(dst)) {
            const std::string & path = fpath_[role][layer];
            auto it = fds_.find(path);
            int fd;
            if (it == fds_.end()) {
                fd = open(path.c_str(), O_RDONLY);
                fds_[path] = fd;
            } else fd = it->second;
            if (fd < 0) throw std::runtime_error("ExpertCache: failed to open shard: " + path);
            if (pread(fd, hp + (size_t) slot * nb2, nb2, (off_t) off) != (ssize_t) nb2)
                throw std::runtime_error("ExpertCache: SSD pread failed");
            stats_.fetch_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            stats_.fetch_bytes += nb2;
            return;
        }
#endif
        // SSD tier: read the slab into a pinned staging buffer, then H2D from it.
        void * stage = stage_host(nb2);
#ifdef _WIN32
        if (direct_) {
            void *& hv = hfiles_[fpath_[role][layer]];
            if (!hv) hv = win_direct_open(fpath_[role][layer]);
            if (!hv || !win_direct_read(hv, stage, nb2, off, abuf_))
                throw std::runtime_error("ExpertCache: direct SSD read failed");
            hsrc = stage;
        } else {
#endif
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
#ifdef _WIN32
        }
#endif
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
// file handles). On unified memory the workers pread straight into the VRAM
// slot (zero-copy); otherwise they stage and the slabs are uploaded serially.
void ExpertCache::fetch_parallel(int layer, std::vector<FetchJob> & jobs) {
    const int n = (int) jobs.size();
    if (n == 0) return;
    const auto t0 = std::chrono::steady_clock::now();

    // ---- coalescing: expert slabs of one (role,layer) tensor are contiguous in
    // the GGUF, and a prefill layer's union covers most of them, so scattered
    // 0.6MB preads are merged into big sequential range reads (several times the
    // drive's scattered throughput). Small gaps are read and discarded; small
    // job sets (decode misses, background refill) keep the per-slab path below.
    // Opt-in (QWEN_COALESCE=1): on drives whose sequential read beats their
    // QD-parallel scattered read, merging a layer's union into range reads
    // wins. Measured here on a drive with a degraded tail region (seq 1.9GB/s
    // head / 0.15GB/s tail), 8-thread scattered slab reads extract more from
    // the controller, so scattered stays the default.
    static const bool use_coal = getenv("QWEN_COALESCE") != nullptr;
    static const int  coal_gap = []{ const char * c = getenv("QWEN_COALESCE_GAP");
                                     const int v = c ? atoi(c) : 8; return v < 0 ? 8 : v; }();
    constexpr size_t SEG_BYTES  = 8u << 20;   // one worker read segment
    constexpr size_t SPAN_BYTES = 64u << 20;  // staging cap: runs split at this size
    constexpr int    COAL_MIN   = 16;         // fewer jobs per role: per-slab path

    struct Run { Pool * pool; int role; int e_lo, e_hi; std::vector<const FetchJob *> jobs; };
    std::vector<Run> runs;
    std::vector<FetchJob> singles;

    if (!use_coal) {
        singles = jobs;
    } else {
        for (int r = 0; r < N_ROLE; ++r) {
            std::vector<const FetchJob *> rj;
            for (const auto & j : jobs) if (j.role == r) rj.push_back(&j);
            if (rj.empty()) continue;
            if ((int) rj.size() < COAL_MIN) {
                for (const FetchJob * j : rj) singles.push_back(*j);
                continue;
            }
            std::sort(rj.begin(), rj.end(),
                      [](const FetchJob * a, const FetchJob * b) { return a->expert < b->expert; });
            const size_t nb2 = rj[0]->pool->t->nb[2];
            std::vector<Run> rruns;
            Run cur{ rj[0]->pool, r, rj[0]->expert, rj[0]->expert, { rj[0] } };
            for (size_t i = 1; i < rj.size(); ++i) {
                const int e = rj[i]->expert;
                if (e - cur.e_hi - 1 <= coal_gap &&
                    (size_t) (e - cur.e_lo + 1) * nb2 <= SPAN_BYTES) {
                    cur.e_hi = e;
                    cur.jobs.push_back(rj[i]);
                } else {
                    rruns.push_back(std::move(cur));
                    cur = Run{ rj[i]->pool, r, e, e, { rj[i] } };
                }
            }
            rruns.push_back(std::move(cur));
            // waste guard: a sparse run reads mostly gap bytes — scattered
            // QD-parallel per-slab reads of just the wanted slabs win there
            for (auto & rr : rruns) {
                const int span_slabs = rr.e_hi - rr.e_lo + 1;
                if ((int) rr.jobs.size() * 2 < span_slabs) {
                    for (const FetchJob * j : rr.jobs) singles.push_back(*j);
                } else {
                    runs.push_back(std::move(rr));
                }
            }
        }
    }

    // coalesced runs: workers read disjoint 4KB-aligned segments of the range
    // straight into the (pinned) staging, then the slabs are uploaded — async
    // DMA when pinned (the sync pageable per-slab copies used to dominate).
    // The staging is reused across runs, so a pending async batch is synced
    // before the next run's reads overwrite it.
    for (auto & run : runs) {
        const size_t nb2  = run.pool->t->nb[2];
        const std::string & path = fpath_[run.role][layer];
        const uint64_t base = foff_[run.role][layer] + (uint64_t) run.e_lo * nb2;
        const size_t span = (size_t) (run.e_hi - run.e_lo + 1) * nb2;
        const uint64_t a0    = base & ~4095ull;          // sector-aligned range start
        const size_t   delta = (size_t) (base - a0);
        const size_t   need  = (delta + span + 4095) & ~(size_t) 4095;

        static const bool coal_dbg = getenv("QWEN_COAL_DEBUG") != nullptr;
        const auto tr0 = std::chrono::steady_clock::now();
        if (coal_async_pending_) {   // staging still feeding async H2D: drain first
            ggml_backend_synchronize(backend_);
            coal_async_pending_ = false;
        }
        uint8_t * cbuf = (uint8_t *) coal_host(need);
        static bool coal_dbg_once = false;
        if (coal_dbg && !coal_dbg_once) {
            fprintf(stderr, "coal: staging %s (%zu MB)\n", coal_pinned_ ? "pinned" : "pageable", need >> 20);
            coal_dbg_once = true;
        }

        std::vector<std::pair<size_t, size_t>> segs;   // (offset into range, length)
        for (size_t o = 0; o < need; o += SEG_BYTES)
            segs.push_back({ o, std::min(SEG_BYTES, need - o) });

        std::atomic<size_t> seg_next{0};
        std::atomic<bool>   seg_ok{true};
        auto rworker = [&]() {
            FILE * f = nullptr;
#ifdef _WIN32
            void * h = nullptr;
            if (direct_) { h = win_direct_open(path); if (!h) { seg_ok = false; return; } }
            if (!h)
#endif
            { f = fopen(path.c_str(), "rb"); if (!f) { seg_ok = false; return; } }
            size_t i;
            while ((i = seg_next.fetch_add(1)) < segs.size()) {
                const size_t o = segs[i].first, len = segs[i].second;
#ifdef _WIN32
                if (h) {
                    // aligned offset/len/buffer: read straight into the staging
                    // (a short read is fine iff it still covers the needed span)
                    OVERLAPPED ov{};
                    ov.Offset     = (DWORD) ((a0 + o) & 0xffffffffull);
                    ov.OffsetHigh = (DWORD) ((a0 + o) >> 32);
                    DWORD got = 0;
                    if (!ReadFile((HANDLE) h, cbuf + o, (DWORD) len, &got, &ov) ||
                        (got < len && a0 + o + got < base + span))
                        seg_ok = false;
                    continue;
                }
                if (_fseeki64(f, (long long) (a0 + o), SEEK_SET) != 0 ||
#else
                if (fseeko(f, (off_t) (a0 + o), SEEK_SET) != 0 ||
#endif
                    fread(cbuf + o, 1, len, f) < std::min(len, (size_t) (base + span - (a0 + o))))
                    seg_ok = false;
            }
            if (f) fclose(f);
#ifdef _WIN32
            if (h) CloseHandle((HANDLE) h);
#endif
        };
        const int nth = std::min((int) segs.size(), prefetch_threads_);
        std::vector<std::thread> rts;
        rts.reserve(nth > 0 ? nth - 1 : 0);
        for (int t = 1; t < nth; ++t) rts.emplace_back(rworker);
        rworker();
        for (auto & t : rts) t.join();
        if (!seg_ok) throw std::runtime_error("ExpertCache: coalesced SSD read failed");
        const auto tr1 = std::chrono::steady_clock::now();

        for (const FetchJob * job : run.jobs) {
            const uint8_t * src = cbuf + delta + (size_t) (job->expert - run.e_lo) * nb2;
            if (run.pool->host) {
                memcpy(run.pool->host + (size_t) job->slot * nb2, src, nb2);
            } else if (coal_pinned_) {
                ggml_backend_tensor_set_async(backend_, run.pool->t, src, (size_t) job->slot * nb2, nb2);
                coal_async_pending_ = true;
            } else {
                ggml_backend_tensor_set(run.pool->t, src, (size_t) job->slot * nb2, nb2);
            }
            stats_.fetch_bytes += nb2;
        }
        if (coal_dbg) {
            const auto tr2 = std::chrono::steady_clock::now();
            fprintf(stderr, "coal: run role=%d e=[%d..%d] read %.1fms (%.2f GB/s) upload %.1fms\n",
                    run.role, run.e_lo, run.e_hi,
                    std::chrono::duration<double, std::milli>(tr1 - tr0).count(),
                    need / 1073741824.0 / std::chrono::duration<double>(tr1 - tr0).count(),
                    std::chrono::duration<double, std::milli>(tr2 - tr1).count());
        }
    }
    // pending async uploads need no drain here: later graph compute and slot
    // writes are ordered behind them on the backend stream; the staging itself
    // is protected by the sync-before-reuse above (coal_async_pending_ persists
    // across calls).

    if (singles.empty()) {
        stats_.fetch_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        return;
    }

    // Per-slab scattered reads (QD-parallel across workers) into batches of the
    // pinned staging, uploaded as async DMA — the sync pageable per-slab copies
    // used to cost more than the SSD reads themselves. The staging is drained
    // before each batch reuses it.
    const int ns = (int) singles.size();
    size_t max_nb2 = 0;
    for (const auto & j : singles) max_nb2 = std::max(max_nb2, (size_t) j.pool->t->nb[2]);
    const int batch = std::max<int>(1, (int) ((64u << 20) / max_nb2));

    for (int b0 = 0; b0 < ns; b0 += batch) {
        const int bn = std::min(batch, ns - b0);
        if (coal_async_pending_) {   // staging still feeding async H2D: drain first
            ggml_backend_synchronize(backend_);
            coal_async_pending_ = false;
        }
        uint8_t * cbuf = (uint8_t *) coal_host((size_t) bn * max_nb2);

        std::atomic<bool> ok{true};
        const int nthreads = std::min(bn, prefetch_threads_);
        auto worker = [&](int tid) {
            std::unordered_map<std::string, FILE *> tf;   // per-thread file cache
#ifdef __APPLE__
            std::unordered_map<std::string, int> tfd;     // per-thread fd cache (direct path)
#endif
#ifdef _WIN32
            std::unordered_map<std::string, void *> th;   // per-thread HANDLE cache (direct path)
            std::vector<uint8_t> tabuf;                   // per-thread aligned bounce
#endif
            for (int j = tid; j < bn; j += nthreads) {
                const FetchJob & job = singles[b0 + j];
                const size_t nb2 = job.pool->t->nb[2];
                const std::string & path = fpath_[job.role][layer];
                const size_t off = foff_[job.role][layer] + (size_t) job.expert * nb2;
                uint8_t * dst = cbuf + (size_t) j * max_nb2;
#ifdef _WIN32
                if (direct_) {
                    void *& hv = th[path];
                    if (!hv) hv = win_direct_open(path);
                    if (!hv || !win_direct_read(hv, dst, nb2, off, tabuf)) ok = false;
                    continue;
                }
#endif
#ifdef __APPLE__
                if (job.pool->host) {   // unified memory: straight into the slot
                    auto it = tfd.find(path);
                    int fd;
                    if (it == tfd.end()) { fd = open(path.c_str(), O_RDONLY); tfd[path] = fd; }
                    else fd = it->second;
                    if (fd < 0 ||
                        pread(fd, job.pool->host + (size_t) job.slot * nb2, nb2, (off_t) off) != (ssize_t) nb2)
                        ok = false;
                    continue;
                }
#endif
                FILE *& f = tf[path];
                if (!f) { f = fopen(path.c_str(), "rb"); if (!f) { ok = false; continue; } }
#ifdef _WIN32
                if (_fseeki64(f, (long long) off, SEEK_SET) != 0 ||
#else
                if (fseeko(f, (off_t) off, SEEK_SET) != 0 ||
#endif
                    fread(dst, 1, nb2, f) != nb2) ok = false;
            }
            for (auto & kv : tf) if (kv.second) fclose(kv.second);
#ifdef __APPLE__
            for (auto & kv : tfd) if (kv.second >= 0) close(kv.second);
#endif
#ifdef _WIN32
            for (auto & kv : th) if (kv.second) CloseHandle((HANDLE) kv.second);
#endif
        };

        std::vector<std::thread> ts;
        ts.reserve(nthreads > 0 ? nthreads - 1 : 0);
        for (int t = 1; t < nthreads; ++t) ts.emplace_back(worker, t);
        worker(0);
        for (auto & t : ts) t.join();
        if (!ok) throw std::runtime_error("ExpertCache: parallel SSD read failed");

        for (int j = 0; j < bn; ++j) {
            const FetchJob & job = singles[b0 + j];
            const size_t nb2 = job.pool->t->nb[2];
            stats_.fetch_bytes += nb2;
            if (job.pool->host) continue;   // already written in place
            const uint8_t * src = cbuf + (size_t) j * max_nb2;
            if (coal_pinned_) {
                ggml_backend_tensor_set_async(backend_, job.pool->t, src, (size_t) job.slot * nb2, nb2);
                coal_async_pending_ = true;
            } else {
                ggml_backend_tensor_set(job.pool->t, src, (size_t) job.slot * nb2, nb2);
            }
        }
    }
    stats_.fetch_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
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

} // namespace questwend
