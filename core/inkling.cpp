// Inkling 2 engine. Graph follows llama.cpp's src/models/inkling2.cpp
// (PR ggml-org/llama.cpp#32695 for the looped variant) op for op, so the two
// can be diffed tensor by tensor.

#include "inkling.h"
#include "model.h"
#include "expert_cache.h"
#include "gguf_util.h"
#include "gguf.h"
#include <map>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <filesystem>

namespace questwend {

namespace {

constexpr int GRAPH_NODES = 65536;
constexpr int UBATCH      = 512;    // prefill chunk; the SWA ring is sized for it

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

} // namespace

struct InklingEngine::Impl {
    Model &             model;
    RuntimeConfig       cfg;
    const HParams &     hp;
    const InklingParams & ik;

    ggml_backend_t gpu = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_threadpool_t cpu_tp = nullptr;
    std::vector<ggml_backend_buffer_t> wbufs_gpu, wbufs_cpu;
    bool offload = false;

    // ---- per-visit state: KV (F16 rows of n_embd_k_gqa) and conv history ----
    ggml_context *        st_ctx = nullptr;
    ggml_backend_buffer_t st_buf = nullptr;
    std::vector<ggml_tensor *> k_l, v_l, conv_l;   // per visit at the current pass count
    // The state there is room for, laid out by the visits of max_pass passes.
    // Fewer passes use the first ones of the core and the same postlude.
    std::vector<ggml_tensor *> st_k, st_v, st_conv;
    int  max_pass = 1;
    void map_state();
    bool set_loops(int n);
    int n_ctx = 0, ring = 0;   // full-attention capacity, SWA ring capacity
    int n_embd_kv = 0, n_head = 0, n_head_kv = 0, n_hd = 0, n_ch = 0;

    int n_past_ = 0;
    // Flash attention with the relative bias folded into a per-head mask (the
    // default); INK_NO_FLASH=1 falls back to the manual path.
    bool flash = getenv("INK_NO_FLASH") == nullptr;
    std::vector<int32_t> kv_toks;
    std::vector<float>   logits;
    std::function<void(int, int)> progress_cb;

    ggml_gallocr_t galloc = nullptr;

    // ---- RAM offload ----
    // Routed experts live in pinned host memory; a VRAM slot cache (keyed by
    // PHYSICAL layer, so every visit of a looped layer shares it) holds a
    // subset. Decode runs per visit: graph A (attention .. router) on the GPU,
    // the selection is read back, cached experts run on the GPU (graph B) and
    // the rest on the CPU from the host copy, and the next visit's graph A
    // starts by finishing this one (post-conv + residual).
    std::unique_ptr<ExpertCache> ec;
    bool cpu_miss = true;       // false: fetch every miss into VRAM before computing
    // Static placement (INK_PLACE=core|lru|front): the pool is filled once, by
    // priority, and never churns -- decode and prefill both compute what is not
    // resident on the CPU. "core" puts the looped layers first: each of their
    // bytes serves n_pass visits per token.
    bool place_static = false;
    std::string place = "lru";
    // Static placement's prefill streams a layer that is not fully resident into
    // this one-layer scratch and runs it on the GPU (as llama.cpp's op offload
    // does): a 512-token chunk reuses every expert dozens of times, so moving the
    // weights beats computing on the CPU, and the pinned pool is never disturbed.
    ggml_context *        scr_ctx = nullptr;
    ggml_backend_buffer_t scr_buf = nullptr;
    std::vector<ggml_tensor *> scr[3];   // per physical layer: gate / up / down views of scr_buf
    bool scr_shared = true;              // the three roles share one region (staged prefill experts)
    ggml_tensor * p_tmp = nullptr;       // [n_ff_exp * n_used * UBATCH] f32: gate output, then the activation
    // Static placement never moves an expert, so a fully resident layer needs no
    // host routing: its router picks are turned into slot ids inside the graph
    // through this fixed map, and a run of such visits -- with "core" placement,
    // every visit of the looped core, which are contiguous -- decodes as one
    // graph. Only the other visits come back to the host (INK_FUSE=0: none fused).
    std::vector<uint8_t> full_res;       // per physical layer
    ggml_context *        map_ctx = nullptr;
    ggml_backend_buffer_t map_buf = nullptr;
    ggml_tensor *         p_slotmap = nullptr;   // i32 [n_exp, 3 * n_phys]: (role, layer) rows
    bool fuse = false;
    // INK_PLACE=hot: prefill runs on the "core" placement and counts every pick
    // per (layer, expert) -- a looped layer's three passes add up on their own --
    // then, before the first decode token, the pool is refilled by that count
    // and stays fixed. The count is the expected value of a slot: picks per
    // token times visits, which is what pinning whole loop layers approximated.
    bool hot = false, replace_pending = false, hot_placed = false;
    // Prefill overlap: a second backend on the same GPU (its own stream) fills
    // the scratch with the next layer while the main stream runs that visit's
    // attention. A 512-token chunk picks ~90% of a layer's experts anyway, so
    // the whole layer is assembled without waiting for the router.
    ggml_backend_t gpu2 = nullptr;
    ggml_gallocr_t galloc2 = nullptr;
    ggml_context * pf_ctx = nullptr;   // the in-flight copy graph (freed after sync)
    int pf_ip = -1;                    // layer the scratch is being filled with
    // Fill scratch[ip] with the experts in `want` (all when null): pool -> D2D,
    // host -> H2D, queued on `be`. Returns the copy graph's context (caller frees
    // it once `be` is synchronized) or null.
    ggml_context * assemble(int ip, const std::vector<uint8_t> * want, ggml_backend_t be, ggml_gallocr_t ga, int role = -1);
    std::vector<uint32_t> pick_cnt;       // [n_phys * n_exp]
    void replace_by_count();
    void refresh_residency();
    // Static placements decide the resident set first and size the slot pools
    // to exactly that set (no role left without its partners), then fill it.
    size_t pool_bytes = 0;
    std::vector<uint8_t> plan_set(const std::vector<double> & value) const;
    void   build_pool(const std::vector<uint8_t> & want);
    int  refill   = 4;          // most-missed experts made resident after each token
    ggml_context *        car_ctx = nullptr;
    ggml_backend_buffer_t car_buf = nullptr;
    std::vector<ggml_tensor *> slots;   // [E, UBATCH] f32 carries
    ggml_tensor * p_tok = nullptr;      // [UBATCH] i32
    ggml_tensor * p_tau = nullptr;      // [UBATCH] f32
    ggml_tensor * p_rel_full = nullptr, * p_mask_full = nullptr;   // n_ctx * UBATCH
    ggml_tensor * p_rel_swa  = nullptr, * p_mask_swa  = nullptr;   // ring  * UBATCH
    ggml_tensor * p_idx_full = nullptr, * p_idx_swa = nullptr;     // [UBATCH] i64
    ggml_tensor * p_vidx_full = nullptr, * p_vidx_swa = nullptr;   // [n_embd_kv * UBATCH] i64
    ggml_tensor * p_sink0 = nullptr;
    ggml_tensor * p_sel = nullptr, * p_w = nullptr;                 // router out [n_used, UBATCH]
    ggml_tensor * p_gpu = nullptr, * p_cpu = nullptr;               // expert sums [E, UBATCH]
    ggml_tensor * p_bpack = nullptr, * p_bw = nullptr;              // graph B ids (packed) / weights
    bool p_cpu_zero = false;
    // Decode does one blocking round trip per visit: graph A's selection,
    // weights and expert input come back in one read (p_rout), and everything
    // the host sends (graph B ids / weights, the CPU sum, zeros) goes up as
    // async copies from pinned memory, ordered on the GPU stream.
    ggml_tensor * p_rout = nullptr;                                 // [2*n_used + E] f32 (decode)
    ggml_backend_buffer_t pin_buf = nullptr;
    float   * pin_rout = nullptr, * pin_w = nullptr, * pin_cpu = nullptr, * pin_zero = nullptr;
    int32_t * pin_ids = nullptr;
    // host copies of the per-expert scales (the CPU expert graphs read them)
    ggml_context * hs_ctx = nullptr; ggml_backend_buffer_t hs_buf = nullptr;
    std::vector<ggml_tensor *> hs_gu, hs_dn;   // per physical layer, CPU tensors
    // one segment graph
    struct Seg {
        ggml_context * ctx = nullptr; ggml_cgraph * gf = nullptr; ggml_gallocr_t ga = nullptr;
        ggml_tensor * out = nullptr;   // logits (head graph)
        ggml_tensor * embd = nullptr;  // graph A of visit 0: image embeddings input
        ggml_tensor * sel = nullptr, * w = nullptr, * x = nullptr;   // graph A outputs (slot views)
        ggml_tensor * xin = nullptr, * ids = nullptr, * win = nullptr;  // CPU expert graph inputs
        int key = -1;
    };
    void free_seg(Seg & sg) {
        if (sg.ga) ggml_gallocr_free(sg.ga);
        if (sg.ctx) ggml_free(sg.ctx);
        sg = Seg{};
    }
    std::vector<Seg> segA;      // per visit (+1: head), kept for the current KV bucket
    std::vector<Seg> segB;      // per (visit, n resident) decode experts
    std::vector<Seg> segC;      // per (physical layer, n missed) CPU experts
    int seg_bucket = 0;
    // per-visit St snapshot after graph A, captured on the first build (deterministic)
    struct Stats {
        uint64_t sel = 0, gpu = 0, cpu = 0, fetched = 0;
        uint64_t core_sel = 0, core_repeat = 0;   // core-layer selections; already chosen by an earlier pass this token
        uint64_t tokens = 0;
        uint64_t streamed_layers = 0, streamed_bytes = 0, d2d_bytes = 0;   // static prefill
        double t_a = 0, t_b = 0, t_c = 0, t_fetch = 0, t_head = 0;
    } stx;
    std::vector<uint64_t> l_sel, l_gpu;   // per physical layer
    void init_offload(size_t budget, size_t st_bytes);
    void forward_offload(const int32_t * toks, int T, bool want_logits);
    void upload_inputs(int T, int n_kv_full);
    void dump_offload_stats();

    Impl(Model & m, const RuntimeConfig & c) : model(m), cfg(c), hp(m.hparams()), ik(m.hparams().ink) {}
    ~Impl();

    void init();
    ggml_tensor * W(const char * fmt, int il) const {
        char nm[160]; snprintf(nm, sizeof(nm), fmt, il);
        ggml_tensor * t = model.tensor(nm);
        if (!t) throw std::runtime_error(std::string("inkling2: missing tensor ") + nm);
        return t;
    }
    ggml_tensor * Wopt(const char * fmt, int il) const {
        char nm[160]; snprintf(nm, sizeof(nm), fmt, il);
        return model.tensor(nm);
    }
    ggml_tensor * G(const char * nm) const {
        ggml_tensor * t = model.tensor(nm);
        if (!t) throw std::runtime_error(std::string("inkling2: missing tensor ") + nm);
        return t;
    }

    // ---- graph pieces ----
    struct Gr {
        ggml_context * ctx = nullptr;
        ggml_cgraph *  gf  = nullptr;
        int T = 0, n_kv_full = 0, n_kv_swa = 0;
        ggml_tensor * tau = nullptr;                    // [1,1,T]
        ggml_tensor * rel_full = nullptr, * rel_swa = nullptr;    // [n_kv, T] i32
        ggml_tensor * mask_full = nullptr, * mask_swa = nullptr;  // [n_kv, T] f32
        ggml_tensor * idx_full = nullptr, * idx_swa = nullptr;    // [T] i64 write rows
        ggml_tensor * vidx_full = nullptr, * vidx_swa = nullptr;  // [n_embd_kv * T] i64 transposed-V elements
        ggml_tensor * sink0 = nullptr;                  // zero sinks [n_head]
    };
    // depth-attention bookkeeping (the PR's blocks / prelude / bank lists)
    struct St {
        std::vector<ggml_tensor *> blocks, prelude_blocks, bank;
        ggml_tensor * prelude_cur = nullptr, * bank_last = nullptr, * cur = nullptr;
    };

    ggml_tensor * clip(Gr & g, ggml_tensor * cur, ggml_tensor * scale, float thr);
    ggml_tensor * mix(Gr & g, const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query);
    ggml_tensor * control(Gr & g, ggml_tensor * cur, ggml_tensor * msg, int ip, int module);
    ggml_tensor * convolve(Gr & g, int iv, ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * dyn, int64_t offset);
    // Visit iv up to the MoE input. Leaves st.cur unchanged; returns ffn_inp
    // (the residual) and *x (normed expert input).
    ggml_tensor * visit_pre(Gr & g, St & st, int iv, ggml_tensor ** x);
    void route(Gr & g, int ip, ggml_tensor * x, ggml_tensor ** sel, ggml_tensor ** w);
    ggml_tensor * situ(Gr & g, ggml_tensor * gate, ggml_tensor * up);
    ggml_tensor * experts(Gr & g, int ip, ggml_tensor * x, ggml_tensor * sel, ggml_tensor * w,
                          ggml_tensor * tg, ggml_tensor * tu, ggml_tensor * td,
                          ggml_tensor * ig = nullptr, ggml_tensor * iu = nullptr, ggml_tensor * id = nullptr,
                          ggml_tensor * sc_gu = nullptr, ggml_tensor * sc_dn = nullptr);
    ggml_tensor * visit_post(Gr & g, int iv, ggml_tensor * ffn_inp, ggml_tensor * moe_out);
    ggml_tensor * head(Gr & g, St & st, bool last_only);

    Gr   new_graph(int T, int n_kv_full);
    Gr   seg_graph(int T, int n_kv_full, int nodes = 8192);   // offload: inputs are views of the persistent p_* tensors
    void set_inputs(Gr & g, int T);
    // the whole stack for T tokens on the GPU (resident weights)
    void build_full(Gr & g, ggml_tensor * inp, bool want_logits, ggml_tensor ** out);
    void forward(const int32_t * toks, int T, bool want_logits);
    // Images: a chunk is either all token ids or all image embeddings, the
    // latter handed to forward() in cur_embd ([n_embd, T], null = token ids).
    std::vector<Runtime::EmbdOverride> embd_ovr;
    const float * cur_embd = nullptr;
    // Sets cur_embd for the chunk at i of a P-token call and returns its length.
    int next_chunk(int i, int P) {
        cur_embd = nullptr;
        int end = P;
        for (const auto & o : embd_ovr) {
            if (i >= o.first && i < o.first + o.count) {
                cur_embd = o.data + (size_t) (i - o.first) * hp.n_embd;
                end = std::min(P, o.first + o.count);
                break;
            }
            if (o.first > i) end = std::min(end, o.first);
        }
        int T = chunk_len(end - i);
        if (cur_embd && T > 2 && end - i - T == 1) --T;   // T == 1 is the decode path
        return T;
    }
    // The overrides belong to one call, also when it throws.
    struct OvrScope {
        Impl & m;
        ~OvrScope() { m.embd_ovr.clear(); m.cur_embd = nullptr; }
    };
    // Prefill chunk: UBATCH, halved while the per-head attention mask of the
    // full-attention visits ([n_kv, T, n_head], gathered in F32 then copied to
    // F16) would pass INK_ATTN_MB (default 400). Its size grows with the KV, so
    // a long context takes shorter chunks instead of running out of VRAM.
    int chunk_len(int remaining) const {
        static const double budget = (getenv("INK_ATTN_MB") ? atof(getenv("INK_ATTN_MB")) : 400.0) * 1048576.0;
        int T = std::min(UBATCH, remaining);
        while (T > 16 && (double) (n_past_ + T) * T * n_head * 6.0 > budget) T /= 2;
        return std::max(1, std::min(T, remaining));
    }

    // Decode graph kept across tokens: sized for a KV bucket (rows past
    // n_past are masked), so its topology and shapes stay fixed and ggml-cuda
    // can replay it as a CUDA graph instead of launching ~11k kernels.
    struct Kept {
        Gr g; ggml_tensor * inp = nullptr, * out = nullptr;
        ggml_gallocr_t ga = nullptr; int bucket = 0, T = 0;
    } kept;
    bool all_logits = false;   // head over every token (a verify step), not just the last

    // ---- DSpark speculative decoding (resident path) ----
    // The draft (arch "dflash", Inkling2 DSpark variant) is a 3-layer transformer
    // whose KV is built from the target's hidden states at four layer visits; one
    // pass over [last token, 7 masks] plus a recurrent head proposes a block.
    struct Dspark {
        bool on = false;
        gguf_context * gg = nullptr;
        ggml_context * wctx = nullptr; ggml_backend_buffer_t wbuf = nullptr;
        std::map<std::string, ggml_tensor *> w;
        int n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, hd = 0, n_ff = 0, block = 0, rank = 0;
        float eps = 1e-6f, rope_base = 1e6f;
        uint32_t n_swa = 0;
        std::vector<uint8_t> is_swa;
        std::vector<int> target;          // visit indices; n_visit means the final normed hidden
        // state + persistent inputs
        ggml_context * sctx = nullptr; ggml_backend_buffer_t sbuf = nullptr;
        std::vector<ggml_tensor *> k, v;  // per draft layer: F16 [n_kv_dim, n_ctx]
        ggml_tensor * frnn = nullptr;     // [E, 16] ring of the recurrent head's context features
        ggml_tensor * feat = nullptr;     // [n_target * E, UBATCH] target features of the last forward
        std::vector<ggml_tensor *> xs;    // per target visit: conv input window [nc-1 + 16, n_ch]
        ggml_tensor * p_tok = nullptr, * p_pos = nullptr, * p_idx = nullptr, * p_ridx = nullptr;
        ggml_tensor * p_crow = nullptr, * p_mask_f = nullptr, * p_mask_s = nullptr;
        ggml_tensor * p_out = nullptr, * p_am = nullptr;
        // graphs
        Seg draft; int draft_bucket = -1, draft_steps = -1;
        std::map<int, Seg> inj;           // by token count
        ggml_tensor * W(const std::string & n) const {
            auto it = w.find(n);
            if (it == w.end()) throw std::runtime_error("dspark: missing tensor " + n);
            return it->second;
        }
        ggml_tensor * Wl(const char * fmt, int il) const { char b[96]; snprintf(b, sizeof(b), fmt, il); return W(b); }
    } ds;
    bool cap_feat = false;   // graphs being built store the target features into ds.feat
    bool xs_save  = false;   // ... and the conv input windows into ds.xs (verify rollback)
    ggml_tensor * am_out = nullptr;   // verify graph: argmax per position -> ds.p_am
    void dspark_init(const std::string & path);
    Kept vkept;                // the verify graph (T = 1 + n_draft), kept per KV bucket
    bool no_keep_now = false;  // forward(): build fresh (the prompt of a speculative run)
    void spec_verify(const int32_t * toks, int T);
    std::function<int32_t(const std::vector<float> &)> spec_pick;   // null: greedy

    // ---- prompt checkpoints (rewind to a position) ----
    // Full-attention KV and the draft KV are indexed by position, so a rewind
    // only lowers n_past for them. The conv states and the SWA rings are
    // overwritten as the sequence moves on, so those are copied to host RAM.
    struct Ckpt { int pos = 0; std::vector<uint8_t> data; };
    std::vector<Ckpt> ckpts;   // oldest first
    std::vector<ggml_tensor *> ckpt_tensors() const {
        std::vector<ggml_tensor *> v(conv_l.begin(), conv_l.end());
        for (uint32_t iv = 0; iv < ik.n_visit; ++iv)
            if (ik.is_swa[iv]) { v.push_back(k_l[iv]); v.push_back(v_l[iv]); }
        if (ds.on) v.push_back(ds.frnn);
        return v;
    }
    void ckpt_snapshot() {
        if (n_past_ <= 0 || (!ckpts.empty() && ckpts.back().pos == n_past_)) return;
        static const int max_n = getenv("INK_CKPTS") ? std::max(0, atoi(getenv("INK_CKPTS"))) : 3;
        if (max_n == 0) return;
        Ckpt c;
        c.pos = n_past_;
        size_t total = 0;
        for (ggml_tensor * t : ckpt_tensors()) total += ggml_nbytes(t);
        c.data.resize(total);
        size_t off = 0;
        for (ggml_tensor * t : ckpt_tensors()) {
            ggml_backend_tensor_get(t, c.data.data() + off, 0, ggml_nbytes(t));
            off += ggml_nbytes(t);
        }
        // a checkpoint past the current position means the state moved on: keep order by pos
        while (!ckpts.empty() && ckpts.back().pos > c.pos) ckpts.pop_back();
        ckpts.push_back(std::move(c));
        while ((int) ckpts.size() > max_n) ckpts.erase(ckpts.begin());
    }
    int ckpt_best(int n) const {
        if (n >= n_past_) return n_past_;
        for (auto it = ckpts.rbegin(); it != ckpts.rend(); ++it) if (it->pos <= n) return it->pos;
        return -1;
    }
    int ckpt_rewind(int n) {
        if (n >= n_past_) return n_past_;
        int k = -1;
        for (int i = (int) ckpts.size() - 1; i >= 0; --i) if (ckpts[(size_t) i].pos <= n) { k = i; break; }
        if (k < 0) return -1;
        const Ckpt & c = ckpts[(size_t) k];
        size_t off = 0;
        for (ggml_tensor * t : ckpt_tensors()) {
            ggml_backend_tensor_set(t, c.data.data() + off, 0, ggml_nbytes(t));
            off += ggml_nbytes(t);
        }
        n_past_ = c.pos;
        kv_toks.resize((size_t) c.pos);
        ckpts.resize((size_t) k + 1);
        return c.pos;
    }
    void dspark_inject(int pos0, int T);           // ds.feat columns 0..T-1 -> draft KV at pos0..
    void dspark_draft(int32_t anchor, int pos0, int n_steps, std::vector<int32_t> & out);
    void dspark_rollback(int keep_tokens);         // conv states after the first keep_tokens of the verify
    void generate_spec(const std::vector<int32_t> & prompt, int max_new, int n_draft,
                       const std::function<bool(int32_t)> & on_token, int32_t * out_pending = nullptr,
                       bool ckpt_after_prefill = false);
    void free_kept() {
        if (kept.ga) ggml_gallocr_free(kept.ga);
        if (kept.g.ctx) ggml_free(kept.g.ctx);
        kept = Kept{};
    }

    // element indices of T new V columns in the transposed caches (channel-major)
    void fill_vidx(int T, std::vector<int64_t> & vf, std::vector<int64_t> & vs) const {
        vf.resize((size_t) n_embd_kv * T);
        vs.resize((size_t) n_embd_kv * T);
        for (int t = 0; t < T; ++t) {
            const int64_t cf = n_past_ + t, cs = (n_past_ + t) % ring;
            for (int c = 0; c < n_embd_kv; ++c) {
                vf[(size_t) t * n_embd_kv + c] = (int64_t) c * n_ctx + cf;
                vs[(size_t) t * n_embd_kv + c] = (int64_t) c * ring  + cs;
            }
        }
    }
    int  pos_of_ring(int cell, int n_total) const {
        // newest position p < n_total with p % ring == cell
        if (cell >= n_total) return -1;
        return cell + ((n_total - 1 - cell) / ring) * ring;
    }
};

InklingEngine::Impl::~Impl() {
    free_kept();
    if (vkept.ga) ggml_gallocr_free(vkept.ga);
    if (vkept.g.ctx) ggml_free(vkept.g.ctx);
    free_seg(ds.draft);
    for (auto & kv : ds.inj) free_seg(kv.second);
    if (ds.sbuf) ggml_backend_buffer_free(ds.sbuf);
    if (ds.sctx) ggml_free(ds.sctx);
    if (ds.wbuf) ggml_backend_buffer_free(ds.wbuf);
    if (ds.wctx) ggml_free(ds.wctx);
    if (ds.gg) gguf_free(ds.gg);
    if (galloc) ggml_gallocr_free(galloc);
    if (st_buf) ggml_backend_buffer_free(st_buf);
    if (st_ctx) ggml_free(st_ctx);
    for (auto b : wbufs_gpu) ggml_backend_buffer_free(b);
    for (auto b : wbufs_cpu) ggml_backend_buffer_free(b);
    for (auto & sg : segA) free_seg(sg);
    for (auto & sg : segB) free_seg(sg);
    for (auto & sg : segC) free_seg(sg);
    ec.reset();
    if (car_buf) ggml_backend_buffer_free(car_buf);
    if (pin_buf) ggml_backend_buffer_free(pin_buf);
    if (scr_buf) ggml_backend_buffer_free(scr_buf);
    if (map_buf) ggml_backend_buffer_free(map_buf);
    if (pf_ctx) ggml_free(pf_ctx);
    if (galloc2) ggml_gallocr_free(galloc2);
    if (gpu2) ggml_backend_free(gpu2);
    if (map_ctx) ggml_free(map_ctx);
    if (scr_ctx) ggml_free(scr_ctx);
    if (car_ctx) ggml_free(car_ctx);
    if (hs_buf) ggml_backend_buffer_free(hs_buf);
    if (hs_ctx) ggml_free(hs_ctx);
    if (cpu && cpu != gpu) ggml_backend_free(cpu);
    if (cpu_tp) ggml_threadpool_free(cpu_tp);
    if (gpu) ggml_backend_free(gpu);
}

void InklingEngine::Impl::init() {
    n_head    = (int) hp.head_count(0);
    n_head_kv = (int) hp.head_count_kv(0);
    n_hd      = (int) hp.n_embd_head;
    n_embd_kv = n_hd * n_head_kv;
    n_ch      = 2 * n_embd_kv + 2 * (int) hp.n_embd;

    // ---- backends ----
    std::vector<ggml_backend_dev_t> devs = cfg.use_cuda ? gpu_devices() : std::vector<ggml_backend_dev_t>{};
    size_t budget = 0;
    if (!devs.empty()) {
        const int di = cfg.gpus.empty() ? 0 : cfg.gpus[0].device;
        if (di < 0 || (size_t) di >= devs.size()) throw std::runtime_error("--gpus: no such device");
        gpu = ggml_backend_dev_init(devs[(size_t) di], nullptr);
        size_t fr = 0, tot = 0;
        ggml_backend_dev_memory(devs[(size_t) di], &fr, &tot);
        size_t mb = cfg.vram_budget_mb;
        if (!cfg.gpus.empty() && cfg.gpus[0].budget_mb != VRAM_BUDGET_AUTO) mb = cfg.gpus[0].budget_mb;
        budget = mb > 0 ? mb * 1024ull * 1024ull : fr;
        fprintf(stderr, "inkling2: GPU %s (%zu MB free, budget %zu MB)\n",
                ggml_backend_dev_description(devs[(size_t) di]), fr >> 20, budget >> 20);
    }
    cpu = ggml_backend_cpu_init();
    int nth = cfg.n_threads > 0 ? cfg.n_threads : (int) std::thread::hardware_concurrency() / 2;
    if (nth <= 0) nth = 4;
    if (const char * e = getenv("INK_CPU_THREADS")) nth = std::max(1, atoi(e));
    ggml_backend_cpu_set_n_threads(cpu, nth);
    // A persistent pool: without one every CPU graph spawns and joins its
    // workers, and decode runs one small CPU expert graph per layer visit.
    if (!getenv("INK_NO_THREADPOOL")) {
        ggml_threadpool_params tpp = ggml_threadpool_params_default(nth);
        if (const char * e = getenv("INK_CPU_POLL")) tpp.poll = (uint32_t) atoi(e);
        cpu_tp = ggml_threadpool_new(&tpp);
        if (cpu_tp) ggml_backend_cpu_set_threadpool(cpu, cpu_tp);
    }
    if (!gpu) { gpu = cpu; fprintf(stderr, "inkling2: CPU only (%d threads)\n", nth); }

    // ---- state sizes ----
    n_ctx = cfg.n_ctx > 0 ? cfg.n_ctx : 4096;
    ring  = std::min<int>(n_ctx, (int) ik.n_swa + UBATCH);
    if (ik.n_pass != ik.n_pass_file)
        fprintf(stderr, "inkling2: --loops %u (the model's own is %u): core layers %u-%u, %u layer visits instead of %u\n",
                ik.n_pass, ik.n_pass_file, ik.core_start, ik.core_end - 1, ik.n_visit,
                hp.n_layer + (ik.n_pass_file - 1) * ik.n_core());
    const int start_pass = (int) ik.n_pass;
    max_pass = start_pass;
    if (ik.has_core() && cfg.max_loops > start_pass) {
        max_pass = cfg.max_loops;
        model.set_loop_passes(max_pass);   // the state is sized by the most visits; back below
        fprintf(stderr, "inkling2: loops switchable 1..%d per request (default %d), state kept for %d\n",
                max_pass, start_pass, max_pass);
    }
    size_t st_bytes = 0;
    for (uint32_t iv = 0; iv < ik.n_visit; ++iv)
        st_bytes += 2ull * n_embd_kv * (ik.is_swa[iv] ? ring : n_ctx) * 2 + (size_t) (ik.n_conv - 1) * n_ch * 4;

    model.set_embd_q8(cfg.embd_q8);
    const size_t wbytes = model.weight_bytes(gpu);
    const size_t reserve = 1200ull << 20;   // compute buffers (prefill attention bias dominates)
    offload = gpu != cpu && wbytes + st_bytes + reserve > budget;
    fprintf(stderr, "inkling2: weights %zu MB, state %zu MB (KV: %u full-attn visits x %d + %u SWA x %d ring)%s\n",
            wbytes >> 20, st_bytes >> 20,
            (unsigned) std::count(ik.is_swa.begin(), ik.is_swa.end(), 0), n_ctx,
            (unsigned) std::count(ik.is_swa.begin(), ik.is_swa.end(), 1), ring,
            offload ? " -> experts offloaded to RAM" : "");
    if (getenv("INK_FORCE_OFFLOAD")) offload = gpu != cpu;

    if (offload) {
        ggml_backend_dev_t gdev = ggml_backend_get_device(gpu);
        ggml_backend_buffer_type_t host_bt = ggml_backend_dev_host_buffer_type(gdev);
        if (!host_bt) host_bt = ggml_backend_cpu_buffer_type();
        model.load_weights_split(gpu, host_bt, wbufs_gpu, wbufs_cpu);
    } else {
        model.load_weights(gpu);   // Model owns the buffer
    }

    // ---- state buffers ----
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * (ik.n_visit * 3 + 4);
    p.no_alloc = true;
    st_ctx = ggml_init(p);
    for (uint32_t iv = 0; iv < ik.n_visit; ++iv) {
        const int cap = ik.is_swa[iv] ? ring : n_ctx;
        st_k.push_back(ggml_new_tensor_2d(st_ctx, GGML_TYPE_F16, n_embd_kv, cap));
        st_v.push_back(flash ? ggml_new_tensor_2d(st_ctx, GGML_TYPE_F16, n_embd_kv, cap)
                             : ggml_new_tensor_2d(st_ctx, GGML_TYPE_F16, cap, n_embd_kv));   // manual path: transposed
        st_conv.push_back(ggml_new_tensor_2d(st_ctx, GGML_TYPE_F32, ik.n_conv - 1, n_ch));
    }
    st_buf = ggml_backend_alloc_ctx_tensors(st_ctx, gpu);
    if (!st_buf) throw std::runtime_error("inkling2: cannot allocate KV / conv state");
    ggml_backend_buffer_clear(st_buf, 0);
    if (max_pass != start_pass) model.set_loop_passes(start_pass);
    map_state();

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
    if (offload) init_offload(budget, st_bytes);

    // --mtp: a DSpark draft next to the model (dspark-*.gguf), or INK_DSPARK=<file>
    if (cfg.use_mtp && (ik.n_pass != ik.n_pass_file || max_pass != start_pass)) {
        // the draft reads fixed visits of the trained loop
        fprintf(stderr, "warning: DSpark only fits the model's own %u loops; ignored with --loops %u%s\n",
                ik.n_pass_file, ik.n_pass, max_pass != start_pass ? " and switchable loops" : "");
    } else if (cfg.use_mtp) {
        std::string dp = getenv("INK_DSPARK") ? getenv("INK_DSPARK") : "";
        if (dp.empty()) {
            namespace fs = std::filesystem;
            const fs::path dir = fs::path(model.path()).parent_path();
            for (const auto & e : fs::directory_iterator(dir.empty() ? fs::path(".") : dir)) {
                const std::string fn = e.path().filename().string();
                if (fn.rfind("dspark", 0) == 0 && e.path().extension() == ".gguf") { dp = e.path().string(); break; }
            }
        }
        if (dp.empty())      fprintf(stderr, "inkling2: --mtp: no dspark-*.gguf next to the model\n");
        else if (offload)    fprintf(stderr, "inkling2: --mtp: DSpark runs on the resident path only (offload is on), skipped\n");
        else                 dspark_init(dp);
    }
}

// ---------------------------------------------------------------------------
// graph pieces
// ---------------------------------------------------------------------------

// scale each row down to an L2 norm of at most thr, after the optional per-channel scale
ggml_tensor * InklingEngine::Impl::clip(Gr & g, ggml_tensor * cur, ggml_tensor * scale, float thr) {
    ggml_context * ctx = g.ctx;
    if (scale) cur = ggml_mul(ctx, cur, scale);
    ggml_tensor * norm = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, cur)));
    norm = ggml_scale_bias(ctx, norm, 1.0f, 1e-6f);
    ggml_tensor * factor = ggml_div(ctx, ggml_fill(ctx, norm, thr), norm);
    return ggml_mul(ctx, cur, ggml_clamp(ctx, factor, 0.0f, 1.0f));
}

// attention over depth: mix the earlier block outputs and cur with learned queries
ggml_tensor * InklingEngine::Impl::mix(Gr & g, const std::vector<ggml_tensor *> & blocks,
                                       ggml_tensor * cur, ggml_tensor * query) {
    ggml_context * ctx = g.ctx;
    const int64_t E = hp.n_embd, T = cur->ne[1];
    ggml_tensor * states = ggml_reshape_3d(ctx, cur, E, 1, T);
    for (ggml_tensor * b : blocks)
        states = ggml_concat(ctx, ggml_reshape_3d(ctx, b, E, 1, T), states, 1);
    states = ggml_rms_norm(ctx, states, 1e-6f);
    const int64_t nq = query->ne[1];
    ggml_tensor * queries = ggml_repeat_4d(ctx, query, E, nq, T, 1);
    ggml_tensor * scores = ggml_mul_mat(ctx, states, queries);
    scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / sqrtf((float) E), 0.0f);
    return ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, states)), scores);   // [E, nq, T]
}

// loop message controller: (1 - tanh(F h))*cur + tanh(G h)*msg, h = D [rms(cur); rms(msg)]
ggml_tensor * InklingEngine::Impl::control(Gr & g, ggml_tensor * cur, ggml_tensor * msg, int ip, int module) {
    static const char * names[2][4] = {
        { "attn",   "k",      "v",      "mlp"      },   // unsloth
        { "q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl" },   // llama.cpp
    };
    const char * name = names[ik.per_layer_msg ? 1 : 0][module];
    ggml_context * ctx = g.ctx;
    const int64_t E = hp.n_embd, T = cur->ne[1];
    char fd[96], fg[96], ff[96];
    snprintf(fd, sizeof(fd), "blk.%%d.loop_%s_down.weight",   name);
    snprintf(fg, sizeof(fg), "blk.%%d.loop_%s_gain.weight",   name);
    snprintf(ff, sizeof(ff), "blk.%%d.loop_%s_forget.weight", name);
    ggml_tensor * h = ggml_concat(ctx, ggml_reshape_3d(ctx, cur, E, 1, T), ggml_reshape_3d(ctx, msg, E, 1, T), 1);
    h = ggml_reshape_2d(ctx, ggml_rms_norm(ctx, h, 1e-6f), 2 * E, T);
    h = ggml_mul_mat(ctx, W(fd, ip), h);
    ggml_tensor * gain   = ggml_tanh(ctx, ggml_mul_mat(ctx, W(fg, ip), h));
    ggml_tensor * forget = ggml_tanh(ctx, ggml_mul_mat(ctx, W(ff, ip), h));
    return ggml_add(ctx, ggml_mul(ctx, ggml_scale_bias(ctx, forget, -1.0f, 1.0f), cur), ggml_mul(ctx, gain, msg));
}

// causal short convolution with a residual; `dyn` adds input-dependent taps per head.
// The last n_conv-1 inputs live in the visit's conv state at channel `offset`.
ggml_tensor * InklingEngine::Impl::convolve(Gr & g, int iv, ggml_tensor * cur, ggml_tensor * weight,
                                            ggml_tensor * dyn, int64_t offset) {
    ggml_context * ctx = g.ctx;
    const int64_t nc = ik.n_conv, C = cur->ne[0], T = cur->ne[1];
    ggml_tensor * state = conv_l[(size_t) iv];
    ggml_tensor * past = ggml_view_2d(ctx, state, nc - 1, C, state->nb[1], (size_t) offset * state->nb[1]);
    ggml_tensor * x = ggml_concat(ctx, past, ggml_transpose(ctx, cur), 0);          // [nc-1+T, C]
    ggml_tensor * last = ggml_view_2d(ctx, x, nc - 1, C, x->nb[1], (size_t) T * x->nb[0]);
    ggml_build_forward_expand(g.gf, ggml_cpy(ctx, last, past));
    if (xs_save) {   // keep the whole window: a rejected tail rolls the state back from it
        ggml_tensor * xsv = ds.xs[(size_t) iv];
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, x, ggml_view_2d(ctx, xsv, nc - 1 + T, C, xsv->nb[1], (size_t) offset * xsv->nb[1])));
    }

    ggml_tensor * out = ggml_ssm_conv(ctx, ggml_reshape_3d(ctx, x, nc - 1 + T, C, 1), weight);
    out = ggml_reshape_2d(ctx, out, C, T);
    if (dyn) {
        const int64_t nh = dyn->ne[0] / nc;
        ggml_tensor * d3 = ggml_reshape_3d(ctx, dyn, nc, nh, T);
        for (int64_t lag = 0; lag < nc; ++lag) {
            ggml_tensor * tap = ggml_view_3d(ctx, d3, 1, nh, T, d3->nb[1], d3->nb[2], (size_t) lag * d3->nb[0]);
            ggml_tensor * win = ggml_view_2d(ctx, x, T, C, x->nb[1], (size_t) (nc - 1 - lag) * x->nb[0]);
            win = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_transpose(ctx, win)), C / nh, nh, T);
            out = ggml_add(ctx, out, ggml_reshape_2d(ctx, ggml_mul(ctx, win, tap), C, T));
        }
    }
    return ggml_scale(ctx, ggml_add(ctx, cur, out), 1.0f / sqrtf((float) ((dyn ? 2 : 1) * nc + 1)));
}

ggml_tensor * InklingEngine::Impl::visit_pre(Gr & g, St & st, int iv, ggml_tensor ** x_out) {
    ggml_context * ctx = g.ctx;
    const int64_t E = hp.n_embd, T = g.T;
    const int  ip  = (int) ik.phys((uint32_t) iv);
    const bool swa = ik.is_swa[(size_t) iv] != 0;
    const float eps = hp.rms_eps;

    // ---- loop bookkeeping: later passes restart from the prelude state ----
    ggml_tensor * messages[4] = { nullptr, nullptr, nullptr, nullptr };
    if (ik.in_core((uint32_t) ip)) {
        const int pass = (int) ik.pass((uint32_t) iv);
        if (ip == (int) ik.core_start) {
            if (pass == 0) {
                st.prelude_blocks = st.blocks;
                st.prelude_cur    = st.cur;
            } else {
                st.bank.assign(st.blocks.begin() + st.prelude_blocks.size() + 1, st.blocks.end());
                st.bank_last = st.cur;
                st.blocks    = st.prelude_blocks;
                st.cur       = st.prelude_cur;
            }
        }
        if (pass > 0 && ik.per_layer_msg) {
            // one query per module and layer: q, k, v, ffn
            ggml_tensor * m = mix(g, st.bank, st.bank_last, W("blk.%d.loop_msg_query.weight", ip));
            for (int j = 0; j < 4; ++j)
                messages[j] = ggml_cont(ctx, ggml_view_2d(ctx, m, E, T, m->nb[2], (size_t) j * m->nb[1]));
        } else if (pass > 0) {
            // the same queries as two model-wide tables, rows (q, ffn) and (k, v) of each layer
            const size_t row = 2 * (size_t) (ip - (int) ik.core_start);
            ggml_tensor * lrq = G("loop_res_query.weight"), * lkq = G("loop_kv_res_query.weight");
            ggml_tensor * q_attn = ggml_view_2d(ctx, lrq, E, 2, lrq->nb[1], row * lrq->nb[1]);
            ggml_tensor * q_kv   = ggml_view_2d(ctx, lkq, E, 2, lkq->nb[1], row * lkq->nb[1]);
            ggml_tensor * m_attn = mix(g, st.bank, st.bank_last, q_attn);
            ggml_tensor * m_kv   = mix(g, st.bank, st.bank_last, q_kv);
            for (int j = 0; j < 2; ++j) {
                messages[j == 0 ? 0 : 3] = ggml_cont(ctx, ggml_view_2d(ctx, m_attn, E, T, m_attn->nb[2], j * m_attn->nb[1]));
                messages[j == 0 ? 1 : 2] = ggml_cont(ctx, ggml_view_2d(ctx, m_kv,   E, T, m_kv->nb[2],   j * m_kv->nb[1]));
            }
        }
    }

    ggml_tensor * inpSA = st.cur;
    if (cap_feat)
        for (size_t j = 0; j < ds.target.size(); ++j)
            if (ds.target[j] == iv)
                ggml_build_forward_expand(g.gf, ggml_cpy(ctx, inpSA, ggml_view_2d(ctx, ds.feat, E, T, ds.feat->nb[1], j * E * 4)));
    ggml_tensor * inputs[3] = { st.cur, st.cur, st.cur };
    if (ggml_tensor * rq = Wopt("blk.%d.attn_res_query.weight", ip)) {
        ggml_tensor * mixed = mix(g, st.blocks, st.cur, rq);
        for (int j = 0; j < 3; ++j)
            inputs[j] = ggml_cont(ctx, ggml_view_2d(ctx, mixed, E, T, mixed->nb[2], (size_t) (swa ? 0 : j) * mixed->nb[1]));
    }
    if (messages[0])
        for (int j = 0; j < 3; ++j) inputs[j] = control(g, inputs[j], messages[j], ip, j);

    auto norm = [&](ggml_tensor * t, ggml_tensor * w) { return ggml_mul(ctx, ggml_rms_norm(ctx, t, eps), w); };
    if (swa && messages[0]) {
        for (int j = 0; j < 3; ++j) inputs[j] = norm(inputs[j], W("blk.%d.attn_norm.weight", ip));
    } else if (swa) {
        inputs[0] = norm(inputs[0], W("blk.%d.attn_norm.weight", ip));
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = norm(inputs[0], W("blk.%d.attn_q_inp_norm.weight", ip));
        inputs[1] = norm(inputs[1], W("blk.%d.attn_k_inp_norm.weight", ip));
        inputs[2] = norm(inputs[2], W("blk.%d.attn_v_inp_norm.weight", ip));
    }

    // ---- self-attention ----
    ggml_tensor * cur;
    {
        ggml_tensor * Q = ggml_mul_mat(ctx, W("blk.%d.attn_q.weight", ip), inputs[0]);
        ggml_tensor * K = convolve(g, iv, ggml_mul_mat(ctx, W("blk.%d.attn_k.weight", ip), inputs[1]),
                                   W("blk.%d.attn_k_conv.weight", ip),
                                   swa ? nullptr : ggml_mul_mat(ctx, W("blk.%d.attn_k_conv_proj.weight", ip), inputs[1]), 0);
        ggml_tensor * V = convolve(g, iv, ggml_mul_mat(ctx, W("blk.%d.attn_v.weight", ip), inputs[2]),
                                   W("blk.%d.attn_v_conv.weight", ip),
                                   swa ? nullptr : ggml_mul_mat(ctx, W("blk.%d.attn_v_conv_proj.weight", ip), inputs[2]),
                                   n_embd_kv);
        ggml_tensor * R = ggml_mul_mat(ctx, W("blk.%d.attn_rel.weight", ip), inputs[0]);

        Q = clip(g, ggml_reshape_3d(ctx, Q, n_hd, n_head,    T), W("blk.%d.attn_q_clip.weight", ip), ik.qk_clip);
        K = clip(g, ggml_reshape_3d(ctx, K, n_hd, n_head_kv, T), nullptr, ik.qk_clip);
        V = ggml_rms_norm(ctx, ggml_reshape_3d(ctx, V, n_hd, n_head_kv, T), eps);
        R = clip(g, ggml_reshape_3d(ctx, R, ik.n_rel, n_head, T), W("blk.%d.attn_rel_clip.weight", ip), ik.rel_clip);
        if (!swa) {
            Q = ggml_mul(ctx, Q, g.tau);
            R = ggml_mul(ctx, R, g.tau);
        }

        // write K / V rows into the visit's cache, then attend over the cached cells
        const int n_kv = swa ? g.n_kv_swa : g.n_kv_full;
        ggml_tensor * idx = swa ? g.idx_swa : g.idx_full;
        ggml_tensor * kc = ggml_set_rows(ctx, k_l[(size_t) iv], ggml_reshape_2d(ctx, K, n_embd_kv, T), idx);
        ggml_tensor * k3 = ggml_view_3d(ctx, kc, n_hd, n_head_kv, n_kv, (size_t) n_hd * ggml_element_size(kc), kc->nb[1], 0);
        ggml_tensor * vl = v_l[(size_t) iv];
        ggml_tensor * v_att = nullptr;   // flash: [hd, n_kv, n_head_kv]; manual: [n_kv, hd, n_head_kv]
        if (flash) {
            ggml_tensor * vc = ggml_set_rows(ctx, vl, ggml_reshape_2d(ctx, V, n_embd_kv, T), idx);
            v_att = ggml_permute(ctx, ggml_view_3d(ctx, vc, n_hd, n_head_kv, n_kv, (size_t) n_hd * 2, vc->nb[1], 0), 0, 2, 1, 3);
        } else {
            // V cached transposed ([cap, n_embd_kv]) so the manual path reads it
            // without a copy: one element per index, like llama.cpp's v_trans cache
            const int64_t cap = vl->ne[0];
            ggml_tensor * vc = ggml_set_rows(ctx, ggml_reshape_2d(ctx, vl, 1, ggml_nelements(vl)),
                                             ggml_reshape_2d(ctx, ggml_cont(ctx, V), 1, (int64_t) n_embd_kv * T),
                                             swa ? g.vidx_swa : g.vidx_full);
            v_att = ggml_view_3d(ctx, vc, n_kv, n_hd, n_head_kv, (size_t) cap * 2, (size_t) cap * 2 * n_hd, 0);
        }

        // relative bias: project R on the bias table, subtract the per-query sink
        // logit, then gather by distance -> [n_kv, T, n_head]
        ggml_tensor * sinks = ggml_mul_mat(ctx, W("blk.%d.attn_sink_proj.weight", ip), inputs[0]);   // [n_head, T]
        ggml_tensor * table = ggml_mul_mat(ctx, ggml_pad(ctx, W("blk.%d.attn_rel_bias.weight", ip), 0, 1, 0, 0), R);
        table = ggml_sub(ctx, table, ggml_reshape_3d(ctx, sinks, 1, n_head, T));
        // row ext+1 = -inf: a masked cell gathers it (the inputs encode masking as
        // that index), so the per-head mask is the gather alone, never a sum
        table = ggml_concat(ctx, table, ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_head, T), -INFINITY), 0);
        table = ggml_cont(ctx, ggml_permute(ctx, table, 1, 0, 2, 3));                // [n_head, ext+2, T]
        ggml_tensor * kq_b = ggml_get_rows(ctx, table, swa ? g.rel_swa : g.rel_full);  // [n_head, n_kv, T]

        ggml_tensor * q  = ggml_permute(ctx, Q, 0, 2, 1, 3);                         // [hd, T, n_head]
        ggml_tensor * k  = ggml_permute(ctx, k3, 0, 2, 1, 3);                        // [hd, n_kv, n_head_kv]
        if (flash) {
            // the bias rides in a per-head F16 mask, as llama.cpp does for this model
            // one copy: [n_head, n_kv, T] f32 -> [n_kv, T, n_head] f16. A cast, not a
            // cpy into a fresh tensor: gallocr keeps such a leaf for the whole graph,
            // so every visit's mask would be alive at once.
            ggml_tensor * m  = ggml_cast(ctx, ggml_permute(ctx, kq_b, 2, 0, 1, 3), GGML_TYPE_F16);
            ggml_tensor * fa = ggml_flash_attn_ext(ctx, q, k, v_att, m, 1.0f / sqrtf((float) n_hd), 0.0f, 0.0f);
            ggml_flash_attn_ext_add_sinks(fa, g.sink0);
            cur = ggml_reshape_2d(ctx, fa, n_hd * n_head, T);                        // fa: [hd, n_head, T]
        } else {
            ggml_tensor * kq = ggml_mul_mat(ctx, k, q);                              // [n_kv, T, n_head]
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float) n_hd));
            kq = ggml_add(ctx, kq, ggml_cont(ctx, ggml_permute(ctx, kq_b, 2, 0, 1, 3)));   // mask included
            kq = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
            ggml_soft_max_add_sinks(kq, g.sink0);
            ggml_tensor * kqv = ggml_mul_mat(ctx, v_att, kq);                        // [hd, T, n_head]
            cur = ggml_cont_2d(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3), n_hd * n_head, T);
        }

        cur = ggml_mul(ctx, cur, ggml_sigmoid(ctx, ggml_mul_mat(ctx, W("blk.%d.attn_gate.weight", ip), inputs[0])));
        cur = convolve(g, iv, ggml_mul_mat(ctx, W("blk.%d.attn_output.weight", ip), cur),
                       W("blk.%d.attn_post_conv.weight", ip), nullptr, 2 * n_embd_kv);
    }

    // the first layer of each block starts a new residual stream; the old one joins the mixed blocks
    if (ip % (int) ik.block_size == 0) st.blocks.push_back(inpSA);
    else                               cur = ggml_add(ctx, inpSA, cur);
    ggml_tensor * ffn_inp = cur;

    ggml_tensor * x = ggml_reshape_2d(ctx, mix(g, st.blocks, cur, W("blk.%d.ffn_res_query.weight", ip)), E, T);
    if (messages[3]) x = control(g, x, messages[3], ip, 3);
    *x_out = norm(x, W("blk.%d.ffn_norm.weight", ip));
    return ffn_inp;
}

void InklingEngine::Impl::route(Gr & g, int ip, ggml_tensor * x, ggml_tensor ** sel, ggml_tensor ** w) {
    ggml_context * ctx = g.ctx;
    const int n_exp = (int) hp.n_expert, n_used = (int) hp.n_expert_used, T = (int) x->ne[1];
    ggml_tensor * logits = ggml_mul_mat(ctx, W("blk.%d.ffn_gate_inp.weight", ip), x);   // [n_exp, T]
    ggml_prec_set_acc(logits, GGML_PREC_F32);
    ggml_tensor * probs = ggml_sigmoid(ctx, logits);
    ggml_tensor * rank = probs;
    if (ggml_tensor * b = Wopt("blk.%d.exp_probs_b.bias", ip)) rank = ggml_add(ctx, probs, b);
    *sel = ggml_argsort_top_k(ctx, rank, n_used);                                        // [n_used, T]
    ggml_tensor * wt = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_exp, T), *sel); // [1, n_used, T]
    wt = ggml_reshape_2d(ctx, wt, n_used, T);
    ggml_tensor * ws = ggml_clamp(ctx, ggml_sum_rows(ctx, wt), 1e-20f, INFINITY);
    wt = ggml_div(ctx, wt, ws);
    *w = ggml_scale(ctx, ggml_reshape_3d(ctx, wt, 1, n_used, T), hp.expert_weights_scale);
}

// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)
ggml_tensor * InklingEngine::Impl::situ(Gr & g, ggml_tensor * gate, ggml_tensor * up) {
    ggml_context * ctx = g.ctx;
    const float b = ik.situ_beta, lb = ik.situ_lin_beta;
    ggml_tensor * act = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, gate, 1.0f / b)), b);
    act = ggml_mul(ctx, act, ggml_sigmoid(ctx, gate));
    if (lb > 0.0f) up = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, up, 1.0f / lb)), lb);
    return ggml_mul(ctx, act, up);
}

// Routed experts over (tg, tu, td) indexed by `slots` (null: the model's own
// tensors, indexed by `sel`). Scales are per logical expert, so they always
// gather by `sel`. Returns the weighted sum [E, T].
ggml_tensor * InklingEngine::Impl::experts(Gr & g, int ip, ggml_tensor * x, ggml_tensor * sel, ggml_tensor * w,
                                           ggml_tensor * tg, ggml_tensor * tu, ggml_tensor * td,
                                           ggml_tensor * ig, ggml_tensor * iu, ggml_tensor * id,
                                           ggml_tensor * sc_gu, ggml_tensor * sc_dn) {
    ggml_context * ctx = g.ctx;
    const int64_t E = hp.n_embd, T = x->ne[1], n_exp = hp.n_expert;
    if (!ig) ig = sel;
    if (!iu) iu = sel;
    if (!id) id = sel;
    ggml_tensor * x3 = ggml_reshape_3d(ctx, x, E, 1, T);
    ggml_tensor * gate = ggml_mul_mat_id(ctx, tg, x3, ig);   // [ff, n_used, T]
    ggml_tensor * up   = ggml_mul_mat_id(ctx, tu, x3, iu);
    static const bool no_scale = getenv("INK_NO_EXP_SCALE") != nullptr;
    auto scale_of = [&](ggml_tensor * s) {
        ggml_tensor * r = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, s, 1, n_exp, 1), 1, n_exp, T, 1);
        return ggml_get_rows(ctx, r, sel);   // [1, n_used, T]
    };
    if (!sc_gu) sc_gu = Wopt("blk.%d.ffn_gate_up_exps.scale", ip);
    if (!sc_dn) sc_dn = Wopt("blk.%d.ffn_down_exps.scale", ip);
    if (!no_scale) {
        if (ggml_tensor * s = sc_gu) {
            ggml_tensor * sr = scale_of(s);
            gate = ggml_mul(ctx, gate, sr);
            up   = ggml_mul(ctx, up,   sr);
        }
    }
    ggml_tensor * ex = ggml_mul_mat_id(ctx, td, situ(g, gate, up), id);    // [E, n_used, T]
    // as llama.cpp does for NVFP4 weights (build_lora_mm_id): BF16 accumulation is allowed
    static const bool nvfp4_bf16 = getenv("INK_NVFP4_F32") == nullptr;
    if (nvfp4_bf16)
        for (ggml_tensor * t : { gate, up, ex })
            if (t->src[0]->type == GGML_TYPE_NVFP4) ggml_prec_set_acc(t, GGML_PREC_BF16);
    if (!no_scale)
        if (ggml_tensor * s = sc_dn) ex = ggml_mul(ctx, ex, scale_of(s));
    ex = ggml_mul(ctx, ex, w);
    ex = ggml_cont(ctx, ggml_permute(ctx, ex, 1, 0, 2, 3));                 // [n_used, E, T]
    return ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ex), E, T);
}

ggml_tensor * InklingEngine::Impl::visit_post(Gr & g, int iv, ggml_tensor * ffn_inp, ggml_tensor * moe_out) {
    const int ip = (int) ik.phys((uint32_t) iv);
    ggml_tensor * cur = convolve(g, iv, moe_out, W("blk.%d.ffn_post_conv.weight", ip), nullptr,
                                 2 * n_embd_kv + (int64_t) hp.n_embd);
    return ggml_add(g.ctx, ffn_inp, cur);
}

ggml_tensor * InklingEngine::Impl::head(Gr & g, St & st, bool last_only) {
    ggml_context * ctx = g.ctx;
    const int64_t E = hp.n_embd;
    std::vector<ggml_tensor *> blocks = st.blocks;
    ggml_tensor * cur = st.cur;
    int64_t T = cur->ne[1];
    if (cap_feat && std::find(ds.target.begin(), ds.target.end(), (int) ik.n_visit) != ds.target.end()) {
        // the draft reads the final normed hidden of every token
        const size_t j = (size_t) (std::find(ds.target.begin(), ds.target.end(), (int) ik.n_visit) - ds.target.begin());
        ggml_tensor * fh = ggml_reshape_2d(ctx, mix(g, blocks, cur, G("output_res_query.weight")), E, T);
        fh = ggml_mul(ctx, ggml_rms_norm(ctx, fh, hp.rms_eps), G("output_norm.weight"));
        ggml_build_forward_expand(g.gf, ggml_cpy(ctx, fh, ggml_view_2d(ctx, ds.feat, E, T, ds.feat->nb[1], j * E * 4)));
    }
    if (last_only && T > 1) {
        auto last = [&](ggml_tensor * t) { return ggml_view_2d(ctx, t, E, 1, t->nb[1], (size_t) (T - 1) * t->nb[1]); };
        for (auto & b : blocks) b = ggml_cont(ctx, last(b));
        cur = ggml_cont(ctx, last(cur));
        T = 1;
    }
    cur = ggml_reshape_2d(ctx, mix(g, blocks, cur, G("output_res_query.weight")), E, T);
    cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, hp.rms_eps), G("output_norm.weight"));
    cur = ggml_mul_mat(ctx, G("output.weight"), cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    if (ik.logit_cap != 0.0f)
        cur = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, cur, 1.0f / ik.logit_cap)), ik.logit_cap);
    return ggml_cont(ctx, ggml_view_2d(ctx, cur, ik.n_vocab, T, cur->nb[1], 0));
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

InklingEngine::Impl::Gr InklingEngine::Impl::new_graph(int T, int n_kv_full) {
    Gr g;
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false);
    p.no_alloc = true;
    g.ctx = ggml_init(p);
    g.gf  = ggml_new_graph_custom(g.ctx, GRAPH_NODES, false);
    g.T   = T;
    g.n_kv_full = n_kv_full;
    g.n_kv_swa  = std::min(ring, n_kv_full);
    auto in = [&](ggml_tensor * t, const char * nm) { ggml_set_input(t); ggml_set_name(t, nm); return t; };
    g.tau       = in(ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, 1, T), "ink_tau");
    g.rel_full  = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, g.n_kv_full, T), "ink_rel_full");
    g.rel_swa   = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, g.n_kv_swa, T), "ink_rel_swa");
    g.mask_full = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, g.n_kv_full, T), "ink_mask_full");
    g.mask_swa  = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, g.n_kv_swa, T), "ink_mask_swa");
    g.idx_full  = in(ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, T), "ink_idx_full");
    g.idx_swa   = in(ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, T), "ink_idx_swa");
    g.vidx_full = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_I64, (int64_t) n_embd_kv * T, 1), "ink_vidx_full");
    g.vidx_swa  = in(ggml_new_tensor_2d(g.ctx, GGML_TYPE_I64, (int64_t) n_embd_kv * T, 1), "ink_vidx_swa");
    g.sink0     = in(ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, n_head), "ink_sink0");
    return g;
}

void InklingEngine::Impl::set_inputs(Gr & g, int T) {
    const int n_total = n_past_ + T;
    std::vector<float> tau(T);
    for (int i = 0; i < T; ++i) {
        const double p = (double) (n_past_ + i) + 1.0;
        tau[(size_t) i] = (float) (1.0 + ik.log_alpha * std::log(std::max(1.0, p / ik.log_floor)));
    }
    ggml_backend_tensor_set(g.tau, tau.data(), 0, tau.size() * sizeof(float));

    const int ext = (int) ik.rel_extent;
    auto fill = [&](int n_kv, bool swa, ggml_tensor * rel, ggml_tensor * mask) {
        std::vector<int32_t> r((size_t) n_kv * T);
        std::vector<float>   m((size_t) n_kv * T);
        for (int i = 0; i < T; ++i) {
            const int pi = n_past_ + i;
            for (int j = 0; j < n_kv; ++j) {
                const int pj = swa ? pos_of_ring(j, n_total) : j;
                const int d  = pi - pj;
                const bool vis = pj >= 0 && d >= 0 && (!swa || d < (int) ik.n_swa);
                r[(size_t) i * n_kv + j] = !vis ? ext + 1 : (d < ext ? d : ext);
                m[(size_t) i * n_kv + j] = vis ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(rel,  r.data(), 0, r.size() * sizeof(int32_t));
        if (mask->buffer)   // the flash path encodes masking in `rel` and never reads it
            ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    };
    fill(g.n_kv_full, false, g.rel_full, g.mask_full);
    fill(g.n_kv_swa,  true,  g.rel_swa,  g.mask_swa);

    std::vector<int64_t> idf(T), ids(T);
    for (int i = 0; i < T; ++i) { idf[(size_t) i] = n_past_ + i; ids[(size_t) i] = (n_past_ + i) % ring; }
    ggml_backend_tensor_set(g.idx_full, idf.data(), 0, idf.size() * sizeof(int64_t));
    ggml_backend_tensor_set(g.idx_swa,  ids.data(), 0, ids.size() * sizeof(int64_t));
    if (!flash) {   // only the manual path's transposed V cache reads these
        std::vector<int64_t> vf, vs;
        fill_vidx(T, vf, vs);
        ggml_backend_tensor_set(g.vidx_full, vf.data(), 0, vf.size() * sizeof(int64_t));
        ggml_backend_tensor_set(g.vidx_swa,  vs.data(), 0, vs.size() * sizeof(int64_t));
    }
    std::vector<float> z((size_t) n_head, 0.0f);
    ggml_backend_tensor_set(g.sink0, z.data(), 0, z.size() * sizeof(float));
}

void InklingEngine::Impl::build_full(Gr & g, ggml_tensor * inp, bool want_logits, ggml_tensor ** out_p) {
    ggml_context * ctx = g.ctx;
    St st;
    if (inp->type == GGML_TYPE_F32) {   // image embeddings: not normalised
        st.cur = inp;
    } else {
        st.cur = ggml_get_rows(ctx, model.tok_embd_rows(), inp);
        st.cur = ggml_mul(ctx, ggml_rms_norm(ctx, st.cur, hp.rms_eps), G("token_embd_norm.weight"));
    }

    for (int iv = 0; iv < (int) ik.n_visit; ++iv) {
        const int ip = (int) ik.phys((uint32_t) iv);
        ggml_tensor * x = nullptr, * sel = nullptr, * w = nullptr;
        ggml_tensor * ffn_inp = visit_pre(g, st, iv, &x);
        route(g, ip, x, &sel, &w);
        ggml_build_forward_expand(g.gf, w);
        ggml_tensor * moe = experts(g, ip, x, sel, w, W("blk.%d.ffn_gate_exps.weight", ip),
                                    W("blk.%d.ffn_up_exps.weight", ip), W("blk.%d.ffn_down_exps.weight", ip), nullptr);
        st.cur = visit_post(g, iv, ffn_inp, moe);
    }
    ggml_tensor * out = nullptr;
    if (want_logits) {
        out = head(g, st, !all_logits);
        ggml_set_output(out);
        ggml_build_forward_expand(g.gf, out);
        if (am_out) ggml_build_forward_expand(g.gf, ggml_cpy(ctx, ggml_argmax(ctx, out), ggml_view_1d(ctx, am_out, out->ne[1], 0)));
    } else {
        ggml_build_forward_expand(g.gf, st.cur);
    }
    *out_p = out;
}

void InklingEngine::Impl::forward(const int32_t * toks, int T, bool want_logits) {
    if (n_past_ + T > n_ctx) throw std::runtime_error("inkling2: context full");
    if (offload) { forward_offload(toks, T, want_logits); return; }
    static const bool prof = getenv("INK_PROF") != nullptr;
    static const bool no_keep = getenv("INK_NO_KEEP") != nullptr;
    auto t0 = clk::now();
    // small batches (a speculative verify) keep their graph too, one per T
    static const bool keep_small = getenv("INK_VERIFY_BENCH") != nullptr;
    const bool use_kept = (T == 1 || (keep_small && T <= 8)) && want_logits && !no_keep && !no_keep_now && !cur_embd;
    Gr g; ggml_tensor * inp = nullptr, * out = nullptr;
    if (use_kept) {
        const int bucket = std::min(n_ctx, (n_past_ + T + 255) / 256 * 256);
        if (kept.bucket != bucket || kept.T != T) {
            free_kept();
            kept.g = new_graph(T, bucket);
            kept.inp = ggml_new_tensor_1d(kept.g.ctx, GGML_TYPE_I32, T);
            ggml_set_input(kept.inp); ggml_set_name(kept.inp, "ink_tokens");
            all_logits = T > 1;
            build_full(kept.g, kept.inp, true, &kept.out);
            all_logits = false;
            kept.T = T;
            kept.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
            if (!ggml_gallocr_alloc_graph(kept.ga, kept.g.gf)) throw std::runtime_error("inkling2: graph alloc failed");
            kept.bucket = bucket;
        }
        g = kept.g; inp = kept.inp; out = kept.out;
    } else {
        g = new_graph(T, n_past_ + T);
        inp = cur_embd ? ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, hp.n_embd, T)
                       : ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, T);
        ggml_set_input(inp); ggml_set_name(inp, "ink_tokens");
        build_full(g, inp, want_logits, &out);
    }
    auto t1 = clk::now();
    if (!use_kept && !ggml_gallocr_alloc_graph(galloc, g.gf)) throw std::runtime_error("inkling2: graph alloc failed");
    auto t2 = clk::now();
    if (cur_embd) ggml_backend_tensor_set(inp, cur_embd, 0, (size_t) T * hp.n_embd * sizeof(float));
    else          ggml_backend_tensor_set(inp, toks, 0, (size_t) T * sizeof(int32_t));
    set_inputs(g, T);
    auto t3 = clk::now();
    if (ggml_backend_graph_compute(gpu, g.gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("inkling2: compute failed");
    if (prof)
        fprintf(stderr, "[ink T=%d n_kv=%d nodes=%d compute buf %.0f MB build %.1f alloc %.1f inputs %.1f compute %.1f ms]\n", T,
                g.n_kv_full, ggml_graph_n_nodes(g.gf),
                ggml_gallocr_get_buffer_size(use_kept ? kept.ga : galloc, 0) / 1048576.0,
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                std::chrono::duration<double, std::milli>(t2 - t1).count(),
                std::chrono::duration<double, std::milli>(t3 - t2).count(), ms_since(t3));
    if (out) {
        logits.resize(ik.n_vocab);
        ggml_backend_tensor_get(out, logits.data(), 0, logits.size() * sizeof(float));
    }
    if (!use_kept) ggml_free(g.ctx);
    const int p0 = n_past_;
    n_past_ += T;
    // the draft needs the target's features of every committed position, whatever
    // path processed it (a server prefill, a plain decode, the prompt of a spec run)
    if (ds.on) dspark_inject(p0, T);
}

void InklingEngine::Impl::spec_verify(const int32_t * toks, int T) {
    if (n_past_ + T > n_ctx) throw std::runtime_error("inkling2: context full");
    const int bucket = std::min(n_ctx, (n_past_ + T + 255) / 256 * 256);
    if (!vkept.g.gf || vkept.bucket != bucket || vkept.T != T) {
        if (vkept.ga) ggml_gallocr_free(vkept.ga);
        if (vkept.g.ctx) ggml_free(vkept.g.ctx);
        vkept = Kept{};
        vkept.g = new_graph(T, bucket);
        vkept.inp = ggml_new_tensor_1d(vkept.g.ctx, GGML_TYPE_I32, T);
        ggml_set_input(vkept.inp);
        build_full(vkept.g, vkept.inp, true, &vkept.out);
        vkept.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
        if (!ggml_gallocr_alloc_graph(vkept.ga, vkept.g.gf)) throw std::runtime_error("inkling2: verify graph alloc failed");
        vkept.bucket = bucket;
        vkept.T = T;
    }
    ggml_backend_tensor_set(vkept.inp, toks, 0, (size_t) T * sizeof(int32_t));
    set_inputs(vkept.g, T);
    if (ggml_backend_graph_compute(gpu, vkept.g.gf) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("inkling2: verify failed");
    n_past_ += T;
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// RAM offload
// ---------------------------------------------------------------------------

namespace {
// Depth-attention state across segment graphs, as carry-slot ids (-1 = none).
struct SSt {
    std::vector<int> blocks, prelude_blocks, bank;
    int prelude_cur = -1, bank_last = -1, cur = -1, ffn_inp = -1, x = -1;
};
}

void InklingEngine::Impl::init_offload(size_t budget, size_t st_bytes) {
    const int E = (int) hp.n_embd, n_used = (int) hp.n_expert_used;
    if (const char * e = getenv("INK_CPU_MISS")) cpu_miss = atoi(e) != 0;
    if (const char * e = getenv("INK_REFILL"))   refill   = atoi(e);

    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * 64;
    p.no_alloc = true;
    car_ctx = ggml_init(p);
    int n_slots = 11;   // peak live values at a routed visit: blocks, prelude, bank, residual, expert input
    if (const char * e = getenv("INK_SLOTS")) n_slots = std::max(4, atoi(e));
    for (int i = 0; i < n_slots; ++i) slots.push_back(ggml_new_tensor_2d(car_ctx, GGML_TYPE_F32, E, UBATCH));
    p_tok       = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I32, UBATCH);
    p_tau       = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, UBATCH);
    p_rel_full  = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I32, (int64_t) n_ctx * UBATCH);
    p_mask_full = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) n_ctx * UBATCH);
    p_rel_swa   = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I32, (int64_t) ring * UBATCH);
    p_mask_swa  = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) ring * UBATCH);
    p_idx_full  = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I64, UBATCH);
    p_idx_swa   = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I64, UBATCH);
    p_vidx_full = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I64, (int64_t) n_embd_kv * UBATCH);
    p_vidx_swa  = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I64, (int64_t) n_embd_kv * UBATCH);
    p_sink0     = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, n_head);
    p_sel       = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I32, (int64_t) n_used * UBATCH);
    p_w         = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) n_used * UBATCH);
    p_gpu       = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) E * UBATCH);
    p_cpu       = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) E * UBATCH);
    // graph B's integer inputs, packed so one upload covers them: [gate|up|down|logical] x (k*T)
    p_bpack     = ggml_new_tensor_1d(car_ctx, GGML_TYPE_I32, (int64_t) 4 * n_used * UBATCH);
    p_bw        = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) n_used * UBATCH);
    p_rout      = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) 2 * n_used + E);
    p_tmp       = ggml_new_tensor_1d(car_ctx, GGML_TYPE_F32, (int64_t) hp.n_ff_exp * n_used * UBATCH);
    car_buf = ggml_backend_alloc_ctx_tensors(car_ctx, gpu);
    if (!car_buf) throw std::runtime_error("inkling2: cannot allocate offload carries");
    ggml_backend_buffer_clear(car_buf, 0);
    p_cpu_zero = true;
    {
        ggml_backend_buffer_type_t hb = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu));
        if (!hb) hb = ggml_backend_cpu_buffer_type();
        const size_t n_f = (size_t) (2 * n_used + E) + n_used + 2 * (size_t) E;
        pin_buf = ggml_backend_buft_alloc_buffer(hb, n_f * 4 + (size_t) 4 * n_used * 4 + 256);
        if (!pin_buf) throw std::runtime_error("inkling2: cannot allocate pinned staging");
        float * base = (float *) ggml_backend_buffer_get_base(pin_buf);
        pin_rout = base;
        pin_w    = pin_rout + 2 * n_used + E;
        pin_cpu  = pin_w + n_used;
        pin_zero = pin_cpu + E;
        pin_ids  = (int32_t *) (pin_zero + E);
        std::fill(pin_zero, pin_zero + E, 0.0f);
    }

    // host copies of the per-expert scales for the CPU expert graphs
    const int n_phys = (int) hp.n_layer;
    ggml_init_params hp2{};
    hp2.mem_size = ggml_tensor_overhead() * (2 * n_phys + 2);
    hp2.no_alloc = true;
    hs_ctx = ggml_init(hp2);
    for (int ip = 0; ip < n_phys; ++ip) {
        hs_gu.push_back(ggml_new_tensor_1d(hs_ctx, GGML_TYPE_F32, hp.n_expert));
        hs_dn.push_back(ggml_new_tensor_1d(hs_ctx, GGML_TYPE_F32, hp.n_expert));
    }
    hs_buf = ggml_backend_alloc_ctx_tensors(hs_ctx, cpu);
    std::vector<float> tmp(hp.n_expert);
    for (int ip = 0; ip < n_phys; ++ip) {
        ggml_tensor * a = Wopt("blk.%d.ffn_gate_up_exps.scale", ip), * b = Wopt("blk.%d.ffn_down_exps.scale", ip);
        std::fill(tmp.begin(), tmp.end(), 1.0f);
        if (a) ggml_backend_tensor_get(a, tmp.data(), 0, tmp.size() * sizeof(float));
        ggml_backend_tensor_set(hs_gu[(size_t) ip], tmp.data(), 0, tmp.size() * sizeof(float));
        std::fill(tmp.begin(), tmp.end(), 1.0f);
        if (b) ggml_backend_tensor_get(b, tmp.data(), 0, tmp.size() * sizeof(float));
        ggml_backend_tensor_set(hs_dn[(size_t) ip], tmp.data(), 0, tmp.size() * sizeof(float));
    }

    // ---- VRAM expert pool: what the budget leaves ----
    size_t gpu_w = 0;
    for (auto b : wbufs_gpu) gpu_w += ggml_backend_buffer_get_size(b);
    size_t reserve = 900ull << 20;
    if (const char * e = getenv("INK_RESERVE_MB")) reserve = (size_t) atoll(e) << 20;
    if (const char * e = getenv("INK_PLACE")) place = e;
    size_t scr_bytes = 0;
    if (place != "lru") {
        static const char * fmt[3] = { "blk.%d.ffn_gate_exps.weight", "blk.%d.ffn_up_exps.weight", "blk.%d.ffn_down_exps.weight" };
        size_t role_max[3] = { 0, 0, 0 };
        for (int ip = 0; ip < n_phys; ++ip)
            for (int r = 0; r < 3; ++r) role_max[r] = std::max(role_max[r], ggml_backend_buft_get_alloc_size(ggml_backend_get_default_buffer_type(gpu), W(fmt[r], ip)));
        const size_t align = 256;
        // One role at a time: prefill runs gate, up and down as three stages over
        // the same region, so the scratch is the largest role, not all three.
        scr_shared = getenv("INK_SCR_FULL") == nullptr;
        scr_bytes = scr_shared ? std::max({ GGML_PAD(role_max[0], align), GGML_PAD(role_max[1], align), GGML_PAD(role_max[2], align) })
                               : GGML_PAD(role_max[0], align) + GGML_PAD(role_max[1], align) + GGML_PAD(role_max[2], align);
        scr_buf = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(gpu), scr_bytes);
        if (!getenv("INK_NO_OVERLAP")) {
            gpu2 = ggml_backend_dev_init(ggml_backend_get_device(gpu), nullptr);
            if (gpu2) galloc2 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu2));
        }
        if (!scr_buf) throw std::runtime_error("inkling2: cannot allocate the prefill expert scratch");
        ggml_init_params sp{};
        sp.mem_size = ggml_tensor_overhead() * (3 * n_phys + 4);
        sp.no_alloc = true;
        scr_ctx = ggml_init(sp);
        char * base = (char *) ggml_backend_buffer_get_base(scr_buf);
        for (int ip = 0; ip < n_phys; ++ip) {
            size_t off = 0;
            for (int r = 0; r < 3; ++r) {
                ggml_tensor * src = W(fmt[r], ip);
                ggml_tensor * t = ggml_new_tensor(scr_ctx, src->type, ggml_n_dims(src), src->ne);
                if (ggml_backend_tensor_alloc(scr_buf, t, base + off) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("inkling2: scratch view alloc failed");
                scr[r].push_back(t);
                if (!scr_shared) off += GGML_PAD(role_max[r], align);
            }
        }
    }
    const size_t used = gpu_w + st_bytes + ggml_backend_buffer_get_size(car_buf) + reserve + scr_bytes;
    const size_t pool = budget > used ? budget - used : 0;
    fprintf(stderr, "inkling2: offload: GPU weights %zu MB + state %zu MB + carries %zu MB + reserve %zu MB"
                    " + prefill scratch %zu MB -> expert pool %zu MB (%s)\n",
            gpu_w >> 20, st_bytes >> 20, ggml_backend_buffer_get_size(car_buf) >> 20, reserve >> 20,
            scr_bytes >> 20, pool >> 20,
            cpu_miss ? "misses computed on the CPU" : "misses fetched into VRAM");
    pool_bytes = pool;
    if (place == "lru")
        ec = std::make_unique<ExpertCache>(gpu, model, n_phys, (int) hp.n_expert, n_used, pool);
    l_sel.assign((size_t) n_phys, 0);
    l_gpu.assign((size_t) n_phys, 0);

    if (place != "lru") {
        place_static = true;
        cpu_miss = true;
        refill = 0;
        // hot: decode may still top the pool up (INK_REFILL); its fused graphs are
        // rebuilt only at re-placement, so refills stay out of fully resident layers
        if (const char * e = getenv("INK_REFILL")) if (place == "hot") refill = atoi(e);
        // value of keeping (layer, expert): its layer's visits per token for
        // "core" (and hot before any prompt has been seen), the layer index for
        // "front" (what --n-cpu-moe's complement looks like: last layers first)
        hot = place == "hot";
        std::vector<double> value((size_t) n_phys * hp.n_expert);
        for (int ip = 0; ip < n_phys; ++ip) {
            double v;
            if (place == "core" || hot) v = (ik.in_core((uint32_t) ip) ? ik.n_pass : 1) * 1000.0 - ip;
            else if (place == "front")  v = ip;
            else throw std::runtime_error("INK_PLACE: core, hot, front or lru");
            for (int e = 0; e < (int) hp.n_expert; ++e) value[(size_t) ip * hp.n_expert + e] = v - e * 1e-3;
        }
        auto t0 = clk::now();
        build_pool(plan_set(value));
        fprintf(stderr, "inkling2: static placement '%s' filled in %.0f ms; resident experts per layer:",
                place.c_str(), ms_since(t0));
        for (int ip = 0; ip < n_phys; ++ip) {
            int n = 0;
            for (int e = 0; e < (int) hp.n_expert; ++e) {
                bool r = true;
                for (int k = 0; k < 3 && r; ++k) r = ec->resident((ExpertCache::Role) k, ip, e);
                n += r;
            }
            fprintf(stderr, " %d%s:%d", ip, ik.in_core((uint32_t) ip) ? "*" : "", n);
            full_res.push_back(n == (int) hp.n_expert ? 1 : 0);
        }
        fprintf(stderr, "\n");

        const char * fz = getenv("INK_FUSE");
        fuse = !(fz && atoi(fz) == 0) && refill == 0;   // refills would move slots under a fused graph
        if (fuse) {
            ggml_init_params mp{};
            mp.mem_size = ggml_tensor_overhead() * 2;
            mp.no_alloc = true;
            map_ctx = ggml_init(mp);
            p_slotmap = ggml_new_tensor_2d(map_ctx, GGML_TYPE_I32, hp.n_expert, 3 * n_phys);
            map_buf = ggml_backend_alloc_ctx_tensors(map_ctx, gpu);
            if (!map_buf) throw std::runtime_error("inkling2: cannot allocate the slot map");
        }
        pick_cnt.assign((size_t) n_phys * hp.n_expert, 0);
        refresh_residency();
    }
}

void InklingEngine::Impl::upload_inputs(int T, int n_kv_full) {
    const int n_total = n_past_ + T;
    const int n_kv_swa = std::min(ring, n_kv_full);
    std::vector<float> tau(T);
    for (int i = 0; i < T; ++i) {
        const double p = (double) (n_past_ + i) + 1.0;
        tau[(size_t) i] = (float) (1.0 + ik.log_alpha * std::log(std::max(1.0, p / ik.log_floor)));
    }
    ggml_backend_tensor_set(p_tau, tau.data(), 0, tau.size() * sizeof(float));
    const int ext = (int) ik.rel_extent;
    auto fill = [&](int n_kv, bool swa, ggml_tensor * rel, ggml_tensor * mask) {
        std::vector<int32_t> r((size_t) n_kv * T);
        std::vector<float>   m((size_t) n_kv * T);
        for (int i = 0; i < T; ++i) {
            const int pi = n_past_ + i;
            for (int j = 0; j < n_kv; ++j) {
                const int pj = swa ? pos_of_ring(j, n_total) : j;
                const int d  = pi - pj;
                const bool vis = pj >= 0 && d >= 0 && (!swa || d < (int) ik.n_swa);
                r[(size_t) i * n_kv + j] = !vis ? ext + 1 : (d < ext ? d : ext);
                m[(size_t) i * n_kv + j] = vis ? 0.0f : -INFINITY;
            }
        }
        ggml_backend_tensor_set(rel,  r.data(), 0, r.size() * sizeof(int32_t));
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    };
    fill(n_kv_full, false, p_rel_full, p_mask_full);
    fill(n_kv_swa,  true,  p_rel_swa,  p_mask_swa);
    std::vector<int64_t> idf(T), ids(T);
    for (int i = 0; i < T; ++i) { idf[(size_t) i] = n_past_ + i; ids[(size_t) i] = (n_past_ + i) % ring; }
    ggml_backend_tensor_set(p_idx_full, idf.data(), 0, idf.size() * sizeof(int64_t));
    ggml_backend_tensor_set(p_idx_swa,  ids.data(), 0, ids.size() * sizeof(int64_t));
    std::vector<int64_t> vf, vs;
    fill_vidx(T, vf, vs);
    ggml_backend_tensor_set(p_vidx_full, vf.data(), 0, vf.size() * sizeof(int64_t));
    ggml_backend_tensor_set(p_vidx_swa,  vs.data(), 0, vs.size() * sizeof(int64_t));
}

InklingEngine::Impl::Gr InklingEngine::Impl::seg_graph(int T, int n_kv_full, int nodes) {
    Gr g;
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false);
    p.no_alloc = true;
    g.ctx = ggml_init(p);
    g.gf  = ggml_new_graph_custom(g.ctx, nodes, false);
    g.T   = T;
    g.n_kv_full = n_kv_full;
    g.n_kv_swa  = std::min(ring, n_kv_full);
    ggml_context * c = g.ctx;
    g.tau       = ggml_view_3d(c, p_tau, 1, 1, T, 4, 4, 0);
    g.rel_full  = ggml_view_2d(c, p_rel_full,  g.n_kv_full, T, (size_t) g.n_kv_full * 4, 0);
    g.mask_full = ggml_view_2d(c, p_mask_full, g.n_kv_full, T, (size_t) g.n_kv_full * 4, 0);
    g.rel_swa   = ggml_view_2d(c, p_rel_swa,   g.n_kv_swa,  T, (size_t) g.n_kv_swa * 4, 0);
    g.mask_swa  = ggml_view_2d(c, p_mask_swa,  g.n_kv_swa,  T, (size_t) g.n_kv_swa * 4, 0);
    g.idx_full  = ggml_view_1d(c, p_idx_full, T, 0);
    g.idx_swa   = ggml_view_1d(c, p_idx_swa,  T, 0);
    g.vidx_full = ggml_view_2d(c, p_vidx_full, (int64_t) n_embd_kv * T, 1, (size_t) n_embd_kv * T * 8, 0);
    g.vidx_swa  = ggml_view_2d(c, p_vidx_swa,  (int64_t) n_embd_kv * T, 1, (size_t) n_embd_kv * T * 8, 0);
    g.sink0     = p_sink0;
    return g;
}

void InklingEngine::Impl::forward_offload(const int32_t * toks, int T, bool want_logits) {
    const int E = (int) hp.n_embd, n_used = (int) hp.n_expert_used, n_exp = (int) hp.n_expert;
    const int NV = (int) ik.n_visit;
    static const bool no_keep = getenv("INK_NO_KEEP") != nullptr;
    const bool decode = T == 1;
    const bool keep   = decode && want_logits && !no_keep && !cur_embd;
    const int  n_kv_full = keep ? std::min(n_ctx, (n_past_ + 1 + 255) / 256 * 256) : n_past_ + T;

    if (decode && replace_pending) {
        replace_pending = false;
        hot_placed = true;
        replace_by_count();
        // older turns count for half as much at every re-placement
        for (auto & c : pick_cnt) c /= 2;
        for (auto & sg : segA) free_seg(sg);   // the routed visits changed
        segA.assign((size_t) NV + 1, Seg{});
    }
    if (keep && seg_bucket != n_kv_full) {
        for (auto & sg : segA) free_seg(sg);
        segA.assign((size_t) NV + 1, Seg{});
        seg_bucket = n_kv_full;
    }
    static std::vector<SSt> sst_after;   // per visit, for kept graphs (deterministic)
    if (sst_after.size() != (size_t) NV + 1) sst_after.assign((size_t) NV + 1, SSt{});
    if (segB.size() != (size_t) NV * (n_used + 1)) segB.assign((size_t) NV * (n_used + 1), Seg{});
    if (segC.size() != (size_t) hp.n_layer * (n_used + 1)) segC.assign((size_t) hp.n_layer * (n_used + 1), Seg{});

    ggml_backend_tensor_set(p_tok, toks, 0, (size_t) T * sizeof(int32_t));
    upload_inputs(T, n_kv_full);

    auto t_tok = clk::now();
    SSt sst;
    std::vector<int32_t> sel((size_t) n_used * T), slot3[3];
    std::vector<float>   w((size_t) n_used * T), xh(E), cpu_out(E);
    for (auto & v : slot3) v.resize((size_t) n_used * T);
    std::vector<std::vector<uint8_t>> chosen(hp.n_layer, std::vector<uint8_t>(n_exp, 0));   // this token, per physical layer
    std::vector<std::pair<int, int>> missed;   // (ip, e) computed on the CPU this token

    // Build graph A for visit iv (iv == NV: the head), starting from `in`.
    // Visits [iv_from, iv) are fully resident and run complete inside this graph.
    auto build_A = [&](Seg & sg, int iv_from, int iv, const SSt & in, SSt & outs) {
        Gr g = seg_graph(T, n_kv_full, 8192 + 640 * (iv - iv_from));
        sg.ctx = g.ctx; sg.gf = g.gf;
        ggml_context * c = g.ctx;
        std::vector<ggml_tensor *> id2v(slots.size(), nullptr);
        auto view = [&](int id) -> ggml_tensor * {
            if (id < 0) return nullptr;
            if (!id2v[(size_t) id]) id2v[(size_t) id] = ggml_view_2d(c, slots[(size_t) id], E, T, slots[(size_t) id]->nb[1], 0);
            return id2v[(size_t) id];
        };
        St st;
        for (int b : in.blocks)         st.blocks.push_back(view(b));
        for (int b : in.prelude_blocks) st.prelude_blocks.push_back(view(b));
        for (int b : in.bank)           st.bank.push_back(view(b));
        st.prelude_cur = view(in.prelude_cur);
        st.bank_last   = view(in.bank_last);
        if (iv_from == 0 && cur_embd) {
            sg.embd = ggml_new_tensor_2d(c, GGML_TYPE_F32, E, T);
            ggml_set_input(sg.embd);
            st.cur = sg.embd;
        } else if (iv_from == 0) {
            ggml_tensor * tk = ggml_view_1d(c, p_tok, T, 0);
            st.cur = ggml_get_rows(c, model.tok_embd_rows(), tk);
            st.cur = ggml_mul(c, ggml_rms_norm(c, st.cur, hp.rms_eps), G("token_embd_norm.weight"));
        } else {
            ggml_tensor * moe = ggml_view_2d(c, p_gpu, E, T, (size_t) E * 4, 0);
            if (decode || place_static) moe = ggml_add(c, moe, ggml_view_2d(c, p_cpu, E, T, (size_t) E * 4, 0));
            st.cur = visit_post(g, iv_from - 1, view(in.ffn_inp), moe);
        }
        for (int v = iv_from; v < iv; ++v) {
            const int vp = (int) ik.phys((uint32_t) v);
            ggml_tensor * vx = nullptr, * vs = nullptr, * vw = nullptr;
            ggml_tensor * vin = visit_pre(g, st, v, &vx);
            route(g, vp, vx, &vs, &vw);
            ggml_tensor * ids[3];
            for (int r = 0; r < 3; ++r) {
                ggml_tensor * row = ggml_view_2d(c, p_slotmap, 1, n_exp, 4,
                                                 ((size_t) r * hp.n_layer + vp) * n_exp * 4);
                ids[r] = ggml_reshape_2d(c, ggml_get_rows(c, row, vs), n_used, T);
            }
            ggml_tensor * moe = experts(g, vp, vx, vs, vw, ec->gate(vp), ec->up(vp), ec->down(vp), ids[0], ids[1], ids[2]);
            st.cur = visit_post(g, v, vin, moe);
        }
        if (iv == NV) {
            sg.out = head(g, st, true);
            ggml_set_output(sg.out);
            ggml_build_forward_expand(g.gf, sg.out);
            outs = SSt{};
            return;
        }
        const int ip = (int) ik.phys((uint32_t) iv);
        ggml_tensor * x = nullptr, * s = nullptr, * wt = nullptr;
        ggml_tensor * ffn_inp = visit_pre(g, st, iv, &x);
        route(g, ip, x, &s, &wt);
        sg.sel = ggml_view_2d(c, p_sel, n_used, T, (size_t) n_used * 4, 0);
        sg.w   = ggml_view_2d(c, p_w,   n_used, T, (size_t) n_used * 4, 0);
        ggml_build_forward_expand(g.gf, ggml_cpy(c, s, sg.sel));
        ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_reshape_2d(c, wt, n_used, T), sg.w));
        if (T == 1) {
            ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_cast(c, s, GGML_TYPE_F32), ggml_view_1d(c, p_rout, n_used, 0)));
            ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_reshape_1d(c, wt, n_used), ggml_view_1d(c, p_rout, n_used, (size_t) n_used * 4)));
            ggml_build_forward_expand(g.gf, ggml_cpy(c, ggml_reshape_1d(c, x, E), ggml_view_1d(c, p_rout, E, (size_t) 2 * n_used * 4)));
        }

        // materialise every live value into a carry slot
        std::vector<char> busy(slots.size(), 0);
        auto mark = [&](int id) { if (id >= 0) busy[(size_t) id] = 1; };
        for (int b : in.blocks) mark(b);
        for (int b : in.prelude_blocks) mark(b);
        for (int b : in.bank) mark(b);
        mark(in.prelude_cur); mark(in.bank_last); mark(in.ffn_inp); mark(in.cur);
        std::vector<std::pair<ggml_tensor *, int>> done;
        auto slot_of = [&](ggml_tensor * t) -> int {
            if (!t) return -1;
            for (size_t i = 0; i < id2v.size(); ++i) if (id2v[i] == t) return (int) i;
            for (auto & d : done) if (d.first == t) return d.second;
            for (size_t i = 0; i < busy.size(); ++i) {
                if (busy[i]) continue;
                busy[i] = 1;
                ggml_build_forward_expand(g.gf, ggml_cpy(c, t, view((int) i)));
                done.push_back({ t, (int) i });
                return (int) i;
            }
            throw std::runtime_error("inkling2: out of carry slots");
        };
        outs = SSt{};
        for (auto * b : st.blocks)         outs.blocks.push_back(slot_of(b));
        for (auto * b : st.prelude_blocks) outs.prelude_blocks.push_back(slot_of(b));
        for (auto * b : st.bank)           outs.bank.push_back(slot_of(b));
        outs.prelude_cur = slot_of(st.prelude_cur);
        outs.bank_last   = slot_of(st.bank_last);
        outs.ffn_inp     = slot_of(ffn_inp);
        outs.x           = slot_of(x);
        sg.x = view(outs.x);
    };

    // Graph B: k experts per token on VRAM slots, ids / weights from p_bids / p_bw.
    auto build_B = [&](Seg & sg, int iv, int k, int x_id, bool accumulate) {
        Gr g = seg_graph(T, n_kv_full);
        sg.ctx = g.ctx; sg.gf = g.gf;
        ggml_context * c = g.ctx;
        const int ip = (int) ik.phys((uint32_t) iv);
        ggml_tensor * x  = ggml_view_2d(c, slots[(size_t) x_id], E, T, slots[(size_t) x_id]->nb[1], 0);
        const size_t blk = (size_t) k * T * 4;
        ggml_tensor * ig = ggml_view_2d(c, p_bpack, k, T, (size_t) k * 4, 0 * blk);
        ggml_tensor * iu = ggml_view_2d(c, p_bpack, k, T, (size_t) k * 4, 1 * blk);
        ggml_tensor * id = ggml_view_2d(c, p_bpack, k, T, (size_t) k * 4, 2 * blk);
        ggml_tensor * bs = ggml_view_2d(c, p_bpack, k, T, (size_t) k * 4, 3 * blk);
        ggml_tensor * bw = ggml_view_3d(c, p_bw, 1, k, T, 4, (size_t) k * 4, 0);
        ggml_tensor * r = experts(g, ip, x, bs, bw, ec->gate(ip), ec->up(ip), ec->down(ip), ig, iu, id);
        ggml_tensor * dst = ggml_view_2d(c, p_gpu, E, T, (size_t) E * 4, 0);
        if (accumulate) r = ggml_add(c, r, dst);
        ggml_build_forward_expand(g.gf, ggml_cpy(c, r, dst));
    };

    auto alloc_run = [&](Seg & sg, bool kept, ggml_backend_t be) {
        if (!sg.ga) {
            sg.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            if (!ggml_gallocr_alloc_graph(sg.ga, sg.gf)) throw std::runtime_error("inkling2: segment alloc failed");
        } else if (!kept) {
            if (!ggml_gallocr_alloc_graph(sg.ga, sg.gf)) throw std::runtime_error("inkling2: segment alloc failed");
        }
    };

    // the visits that come back to the host (+ NV, the head); all of them unless fusing
    std::vector<int> routed;
    const bool fused = keep && fuse && place_static;
    for (int v = 0; v < NV; ++v)
        if (!fused || !full_res[ik.phys((uint32_t) v)]) routed.push_back(v);
    routed.push_back(NV);

    for (size_t j = 0; j < routed.size(); ++j) {
        const int iv = routed[j];
        const int iv_from = j == 0 ? 0 : routed[j - 1] + 1;
        if (iv > iv_from) {   // fused visits: every pick is on the GPU
            stx.sel += (uint64_t) n_used * (iv - iv_from);
            stx.gpu += (uint64_t) n_used * (iv - iv_from);
            for (int v = iv_from; v < iv; ++v) {
                l_sel[ik.phys((uint32_t) v)] += n_used;
                l_gpu[ik.phys((uint32_t) v)] += n_used;
            }
        }
        // prefill: start filling the scratch for this visit's layer on the second stream
        if (!decode && place_static && gpu2 && T >= 256 && iv < NV) {
            const int pip = (int) ik.phys((uint32_t) iv);
            bool full = true;
            for (int e = 0; e < n_exp && full; ++e)
                for (int r = 0; r < 3 && full; ++r) full = ec->resident((ExpertCache::Role) r, pip, e);
            if (!full) {
                pf_ctx = assemble(pip, nullptr, gpu2, galloc2, scr_shared ? 0 : -1);
                pf_ip  = pip;
            }
        }
        // ---- graph A ----
        auto t0 = clk::now();
        Seg tmpA;
        Seg * A = &tmpA;
        SSt next;
        if (keep) {
            A = &segA[(size_t) iv];
            if (!A->gf) {
                build_A(*A, iv_from, iv, sst, next);
                sst_after[(size_t) iv] = next;
                alloc_run(*A, true, gpu);
            }
            next = sst_after[(size_t) iv];
        } else {
            build_A(*A, iv_from, iv, sst, next);
            if (!ggml_gallocr_alloc_graph(galloc, A->gf)) throw std::runtime_error("inkling2: graph A alloc failed");
            if (A->embd) ggml_backend_tensor_set(A->embd, cur_embd, 0, (size_t) T * E * sizeof(float));
        }
        if (decode && iv < NV) {
            if (ggml_backend_graph_compute_async(gpu, A->gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("inkling2: graph A failed");
            ggml_backend_tensor_get_async(gpu, p_rout, pin_rout, 0, (size_t) (2 * n_used + E) * 4);
            ggml_backend_synchronize(gpu);
        } else if (ggml_backend_graph_compute(gpu, A->gf) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("inkling2: graph A failed");
        }
        stx.t_a += ms_since(t0);
        if (iv == NV) {
            if (want_logits) {
                logits.resize(ik.n_vocab);
                ggml_backend_tensor_get(A->out, logits.data(), 0, logits.size() * sizeof(float));
            }
            if (!keep) ggml_free(A->ctx);
            break;
        }
        sst = next;
        const int ip = (int) ik.phys((uint32_t) iv);
        if (!keep) ggml_free(A->ctx);

        if (decode) {
            for (int k = 0; k < n_used; ++k) {
                sel[(size_t) k] = (int32_t) pin_rout[k];
                w[(size_t) k]   = pin_rout[n_used + k];
            }
        } else {
            ggml_backend_tensor_get(p_sel, sel.data(), 0, sel.size() * sizeof(int32_t));
            ggml_backend_tensor_get(p_w,   w.data(),   0, w.size() * sizeof(float));
        }

        for (int t = 0; t < T; ++t) {
            for (int k = 0; k < n_used; ++k) {
                const int e = sel[(size_t) t * n_used + k];
                if (decode && ik.in_core((uint32_t) ip)) {
                    stx.core_sel++;
                    if (chosen[(size_t) ip][(size_t) e]) stx.core_repeat++;
                }
                if (decode) chosen[(size_t) ip][(size_t) e] = 1;
            }
        }

        if (decode) {
            // ---- split the selection: resident on the GPU, the rest on the CPU ----
            auto t1 = clk::now();
            if (!cpu_miss) {
                const uint64_t m0 = ec->stats().misses;
                ec->ensure(ip, sel.data(), n_used, slot3[0].data(), slot3[1].data(), slot3[2].data());
                stx.fetched += (ec->stats().misses - m0) / 3;
                stx.t_fetch += ms_since(t1);
            }
            std::vector<int32_t> hg, hu, hd, hs, ms;
            std::vector<float>   hw, mw;
            for (int k = 0; k < n_used; ++k) {
                const int e = sel[(size_t) k];
                bool res = true;
                for (int r = 0; r < 3 && res; ++r) res = ec->resident((ExpertCache::Role) r, ip, e);
                if (!cpu_miss) res = true;
                if (res) {
                    if (cpu_miss) {
                        for (int r = 0; r < 3; ++r) ec->touch((ExpertCache::Role) r, ip, e);
                        hg.push_back(ec->slot_of_row(ExpertCache::GATE, ip)[e]);
                        hu.push_back(ec->slot_of_row(ExpertCache::UP,   ip)[e]);
                        hd.push_back(ec->slot_of_row(ExpertCache::DOWN, ip)[e]);
                    } else {
                        hg.push_back(slot3[0][(size_t) k]); hu.push_back(slot3[1][(size_t) k]); hd.push_back(slot3[2][(size_t) k]);
                    }
                    hs.push_back(e); hw.push_back(w[(size_t) k]);
                } else {
                    ms.push_back(e); mw.push_back(w[(size_t) k]);
                }
            }
            const int nr = (int) hs.size(), nm = (int) ms.size();
            stx.sel += n_used; stx.gpu += nr; stx.cpu += nm;
            l_sel[(size_t) ip] += n_used; l_gpu[(size_t) ip] += nr;

            if (nr > 0) {
                std::copy(hg.begin(), hg.end(), pin_ids);
                std::copy(hu.begin(), hu.end(), pin_ids + nr);
                std::copy(hd.begin(), hd.end(), pin_ids + 2 * nr);
                std::copy(hs.begin(), hs.end(), pin_ids + 3 * nr);
                std::copy(hw.begin(), hw.end(), pin_w);
                ggml_backend_tensor_set_async(gpu, p_bpack, pin_ids, 0, (size_t) 4 * nr * 4);
                ggml_backend_tensor_set_async(gpu, p_bw,    pin_w,   0, (size_t) nr * 4);
                Seg tmpB;
                Seg * B = &tmpB;
                if (keep) {
                    B = &segB[(size_t) iv * (n_used + 1) + nr];
                    if (!B->gf) { build_B(*B, iv, nr, sst.x, false); alloc_run(*B, true, gpu); }
                } else {
                    build_B(*B, iv, nr, sst.x, false);
                    if (!ggml_gallocr_alloc_graph(galloc, B->gf)) throw std::runtime_error("inkling2: graph B alloc failed");
                }
                ggml_backend_graph_compute_async(gpu, B->gf);
                if (!keep) { ggml_backend_synchronize(gpu); ggml_free(B->ctx); }
            } else {
                ggml_backend_tensor_set_async(gpu, p_gpu, pin_zero, 0, (size_t) E * 4);
            }
            auto t2 = clk::now();
            stx.t_b += std::chrono::duration<double, std::milli>(t2 - t1).count();

            if (nm > 0) {
                std::copy(pin_rout + 2 * n_used, pin_rout + 2 * n_used + E, xh.begin());
                Seg & C = segC[(size_t) ip * (n_used + 1) + nm];
                if (!C.gf) {
                    Gr g;
                    ggml_init_params cp{};
                    cp.mem_size = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false);
                    cp.no_alloc = true;
                    g.ctx = ggml_init(cp);
                    g.gf  = ggml_new_graph_custom(g.ctx, 512, false);
                    g.T = 1;
                    C.ctx = g.ctx; C.gf = g.gf;
                    C.xin = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, E, 1);  ggml_set_input(C.xin);
                    C.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, nm, 1); ggml_set_input(C.ids);
                    C.win = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, 1, nm, 1); ggml_set_input(C.win);
                    C.out = experts(g, ip, C.xin, C.ids, C.win, W("blk.%d.ffn_gate_exps.weight", ip),
                                    W("blk.%d.ffn_up_exps.weight", ip), W("blk.%d.ffn_down_exps.weight", ip),
                                    nullptr, nullptr, nullptr, hs_gu[(size_t) ip], hs_dn[(size_t) ip]);
                    ggml_set_output(C.out);
                    ggml_build_forward_expand(g.gf, C.out);
                    C.ga = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
                    if (!ggml_gallocr_alloc_graph(C.ga, C.gf)) throw std::runtime_error("inkling2: CPU expert alloc failed");
                }
                ggml_backend_tensor_set(C.xin, xh.data(), 0, (size_t) E * 4);
                ggml_backend_tensor_set(C.ids, ms.data(), 0, (size_t) nm * 4);
                ggml_backend_tensor_set(C.win, mw.data(), 0, (size_t) nm * 4);
                ggml_backend_graph_compute(cpu, C.gf);
                ggml_backend_tensor_get(C.out, pin_cpu, 0, (size_t) E * 4);
                ggml_backend_tensor_set_async(gpu, p_cpu, pin_cpu, 0, (size_t) E * 4);
                p_cpu_zero = false;
                for (int e : ms) missed.push_back({ ip, e });
            } else if (!p_cpu_zero) {
                ggml_backend_tensor_set_async(gpu, p_cpu, pin_zero, 0, (size_t) E * 4);
                p_cpu_zero = true;
            }
            // no sync: the next graph A is queued behind graph B and these copies
            stx.t_c += ms_since(t2);
        } else if (place_static) {
            // ---- prefill, static pool: a fully resident layer runs from its slots;
            // any other layer is streamed whole into the scratch and runs there ----
            auto t1 = clk::now();
            for (int32_t e : sel) pick_cnt[(size_t) ip * n_exp + e]++;
            if (hot) replace_pending = true;
            bool full = true;
            for (int e = 0; e < n_exp && full; ++e)
                for (int r = 0; r < 3 && full; ++r) full = ec->resident((ExpertCache::Role) r, ip, e);
            Seg B;
            if (full) {
                for (size_t i = 0; i < sel.size(); ++i) {
                    slot3[0][i] = ec->slot_of_row(ExpertCache::GATE, ip)[sel[i]];
                    slot3[1][i] = ec->slot_of_row(ExpertCache::UP,   ip)[sel[i]];
                    slot3[2][i] = ec->slot_of_row(ExpertCache::DOWN, ip)[sel[i]];
                }
                std::vector<int32_t> pk;
                for (int r = 0; r < 3; ++r) pk.insert(pk.end(), slot3[r].begin(), slot3[r].end());
                pk.insert(pk.end(), sel.begin(), sel.end());
                ggml_backend_tensor_set(p_bpack, pk.data(), 0, pk.size() * 4);
                ggml_backend_tensor_set(p_bw, w.data(), 0, w.size() * 4);
                build_B(B, iv, n_used, sst.x, false);
            } else {
                // Assemble the layer in the scratch, at each expert's own index:
                // the chunk's picks that sit in the pool are copied on the GPU,
                // only the rest cross PCIe. INK_STREAM_WHOLE=1: the whole layer
                // over PCIe, as before.
                static const char * fmt[3] = { "blk.%d.ffn_gate_exps.weight", "blk.%d.ffn_up_exps.weight", "blk.%d.ffn_down_exps.weight" };
                static const bool whole = getenv("INK_STREAM_WHOLE") != nullptr;
                if (whole && !scr_shared) {   // needs INK_SCR_FULL=1 (three regions)
                    for (int r = 0; r < 3; ++r) {
                        ggml_tensor * src = W(fmt[r], ip);
                        ggml_backend_tensor_set_async(gpu, scr[r][(size_t) ip], src->data, 0, ggml_nbytes(src));
                        stx.streamed_bytes += ggml_nbytes(src);
                    }
                } else if (!scr_shared) {
                    if (pf_ip == ip) {   // filled on the second stream while graph A ran
                        ggml_backend_synchronize(gpu2);
                        if (pf_ctx) { ggml_free(pf_ctx); pf_ctx = nullptr; }
                        pf_ip = -1;
                    } else {
                        std::vector<uint8_t> want(n_exp, 0);
                        for (int32_t e : sel) want[(size_t) e] = 1;
                        if (ggml_context * cc = assemble(ip, &want, gpu, galloc)) ggml_free(cc);
                    }
                }
                stx.streamed_layers++;
                if (!scr_shared) {
                    Gr g = seg_graph(T, n_kv_full);
                    B.ctx = g.ctx; B.gf = g.gf;
                    ggml_context * c = g.ctx;
                    ggml_tensor * x  = ggml_view_2d(c, slots[(size_t) sst.x], E, T, slots[(size_t) sst.x]->nb[1], 0);
                    ggml_tensor * s2 = ggml_view_2d(c, p_sel, n_used, T, (size_t) n_used * 4, 0);
                    ggml_tensor * w3 = ggml_view_3d(c, p_w, 1, n_used, T, 4, (size_t) n_used * 4, 0);
                    ggml_tensor * r2 = experts(g, ip, x, s2, w3, scr[0][(size_t) ip], scr[1][(size_t) ip], scr[2][(size_t) ip]);
                    ggml_build_forward_expand(g.gf, ggml_cpy(c, r2, ggml_view_2d(c, p_gpu, E, T, (size_t) E * 4, 0)));
                } else {
                    // gate, up, down in turn through the one region; the gate output
                    // and then the activation wait in p_tmp between stages
                    std::vector<uint8_t> want(n_exp, 0);
                    for (int32_t e : sel) want[(size_t) e] = 1;
                    const int64_t ff = hp.n_ff_exp;
                    auto load = [&](int r) {
                        if (r == 0 && pf_ip == ip) {
                            ggml_backend_synchronize(gpu2);
                            if (pf_ctx) { ggml_free(pf_ctx); pf_ctx = nullptr; }
                            pf_ip = -1;
                        } else if (ggml_context * cc = assemble(ip, &want, gpu, galloc, r)) {
                            ggml_free(cc);
                        }
                    };
                    auto scale_rows = [&](Gr & g, const char * fmt, ggml_tensor * s2) -> ggml_tensor * {
                        ggml_tensor * sc = Wopt(fmt, ip);
                        if (!sc || getenv("INK_NO_EXP_SCALE")) return nullptr;
                        ggml_tensor * rr = ggml_repeat_4d(g.ctx, ggml_reshape_3d(g.ctx, sc, 1, n_exp, 1), 1, n_exp, T, 1);
                        return ggml_get_rows(g.ctx, rr, s2);
                    };
                    for (int stage = 0; stage < 3; ++stage) {
                        load(stage);
                        Gr g = seg_graph(T, n_kv_full);
                        ggml_context * c = g.ctx;
                        ggml_tensor * x  = ggml_view_2d(c, slots[(size_t) sst.x], E, T, slots[(size_t) sst.x]->nb[1], 0);
                        ggml_tensor * x3 = ggml_reshape_3d(c, x, E, 1, T);
                        ggml_tensor * s2 = ggml_view_2d(c, p_sel, n_used, T, (size_t) n_used * 4, 0);
                        ggml_tensor * tmp = ggml_view_3d(c, p_tmp, ff, n_used, T, (size_t) ff * 4, (size_t) ff * n_used * 4, 0);
                        if (stage == 0) {
                            ggml_tensor * gt = ggml_mul_mat_id(c, scr[0][(size_t) ip], x3, s2);
                            if (ggml_tensor * sr = scale_rows(g, "blk.%d.ffn_gate_up_exps.scale", s2)) gt = ggml_mul(c, gt, sr);
                            ggml_build_forward_expand(g.gf, ggml_cpy(c, gt, tmp));
                        } else if (stage == 1) {
                            ggml_tensor * up = ggml_mul_mat_id(c, scr[1][(size_t) ip], x3, s2);
                            if (ggml_tensor * sr = scale_rows(g, "blk.%d.ffn_gate_up_exps.scale", s2)) up = ggml_mul(c, up, sr);
                            ggml_build_forward_expand(g.gf, ggml_cpy(c, situ(g, tmp, up), tmp));
                        } else {
                            ggml_tensor * ex = ggml_mul_mat_id(c, scr[2][(size_t) ip], tmp, s2);   // [E, n_used, T]
                            if (scr[2][(size_t) ip]->type == GGML_TYPE_NVFP4 && !getenv("INK_NVFP4_F32")) ggml_prec_set_acc(ex, GGML_PREC_BF16);
                            if (ggml_tensor * sr = scale_rows(g, "blk.%d.ffn_down_exps.scale", s2)) ex = ggml_mul(c, ex, sr);
                            ex = ggml_mul(c, ex, ggml_view_3d(c, p_w, 1, n_used, T, 4, (size_t) n_used * 4, 0));
                            ex = ggml_cont(c, ggml_permute(c, ex, 1, 0, 2, 3));
                            ggml_tensor * r2 = ggml_reshape_2d(c, ggml_sum_rows(c, ex), E, T);
                            ggml_build_forward_expand(g.gf, ggml_cpy(c, r2, ggml_view_2d(c, p_gpu, E, T, (size_t) E * 4, 0)));
                        }
                        if (stage < 2) {
                            if (!ggml_gallocr_alloc_graph(galloc, g.gf)) throw std::runtime_error("inkling2: staged experts alloc failed");
                            if (ggml_backend_graph_compute(gpu, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("inkling2: staged experts failed");
                            ggml_free(g.ctx);
                        } else {
                            B.ctx = g.ctx; B.gf = g.gf;   // the last stage runs below like graph B
                        }
                    }
                }
            }
            if (!ggml_gallocr_alloc_graph(galloc, B.gf)) throw std::runtime_error("inkling2: graph B alloc failed");
            if (ggml_backend_graph_compute(gpu, B.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("inkling2: graph B failed");
            ggml_free(B.ctx);
            if (!p_cpu_zero) {   // the static tail adds p_cpu; nothing went to the CPU here
                ggml_backend_tensor_memset(p_cpu, 0, 0, (size_t) E * UBATCH * 4);
                p_cpu_zero = true;
            }
            stx.t_fetch += ms_since(t1);
        } else {
            // ---- prefill: the chunk's union of experts, in groups the pool can hold ----
            auto t1 = clk::now();
            std::vector<int> uni;
            std::vector<uint8_t> in_u(n_exp, 0);
            for (int32_t e : sel) if (!in_u[(size_t) e]) { in_u[(size_t) e] = 1; uni.push_back(e); }
            const int cap = std::max(1, ec->capacity(ip));
            const uint64_t m0 = ec->stats().misses;
            bool first = true;
            // Balanced groups, each of at least n_used experts: a token's
            // out-of-group picks need distinct stand-in slots (this ggml's CUDA
            // mul_mat_id does not take a repeated id within a row).
            if (cap < n_used) throw std::runtime_error("inkling2: expert pool smaller than one token's experts");
            const size_t n_groups = (uni.size() + cap - 1) / cap;
            const size_t gsz = (uni.size() + n_groups - 1) / n_groups;
            for (size_t g0 = 0; g0 < uni.size(); g0 += gsz) {
                const int ng = (int) std::min(uni.size() - g0, gsz);
                std::vector<int32_t> gs[3];
                for (auto & v : gs) v.resize((size_t) ng);
                ec->ensure(ip, uni.data() + g0, ng, gs[0].data(), gs[1].data(), gs[2].data());
                static const bool dbg = getenv("INK_DBG") != nullptr;
                if (dbg) {
                    int bad = 0;
                    for (int r = 0; r < 3; ++r)
                        for (int i = 0; i < ng; ++i)
                            bad += gs[r][(size_t) i] < 0 || gs[r][(size_t) i] >= ec->tensor((ExpertCache::Role) r, ip)->ne[2];
                    fprintf(stderr, "[ink prefill iv %d ip %d union %zu cap %d group %zu+%d bad %d]\n",
                            iv, ip, uni.size(), cap, g0, ng, bad);
                }
                std::vector<int> pos_in_group(n_exp, -1);
                for (int i = 0; i < ng; ++i) pos_in_group[(size_t) uni[g0 + (size_t) i]] = i;
                std::vector<float> bw((size_t) n_used * T);
                for (int t = 0; t < T; ++t) {
                    // group members this row already uses, then fill the rest with unused ones
                    std::vector<uint8_t> used_in_row((size_t) ng, 0);
                    for (int k = 0; k < n_used; ++k) {
                        const int pi = pos_in_group[(size_t) sel[(size_t) t * n_used + k]];
                        if (pi >= 0) used_in_row[(size_t) pi] = 1;
                    }
                    int next_free = 0;
                    for (int k = 0; k < n_used; ++k) {
                        const size_t i = (size_t) t * n_used + k;
                        int pi = pos_in_group[(size_t) sel[i]];
                        bw[i] = pi >= 0 ? w[i] : 0.0f;
                        if (pi < 0) {
                            while (used_in_row[(size_t) next_free]) ++next_free;
                            pi = next_free;
                            used_in_row[(size_t) pi] = 1;
                        }
                        for (int r = 0; r < 3; ++r) slot3[r][i] = gs[r][(size_t) pi];
                    }
                }
                std::vector<int32_t> pk;
                for (int r = 0; r < 3; ++r) pk.insert(pk.end(), slot3[r].begin(), slot3[r].end());
                pk.insert(pk.end(), sel.begin(), sel.end());
                ggml_backend_tensor_set(p_bpack, pk.data(), 0, pk.size() * 4);
                ggml_backend_tensor_set(p_bw, bw.data(), 0, bw.size() * 4);
                Seg B;
                build_B(B, iv, n_used, sst.x, !first);
                if (!ggml_gallocr_alloc_graph(galloc, B.gf)) throw std::runtime_error("inkling2: graph B alloc failed");
                if (ggml_backend_graph_compute(gpu, B.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("inkling2: graph B failed");
                ggml_free(B.ctx);
                first = false;
            }
            stx.fetched += (ec->stats().misses - m0) / 3;
            stx.t_fetch += ms_since(t1);
        }
    }

    if (decode) {
        stx.tokens++;
        // refill: make a few of this token's CPU experts resident (most recent first)
        int left = refill;
        for (auto it = missed.rbegin(); it != missed.rend() && left > 0; ++it, --left)
            ec->ensure_resident(it->first, it->second);
    }
    (void) t_tok;
    n_past_ += T;
}

ggml_context * InklingEngine::Impl::assemble(int ip, const std::vector<uint8_t> * want, ggml_backend_t be, ggml_gallocr_t ga, int role) {
    static const char * fmt[3] = { "blk.%d.ffn_gate_exps.weight", "blk.%d.ffn_up_exps.weight", "blk.%d.ffn_down_exps.weight" };
    const int n_exp = (int) hp.n_expert;
    // Zero it first when only some experts are written: CUDA mul_mat_id reads
    // past the picked ones, and what an earlier layer left there (another type)
    // can be NaN.
    if (want) ggml_backend_buffer_clear(scr_buf, 0);
    ggml_init_params cp{};
    const int nodes = 1024 + 8 * n_exp;
    cp.mem_size = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false);
    cp.no_alloc = true;
    ggml_context * cc = ggml_init(cp);
    ggml_cgraph * cg = ggml_new_graph_custom(cc, nodes, false);
    int n_d2d = 0;
    for (int e = 0; e < n_exp; ++e) {
        if (want && !(*want)[(size_t) e]) continue;
        bool r_all = true;
        for (int r = 0; r < 3 && r_all; ++r) r_all = ec->resident((ExpertCache::Role) r, ip, e);
        for (int r = 0; r < 3; ++r) {
            if (role >= 0 && r != role) continue;
            ggml_tensor * dst = scr[r][(size_t) ip];
            const size_t nb2 = dst->nb[2];
            if (r_all) {
                ggml_tensor * pool = ec->tensor((ExpertCache::Role) r, ip);
                const int slot = ec->slot_of_row((ExpertCache::Role) r, ip)[e];
                ggml_tensor * sv = ggml_view_3d(cc, pool, pool->ne[0], pool->ne[1], 1, pool->nb[1], pool->nb[2], (size_t) slot * pool->nb[2]);
                ggml_tensor * dv = ggml_view_3d(cc, dst,  dst->ne[0],  dst->ne[1],  1, dst->nb[1],  dst->nb[2],  (size_t) e * nb2);
                ggml_build_forward_expand(cg, ggml_cpy(cc, sv, dv));
                stx.d2d_bytes += nb2;
            } else {
                ggml_tensor * src = W(fmt[r], ip);
                ggml_backend_tensor_set_async(be, dst, (const char *) src->data + (size_t) e * nb2, (size_t) e * nb2, nb2);
                stx.streamed_bytes += nb2;
            }
        }
        n_d2d += r_all;
    }
    if (n_d2d > 0) {
        if (!ggml_gallocr_alloc_graph(ga, cg)) throw std::runtime_error("inkling2: scratch copy alloc failed");
        if (ggml_backend_graph_compute_async(be, cg) != GGML_STATUS_SUCCESS) throw std::runtime_error("inkling2: scratch copy failed");
    }
    return cc;
}

void InklingEngine::Impl::refresh_residency() {
    const int n_phys = (int) hp.n_layer, n_exp = (int) hp.n_expert;
    {   // how much of the pool holds experts whose three roles are all resident
        static const char * fmt[3] = { "blk.%d.ffn_gate_exps.weight", "blk.%d.ffn_up_exps.weight", "blk.%d.ffn_down_exps.weight" };
        double full_b = 0, part_b = 0;
        for (int ip = 0; ip < n_phys; ++ip)
            for (int e = 0; e < n_exp; ++e) {
                int nr = 0; double b = 0;
                for (int r = 0; r < 3; ++r) {
                    const double sb = (double) W(fmt[r], ip)->nb[2];
                    if (ec->resident((ExpertCache::Role) r, ip, e)) { ++nr; b += sb; }
                }
                if (nr == 3) full_b += b; else part_b += b;
            }
        fprintf(stderr, "inkling2: pool holds %.0f MB of complete experts and %.0f MB of incomplete ones (unusable)\n",
                full_b / 1048576.0, part_b / 1048576.0);
    }
    full_res.assign((size_t) n_phys, 0);
    for (int ip = 0; ip < n_phys; ++ip) {
        int n = 0;
        for (int e = 0; e < n_exp; ++e) {
            bool r = true;
            for (int k = 0; k < 3 && r; ++k) r = ec->resident((ExpertCache::Role) k, ip, e);
            n += r;
        }
        full_res[(size_t) ip] = n == n_exp ? 1 : 0;
    }
    if (!fuse || !p_slotmap) return;
    std::vector<int32_t> mv((size_t) n_exp * 3 * n_phys, 0);
    for (int r = 0; r < 3; ++r)
        for (int ip = 0; ip < n_phys; ++ip) {
            const int32_t * row = ec->slot_of_row((ExpertCache::Role) r, ip);
            std::copy(row, row + n_exp, mv.begin() + ((size_t) r * n_phys + ip) * n_exp);
        }
    ggml_backend_tensor_set(p_slotmap, mv.data(), 0, mv.size() * 4);
    int nf = 0;
    for (uint32_t iv = 0; iv < ik.n_visit; ++iv) nf += full_res[ik.phys(iv)];
    fprintf(stderr, "inkling2: %d of %u visits fully resident -> decoded inside fused graphs\n", nf, ik.n_visit);
}

std::vector<uint8_t> InklingEngine::Impl::plan_set(const std::vector<double> & value) const {
    static const char * fmt[3] = { "blk.%d.ffn_gate_exps.weight", "blk.%d.ffn_up_exps.weight", "blk.%d.ffn_down_exps.weight" };
    const int n_phys = (int) hp.n_layer, n_exp = (int) hp.n_expert;
    std::vector<size_t> eb((size_t) n_phys, 0);   // bytes of one expert, all three roles
    for (int ip = 0; ip < n_phys; ++ip)
        for (int r = 0; r < 3; ++r) eb[(size_t) ip] += W(fmt[r], ip)->nb[2];
    std::vector<int> keys((size_t) n_phys * n_exp);
    for (size_t k = 0; k < keys.size(); ++k) keys[k] = (int) k;
    std::stable_sort(keys.begin(), keys.end(), [&](int a, int b) { return value[(size_t) a] > value[(size_t) b]; });
    std::vector<uint8_t> want(keys.size(), 0);
    size_t used = 0;
    for (int k : keys) {
        const size_t need = eb[(size_t) (k / n_exp)];
        if (used + need > pool_bytes) continue;   // a smaller expert further down may still fit
        used += need;
        want[(size_t) k] = 1;
    }
    return want;
}

void InklingEngine::Impl::build_pool(const std::vector<uint8_t> & want) {
    const int n_phys = (int) hp.n_layer, n_exp = (int) hp.n_expert;
    // every graph that names a pool tensor goes with the old cache
    for (auto & sg : segA) free_seg(sg);
    for (auto & sg : segB) free_seg(sg);
    seg_bucket = 0;
    ec.reset();
    ec = std::make_unique<ExpertCache>(gpu, model, n_phys, n_exp, (int) hp.n_expert_used, pool_bytes,
                                       false, nullptr, &want);
    for (int k = 0; k < n_phys * n_exp; ++k)
        if (want[(size_t) k]) ec->ensure_resident(k / n_exp, k % n_exp);
    ggml_backend_synchronize(gpu);
}

void InklingEngine::Impl::replace_by_count() {
    const int n_phys = (int) hp.n_layer, n_exp = (int) hp.n_expert;
    auto t0 = clk::now();
    std::vector<double> value((size_t) n_phys * n_exp);
    for (int k = 0; k < n_phys * n_exp; ++k)
        value[(size_t) k] = pick_cnt[(size_t) k] + 1e-3 * (ik.in_core((uint32_t) (k / n_exp)) ? ik.n_pass : 1);
    build_pool(plan_set(value));
    uint64_t total = 0, covered = 0;
    for (int k = 0; k < n_phys * n_exp; ++k) {
        total += pick_cnt[(size_t) k];
        bool r = true;
        for (int q = 0; q < 3 && r; ++q) r = ec->resident((ExpertCache::Role) q, k / n_exp, k % n_exp);
        if (r) covered += pick_cnt[(size_t) k];
    }
    fprintf(stderr, "inkling2: pool rebuilt by prefill pick count in %.0f ms; it covers %.1f%% of the prompt's picks."
                    " resident per layer:", ms_since(t0), total ? 100.0 * covered / total : 0.0);
    for (int ip = 0; ip < n_phys; ++ip) {
        int n = 0;
        for (int e = 0; e < n_exp; ++e) {
            bool r = true;
            for (int q = 0; q < 3 && r; ++q) r = ec->resident((ExpertCache::Role) q, ip, e);
            n += r;
        }
        fprintf(stderr, " %d%s:%d", ip, ik.in_core((uint32_t) ip) ? "*" : "", n);
    }
    fprintf(stderr, "\n");
    refresh_residency();
}

void InklingEngine::Impl::dump_offload_stats() {
    if (!stx.sel) return;
    {
        size_t ka = 0, kb = 0, kc = 0;
        for (auto & sg : segA) if (sg.ga) ka += ggml_gallocr_get_buffer_size(sg.ga, 0);
        for (auto & sg : segB) if (sg.ga) kb += ggml_gallocr_get_buffer_size(sg.ga, 0);
        for (auto & sg : segC) if (sg.ga) kc += ggml_gallocr_get_buffer_size(sg.ga, 0);
        fprintf(stderr, "  kept graph compute buffers: A %.1f MB, B %.1f MB (GPU), C %.1f MB (CPU); shared galloc %.1f MB\n",
                ka / 1048576.0, kb / 1048576.0, kc / 1048576.0, ggml_gallocr_get_buffer_size(galloc, 0) / 1048576.0);
    }
    const double n = (double) std::max<uint64_t>(1, stx.tokens);
    fprintf(stderr, "inkling2 offload: %llu decode tokens, expert visits %.1f/token: GPU %.1f%%, CPU %.1f%%;"
                    " core-layer picks already made by an earlier pass this token %.1f%%\n",
            (unsigned long long) stx.tokens, stx.sel / n, 100.0 * stx.gpu / stx.sel, 100.0 * stx.cpu / stx.sel,
            stx.core_sel ? 100.0 * stx.core_repeat / stx.core_sel : 0.0);
    fprintf(stderr, "  per token: graph A %.2f ms, GPU experts %.2f ms, CPU experts+sync %.2f ms, fetch %.2f ms;"
                    " prefill/fetch total %.0f ms, %llu experts fetched\n",
            stx.t_a / n, stx.t_b / n, stx.t_c / n, 0.0, stx.t_fetch, (unsigned long long) stx.fetched);
    if (stx.streamed_layers)
        fprintf(stderr, "  prefill assembled %llu layer visits in the scratch: %.1f GB over PCIe, %.1f GB copied from the pool\n",
                (unsigned long long) stx.streamed_layers, stx.streamed_bytes / 1e9, stx.d2d_bytes / 1e9);
    fprintf(stderr, "  GPU share by physical layer:");
    for (size_t ip = 0; ip < l_sel.size(); ++ip)
        fprintf(stderr, " %zu%s:%.0f", ip, ik.in_core((uint32_t) ip) ? "*" : "", l_sel[ip] ? 100.0 * l_gpu[ip] / l_sel[ip] : 0.0);
    fprintf(stderr, "  (* = looped)\n");
}

// ---------------------------------------------------------------------------
// DSpark (Inkling2 variant of the dflash draft): llama.cpp's graph_inkling2 in
// src/models/dspark.cpp, branch pwilkin/inkling2-dspark.
// ---------------------------------------------------------------------------

void InklingEngine::Impl::dspark_init(const std::string & path) {
    ggml_context * meta = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx = &meta;
    ds.gg = gguf_init_from_file(path.c_str(), gp);
    if (!ds.gg) throw std::runtime_error("dspark: cannot open " + path);
    const gguf_context * g = ds.gg;
    if (gguf_str(g, "general.architecture") != "dflash")
        throw std::runtime_error("dspark: " + path + " is not a dflash draft");
    ds.n_layer   = (int) gguf_u32(g, "dflash.block_count");
    ds.n_embd    = (int) gguf_u32(g, "dflash.embedding_length");
    ds.n_ff      = (int) gguf_u32(g, "dflash.feed_forward_length");
    ds.n_head    = (int) gguf_u32(g, "dflash.attention.head_count");
    ds.n_head_kv = (int) gguf_u32(g, "dflash.attention.head_count_kv");
    ds.hd        = (int) gguf_u32(g, "dflash.attention.key_length");
    ds.block     = (int) gguf_u32(g, "dflash.block_size");
    ds.eps       = gguf_f32(g, "dflash.attention.layer_norm_rms_epsilon", 1e-6f);
    ds.rope_base = gguf_f32(g, "dflash.rope.freq_base", 1e6f);
    ds.n_swa     = gguf_u32(g, "dflash.attention.sliding_window") + 1;   // the anchor row includes the anchor
    for (bool b : gguf_bool_array(g, "dflash.attention.sliding_window_pattern")) ds.is_swa.push_back(b ? 1 : 0);
    for (int32_t t : gguf_i32_array(g, "dflash.target_layers")) ds.target.push_back(t);
    if (ds.n_embd != (int) hp.n_embd || ds.block != 8 || (int) ds.is_swa.size() != ds.n_layer || ds.target.empty() ||
        ds.target.back() != (int) ik.n_visit || gguf_bool(g, "dflash.sample_from_anchor", false))
        throw std::runtime_error("dspark: " + path + " does not fit this model");

    // weights: one GPU buffer, data read straight from the file
    ds.wctx = meta;
    for (ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) ds.w[ggml_get_name(t)] = t;
    ds.wbuf = ggml_backend_alloc_ctx_tensors(meta, gpu);
    if (!ds.wbuf) throw std::runtime_error("dspark: cannot allocate the draft weights");
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("dspark: cannot read " + path);
    std::vector<uint8_t> buf;
    const size_t data0 = gguf_get_data_offset(g);
    for (auto & kv : ds.w) {
        const int64_t tid = gguf_find_tensor(g, kv.first.c_str());
        const size_t nb = ggml_nbytes(kv.second);
        buf.resize(nb);
#ifdef _WIN32
        _fseeki64(f, (long long) (data0 + gguf_get_tensor_offset(g, tid)), SEEK_SET);
#else
        fseeko(f, (off_t) (data0 + gguf_get_tensor_offset(g, tid)), SEEK_SET);
#endif
        if (fread(buf.data(), 1, nb, f) != nb) { fclose(f); throw std::runtime_error("dspark: short read"); }
        ggml_backend_tensor_set(kv.second, buf.data(), 0, nb);
    }
    fclose(f);
    ds.rank = (int) ds.W("markov_w1.weight")->ne[0];

    // state
    const int E = ds.n_embd, nkv = ds.n_head_kv * ds.hd, nt = (int) ds.target.size();
    ggml_init_params sp{};
    sp.mem_size = ggml_tensor_overhead() * (2 * ds.n_layer + ik.n_visit + 16);
    sp.no_alloc = true;
    ds.sctx = ggml_init(sp);
    for (int l = 0; l < ds.n_layer; ++l) {
        ds.k.push_back(ggml_new_tensor_2d(ds.sctx, GGML_TYPE_F16, nkv, n_ctx));
        ds.v.push_back(ggml_new_tensor_2d(ds.sctx, GGML_TYPE_F16, nkv, n_ctx));
    }
    ds.frnn   = ggml_new_tensor_2d(ds.sctx, GGML_TYPE_F32, E, 16);
    ds.feat   = ggml_new_tensor_2d(ds.sctx, GGML_TYPE_F32, (int64_t) nt * E, UBATCH);
    for (uint32_t iv = 0; iv < ik.n_visit; ++iv) ds.xs.push_back(ggml_new_tensor_2d(ds.sctx, GGML_TYPE_F32, ik.n_conv - 1 + 16, n_ch));
    ds.p_tok  = ggml_new_tensor_1d(ds.sctx, GGML_TYPE_I32, 1);
    ds.p_pos  = ggml_new_tensor_1d(ds.sctx, GGML_TYPE_I32, UBATCH);
    ds.p_idx  = ggml_new_tensor_1d(ds.sctx, GGML_TYPE_I64, UBATCH);
    ds.p_ridx = ggml_new_tensor_1d(ds.sctx, GGML_TYPE_I64, UBATCH);
    ds.p_crow = ggml_new_tensor_1d(ds.sctx, GGML_TYPE_I32, 1);
    ds.p_mask_f = ggml_new_tensor_1d(ds.sctx, GGML_TYPE_F16, (int64_t) n_ctx * ds.block);
    ds.p_mask_s = ggml_new_tensor_1d(ds.sctx, GGML_TYPE_F16, (int64_t) n_ctx * ds.block);
    ds.p_out  = ggml_new_tensor_1d(ds.sctx, GGML_TYPE_I32, ds.block);
    ds.p_am   = ggml_new_tensor_1d(ds.sctx, GGML_TYPE_I32, 16);
    ds.sbuf = ggml_backend_alloc_ctx_tensors(ds.sctx, gpu);
    if (!ds.sbuf) throw std::runtime_error("dspark: cannot allocate the draft state");
    ggml_backend_buffer_clear(ds.sbuf, 0);
    ds.on = true;
    cap_feat = true;   // from here on every target graph stores the draft's features
    fprintf(stderr, "inkling2: DSpark draft %s: %d layers, block %d, target visits", path.c_str(), ds.n_layer, ds.block);
    for (int t : ds.target) fprintf(stderr, " %d", t);
    fprintf(stderr, " (%.0f MB)\n", ggml_backend_buffer_get_size(ds.wbuf) / 1048576.0 + ggml_backend_buffer_get_size(ds.sbuf) / 1048576.0);
}

// mix the target features with fuse row `row`, then RMS-norm with `norm`
static ggml_tensor * ds_mix(ggml_context * ctx, ggml_tensor * fuse, ggml_tensor * feat, int nt, int E, int T, int row,
                            ggml_tensor * norm, float eps) {
    ggml_tensor * wts = ggml_soft_max(ctx, ggml_view_1d(ctx, fuse, nt, (size_t) row * fuse->nb[1]));
    ggml_tensor * x = ggml_reshape_3d(ctx, ggml_cont(ctx, feat), E, nt, T);
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));                 // [nt, E, T]
    x = ggml_sum_rows(ctx, ggml_mul(ctx, x, wts));                        // [1, E, T]
    return ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, x, E, T), eps), norm);
}

void InklingEngine::Impl::dspark_inject(int pos0, int T) {
    const int E = ds.n_embd, nt = (int) ds.target.size(), nkv = ds.n_head_kv * ds.hd;
    Seg & sg = ds.inj[T];
    if (!sg.gf) {
        ggml_init_params p{};
        const int nodes = 2048;
        p.mem_size = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false);
        p.no_alloc = true;
        sg.ctx = ggml_init(p);
        sg.gf  = ggml_new_graph_custom(sg.ctx, nodes, false);
        ggml_context * c = sg.ctx;
        ggml_tensor * feat = ggml_view_2d(c, ds.feat, (int64_t) nt * E, T, ds.feat->nb[1], 0);
        ggml_tensor * pos  = ggml_view_1d(c, ds.p_pos, T, 0);
        ggml_tensor * idx  = ggml_view_1d(c, ds.p_idx, T, 0);
        ggml_tensor * fuse = ds.W("dspark_fuse.weight");
        for (int l = 0; l < ds.n_layer; ++l) {
            ggml_tensor * ctxf = ds_mix(c, fuse, feat, nt, E, T, l, ds.Wl("blk.%d.dflare_norm.weight", l), ds.eps);
            ggml_tensor * k = ggml_reshape_3d(c, ggml_mul_mat(c, ds.Wl("blk.%d.attn_k_ctx.weight", l), ctxf), ds.hd, ds.n_head_kv, T);
            ggml_tensor * v = ggml_mul_mat(c, ds.Wl("blk.%d.attn_v_ctx.weight", l), ctxf);
            k = ggml_mul(c, ggml_rms_norm(c, k, ds.eps), ds.Wl("blk.%d.attn_k_ctx_norm.weight", l));
            k = ggml_rope_ext(c, k, pos, nullptr, ds.hd, GGML_ROPE_TYPE_NEOX, 0, ds.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_build_forward_expand(sg.gf, ggml_set_rows(c, ds.k[(size_t) l], ggml_reshape_2d(c, k, nkv, T), idx));
            ggml_build_forward_expand(sg.gf, ggml_set_rows(c, ds.v[(size_t) l], v, idx));
        }
        ggml_tensor * fr = ds_mix(c, fuse, feat, nt, E, T, ds.n_layer, ds.W("dspark_norm.weight"), ds.eps);
        ggml_build_forward_expand(sg.gf, ggml_set_rows(c, ds.frnn, fr, ggml_view_1d(c, ds.p_ridx, T, 0)));
        sg.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
        if (!ggml_gallocr_alloc_graph(sg.ga, sg.gf)) throw std::runtime_error("dspark: inject alloc failed");
    }
    std::vector<int32_t> pos((size_t) T);
    std::vector<int64_t> idx((size_t) T), ridx((size_t) T);
    for (int i = 0; i < T; ++i) { pos[(size_t) i] = pos0 + i; idx[(size_t) i] = pos0 + i; ridx[(size_t) i] = (pos0 + i) % 16; }
    ggml_backend_tensor_set(ds.p_pos,  pos.data(),  0, pos.size() * 4);
    ggml_backend_tensor_set(ds.p_idx,  idx.data(),  0, idx.size() * 8);
    ggml_backend_tensor_set(ds.p_ridx, ridx.data(), 0, ridx.size() * 8);
    if (ggml_backend_graph_compute(gpu, sg.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("dspark: inject failed");
}

void InklingEngine::Impl::dspark_draft(int32_t anchor, int pos0, int n_steps, std::vector<int32_t> & out) {
    const int E = ds.n_embd, B = ds.block, nkv = ds.n_head_kv * ds.hd;
    const int bucket = std::min(n_ctx, (pos0 + B + 255) / 256 * 256);
    if (!ds.draft.gf || ds.draft_bucket != bucket || ds.draft_steps != n_steps) {
        free_seg(ds.draft);
        ggml_init_params p{};
        const int nodes = 4096;
        p.mem_size = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false);
        p.no_alloc = true;
        ds.draft.ctx = ggml_init(p);
        ds.draft.gf  = ggml_new_graph_custom(ds.draft.ctx, nodes, false);
        ggml_context * c = ds.draft.ctx;
        ggml_cgraph * gf = ds.draft.gf;
        const int n_kv = bucket;
        ggml_tensor * pos = ggml_view_1d(c, ds.p_pos, B, 0);
        ggml_tensor * idx = ggml_view_1d(c, ds.p_idx, B, 0);
        ggml_tensor * mf  = ggml_view_2d(c, ds.p_mask_f, n_kv, B, (size_t) n_kv * 2, 0);
        ggml_tensor * ms  = ggml_view_2d(c, ds.p_mask_s, n_kv, B, (size_t) n_kv * 2, 0);

        // [anchor, mask x (B-1)]: the anchor through the target's embedding and its norm
        ggml_tensor * a = ggml_get_rows(c, model.tok_embd_rows(), ds.p_tok);
        a = ggml_mul(c, ggml_rms_norm(c, a, hp.rms_eps), G("token_embd_norm.weight"));
        ggml_tensor * masks = ggml_repeat_4d(c, ggml_reshape_2d(c, ds.W("mask_embd.weight"), E, 1), E, B - 1, 1, 1);
        ggml_tensor * cur = ggml_concat(c, a, masks, 1);                     // [E, B]
        for (int l = 0; l < ds.n_layer; ++l) {
            ggml_tensor * res = cur;
            ggml_tensor * h = ggml_mul(c, ggml_rms_norm(c, cur, ds.eps), ds.Wl("blk.%d.attn_norm.weight", l));
            ggml_tensor * q = ggml_reshape_3d(c, ggml_mul_mat(c, ds.Wl("blk.%d.attn_q.weight", l), h), ds.hd, ds.n_head, B);
            ggml_tensor * k = ggml_reshape_3d(c, ggml_mul_mat(c, ds.Wl("blk.%d.attn_k.weight", l), h), ds.hd, ds.n_head_kv, B);
            ggml_tensor * v = ggml_mul_mat(c, ds.Wl("blk.%d.attn_v.weight", l), h);
            q = ggml_mul(c, ggml_rms_norm(c, q, ds.eps), ds.Wl("blk.%d.attn_q_norm.weight", l));
            k = ggml_mul(c, ggml_rms_norm(c, k, ds.eps), ds.Wl("blk.%d.attn_k_norm.weight", l));
            q = ggml_rope_ext(c, q, pos, nullptr, ds.hd, GGML_ROPE_TYPE_NEOX, 0, ds.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(c, k, pos, nullptr, ds.hd, GGML_ROPE_TYPE_NEOX, 0, ds.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * kc = ggml_set_rows(c, ds.k[(size_t) l], ggml_reshape_2d(c, k, nkv, B), idx);
            ggml_tensor * vc = ggml_set_rows(c, ds.v[(size_t) l], v, idx);
            ggml_tensor * K = ggml_permute(c, ggml_view_3d(c, kc, ds.hd, ds.n_head_kv, n_kv, (size_t) ds.hd * 2, kc->nb[1], 0), 0, 2, 1, 3);
            ggml_tensor * V = ggml_permute(c, ggml_view_3d(c, vc, ds.hd, ds.n_head_kv, n_kv, (size_t) ds.hd * 2, vc->nb[1], 0), 0, 2, 1, 3);
            ggml_tensor * fa = ggml_flash_attn_ext(c, ggml_permute(c, q, 0, 2, 1, 3), K, V, ds.is_swa[(size_t) l] ? ms : mf,
                                                   1.0f / sqrtf((float) ds.hd), 0.0f, 0.0f);
            ggml_tensor * o = ggml_mul_mat(c, ds.Wl("blk.%d.attn_output.weight", l), ggml_reshape_2d(c, fa, ds.hd * ds.n_head, B));
            cur = ggml_add(c, res, o);
            res = cur;
            h = ggml_mul(c, ggml_rms_norm(c, cur, ds.eps), ds.Wl("blk.%d.ffn_norm.weight", l));
            ggml_tensor * ff = ggml_mul_mat(c, ds.Wl("blk.%d.ffn_down.weight", l),
                ggml_mul(c, ggml_silu(c, ggml_mul_mat(c, ds.Wl("blk.%d.ffn_gate.weight", l), h)), ggml_mul_mat(c, ds.Wl("blk.%d.ffn_up.weight", l), h)));
            cur = ggml_add(c, res, ff);
        }
        cur = ggml_mul(c, ggml_rms_norm(c, cur, ds.eps), ds.W("output_norm.weight"));   // [E, B]

        // recurrent head over block positions 1..n_steps
        ggml_tensor * out_w = G("output.weight");
        ggml_tensor * context = ggml_get_rows(c, ds.frnn, ds.p_crow);                   // [E, 1]
        ggml_tensor * state = ggml_fill(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, ds.rank, 1), 0.0f);
        ggml_tensor * prev = ds.p_tok;
        const float cap = ik.logit_cap;
        for (int i = 1; i <= n_steps; ++i) {
            ggml_tensor * hi = ggml_cont(c, ggml_view_2d(c, cur, E, 1, cur->nb[1], (size_t) i * cur->nb[1]));
            ggml_tensor * base = ggml_mul_mat(c, out_w, hi);
            ggml_prec_set_acc(base, GGML_PREC_F32);
            ggml_tensor * joint = ggml_concat(c, state, ggml_get_rows(c, ds.W("markov_w1.weight"), prev), 0);
            joint = ggml_concat(c, joint, ggml_concat(c, hi, context, 0), 0);
            joint = ggml_add(c, ggml_mul_mat(c, ds.W("joint_proj.weight"), joint), ds.W("joint_proj.bias"));
            auto chunk = [&](int j) { return ggml_cont(c, ggml_view_2d(c, joint, ds.rank, 1, joint->nb[1], (size_t) j * ds.rank * 4)); };
            ggml_tensor * gate = ggml_sigmoid(c, chunk(0));
            ggml_tensor * cand = ggml_tanh(c, chunk(1));
            state = ggml_add(c, cand, ggml_mul(c, gate, ggml_sub(c, state, cand)));
            ggml_tensor * latent = ggml_tanh(c, chunk(2));
            ggml_tensor * corr = ggml_add(c, ggml_mul_mat(c, ds.W("dspark_hidden_out.weight"), latent), ds.W("dspark_hidden_out.bias"));
            ggml_tensor * bias = ggml_mul_mat(c, out_w, corr);
            ggml_prec_set_acc(bias, GGML_PREC_F32);
            ggml_tensor * lg = ggml_add(c, base, bias);
            lg = ggml_scale(c, ggml_tanh(c, ggml_scale(c, lg, 1.0f / cap)), cap);
            prev = ggml_argmax(c, ggml_cont(c, ggml_view_2d(c, lg, ik.n_vocab, 1, lg->nb[1], 0)));
            ggml_build_forward_expand(gf, ggml_cpy(c, prev, ggml_view_1d(c, ds.p_out, 1, (size_t) (i - 1) * 4)));
        }
        ds.draft.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
        if (!ggml_gallocr_alloc_graph(ds.draft.ga, gf)) throw std::runtime_error("dspark: draft alloc failed");
        ds.draft_bucket = bucket;
        ds.draft_steps = n_steps;
    }
    // inputs: positions pos0.., write rows, and the shared mask row of the anchor
    const int n_kv = bucket;
    std::vector<int32_t> pos((size_t) B);
    std::vector<int64_t> idx((size_t) B);
    for (int i = 0; i < B; ++i) { pos[(size_t) i] = pos0 + i; idx[(size_t) i] = pos0 + i; }
    std::vector<ggml_fp16_t> mf((size_t) n_kv * B), ms((size_t) n_kv * B);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int j = 0; j < n_kv; ++j) {
        const bool vis  = j < pos0 + B;                                // committed context + the whole block
        const bool visw = vis && pos0 - j < (int) ds.n_swa;           // ... within the anchor's window
        for (int r = 0; r < B; ++r) {
            mf[(size_t) r * n_kv + j] = vis  ? z : ninf;
            ms[(size_t) r * n_kv + j] = visw ? z : ninf;
        }
    }
    const int32_t crow = (pos0 - 1 + 16) % 16;
    ggml_backend_tensor_set(ds.p_tok,  &anchor, 0, 4);
    ggml_backend_tensor_set(ds.p_crow, &crow,   0, 4);
    ggml_backend_tensor_set(ds.p_pos,  pos.data(), 0, pos.size() * 4);
    ggml_backend_tensor_set(ds.p_idx,  idx.data(), 0, idx.size() * 8);
    ggml_backend_tensor_set(ds.p_mask_f, mf.data(), 0, mf.size() * 2);
    ggml_backend_tensor_set(ds.p_mask_s, ms.data(), 0, ms.size() * 2);
    if (ggml_backend_graph_compute(gpu, ds.draft.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("dspark: draft failed");
    out.resize((size_t) n_steps);
    ggml_backend_tensor_get(ds.p_out, out.data(), 0, (size_t) n_steps * 4);
}

void InklingEngine::Impl::dspark_rollback(int keep_tokens) {
    // state = the conv inputs of the last nc-1 kept tokens: rows keep..keep+nc-2 of each window
    ggml_init_params p{};
    const int nodes = 4 * (int) ik.n_visit + 16;
    p.mem_size = ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false);
    p.no_alloc = true;
    ggml_context * c = ggml_init(p);
    ggml_cgraph * gf = ggml_new_graph_custom(c, nodes, false);
    for (uint32_t iv = 0; iv < ik.n_visit; ++iv) {
        ggml_tensor * xsv = ds.xs[iv];
        ggml_tensor * src = ggml_view_2d(c, xsv, ik.n_conv - 1, n_ch, xsv->nb[1], (size_t) keep_tokens * 4);
        ggml_build_forward_expand(gf, ggml_cpy(c, src, conv_l[iv]));
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) throw std::runtime_error("dspark: rollback alloc failed");
    if (ggml_backend_graph_compute(gpu, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("dspark: rollback failed");
    ggml_free(c);
}

void InklingEngine::Impl::generate_spec(const std::vector<int32_t> & prompt, int max_new, int n_draft,
                                        const std::function<bool(int32_t)> & on_token, int32_t * out_pending,
                                        bool ckpt_after_prefill) {
    if (out_pending) *out_pending = -1;
    n_draft = std::max(1, std::min(n_draft, ds.block - 1));
    static const bool stats = getenv("INK_SPEC_STATS") != nullptr;
    // prompt: forward with the features captured, every chunk injected
    const int P = (int) prompt.size();
    if (P == 0) throw std::runtime_error("inkling2: speculative decode needs at least one new token");
    {
        OvrScope scope{ *this };
        for (int i = 0; i < P; ) {
            const int T = next_chunk(i, P);
            no_keep_now = true;
            forward(prompt.data() + i, T, i + T == P);   // injects the draft features itself
            no_keep_now = false;
            i += T;
        }
    }
    kv_toks.insert(kv_toks.end(), prompt.begin(), prompt.end());
    if (ckpt_after_prefill) ckpt_snapshot();   // the next turn rewinds here
    // Sampling stays exact with a deterministic draft: sample each verified
    // position from the target as usual and keep the draft while they agree.
    auto pick = [&](const std::vector<float> & l) -> int32_t {
        if (spec_pick) return spec_pick(l);
        return (int32_t) (std::max_element(l.begin(), l.end()) - l.begin());
    };
    int32_t last = pick(logits);
    std::vector<float> all, row;

    const int T = n_draft + 1;
    uint64_t steps = 0, drafted = 0, accepted = 0;
    std::vector<int32_t> draft, toks((size_t) T), am((size_t) T);
    int produced = 0;
    while (produced < max_new) {
        if (!on_token(last)) { if (out_pending) *out_pending = last; break; }   // not decoded yet
        ++produced;
        if (n_past_ + T + ds.block > n_ctx) break;
        const int pos0 = n_past_;
        dspark_draft(last, pos0, n_draft, draft);
        toks[0] = last;
        for (int i = 0; i < n_draft; ++i) toks[(size_t) i + 1] = draft[(size_t) i];

        // verify: one kept forward over [last, drafts], every position's argmax
        // and features, and the conv windows for a rollback
        xs_save = true;
        am_out = ds.p_am;
        all_logits = true;
        spec_verify(toks.data(), T);
        all_logits = false;
        am_out = nullptr;
        xs_save = false;
        int a = 0;
        int32_t bonus = -1;
        if (spec_pick) {
            all.resize((size_t) ik.n_vocab * T);
            ggml_backend_tensor_get(vkept.out, all.data(), 0, all.size() * sizeof(float));
            for (int i = 0; i <= n_draft; ++i) {
                row.assign(all.begin() + (size_t) i * ik.n_vocab, all.begin() + (size_t) (i + 1) * ik.n_vocab);
                const int32_t t = spec_pick(row);
                if (i < n_draft && t == draft[(size_t) i]) { ++a; continue; }
                bonus = t;
                break;
            }
        } else {
            ggml_backend_tensor_get(ds.p_am, am.data(), 0, (size_t) T * 4);
            while (a < n_draft && am[(size_t) a] == draft[(size_t) a]) ++a;
            bonus = am[(size_t) a];
        }
        ++steps; drafted += n_draft; accepted += a;

        // keep [last, drafts 0..a-1]
        const int keep = a + 1;
        if (keep < T) {
            dspark_rollback(keep);
            n_past_ -= T - keep;
        }
        kv_toks.insert(kv_toks.end(), toks.begin(), toks.begin() + keep);
        dspark_inject(pos0, keep);
        bool stop = false;
        for (int i = 0; i < a && !stop; ++i) {
            if (produced >= max_new) { stop = true; break; }
            // the rest of the accepted drafts are in kv_tokens(); the bonus is the next undecoded one
            if (!on_token(draft[(size_t) i])) { if (out_pending) *out_pending = bonus; stop = true; break; }
            ++produced;
        }
        if (stop) break;
        last = bonus;
    }
    if (stats && steps)
        fprintf(stderr, "[dspark] %llu steps, draft %d: accepted %llu of %llu (%.0f%%), %.2f tokens per step\n",
                (unsigned long long) steps, n_draft, (unsigned long long) accepted, (unsigned long long) drafted,
                100.0 * accepted / drafted, 1.0 + (double) accepted / steps);
}

// k_l / v_l / conv_l = the part of the state the current pass count's visits use
void InklingEngine::Impl::map_state() {
    const uint32_t core_last = ik.core_start + ik.n_pass * ik.n_core();   // first postlude visit
    const uint32_t shift = ((uint32_t) max_pass - ik.n_pass) * ik.n_core();
    k_l.clear(); v_l.clear(); conv_l.clear();
    for (uint32_t iv = 0; iv < ik.n_visit; ++iv) {
        const size_t s = iv < core_last ? iv : iv + shift;
        k_l.push_back(st_k[s]);
        v_l.push_back(st_v[s]);
        conv_l.push_back(st_conv[s]);
    }
}

bool InklingEngine::Impl::set_loops(int n) {
    if (n == (int) ik.n_pass) return true;
    if (n < 1 || n > max_pass || ds.on) return false;   // (the draft's state follows fixed visits)
    // every kept graph is wired to the old visits; the weights and the expert pool stay
    free_kept();
    for (auto * v : { &segA, &segB, &segC }) {
        for (auto & sg : *v) free_seg(sg);
        v->clear();
    }
    seg_bucket = 0;
    model.set_loop_passes(n);
    map_state();
    return true;
}

InklingEngine::InklingEngine(Model & model, const RuntimeConfig & cfg)
    : impl_(std::make_unique<Impl>(model, cfg)) { impl_->init(); }
InklingEngine::~InklingEngine() {
    if (impl_ && impl_->ec) impl_->dump_offload_stats();
}

const std::vector<float> & InklingEngine::decode(const std::vector<int32_t> & tokens) {
    Impl & m = *impl_;
    const int P = (int) tokens.size();
    // INK_SPLIT_PREFILL=k: break the chunks at k too (checks chunking against a rewind)
    static const int split = getenv("INK_SPLIT_PREFILL") ? atoi(getenv("INK_SPLIT_PREFILL")) : 0;
    {
        Impl::OvrScope scope{ m };
        for (int i = 0; i < P; ) {
            int T = m.next_chunk(i, P);
            if (split > i && split < i + T) T = split - i;
            m.forward(tokens.data() + i, T, i + T == P);
            if (m.progress_cb && P > UBATCH) m.progress_cb(i + T, P);
            i += T;
        }
    }
    m.kv_toks.insert(m.kv_toks.end(), tokens.begin(), tokens.end());
    // INK_VERIFY_BENCH=1: after the prompt, time forwards of 1..8 tokens (what a
    // speculative verify step costs). It advances the state; for measuring only.
    if (P > 1 && getenv("INK_VERIFY_BENCH")) {
        for (int k : { 1, 2, 3, 4, 5, 8, 1 }) {
            std::vector<int32_t> t((size_t) k, tokens.back());
            double best = 1e9;
            for (int rep = 0; rep < 4; ++rep) {
                auto t0 = clk::now();
                m.forward(t.data(), k, true);
                best = std::min(best, ms_since(t0));
            }
            fprintf(stderr, "[verify bench] %d token(s): %.1f ms (%.1f ms/token)\n", k, best, best / k);
        }
    }
    return m.logits;
}
void InklingEngine::prefill(const std::vector<int32_t> & tokens) {
    Impl & m = *impl_;
    const int P = (int) tokens.size();
    Impl::OvrScope scope{ m };
    for (int i = 0; i < P; ) {
        const int T = m.next_chunk(i, P);
        m.forward(tokens.data() + i, T, false);
        if (m.progress_cb) m.progress_cb(i + T, P);
        i += T;
    }
    m.kv_toks.insert(m.kv_toks.end(), tokens.begin(), tokens.end());
}
bool InklingEngine::has_draft() const { return impl_->ds.on; }
void InklingEngine::set_sampler(std::function<int32_t(const std::vector<float> &)> fn) { impl_->spec_pick = std::move(fn); }
void InklingEngine::generate_spec(const std::vector<int32_t> & prompt, int max_new, int n_draft,
                                  const std::function<bool(int32_t)> & on_token, int32_t * out_pending,
                                  bool ckpt_after_prefill) {
    impl_->generate_spec(prompt, max_new, n_draft, on_token, out_pending, ckpt_after_prefill);
}
void InklingEngine::set_embd_overrides(std::vector<Runtime::EmbdOverride> ovr) { impl_->embd_ovr = std::move(ovr); }
void InklingEngine::snapshot_ckpt() { impl_->ckpt_snapshot(); }
int  InklingEngine::best_ckpt(int n) const { return impl_->ckpt_best(n); }
int  InklingEngine::rewind_to(int n) { return impl_->ckpt_rewind(n); }
void InklingEngine::reset() {
    impl_->ckpts.clear();
    impl_->free_kept();
    impl_->n_past_ = 0;
    impl_->embd_ovr.clear();
    impl_->kv_toks.clear();
    ggml_backend_buffer_clear(impl_->st_buf, 0);
}
int InklingEngine::n_past() const { return impl_->n_past_; }
const std::vector<int32_t> & InklingEngine::kv_tokens() const { return impl_->kv_toks; }
bool InklingEngine::has_expert_cache() const { return impl_->ec != nullptr; }
int  InklingEngine::loops() const     { return impl_->ik.has_core() ? (int) impl_->ik.n_pass : 0; }
int  InklingEngine::max_loops() const { return impl_->ik.has_core() && !impl_->ds.on ? impl_->max_pass : 0; }
bool InklingEngine::set_loops(int n) {
    if (n == loops()) return true;
    if (!impl_->set_loops(n)) return false;
    reset();
    return true;
}
Runtime::CacheStats InklingEngine::cache_stats() const {
    Runtime::CacheStats c;
    if (impl_->ec) {
        c.hits   = impl_->stx.gpu;
        c.misses = impl_->stx.cpu + impl_->stx.fetched;
        c.fetch_bytes = impl_->ec->stats().fetch_bytes;
        c.fetch_ms    = impl_->ec->stats().fetch_ms;
    }
    return c;
}
void InklingEngine::set_progress_cb(std::function<void(int, int)> cb) { impl_->progress_cb = std::move(cb); }

} // namespace questwend
