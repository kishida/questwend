# QuestWend

*English | [日本語](README.ja.md)*

A from-scratch inference engine for Qwen3 / Qwen3.5 / Qwen3.6, built on vendored ggml.

- **Architectures**: `qwen3` (dense), `qwen3moe` (MoE), `qwen35` / `qwen35moe` (Gated DeltaNet hybrid), `qwen3next`
- **Backends**: CUDA (Windows/Linux), Metal (macOS), HIP/ROCm (AMD, untested), CPU fallback
- **Features**:
  - MoE expert offloading (**RAM tier** / **SSD tier**) with a dynamic VRAM expert cache
    → runs models like 35B-A3B MoE on a 16GB GPU
  - Persistent hot-expert residency profiles (`--cache-profile`) for fast warm starts
  - MTP (Multi-Token Prediction / nextn) self-speculative decoding (`--mtp`, `--draft N`)
  - **Image input** (Qwen3-VL family + mmproj GGUF; the vision tower runs on ggml/GPU. CLI `--image`, server takes OpenAI-style `image_url`)
  - **Tool calling** (OpenAI-compatible: `tools` / `tool_calls` / `role:"tool"`, with two-way conversion of Qwen3.6's `<function=...>` form)
  - Sharded GGUF support (`-NNNNN-of-MMMMM.gguf`)
  - K-quant embeddings on large-vocab models are converted to F16 in VRAM only when the backend's `get_rows` does not support the type (CUDA only; Metal/CPU support them natively so no copy is made. `--embd-q8` gives Q8_0 instead)
  - OpenAI-compatible server (`qw-server`) with a browser chat UI

---

## 1. Requirements

| | Windows (CUDA) | macOS (Metal) | CPU only |
|---|---|---|---|
| Compiler | Visual Studio 2022 (C++) | Xcode Command Line Tools | any C++17 |
| CMake | 3.20+ (the one bundled with VS works) | 3.20+ (`brew install cmake`) | 3.20+ |
| Other | CUDA Toolkit 12.x | — | — |

The repository vendors ggml and is self-contained (`third_party/ggml`); no submodule checkout is needed.
The upstream commit it was vendored from is recorded in `third_party/ggml/GGML_VERSION`, and the
qwencpp-specific patches applied on top live in [`patches/`](patches/README.md), together with the
re-vendoring procedure and its pitfalls.

---

## 2. Build

`CMAKE_BUILD_TYPE` does not need to be passed: the project forces `Release` when it is
unset. (Visual Studio is multi-config, so there you pick the config at build time with
`--config Release`.)

### Windows (CUDA)

```powershell
# configure (enable the CUDA backend)
cmake -B build -DQW_CUDA=ON

# build (Release)
cmake --build build --config Release --target qw-cli qw-server
```

- Outputs: `build\Release\qw-cli.exe`, `build\Release\qw-server.exe`. The ggml DLLs are built into
  `build\bin\Release\` and copied next to the executables after linking, so run the exe in place.
- The CUDA architecture is detected natively (e.g. RTX 4060 Ti = sm_89, RTX 3050 = sm_86).
- CMake configures `CUDA graphs ON`, `Flash Attention ON`.
- Using the cmake bundled with VS:
  `"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"`

### macOS (Metal)

```bash
cmake -B build -DQW_METAL=ON
cmake --build build -j --target qw-cli qw-server
```

- Outputs: `build/qw-cli`, `build/qw-server`.
- Metal is active if startup prints `backend: GPU [MTL0] ...`.

### AMD (HIP / ROCm)

```bash
cmake -B build -DQW_HIP=ON
cmake --build build -j --target qw-cli qw-server
```

- The GPU architecture is normally detected for you. Pass `-DGPU_TARGETS=gfx1100` only when the
  detection misses, when building for a machine with a different GPU, or when you want several
  architectures in one binary (`rocminfo | grep gfx` prints yours; RX 7900 series = `gfx1100`,
  MI300 = `gfx942`). The older `AMDGPU_TARGETS` spelling also works.
- HIP graphs are on by default in ggml (`GGML_HIP_GRAPHS`), so unlike CUDA there is nothing to pass.
- To avoid installing ROCm, Vulkan should also work: `-DGGML_VULKAN=ON` (there is no `QW_*` wrapper,
  so pass ggml's option directly).
- **Untested**: backend selection is generic (it takes the first GPU device), so it should work with
  no code changes, but the expert-offload paths have only been exercised on CUDA and Metal.
  Reports welcome.

### CPU only (any platform)

```bash
cmake -B build
cmake --build build -j --target qw-cli
# pass --cpu at runtime, or build without a GPU backend and it runs on CPU automatically
```

---

## 3. Usage (CLI: `qw-cli`)

```bash
# one-shot generation
qw-cli -m model.gguf -p "The capital of France is" -n 128

# chat (wrapped in ChatML)
qw-cli -m model.gguf --chat -p "Explain photosynthesis." -n 256
qw-cli -m model.gguf -i                      # interactive mode
qw-cli -m model.gguf --info                  # print model info and exit
```

### Main options

| Option | Description |
|---|---|
| `-m <path>` | model GGUF (for sharded models, pass the first shard) |
| `-p <text>` | prompt |
| `-n <N>` | tokens to generate (default 128) |
| `-i` | interactive chat |
| `--chat` | wrap `-p` in ChatML |
| `--reasoning <on\|off>` | thinking mode (default on; `off` terminates with `<think>\n\n</think>`) |
| `--n-ctx <N>` | context length (default 4096) |
| `--temp <f>` / `--top-p <f>` / `--top-k <N>` / `--seed <N>` | sampling (default greedy: temp=0) |
| `--embd-q8` | use Q8_0 instead of F16 for the embedding `get_rows` fallback (saves VRAM; slight quantization error) |
| `--image <path>` | attach an image to the prompt (repeatable; needs a VL model + mmproj) |
| `--mmproj <gguf>` | vision tower GGUF (defaults to `mmproj-*.gguf` next to the model) |
| `--vision-test` | encode the image with the vision tower only and print statistics (numerical checks; no LLM) |
| `--log-tokens-per-sec` | print prefill / generation tok/s |
| `--cpu` | force the CPU backend |

### Expert offloading (large MoE on limited VRAM)

```bash
# RAM tier: non-expert weights on the GPU, experts in pinned host RAM,
#           streamed through the dynamic VRAM cache (PCIe)
qw-cli -m moe.gguf -p "..." -n 128 --vram-budget 15

# SSD tier: experts read straight from the GGUF on disk (no RAM copy)
qw-cli -m moe.gguf -p "..." -n 128 --vram-budget 12 --experts-ssd

# residency profile: saved on the first run, then hot experts are preloaded at startup
qw-cli -m moe.gguf -p "..." -n 128 --vram-budget 15 --cache-profile hot.prof
```

- `--vram-budget <GB>`: VRAM budget for the slot pools (offloading is enabled when > 0).
  The unit is **GB** and fractions work (`13.5`); an explicit `M`/`G` suffix always wins
  (`13500M`, `14G`). A bare number of 512 or more is read as MB, so command lines written
  before the unit changed keep working (with a note pointing at the GB spelling).
- `--experts-ssd`: stream experts from disk (for models too large for RAM).
- `--cache-profile <file>`: persist the hot-expert frequency profile. **On the same or a similar
  workload the hit rate reaches ~100%, streaming essentially disappears and throughput jumps.**

Prefill runs **layer-major** by default (each layer fetches its experts once and processes every
token), which is ~4x faster than the old scheme on the RAM tier and ~8x on the SSD tier. Further
knobs, all either lossy or behavior-changing (measurements in
[`prefill_layer_major.md`](prefill_layer_major.md)):

- `--resident-decode`: **run decode as one fused graph with routing restricted to resident experts**.
  After a short warmup the router is masked to the resident set and never misses again (residency
  verification, state backup and the fallback path all disappear). Experts the unmasked router
  actually wanted are refilled in the background every token and join the mask, so the set follows
  topic drift with a few tokens of lag. Measured: RAM tier 20.8 → 36-45 tok/s, SSD tier 3.1 → 25 tok/s
  (with true SSD reads). Lossy. See [`docs/resident_decode.md`](docs/resident_decode.md) for the
  mechanism, how experts get loaded, and the quality characteristics (degradation on domain switches)
  with mitigations.
- `--resident-refill <N>`: experts refilled per token (summed over all layers; default 8, `0` freezes
  the set completely — not recommended).
- `--resident-warmup <N>`: decode tokens before the mask locks in (default 32).
- `--prefill-prune <eps>`: during prefill, skip fetching non-resident experts whose summed router
  weight is below `eps` of the layer total (e.g. `0.05`; especially effective on the SSD tier). Lossy.
- `--ssd-direct`: bypass the OS page cache for SSD reads (Windows, unbuffered I/O). Use it to measure
  real performance when the model does not fit in RAM, or to avoid double caching.

### MTP self-speculative decoding (models with a nextn block)

```bash
qw-cli -m model-MTP.gguf -p "..." -n 128 --mtp            # look ahead 1 token
qw-cli -m model-MTP.gguf -p "..." -n 128 --mtp --draft 2  # look ahead 2 tokens (recommended)
```

- Output matches plain decoding exactly (greedy lossless).
- When compute-bound (whole model resident in VRAM), `--draft 2`-`3` gives roughly **+30%** over plain
  (27B Q2, ctx=2000: 18.2 → 24.2 tok/s). It is a loss when offloading (fetch-bound).
  Details in [`docs/mtp.md`](docs/mtp.md).

---

## 4. Usage (server: `qw-server`)

Provides an OpenAI-compatible `/v1/chat/completions` (with SSE streaming) plus a browser chat UI.

```bash
qw-server -m model.gguf --host 0.0.0.0 --port 8080 --vram-budget 15
```

Open `http://<host>:<port>/` for the chat UI (shows TTFT / tok/s / prefill tok/s / output tokens).

| Option | Description |
|---|---|
| `-m <path>` | model GGUF |
| `--host <addr>` | bind address (default `127.0.0.1`; use `0.0.0.0` to expose on the LAN) |
| `--port <N>` | port (default 8080) |
| `--n-ctx <N>` | context length |
| `--vram-budget <GB>` / `--experts-ssd` | expert offloading (GB, fractions ok; `M`/`G` suffix to be explicit) |
| `--cache-profile <file>` | residency profile (**the server only reads it**, never overwrites) |
| `--reasoning <on\|off>` | default thinking mode (also reflected in the UI checkbox) |
| `--mtp` | MTP self-speculative decoding (models with a nextn block; enabled at startup) |
| `--draft <N>` | MTP draft length (default 1) |
| `--embd-q8` | use Q8_0 for the embedding `get_rows` fallback (saves VRAM) |
| `--mmproj <gguf>` | vision tower for image input (defaults to `mmproj-*.gguf` next to the model) |
| `--no-mmproj` | disable image input even if an mmproj file is present |
| `--cache-slots <N>` | extra prefix-cache slots (default 0; for interleaved conversations) |
| `--cache-slots-dir <dir>` | store slots on disk instead of RAM (survives a server restart) |
| `--time-slice <N>` | interleave concurrent streams every N generated tokens (default 64; effective with `--cache-slots`. `0` serializes completely = throughput first) |
| `--cpu` | force the CPU backend |
| `--resident-decode` and friends | offload tuning knobs (`--resident-refill/-warmup`, `--prefill-prune`, `--batch-chunk`, `--ssd-direct`; shared with the CLI, see the offloading section above) |
| `--pf-chunk <N>` | server prefill slice length (disconnect-detection granularity; default 4096) |

> `max_tokens` is optional: when it is omitted (or <= 0) generation runs until the context is
> exhausted, matching llama.cpp's server — set a budget only when you want one. `finish_reason` is
> `"length"` when generation was cut by either the budget or the context. OpenAI's newer
> `max_completion_tokens` spelling is accepted as an alias.

> Images are accepted in OpenAI form (a base64 data URI in `image_url` inside the `content` array).
> The browser UI has a 📎 button. Image input also works with MTP enabled.
> Loading an mmproj subtracts the vision tower's GPU usage (weights + compute buffers, logged at
> startup) from `--vram-budget` automatically, so a budget set to Metal's working-set limit will not
> OOM. For text-only use, `--no-mmproj` returns that memory to the expert cache.
> For tool calling, pass `tools` and `<tool_call>` output is converted to OpenAI `tool_calls`
> (`finish_reason: "tool_calls"`). `role:"tool"` response messages are supported.

> MTP keeps the nextn block resident in VRAM from model load, so it is a **server startup flag**, not a
> per-request option. Both streaming and non-streaming work. **MTP is greedy-only**: a request's
> `temperature` / `top_p` / `top_k` are ignored and output is always as if temp=0 (also stated in the
> startup log). Start without `--mtp` if you need sampling.

> Plain decoding (without `--mtp`) honors a request's `repetition_penalty` (or `repeat_penalty`) /
> `presence_penalty` / `frequency_penalty` / `repeat_last_n` (window, default 64). Penalties are
> applied at the logits stage, so **repetition control works even under greedy (temp=0)**. They are
> ignored under MTP.

> **Prompt prefix cache**: the server keeps the previous request's KV cache / GDN state, and when a new
> prompt continues the previous "prompt + generation" (an ordinary chat continuation) only the delta
> tokens are prefilled. However long the conversation grows, each turn's prefill costs only the new
> input. `timings.cached_tokens` (`N cached` in the UI) shows how much was reused. If retokenization
> shifts token boundaries, matching falls back to the text level. Images are identified by a byte
> hash (a different image forces a full prefill). `<think>` blocks in the history are preserved and
> reconstructed so the cache still matches (`preserve_thinking`). Editing an earlier turn or switching
> to a different conversation resets fully and automatically.

> **Interleaved conversations** (agents etc.): by default there is only one live state, so alternating
> between conversations A and B falls back to a full prefill each time. `--cache-slots N` adds
> eviction slots: on a switch the live state is parked in a slot and restored when that conversation
> comes back, continuing with only a delta prefill. Slots live in RAM by default (per slot: the GDN
> state plus KV for the conversation length). `--cache-slots-dir <dir>` moves them to disk, consuming
> no RAM and **surviving a server restart** (slots whose model / n-ctx changed are discarded by a
> header check). Passing only `--cache-slots-dir` implies 4 slots.

> **Request logging** (stderr): each request prints a start and a completion line. Start:
> `[time] req: N tok (+img), KV reuse M (live/slot/none), prefill K (resident)` — tokens accepted,
> how much the prefix cache reused and from where, and how many tokens will actually be prefilled.
> Completion: `[time] done: prefill X tok/s, gen Y tok/s (Z tok), expert hit P%` — prefill/decode
> speed, output tokens, and (when offloading) that request's expert cache hit rate. During long
> offloaded prefills a `prefill done/total (%)` progress line appears every 10 seconds (resident
> models finish in one shot, so no progress line).

> **Time slicing**: by default requests are fully serialized (a later request waits for the previous
> generation to finish). With `--time-slice N`, when concurrent streams compete the runtime is handed
> over every N tokens so both streams progress (output is unaffected; state is parked and restored
> exactly through slots). **Prefill is preemptible too**: a long prompt's prefill advances in chunks of
> `max(N, 256)` tokens and yields at chunk boundaries if someone is waiting (coarser than generation to
> keep batch efficiency). There is zero overhead when nothing is competing. Switching costs tens of
> milliseconds depending on state size, so N=10-50 is a reasonable range. Use it together with
> `--cache-slots` (RAM slots recommended for slicing). It works without slots, but the interrupted side
> re-prefills its whole context on every resume.

---

## 5. Sharded GGUF

For models split as `model-00001-of-00003.gguf`, pass the **first shard** to `-m` and the remaining
shards are discovered and merged automatically (5-digit zero-padded naming, starting at 00001).

```bash
qw-cli -m Qwen3.5-122B-A10B-00001-of-00005.gguf -p "..." --vram-budget 40 --experts-ssd
```

---

## 6. Environment variables (debugging / tuning)

Not normally needed — for profiling, A/B checks and troubleshooting.

### Profiling

| Variable | Effect |
|---|---|
| `QWEN_PROF=1` | print the time breakdown of single-token decode (decode_reuse) |
| `QWEN_PROF_DC=1` | print offloaded decode's wall / GPU compute / host time at exit |
| `QWEN_PROF_MTP=1` | print MTP's per-cycle phase breakdown (draft / verify / settle / resync) |
| `QWEN_PROF_MTP2=1` | print compute / readback time per MTP draft (averaged over 100) |

### Behavior switches (A/B, fallbacks)

| Variable | Effect |
|---|---|
| `QWEN_NO_FLASH=1` | disable fused flash attention |
| `QWEN_NO_REUSE=1` | disable persistent graph reuse (CUDA graphs) for single-token decode |
| `QWEN_NO_DIRECT_FETCH=1` | disable zero-copy SSD reads into unified memory (use the staging path) |
| `QWEN_SYNC_FETCH=1` | make RAM-tier H2D copies synchronous (to A/B the async DMA) |
| `QWEN_PREFETCH_THREADS=N` | worker count for parallel SSD reads (default 8, 1-64) |
| `QWEN_GGML_DEBUG=1` | show all raw ggml logs (by default DEBUG/INFO are dropped and repeated WARNs collapsed) |
| `QWEN_CACHE_DEBUG=1` | on a server prefix-cache miss, print where it diverged and the tokens |

### Experimental / test

| Variable | Effect |
|---|---|
| `QWEN_NO_BATCH_PREFILL=1` | revert SSD-tier prefill to the old token-by-token path (default is batched chunks) |
| `QWEN_BATCH_CHUNK=N` | offloaded prefill chunk length (default 4096; layer-major makes expert traffic scale with the chunk count) |
| `QWEN_SEGA_CHUNK=N` | attention sub-chunk length in layer-major prefill (default 256; a smaller T uses the old fused path) |
| `QWEN_SEGB_SLICE=N` | max tokens per FFN slice in layer-major prefill (default 1024; caps MoE activation memory) |
| `QWEN_PREFILL_STATS=1` | print per (chunk, layer) expert union / fetch bytes / slice count to stderr |
| `QWEN_PREFILL_PRUNE=eps` | = `--prefill-prune` (mass-based expert pruning in prefill; lossy) |
| `QWEN_RESIDENT_DECODE=1` | = `--resident-decode` (resident-only routing decode; lossy) |
| `QWEN_RESIDENT_MIN=N` | residents per layer required before masking (default 32; layers below it keep warming up) |
| `QWEN_RESIDENT_WARMUP=N` | = `--resident-warmup` (after this many decode tokens the mask locks in regardless; default 32) |
| `QWEN_RESIDENT_REFILL=N` | = `--resident-refill` (experts refilled per token while masked, all layers combined; default 8, 0 = frozen) |
| `QWEN_SSD_DIRECT=1` | = `--ssd-direct` (Windows: unbuffered reads bypassing the page cache) |
| `QWEN_COALESCE=1` | merge a layer's expert union into large sequential range reads (for drives where sequential beats random QD8) |
| `QWEN_COAL_DEBUG=1` | print per-run read/upload times for coalesced reads |
| `QWEN_PF_CHUNK=N` | = `--pf-chunk` (server prefill slice length; default 4096) |
| `QWEN_PREFILL_CHUNK=N` | chunk length for resident / RAM-tier build_graph prefill (default 512) |
| `QWEN_CPU_PREFILL=1` | revert RAM-tier prefill to running experts on CPU (scheduler). The old behavior: the GPU idles but H2D transfers are avoided |
| `QWEN_MTP_NO_BATCH_PREFILL=1` | revert MTP prefill to token-by-token (default is batched; forced batched with image input) |
| `QWEN_FASTCACHE=1` | optimistic single-graph decode (assumes all experts resident, falls back on a miss; `--resident-decode` without the mask) |
| `QWEN_GDN_TEST=1` | run the GDN multi-token / token-by-token equivalence check and exit |
| `QWEN_MTP_TEST=1` | MTP draft acceptance-rate measurement mode (keeps nextn resident even without `--mtp`) |
| `QWEN_MTP_NOACCEPT=1` | force-reject every MTP draft (output should then match plain; for lossless verification) |

---

## 7. Documentation

- [`docs/mtp.md`](docs/mtp.md) — MTP (self-speculative decoding): implementation, evaluation, benchmarks
- [`docs/resident_decode.md`](docs/resident_decode.md) — resident-only routing decode: mechanism, expert load paths, quality characteristics
- [`docs/metal_unified_memory.md`](docs/metal_unified_memory.md) — tuning log for running 122B-A10B on a 48GB Mac (Metal / unified memory / zero-copy SSD reads)
- [`offload_optimize.md`](offload_optimize.md) — measurement log for expert-offload speedups (pinned / async / graph fusion)
- [`prefill_layer_major.md`](prefill_layer_major.md) — layer-major prefill / expert pruning / resident-only decode: design, measurements, pitfalls (2026-07)
- [`docs/gpu_performance_guide.md`](docs/gpu_performance_guide.md) — GPU performance notes
- [`patches/README.md`](patches/README.md) — patches applied to the vendored ggml and how to re-vendor

---

## 8. Notes / limitations

- Sampling is off by default (greedy, `temp=0`) for reproducibility.
- For large vocabularies (e.g. 248k), K-quant embeddings are kept as an F16 conversion in VRAM only
  when the backend's `get_rows` does not support that type (effectively lossless). CUDA does not
  support K-quant/IQ there, so the conversion runs; Metal and CPU support them natively so no copy is
  made at all (saving memory). When VRAM is tight, `--embd-q8` uses Q8_0 instead (about half of F16,
  with slight quantization error).
- The offload RAM tier uses pinned host memory. A single `cudaHostAlloc` has an upper limit
  (environment-dependent, e.g. ~15.5GB), so experts are pinned in several chunks.
- CUDA graphs and Flash Attention are enabled by default in CUDA builds.
- `--cache-profile` is overwritten on every CLI run (the server only reads it). Mixing runs with
  different access patterns (MTP vs plain) pollutes the profile, so copying the best profile to a
  fixed file is recommended.
- On unified memory (Apple Silicon), SSD-tier misses are `pread` directly into slot memory
  (zero copy).
- The chat template implements logic equivalent to the official Qwen3.6 template directly (no jinja
  engine). It covers tools, `<think>` splitting, image placeholders and reasoning on/off. Video input
  is not supported.
- Prefill is batched (for plain decoding, MTP and SSD offloading alike). MTP prefill captures every
  token's hidden state in one main forward pass and builds the nextn KV in a batch too, so image +
  MTP works together.
