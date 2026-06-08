# Qwen3/3.5 推論エンジン 実装計画

`spec.md` を実現するための実装計画。llama.cpp(`D:\dev\llama.cpp`) と Java実装(`C:\Users\naoki\Desktop\qwenjava`) の資産を最大限流用する前提でまとめる。

---

## 0. 現状調査でわかった重要事実

### llama.cpp は Qwen3.5 を「すでに実装済み」
| ファイル | 内容 | 本プロジェクトでの扱い |
|---|---|---|
| `src/models/qwen3.cpp` | Qwen3 Dense (Step0相当) | 流用 |
| `src/models/qwen35.cpp` (649行) | Qwen3.5 Dense + GDN (Step1相当) | 流用 |
| `src/models/qwen35moe.cpp` (744行) | Qwen3.5 MoE + GDN (Step2-4相当) | 流用 |
| `src/models/delta-net-base.cpp` (607行) | Gated DeltaNet の **chunked ggmlグラフ実装** | 流用（最重要） |
| `src/models/qwen3moe.cpp` / `qwen3next.cpp` | 関連MoE/ハイブリッド | 参照 |
| `ggml/src/ggml-cuda`, `ggml-metal` | GDN/SSM/mul_mat_id のカーネル | そのまま使う |

→ **GDN forward を自作する必要はほぼ無い。** Java実装で苦労した GDN の数値検証(L2norm, query scaling, ssm_norm位置=GDN_IMPLEMENTATION.md の Bug1-4)は、llama.cpp 側が既に通している。

### 結論：本プロジェクトの「新規実装」は2点に集中する
1. **Expert Cache Manager** — VRAM/RAM/SSD の3段キャッシュ + 統計ベースMarkovプリフェッチ。
   llama.cpp には存在しない（`-ot`/`--cpu-moe` による静的オフロードはあるが、動的・統計駆動キャッシュは無い）。
2. **マルチGPU 自動レイアウト** — TFLOPS差判定による LayerSplit / ExpertPool 切替。
   llama.cpp の `-ts`/`-sm` は手動。自動判定とExpertPoolロールは新規。

その他（モデルロード、トークナイザ、サンプラ、CLI/Server、forward）は流用・薄いラッパーで済む。

---

## 1. アーキテクチャ方針の決定（最重要）

`spec.md` のディレクトリ構成 (`core/forward.cpp`, `core/deltanet.cpp`, `core/cache.cpp` …) と
「ggmlを使って」「使えるものはなるべく使いたい」を踏まえ、3つの戦略を比較した。

| 戦略 | 内容 | Expert Cache 実装難度 | 流用度 | 評価 |
|---|---|---|---|---|
| **A: libllama リンク** | `libllama` をそのままリンクし Engine API でラップ | ★★★ 困難（静的mmap+静的グラフに動的キャッシュを差し込めない） | 最大 | ✕ 核心機能が入らない |
| **B: ggml直叩き + グラフ移植** | ggml だけ使い、`qwen35*.cpp`/`delta-net-base.cpp` のグラフ構築コードを `core/` に移植。weight配置を自前管理 | ★ 制御可能 | 大（グラフ式は丸ごとコピー可） | ◎ 推奨 |
| **C: llama.cpp フォーク改造** | llama.cpp 本体を fork して loader/backend を改造 | ★★ 中 | 最大だが本体追従が地獄 | △ |

### 採用：戦略B / サブ選択 B2ハイブリッド（確定 2026-06-06）
- ggml/ggml-cuda/ggml-metal は **submodule として `D:\dev\llama.cpp` から取り込む**（自作しない）。
- **`llm_graph_context`(約3,100行)・`llama-memory-recurrent`・kv-cache の框組みは丸ごと移植しない。**
  - 理由：model build() が薄いのはこの框組みに処理が集約されているため。框組み移植は libllama の半分を
    持ち込むことになり、Expert Cache の差し込みが抽象の壁と衝突する。
- 代わりに **B2ハイブリッド**：
  - GDN(`delta-net-base.cpp`) と各archのグラフ合成(`qwen3/qwen35/qwen3next`)は **数式を写経して core/ へ持ち込む**。
  - `build_attn`/`build_norm`/`build_rope`/`build_moe_ffn` は **kv-cache/memory抽象に依存しない薄い自前ヘルパ**として再実装。
  - GDN/Attention の状態（recurrent state / KVキャッシュ）は spec に沿って自前の固定バッファで保持。
  - MoE の `ggml_mul_mat_id` に渡す **expert weight を Cache Manager が差し替える** ことで3段キャッシュを実現（核心の接続点）。
- 利点：流用は最大化（GDN数値検証・量子化・トークナイザ・グラフ式）しつつ、expert weight の配置を完全掌握できる。

---

## 1.5. 依存方針：このフォルダで自己完結（確定 2026-06-06）

**`D:\dev\llama.cpp` はビルド/実行の依存にしない。** このリポジトリ単体で `cmake → build` が完結する。

- **vendor するのは `ggml/` サブツリーのみ**（→ `third_party/ggml/`）。スナップショットでコピー（git submodule不使用＝ネット/git不要）。
  - 元コミットハッシュを `third_party/ggml/GGML_VERSION` に記録、`LICENSE`(MIT)同梱。
- GDNの特殊op（`ggml_gated_delta_net` / `ggml_solve_tri` / `ggml_ssm_conv` / `ggml_cumsum` / `ggml_tri` / `ggml_softplus`）は
  **すべて `ggml/` 内で完結**することを確認済み：
  - 定義 `ggml/include/ggml.h` + `ggml/src/ggml.c`
  - **CUDA**: `ggml-cuda/{gated_delta_net,solve_tri,ssm-conv,cumsum}.cu` ✅
  - **Metal**: `ggml-metal-device.cpp` 等に全op実装 ✅
- 移植するグラフコード（delta-net / qwen3 / qwen35 / qwen3next）は写経した時点で **core/ の自前ソース**になり依存は残らない。
- `D:\dev\llama.cpp` が要るのは「開発時のみ」の2用途だけ（ビルド成果物に残らない）：
  1. グラフ式の写経の参照元（作成時のみ）
  2. `--compare-ref` の数値検証で llama-cli を実行（任意の開発ツール）

---

## 2. モジュール構成（spec.md 準拠 + 補足）

```
qwencpp/
  core/
    engine.{h,cpp}      外部API（pimpl）          §Engine API
    model.{h,cpp}       GGUFロード/メタ/テンソル配置   ← gguf.cpp流用 + Java GGUFReader.java参照
    forward.{h,cpp}     forward pass全体（層ループ）    ← qwen35*.cpp移植
    deltanet.{h,cpp}    GDNグラフ構築               ← delta-net-base.cpp移植
    moe.{h,cpp}         MoE dispatch + cache接続点    ← qwen35moe.cpp移植
    cache.{h,cpp}       Expert Cache Manager【新規・核心】
    ssd.{h,cpp}         SSDバックエンド(pread/mmap/threadpool, Linux io_uring)【新規】
    prefetch.{h,cpp}    Markov遷移予測 + 非同期転送【新規】
    sampler.{h,cpp}     argmax/top-p/temp           ← Java Qwen.java/llama-sampling参照
    stats.{h,cpp}       統計収集（hit count, transition, hit rate）【新規】
    gpu_layout.{h,cpp}  マルチGPU自動レイアウト判定【新規】
    tokenizer.{h,cpp}   tiktokenBPE                ← llama-vocab.cpp流用 or Java移植
  cli/main.cpp          ← spec CLIオプション
  server/main.cpp,handlers.cpp  OpenAI互換(SSE)   ← cpp-httplib
  third_party/ggml/     submodule (llama.cppのggmlサブツリー)
  third_party/cpp-httplib/
  tests/                数値検証（llama-cli突き合わせ）
  CMakeLists.txt
```

---

## 実装進捗（2026-06-06）

- [x] **Phase 0 完了**：ggml vendor(commit e674b12)、CMake(MSVC+ninja)、`core/gguf_util`・`core/model`、CPUビルド成功。
- [x] **Phase A / qwen3 Dense スライス完成・検証済み**：
  - `core/model.cpp` 重みロード（GGUF→backend buffer、64bit seek）
  - `core/runtime.cpp` forwardグラフ（GQA + per-head Q/K RMSNorm + RoPE NEOX + KVキャッシュ + SwiGLU）+ greedy生成
  - `core/tokenizer.cpp` GPT2 byte-level BPE（encode/decode）
  - 検証：「The capital of France is」→「Paris. The capital of Italy is Rome...」、Q8_0/Q4_K_M両方で正常生成・exit 0
- [x] **Phase A / qwen35(GDN Dense) スライス完成・検証済み**：
  - `core/runtime.cpp` に GDN層（融合op `ggml_gated_delta_net` + causal conv1d + recurrent state(conv/ssm)）、gated attention（gate sigmoid + partial RoPE n_rot=64 + post_attention_norm 残差構造）を実装
  - recurrent state（conv_state[3,6144]/ssm_state[128,128,16]）をst_bufに常駐、reset時ゼロ化
  - imrope sections[11,11,10,0]はテキスト等位置のためNEOX partial ropeで等価処理
  - 検証：「Once upon a time」→「in a world where everything was made of atoms... a special kind of atom called a "superatom"...」高品質生成。Q8_0/Q4_K_M両OK、exit 0
- [x] **Phase A / ツール群 完成・検証済み**：
  - `core/sampler.cpp`：greedy/temperature/top-k/top-p(nucleus)、seed対応
  - `core/chat.h`：ChatML(`<|im_start|>`…)プロンプト構築（special token id をvocabから解決）
  - `cli/main.cpp`：`-p`/`--chat`/`-i`対話/`--temp`/`--top-p`/`--top-k`/`--seed`/`--log-tokens-per-sec`
  - `server/main.cpp`：OpenAI互換（httplib+nlohmann json vendor）。`/health`・`/v1/models`・`/v1/chat/completions`（**SSEストリーミング**＋非ストリーミング、usage付き）
  - 検証：`--chat`でQwen3.5 thinkingモード発動「Red/Green/Blue」正答、サーバSSEチャンク正常
- [x] **MoE (qwen3moe / qwen35moe) 実装**：
  - `core/runtime.cpp` に `build_moe`（softmax gating + `ggml_argsort_top_k` + `ggml_mul_mat_id` × gate/up/down + 正規化weight + expert集約）
  - shared expert（qwen35moe：gated SwiGLU `ffn_*_shexp` を sigmoid gate付きで加算）対応
  - GDN の H_k≠H_v（qwen35moe: H_k=16/H_v=32）は融合opが内部GQA処理
  - 検証：qwen3moe Qwen3-30B-A3B Q2_K →「Paris. This is a well...」正答 **6.7 tok/s**(CPU)；
    qwen35moe Qwen3.6-35B-A3B Q3_K_S →「Paris, a city renowned for its rich history...」正答 **5.7 tok/s**(CPU)
- **対応アーキ進捗：qwen3 / qwen35 / qwen3moe / qwen35moe の4種完成。残り qwen3next。**
- [x] **GPU(CUDA)バックエンド有効化・検証済み**：
  - `core/runtime.cpp`：`ggml_backend_dev_by_type(GPU)`→`ggml_backend_dev_init` でGPU選択、無ければCPUフォールバック。`RuntimeConfig::use_cuda`（既定true）、CLI/Server に `--cpu`
  - CMake `-DQWENCPP_CUDA=ON` で ggml-cuda ビルド（CUDA 12.8、arch 89/86）
  - GPU2枚検出（RTX 3050 6GB / RTX 4060 Ti 16GB）。CUDA0=4060 Ti を選択（現状は最初のGPU device）
  - 検証：qwen35 0.8B GPU **71 tok/s** vs CPU 26 tok/s（約2.7倍）。出力コヒーレント（CPU/CUDAのFP差でgreedy分岐は正常）
  - ※全テンソルを単一バックエンドbufferに載せる方式。大MoEは要VRAM。マルチGPU/段階配置は今後（Expert Cache）
  - **GPU固有バグ2件を修正**：
    (1) q2_K等の埋め込みで `get_rows` がCUDA未対応 → `token_embd` を F32 にdequantして使用（`Model::tok_embd_rows()`）
    (2) **転置V cacheの ne0==1 書き込みがCUDAで不正**（単一トークンdecodeで出力崩壊）→ V cacheを非転置格納（K同様contiguous書き込み）し、転置を読み出し時に実施
  - CPU/GPU argmax軌跡を突き合わせる内部比較で根本原因を特定（prefillはOK→step1のn_tokens=1 decodeで乖離→転置V書き込みと判明）
  - 検証（GPU, 全てCPUと一致）：0.6B dense 52 tok/s、0.8B GDN 56 tok/s、30B-A3B MoE Q2_K **36.5 tok/s（CPU 6.7の約5.4倍）**
- [x] **decode性能最適化（graph再利用 + Flash Attention）**：
  - 計測：30B-A3B MoE は prefill 459 tok/s に対し decode 39.5 tok/s（11.6倍差）＝decode固定オーバーヘッド支配。内訳 build1.4+alloc1.1+compute18.5ms。compute=約2400極小カーネルのGPU側逐次実行が主因
  - **グラフ再利用**(`decode_reuse`)：decodeグラフを1回構築・同一オブジェクト再利用、KV書込を`ggml_set_rows`+index入力(I64)で動的化、n_kvをKV_BUCKET=256でバケット化、KVキャッシュゼロ初期化
  - **Flash Attention**(`ggml_flash_attn_ext`)：attentionの約8カーネル/層を1つに融合、mask F16化。head128/256とも対応
  - 追加調査：ggml-cuda の融合は既に発火（topk-moe router / gate-up-swiglu(GLU) / add連鎖 / rms_norm+mul）。融合on/off で 17→26ms＝カーネルオーバーヘッド律速を確認。
  - MoE集約をGEMV化（8→2カーネル）も効果なし＝add連鎖は既に融合済み。
  - **CUDA Graphは効いていない**（reuse compute 15.8 vs rebuild 14.1ms）＝GPU側per-kernel律速でCPU起動削減が効かない。reuseの利点はbuild/alloc削減のみ。
  - KV_BUCKET=256→**32**：dense/GDNのattention無駄削減。
  - **【真因特定】CUDA Graphが効かないのではなく、ggml標準で「コンパイル除外」されていた**：`GGML_CUDA_GRAPHS` 既定OFF（ggml CMake、コメント"llama.cpp only"）→ `USE_CUDA_GRAPH` 未定義 → CUDA Graph機構ごと無し。
    - 診断：ggml-cuda.cu に env(QWEN_CG_DEBUG)計装＋cg.logファイル直書きで「関数は呼ばれるが #ifdef USE_CUDA_GRAPH 内が空」を確認。
    - 対処：CMakeで `set(GGML_CUDA_GRAPHS ON)`（QWENCPP_CUDA時）。
    - 効果：[cg]ログで warmup完了→CUDA Graph再生を確認。decode compute 14→9.2ms。グラフ再利用+bucket化が正しい土台だった（再利用でグラフ安定→warmup成立）。
  - **最終結果（GPU, CUDA Graph有効）：0.6B dense ~196、qwen35 0.8B GDN ~154、30B-A3B MoE ~91 tok/s**。出力全て正常、CPU回帰なし。**llama.cpp(30B ~100)にほぼ到達**。
- [ ] **Phase A 残り**：qwen3next(GDN+MoE)、`--compare-ref`数値検証、マルチGPU自動レイアウト

---

## 3. 実装ステップ（流用を踏まえて圧縮 / 確定版）

> 方針：spec の Step0-1（GDNの漸進的立ち上げ）は llama.cpp 流用でほぼ消化されるため
> **「Phase A: 3アーキ同時立ち上げ＋ツール完成」に統合**し、リソースを Phase B以降の
> Expert Cache（spec Step2-4 の核心）に集中する。3アーキ(qwen3/qwen3next/qwen35)は
> いずれも `llm_build_delta_net_base` を共有 → GDN移植は1回で全対応。

### Phase 0: 足場づくり
- [ ] ggml を submodule 化（**CUDA と Metal の両ビルドを最初から通す**＝クロスプラットフォーム確定方針）
- [ ] CMake 雛形、`Engine`/`InferConfig` ヘッダのスケルトン（spec API通り）
- [ ] `model.cpp`：GGUFリーダ（`gguf.cpp`流用）でメタ・テンソル一覧。Java `GGUFReader.java` のアライメント知見を踏襲
- [ ] `tokenizer`：`llama-vocab.cpp` 移植で round-trip テスト
- [ ] **B2ハイブリッド方針の自前グラフヘルパ雛形**：`build_attn`/`build_norm`/`build_rope`/`build_moe_ffn` を
      `llama-graph.cpp` から**数式だけ写経**した薄い関数として用意（kv-cache/memory抽象には依存しない）

### Phase A: 3アーキ同時立ち上げ ＋ ツール完成（旧 Step0+Step1 を統合）
- [ ] `deltanet.cpp`：`delta-net-base.cpp` の chunked GDN を移植（**1回で qwen35/qwen3next 両対応の基底**）
- [ ] `forward.cpp`：3アーキ分の build() 合成を移植
  - **qwen3 (Dense)**：GQA+RoPE+SwiGLU（RoPE base=1e6, eps=1e-6, EOS=151645）
  - **qwen35 (GDN Dense)**：`l%4==3`→Gated Attention(sigmoidゲート)、他→GDN
  - **qwen3next (GDN+MoE ハイブリッド)**：`qwen3next.cpp` の層パターン・gating
- [ ] GDN状態保持を自前の固定サイズ state バッファで実装（recurrent state、Java `gdn_state` 相当）
- [ ] サンプラ、CLI(`-m -p -i --n-ctx --compare-ref`)、Server(OpenAI互換+SSE) 完成 ← **以降ツールは凍結**
- [ ] **検証**: 3アーキそれぞれ llama-cli と greedy一致 / GDN中間テンソル(rms)一致
      （Java `GDN_IMPLEMENTATION.md` の o rms・gamma・beta を回帰テストに転記）
- [ ] **検証**: GDN層のKVキャッシュ削減を実測

### Phase B: Expert Cache 基本（MoE、RAM全量 / SSDなし）← 旧 Step2・本番開始
- [ ] `moe.cpp`：`qwen35moe.cpp` の MoE dispatch を移植。**expert weight を Cache Manager 経由で供給**する形に改造
- [ ] **最優先PoC**: `ggml_mul_mat_id` への expert 差し替え成立性（→§4のリスク。Phase A中に前倒し着手可）
- [ ] `cache.cpp` v1：Stage1(VRAM LRU) + Stage2(RAM常駐) の2段
- [ ] `stats.cpp`：expert hit count / transition 行列（spec JSON形式）
- [ ] `prefetch.cpp`：Markov遷移確率で次tokenを予測 → 非同期(別stream)転送
- [ ] **検証**: 出力が「キャッシュ無効(全VRAM)時とビット一致」（キャッシュは速度最適化、数値不変）
- [ ] **検証**: expert使用分布、プリフェッチ命中率、cache hit率(gpu/ram)

### Phase C: SSD tier（RAM制限）← 旧 Step3
- [ ] `ssd.cpp`：mmap + pread + スレッドプール。Linuxは io_uring 追加最適化
- [ ] `cache.cpp` v2：Stage3(SSD) 追加、3段LRU/ピン留め完成
- [ ] 静的ピン留め（統計上位N expertを起動時RAM展開）、オンライン学習(EMA)
- [ ] **検証**: RAMを絞りSSDアクセス誘発、プリフェッチ精度・hit率測定、122Bへ外挿予測

### Phase D: 最終目標 122B-A10B Q3KM ← 旧 Step4
- [ ] Q3KM の Cache転送・dequant経路確認（dequant自体はggml任せ＝実装不要）
- [ ] VRAM16GB/RAM32GB 配分チューニング（40 expert GPU / 137 RAM / 69%カバー）
- [ ] **目標**: hit率70%以上、8tok/sec（楽観）/ 4-6tok/sec（現実）

### 横断: マルチGPU（Phase B以降で並行）
- [ ] `gpu_layout.cpp`：cudaDeviceProp から FP16 TFLOPS推定 → `min/max>0.8` で LayerSplit / 否で ExpertPool
- [ ] LayerSplit：層を均等分割、各GPUが独立にExpert管理（GPU間hidden state転送のみ）
- [ ] ExpertPool：高性能GPUが計算、低性能GPUをExpert VRAMプールとして使用
- [ ] CLI `--gpu-ids` / `--gpu-layout` 配線

---

## 4. Expert Cache Manager 詳細設計（核心）

### 接続点
MoE層の `ggml_mul_mat_id(ctx, experts_tensor, ...)` に渡す `experts_tensor` を、
全expertの連続バッファではなく **「今選ばれたexpertだけVRAMに集めたバッファ」** にする。
- gating で選ばれた top-k expert id を取得 → Cache Manager に要求
- Manager は VRAM(Stage1) に無ければ RAM(Stage2)→SSD(Stage3) から昇格、LRUで追い出し
- 次token用に prefetch が裏で先回り転送

### データ構造
- `slot table`：VRAMスロット（~40個）の {expert_id, layer, last_used, dirty}
- `ram pool`：pinned host mem（`cudaMallocHost`）/ Metalは通常malloc。常駐expert群
- `transition[layer][from_expert] -> distribution`：Markov行列（オンラインEMA更新）
- `hit counter[layer][expert]`：統計、stats.json入出力

### プラットフォーム差分（spec通り）
```cpp
void* alloc_host_memory(size_t n){
#ifdef GGML_USE_CUDA
  void* p; cudaMallocHost(&p,n); return p;     // pinned
#else
  return malloc(n);                            // Metal unified
#endif
}
```
Metal は Unified Memory なので Stage1↔Stage2 転送実質ゼロ。CUDA は PCIe DMA を非同期streamで。

### リスク
- mul_mat_id がexpertテンソルの**連続レイアウト**を前提にしている → スロットへの gather/再配置が必要。最大の技術的不確実性。Step2 で PoC を最優先で潰す。

---

## 5. 流用マッピング早見表

| 必要機能 | 流用元 |
|---|---|
| ggml本体/CUDA/Metalカーネル | `D:\dev\llama.cpp\ggml\` をsubmodule |
| GGUFパース | `ggml/src/gguf.cpp` ＋ Java `GGUFReader.java`(アライメント知見) |
| Qwen3/3.5 forwardグラフ | `src/models/qwen3.cpp` `qwen35.cpp` `qwen35moe.cpp` |
| Gated DeltaNet | `src/models/delta-net-base.cpp` |
| トークナイザ(tiktoken BPE) | `src/llama-vocab.cpp` or Java `tokenizer.bin`+BPE実装 |
| サンプラ | `src/llama-sampling.cpp` or Java `Qwen.java` |
| GDN数値検証の落とし穴 | Java `GDN_IMPLEMENTATION.md`（Bug1-4） |
| Server(HTTP/SSE) | `tools/server` 参照 + cpp-httplib |

---

## 6. 検証戦略

- 各Stepで `--compare-ref /path/to/llama-cli` を必須化。
  - logits L2誤差・greedy生成一致・GDN中間テンソル(rms)を突き合わせ。
- Java `GDN_IMPLEMENTATION.md` の検証点（o rms, gamma, beta 等）を回帰テストに転記。
- Expert Cache は「キャッシュ無効(全VRAM)時と出力ビット一致」を不変条件にする（キャッシュは速度最適化であり数値を変えない）。

---

## 7. 直近の着手順（最初の2週間想定）

1. ggml submodule化 + CMakeでWindows/CUDAビルド成功
2. `model.cpp` GGUFロード + `tokenizer` round-trip
3. `forward.cpp` に qwen3.cpp 移植 → Qwen3-0.6B/8B で生成、llama-cli一致（Step0 CLI最小）
4. **mul_mat_id へのexpert差し替え PoC**（小さいMoEで成立性を早期検証＝最大リスクの前倒し）

---

## 8. 確定した方針（2026-06-06）

- **アーキテクチャ：戦略B（ggml直叩き + llama.cppグラフ移植）で確定。**
- **対象プラットフォーム：Windows+CUDA と macOS+Metal を最初からクロスプラットフォーム前提で並行。**
  - `alloc_host_memory` 等のプラットフォーム差分を初期から `#ifdef GGML_USE_CUDA` で分岐。
  - ggml submodule のビルドも CUDA / Metal 両方を Phase0 で通す。
  - Metal の Unified Memory（Stage1↔2転送ゼロ）の利点を早期に検証できる。
- **トークナイザ/サンプラ：llama.cpp移植（`llama-vocab.cpp` / `llama-sampling.cpp`）を基準にする。**
  - llama-cli との一致検証がしやすいため。Java実装は補助参照。
