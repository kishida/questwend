# QuestWend

Qwen3 / Qwen3.5 / Qwen3.6 をスクラッチ実装した推論エンジン（vendored ggml の上に構築）。

- **対応アーキテクチャ**: `qwen3`（dense）, `qwen3moe`（MoE）, `qwen35` / `qwen35moe`（Gated DeltaNet ハイブリッド）, `qwen3next`
- **バックエンド**: CUDA（Windows/Linux）, Metal（macOS）, CPU フォールバック
- **主な機能**:
  - MoE エキスパートのオフロード（**RAM 階層** / **SSD 階層**）＋動的 VRAM エキスパートキャッシュ
    → 16GB GPU で 35B-A3B MoE などを実行可能
  - エキスパート常駐プロファイルの永続化（`--cache-profile`）でウォーム起動を高速化
  - MTP（Multi-Token Prediction / nextn）自己推測デコード（`--mtp`, `--draft N`）
  - **画像入力**（Qwen3-VL 系 + mmproj GGUF; vision tower を ggml/GPU で実行。CLI `--image`、サーバーは OpenAI 形式の `image_url`）
  - **tool calling**（OpenAI 互換: `tools` / `tool_calls` / `role:"tool"`、Qwen3.6 の `<function=...>` 形式を双方向変換）
  - 分割（sharded）GGUF 対応（`-NNNNN-of-MMMMM.gguf`）
  - 大語彙モデルの K-quant 埋め込みは、バックエンドの get_rows が非対応の場合のみ F16 変換版を VRAM に保持（CUDA のみ; Metal/CPU はネイティブ対応のためコピー不要。`--embd-q8` で Q8_0 化も可）
  - OpenAI 互換サーバー（`qw-server`）＋ブラウザ・チャット UI

---

## 1. 必要なもの

| | Windows (CUDA) | macOS (Metal) | CPU のみ |
|---|---|---|---|
| コンパイラ | Visual Studio 2022 (C++) | Xcode Command Line Tools | 任意の C++17 |
| CMake | 3.20+（VS 同梱のものでも可） | 3.20+（`brew install cmake`） | 3.20+ |
| その他 | CUDA Toolkit 12.x | — | — |

リポジトリは vendored ggml を含み自己完結しています（`third_party/ggml`）。サブモジュール取得は不要。

---

## 2. ビルド

### Windows（CUDA）

```powershell
# 構成（CUDA バックエンドを有効化）
cmake -B build -DQW_CUDA=ON

# ビルド（Release）
cmake --build build --config Release --target qw-cli qw-server
```

- 生成物: `build\Release\qw-cli.exe`, `build\Release\qw-server.exe`（ggml の DLL は `build\bin\Release\` に置かれ、exe と同じ場所から実行すれば解決されます）。
- CUDA アーキは native 検出（例: RTX 4060 Ti = sm_89, RTX 3050 = sm_86）。
- CMake は `CUDA graphs ON`, `Flash Attention ON` で構成されます。
- VS 同梱の cmake を使う場合の例:
  `"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"`

### macOS（Metal）

```bash
cmake -B build -DQW_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target qw-cli qw-server
```

- 生成物: `build/qw-cli`, `build/qw-server`。
- 起動時に `backend: GPU [MTL0] ...` と出れば Metal が有効。

### CPU のみ（任意プラットフォーム）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target qw-cli
# 実行時に --cpu を付けるか、GPU バックエンド無効ビルドで自動的に CPU で動作
```

---

## 3. 使い方（CLI: `qw-cli`）

```bash
# 一回限りの生成
qw-cli -m model.gguf -p "The capital of France is" -n 128

# チャット（ChatML でラップ）
qw-cli -m model.gguf --chat -p "Explain photosynthesis." -n 256
qw-cli -m model.gguf -i                      # 対話モード
qw-cli -m model.gguf --info                  # モデル情報を表示して終了
```

### 主なオプション

| オプション | 説明 |
|---|---|
| `-m <path>` | モデル GGUF（分割 GGUF は先頭シャードを指定） |
| `-p <text>` | プロンプト |
| `-n <N>` | 生成トークン数（既定 128） |
| `-i` | 対話チャット |
| `--chat` | `-p` を ChatML でラップ |
| `--reasoning <on\|off>` | thinking モード（既定 on; off で `<think>\n\n</think>` 終端） |
| `--n-ctx <N>` | コンテキスト長（既定 4096） |
| `--temp <f>` / `--top-p <f>` / `--top-k <N>` / `--seed <N>` | サンプリング（既定 greedy: temp=0） |
| `--embd-q8` | 埋め込み get_rows フォールバックを F16 でなく Q8_0 にする（VRAM 節約; わずかな量子化誤差） |
| `--image <path>` | 画像をプロンプトに添付（複数可。VL モデル + mmproj が必要） |
| `--mmproj <gguf>` | vision tower の GGUF（省略時はモデルと同じフォルダの `mmproj-*.gguf` を自動発見） |
| `--vision-test` | 画像を vision tower だけでエンコードして統計表示（数値検証用; LLM 不要） |
| `--log-tokens-per-sec` | prefill / 生成の tok/s を表示 |
| `--cpu` | CPU バックエンドを強制 |

### エキスパート・オフロード（大きい MoE を限られた VRAM で）

```bash
# RAM 階層: 非エキスパート重みを GPU、エキスパートは pinned host RAM に置き
#           動的 VRAM キャッシュ経由でストリーミング（PCIe）
qw-cli -m moe.gguf -p "..." -n 128 --vram-budget 15000

# SSD 階層: エキスパートを GGUF からディスク直読（RAM コピーなし）
qw-cli -m moe.gguf -p "..." -n 128 --vram-budget 12000 --experts-ssd

# 常駐プロファイル: 1回目に保存、2回目以降は起動時にホットなエキスパートを事前常駐
qw-cli -m moe.gguf -p "..." -n 128 --vram-budget 15000 --cache-profile hot.prof
```

- `--vram-budget <MB>`: スロットプールに使う VRAM 予算（>0 でオフロード有効）。
- `--experts-ssd`: エキスパートをディスクからストリーミング（RAM に載らない巨大モデル向け）。
- `--cache-profile <file>`: ホットエキスパートの頻度プロファイルを永続化。**同一/類似ワークロードで hit 率が ~100% になり、ストリーミングがほぼ消えて大幅高速化**。

prefill は **layer-major 実行**（層ごとにエキスパートを一度だけ fetch して全トークンを処理）が既定で、
旧方式比 RAM 階層 ~4倍 / SSD 階層 ~8倍。さらに速度を求めるときの調整ノブ（いずれも lossy または挙動変更、
計測は [`prefill_layer_major.md`](prefill_layer_major.md)）:

- `--resident-decode`: **decode を常駐エキスパート限定ルーティングの融合単一グラフで実行**。
  数トークンのウォームアップ後にルーターを常駐集合へマスクし、以後ミスなし
  （検証・状態バックアップ・フォールバックも消える）。「本来選びたかった」エキスパートは
  毎トークン裏で補充されマスクに合流するので、話題の変化にも数トークン遅れで追従する。
  実測: RAM 階層 20.8 → 36〜45 tok/s、SSD 階層 3.1 → 25 tok/s（真のSSD読み時）。lossy。
- `--resident-refill <N>`: 補充するエキスパート数/トークン（既定 RAM 8 / SSD 4、`0` で完全凍結=非推奨）。
- `--resident-warmup <N>`: マスク固定までの decode トークン数（既定 32）。
- `--prefill-prune <eps>`: prefill で router 重み合計が層の eps 未満の非常駐エキスパートの fetch を
  スキップ（例 `0.05`。SSD 階層で特に有効）。lossy。
- `--ssd-direct`: SSD 読みで OS のページキャッシュをバイパス（Windows, unbuffered I/O）。
  モデルが RAM に収まらない場合の実性能を測るとき・二重キャッシュを避けたいときに。

### MTP 自己推測デコード（nextn ブロックを持つモデル）

```bash
qw-cli -m model-MTP.gguf -p "..." -n 128 --mtp            # 1 トークン先読み
qw-cli -m model-MTP.gguf -p "..." -n 128 --mtp --draft 2  # 2 トークン先読み（推奨）
```

- 出力は通常デコードと完全一致（greedy ロスレス）。
- 計算律速（モデル全量 VRAM 常駐）では `--draft 2`〜`3` で plain 比 **+30%程度**
  （27B Q2, ctx=2000: 18.2 → 24.2 tok/s）。オフロード（fetch 律速）では不利。
  詳細は [`docs/mtp.md`](docs/mtp.md)。

---

## 4. 使い方（サーバー: `qw-server`）

OpenAI 互換の `/v1/chat/completions`（SSE ストリーミング対応）と、ブラウザ・チャット UI を提供。

```bash
qw-server -m model.gguf --host 0.0.0.0 --port 8080 --vram-budget 15000
```

ブラウザで `http://<host>:<port>/` を開くとチャット UI（TTFT / tok/s / prefill tok/s / 出力トークン数を表示）。

| オプション | 説明 |
|---|---|
| `-m <path>` | モデル GGUF |
| `--host <addr>` | バインドアドレス（既定 `127.0.0.1`; LAN 公開は `0.0.0.0`） |
| `--port <N>` | ポート（既定 8080） |
| `--n-ctx <N>` | コンテキスト長 |
| `--vram-budget <MB>` / `--experts-ssd` | エキスパート・オフロード |
| `--cache-profile <file>` | 常駐プロファイル（**サーバーは読み込みのみ**, 上書きしない） |
| `--reasoning <on\|off>` | thinking モードの既定（UI のチェックにも反映） |
| `--mtp` | MTP 自己推測デコード（nextn ブロックを持つモデル; 起動時に有効化） |
| `--draft <N>` | MTP draft 長（既定 1） |
| `--embd-q8` | 埋め込み get_rows フォールバックを Q8_0 にする（VRAM 節約） |
| `--mmproj <gguf>` | 画像入力用 vision tower（省略時はモデルと同じフォルダの `mmproj-*.gguf` を自動発見） |
| `--no-mmproj` | mmproj があっても画像入力を無効化 |
| `--cache-slots <N>` | プレフィックスキャッシュの追加スロット数（既定 0; 複数会話の並行アクセス用） |
| `--cache-slots-dir <dir>` | スロットの退避先を RAM ではなくディスクにする（再起動後もキャッシュが残る） |
| `--time-slice <N>` | 同時ストリーミングを N トークンごとに交互実行（既定 0 = 完全直列） |
| `--cpu` | CPU バックエンドを強制 |
| `--resident-decode` ほか | オフロード調整ノブ（`--resident-refill/-warmup`, `--prefill-prune`, `--batch-chunk`, `--ssd-direct`; CLI と共通、上のオフロード節を参照） |
| `--pf-chunk <N>` | サーバー prefill のスライス長（切断検出の粒度; 既定 4096） |

> 画像は OpenAI 形式（`content` 配列の `image_url` に base64 data URI）で受け付け。ブラウザ UI にも 📎 ボタンあり。MTP 有効時もそのまま画像入力可。
> mmproj をロードすると vision tower の GPU 使用分（重み + 計算バッファ; 起動時にログ表示）が `--vram-budget` から自動で差し引かれる。Metal の作業セット上限いっぱいに budget を張っていても OOM しない。テキストのみで使うときは `--no-mmproj` でその分を expert キャッシュに戻せる。
> tool calling は `tools` を渡すと `<tool_call>` 出力を OpenAI の `tool_calls` に変換して返す（`finish_reason: "tool_calls"`）。`role:"tool"` の応答メッセージにも対応。

> MTP はモデルロード時に nextn ブロックを VRAM 常駐させるため、リクエスト単位ではなく**サーバー起動フラグ**で指定します。ストリーミング/非ストリーミングの両方に対応。**MTP は greedy デコード専用**: リクエストの `temperature` / `top_p` / `top_k` は無視され、常に temp=0 相当の出力になります（起動ログにも明示）。サンプリングが必要な場合は `--mtp` なしで起動してください。

> **プロンプト・プレフィックスキャッシュ**: 前リクエストの KV キャッシュ / GDN 状態をサーバーが保持し、新しいプロンプトが前回の「プロンプト + 生成」の続き（通常のチャット継続）なら差分トークンだけを prefill する。会話が長くなっても各ターンの prefill コストは新規入力分のみ。`timings.cached_tokens`（UI では `N cached`）で再利用量を確認できる。トークン分割が再トークナイズで揺れた場合はテキストレベルで照合して吸収。画像はバイト列ハッシュで同一性を確認（別画像なら全 prefill）。履歴の `<think>` ブロックはキャッシュ一致のため保持して再構成する（`preserve_thinking`）。途中ターンの編集や別会話への切り替えは自動でフルリセット。

> **複数会話の並行アクセス**（エージェント等）: 既定ではライブ状態 1 つだけなので、会話 A/B が交互に来ると毎回フル prefill に戻る。`--cache-slots N` で退避スロットを持たせると、別会話への切り替え時にライブ状態をスロットに退避し、戻ってきたとき復元して差分 prefill だけで継続できる。退避先は既定で RAM（1 スロットあたり GDN 状態 + 会話長ぶんの KV）。`--cache-slots-dir <dir>` を指定するとディスク退避になり RAM を消費せず、**サーバー再起動後もキャッシュが残る**（モデル / n-ctx が変わったスロットはヘッダ検証で自動破棄）。`--cache-slots-dir` のみ指定した場合はスロット数 4。

> **リクエストログ**（stderr）: 各リクエストで開始行と完了行を出力する。開始行は `[時刻] req: N tok (+img), KV reuse M (live/slot/none), prefill K (resident)` — 受付トークン数、プレフィックスキャッシュ再利用量とその出所、実際に prefill するトークン数。完了行は `[時刻] done: prefill X tok/s, gen Y tok/s (Z tok), expert hit P%` — プレフィル/デコード速度、出力トークン数、（オフロード時のみ）そのリクエストでの expert キャッシュヒット率。オフロードの長いプレフィル中は 10 秒ごとに `prefill done/total (%)` の進捗行も出る（常駐モデルは一括で速いので進捗行なし）。

> **タイムスライス**: 既定ではリクエストは完全直列（後着は前の生成完了まで待つ）。`--time-slice N` を指定すると、同時ストリーミングが競合したとき N トークンごとにランタイムを交互に明け渡し、両方のストリームが並行に進む（出力は無劣化; 状態はスロット経由で正確に退避/復元される）。**prefill も割り込み可能**: 長いプロンプトの prefill は `max(N, 256)` トークンのチャンク単位で進み、チャンク境界で待機者がいれば明け渡す（バッチ効率を保つため生成時より粗い粒度）。競合していないときはオーバーヘッドゼロ。切り替えコストは状態サイズ次第で数十 ms 程度なので N=10〜50 が目安。`--cache-slots`（スライス退避は RAM 推奨）と併用すること。スロットなしでも動くが、中断された側は再開のたびに全コンテキストを再 prefill する。

---

## 5. 分割（sharded）GGUF

`model-00001-of-00003.gguf` のように分割されたモデルは、**先頭シャード**を `-m` に渡すだけで自動的に全シャードを探索・統合します（命名は 5 桁ゼロ詰め、00001 開始）。

```bash
qw-cli -m Qwen3.5-122B-A10B-00001-of-00005.gguf -p "..." --vram-budget 40000 --experts-ssd
```

---

## 6. 環境変数（デバッグ / チューニング）

通常は不要。プロファイルや A/B 検証、トラブルシュート用。

### プロファイリング

| 変数 | 効果 |
|---|---|
| `QWEN_PROF=1` | 単発デコード（decode_reuse）の所要時間内訳を表示 |
| `QWEN_PROF_DC=1` | オフロード decode の wall / GPU計算 / ホスト時間を exit 時に表示 |
| `QWEN_PROF_MTP=1` | MTP の per-cycle フェーズ分解（draft / verify / settle / resync）を表示 |
| `QWEN_PROF_MTP2=1` | MTP draft 1回ごとの compute / readback 時間（100回平均）を表示 |

### 動作切り替え（A/B・フォールバック）

| 変数 | 効果 |
|---|---|
| `QWEN_NO_FLASH=1` | fused flash attention を無効化 |
| `QWEN_NO_REUSE=1` | 単発デコードの永続グラフ再利用（CUDA グラフ化）を無効化 |
| `QWEN_NO_DIRECT_FETCH=1` | ユニファイドメモリへのゼロコピー SSD 読みを無効化（staging 経路に戻す） |
| `QWEN_SYNC_FETCH=1` | RAM 階層の H2D を同期コピーに（async DMA の A/B 用） |
| `QWEN_PREFETCH_THREADS=N` | SSD 並列読みのワーカー数（既定 8, 1〜64） |
| `QWEN_GGML_DEBUG=1` | ggml の生ログを全表示（既定は DEBUG/INFO 破棄、同一 WARN 連続は1回） |
| `QWEN_CACHE_DEBUG=1` | サーバーのプレフィックスキャッシュがミスしたとき分岐位置とトークンを表示 |

### 実験的 / テスト

| 変数 | 効果 |
|---|---|
| `QWEN_NO_BATCH_PREFILL=1` | SSD 階層の prefill を旧来の token-by-token に戻す（既定はバッチチャンク実行） |
| `QWEN_BATCH_CHUNK=N` | オフロード prefill のチャンク長（既定 4096; layer-major で expert 転送はチャンク数に比例） |
| `QWEN_SEGA_CHUNK=N` | layer-major prefill の attention サブチャンク長（既定 256; これ以下の T は旧・融合経路） |
| `QWEN_SEGB_SLICE=N` | layer-major prefill の FFN スライス上限トークン数（既定 1024; MoE 活性メモリの上限） |
| `QWEN_PREFILL_STATS=1` | prefill のチャンク×層ごとの expert 和集合 / fetch 量 / スライス数を stderr に表示 |
| `QWEN_PREFILL_PRUNE=eps` | = `--prefill-prune`（prefill の質量ベース expert pruning; lossy） |
| `QWEN_RESIDENT_DECODE=1` | = `--resident-decode`（常駐限定ルーティング decode; lossy） |
| `QWEN_RESIDENT_MIN=N` | マスク発動に必要な層あたり常駐数（既定 32; 揃わない層はウォームアップ継続） |
| `QWEN_RESIDENT_WARMUP=N` | = `--resident-warmup`（この decode トークン数の後は常駐数を問わず固定; 既定 32） |
| `QWEN_RESIDENT_REFILL=N` | = `--resident-refill`（マスク中の補充 expert 数/トークン; 既定 RAM 8 / SSD 4, 0=凍結） |
| `QWEN_SSD_DIRECT=1` | = `--ssd-direct`（Windows: unbuffered read でページキャッシュをバイパス） |
| `QWEN_COALESCE=1` | SSD 読みで層の和集合を大きな連続レンジ読みに合体（シーケンシャルがランダム QD8 より速いドライブ向け） |
| `QWEN_COAL_DEBUG=1` | 合体読みの run ごとの read/upload 時間を表示 |
| `QWEN_PF_CHUNK=N` | = `--pf-chunk`（サーバー prefill のスライス長; 既定 4096） |
| `QWEN_PREFILL_CHUNK=N` | 常駐／RAM階層 build_graph prefill のチャンク長（既定 512） |
| `QWEN_CPU_PREFILL=1` | RAM 階層の prefill をエキスパート CPU 実行（sched）に戻す（既定は GPU キャッシュ経路。GPU が遊ぶ代わりに H2D 転送を省く旧挙動） |
| `QWEN_MTP_NO_BATCH_PREFILL=1` | MTP の prefill を旧来の token-by-token に戻す（既定はバッチ。画像入力時はバッチ強制） |
| `QWEN_FASTCACHE=1` | 楽観単一グラフデコード（全 expert 常駐前提 + ミス時フォールバック; `--resident-decode` のマスク無し版） |
| `QWEN_GDN_TEST=1` | GDN の multi-token / token-by-token 等価性チェックを実行して終了 |
| `QWEN_MTP_TEST=1` | MTP の draft 受理率計測モード（`--mtp` なしでも nextn を常駐させる） |
| `QWEN_MTP_NOACCEPT=1` | MTP の draft 受理を強制無効化（= 出力は plain と一致するはず。ロスレス確認用） |

---

## 7. ドキュメント

- [`docs/mtp.md`](docs/mtp.md) — MTP（自己推測デコード）の実装・評価・ベンチ
- [`docs/metal_unified_memory.md`](docs/metal_unified_memory.md) — 48GB Mac で 122B-A10B を動かすチューニング記録（Metal / ユニファイドメモリ / ゼロコピー SSD 読み）
- [`offload_optimize.md`](offload_optimize.md) — エキスパート・オフロードの高速化（pinned/async/グラフ融合）実測ログ
- [`prefill_layer_major.md`](prefill_layer_major.md) — layer-major prefill / expert pruning / 常駐限定 decode の設計・実測・落とし穴（2026-07）
- [`docs/gpu_performance_guide.md`](docs/gpu_performance_guide.md) — GPU パフォーマンス指針

---

## 8. メモ / 制限

- 既定はサンプリング無し（greedy, `temp=0`）で再現性重視。
- 大語彙（例: 248k）の K-quant 埋め込みは、バックエンドの get_rows がその型に非対応の場合のみ F16 変換版を VRAM に持つ（実質ロスレス）。CUDA は K-quant/IQ 非対応のため変換が走る。Metal / CPU はネイティブ対応なのでコピー自体を作らない（メモリ節約）。VRAM が厳しい場合は `--embd-q8` で Q8_0 化（F16 比 約半分、わずかな量子化誤差あり）。
- オフロードの RAM 階層は host pinned メモリを使う。単一の `cudaHostAlloc` には上限（環境依存, 例 ~15.5GB）があるため、エキスパートは複数チャンクに分けて pin する。
- CUDA グラフ / Flash Attention は CUDA ビルドで既定有効。
- `--cache-profile` は CLI では実行のたびに上書き保存される（サーバーは読み込みのみ）。アクセスパターンの異なる実行（MTP と plain 等）を混ぜるとプロファイルが汚れるので、ベスト状態のファイルをコピーして固定する運用を推奨。
- ユニファイドメモリ（Apple Silicon）では SSD 階層のミスをスロット実メモリへ直接 `pread` する（ゼロコピー）。
- チャットテンプレートは Qwen3.6 公式テンプレート等価のロジックを直接実装（jinja エンジンは不使用）。tools・`<think>` 分割・画像プレースホルダ・reasoning on/off に対応。動画入力は未対応。
- prefill はバッチ実行（通常デコード・MTP・SSD オフロードとも）。MTP の prefill はメイン一括 forward で全トークンの hidden を取得し、nextn KV もバッチ構築するため、画像 + MTP の併用も可。
