# qwencpp

Qwen3 / Qwen3.5 / Qwen3.6 をスクラッチ実装した推論エンジン（vendored ggml の上に構築）。

- **対応アーキ**: `qwen3`（dense）, `qwen3moe`（MoE）, `qwen35` / `qwen35moe`（Gated DeltaNet ハイブリッド）, `qwen3next`
- **バックエンド**: CUDA（Windows/Linux）, Metal（macOS）, CPU フォールバック
- **主な機能**:
  - MoE エキスパートのオフロード（**RAM 階層** / **SSD 階層**）＋動的 VRAM エキスパートキャッシュ
    → 16GB GPU で 35B-A3B MoE などを実行可能
  - エキスパート常駐プロファイルの永続化（`--cache-profile`）でウォーム起動を高速化
  - MTP（Multi-Token Prediction / nextn）自己推測デコード（`--mtp`, `--draft N`）
  - 分割（sharded）GGUF 対応（`-NNNNN-of-MMMMM.gguf`）
  - 大語彙モデルの K-quant 埋め込みは、バックエンドの get_rows が非対応の場合のみ F16 変換版を VRAM に保持（CUDA のみ; Metal/CPU はネイティブ対応のためコピー不要。`--embd-q8` で Q8_0 化も可）
  - OpenAI 互換サーバー（`infer-server`）＋ブラウザ・チャット UI

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
cmake -B build -DQWENCPP_CUDA=ON

# ビルド（Release）
cmake --build build --config Release --target infer infer-server
```

- 生成物: `build\Release\infer.exe`, `build\Release\infer-server.exe`（ggml の DLL は `build\bin\Release\` に置かれ、exe と同じ場所から実行すれば解決されます）。
- CUDA アーキは native 検出（例: RTX 4060 Ti = sm_89, RTX 3050 = sm_86）。
- CMake は `CUDA graphs ON`, `Flash Attention ON` で構成されます。
- VS 同梱の cmake を使う場合の例:
  `"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"`

### macOS（Metal）

```bash
cmake -B build -DQWENCPP_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target infer infer-server
```

- 生成物: `build/infer`, `build/infer-server`。
- 起動時に `backend: GPU [MTL0] ...` と出れば Metal が有効。

### CPU のみ（任意プラットフォーム）

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target infer
# 実行時に --cpu を付けるか、GPU バックエンド無効ビルドで自動的に CPU で動作
```

---

## 3. 使い方（CLI: `infer`）

```bash
# 一回限りの生成
infer -m model.gguf -p "The capital of France is" -n 128

# チャット（ChatML でラップ）
infer -m model.gguf --chat -p "Explain photosynthesis." -n 256
infer -m model.gguf -i                      # 対話モード
infer -m model.gguf --info                  # モデル情報を表示して終了
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
| `--log-tokens-per-sec` | prefill / 生成の tok/s を表示 |
| `--cpu` | CPU バックエンドを強制 |

### エキスパート・オフロード（大きい MoE を限られた VRAM で）

```bash
# RAM 階層: 非エキスパート重みを GPU、エキスパートは pinned host RAM に置き
#           動的 VRAM キャッシュ経由でストリーミング（PCIe）
infer -m moe.gguf -p "..." -n 128 --vram-budget 15000

# SSD 階層: エキスパートを GGUF からディスク直読（RAM コピーなし）
infer -m moe.gguf -p "..." -n 128 --vram-budget 12000 --experts-ssd

# 常駐プロファイル: 1回目に保存、2回目以降は起動時にホットなエキスパートを事前常駐
infer -m moe.gguf -p "..." -n 128 --vram-budget 15000 --cache-profile hot.prof
```

- `--vram-budget <MB>`: スロットプールに使う VRAM 予算（>0 でオフロード有効）。
- `--experts-ssd`: エキスパートをディスクからストリーミング（RAM に載らない巨大モデル向け）。
- `--cache-profile <file>`: ホットエキスパートの頻度プロファイルを永続化。**同一/類似ワークロードで hit 率が ~100% になり、ストリーミングがほぼ消えて大幅高速化**。

### MTP 自己推測デコード（nextn ブロックを持つモデル）

```bash
infer -m model-MTP.gguf -p "..." -n 128 --mtp            # 1 トークン先読み
infer -m model-MTP.gguf -p "..." -n 128 --mtp --draft 2  # 2 トークン先読み（推奨）
```

- 出力は通常デコードと完全一致（greedy ロスレス）。
- 計算律速（モデル全量 VRAM 常駐）では `--draft 2`〜`3` で plain 比 **+30%程度**
  （27B Q2, ctx=2000: 18.2 → 24.2 tok/s）。オフロード（fetch 律速）では不利。
  詳細は [`docs/mtp.md`](docs/mtp.md)。

---

## 4. 使い方（サーバー: `infer-server`）

OpenAI 互換の `/v1/chat/completions`（SSE ストリーミング対応）と、ブラウザ・チャット UI を提供。

```bash
infer-server -m model.gguf --host 0.0.0.0 --port 8080 --vram-budget 15000
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
| `--cpu` | CPU バックエンドを強制 |

> MTP はモデルロード時に nextn ブロックを VRAM 常駐させるため、リクエスト単位ではなく**サーバー起動フラグ**で指定します。ストリーミング/非ストリーミングの両方に対応。

---

## 5. 分割（sharded）GGUF

`model-00001-of-00003.gguf` のように分割されたモデルは、**先頭シャード**を `-m` に渡すだけで自動的に全シャードを探索・統合します（命名は 5 桁ゼロ詰め、00001 開始）。

```bash
infer -m Qwen3.5-122B-A10B-00001-of-00005.gguf -p "..." --vram-budget 40000 --experts-ssd
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

### 実験的 / テスト

| 変数 | 効果 |
|---|---|
| `QWEN_BATCH=1` | SSD 階層の prefill をバッチ実行（既定 token-by-token; MTP verify は常にバッチ） |
| `QWEN_BATCH_CHUNK=N` | 上記バッチのチャンク長を固定 |
| `QWEN_FASTCACHE=1` | 楽観単一グラフデコード（全 expert 常駐前提 + ミス時フォールバック; 実験的） |
| `QWEN_GDN_TEST=1` | GDN の multi-token / token-by-token 等価性チェックを実行して終了 |
| `QWEN_MTP_TEST=1` | MTP の draft 受理率計測モード（`--mtp` なしでも nextn を常駐させる） |
| `QWEN_MTP_NOACCEPT=1` | MTP の draft 受理を強制無効化（= 出力は plain と一致するはず。ロスレス確認用） |

---

## 7. ドキュメント

- [`docs/mtp.md`](docs/mtp.md) — MTP（自己推測デコード）の実装・評価・ベンチ
- [`docs/metal_unified_memory.md`](docs/metal_unified_memory.md) — 48GB Mac で 122B-A10B を動かすチューニング記録（Metal / ユニファイドメモリ / ゼロコピー SSD 読み）
- [`offload_optimize.md`](offload_optimize.md) — エキスパート・オフロードの高速化（pinned/async/グラフ融合）実測ログ
- [`docs/gpu_performance_guide.md`](docs/gpu_performance_guide.md) — GPU パフォーマンス指針

---

## 8. メモ / 制限

- 既定はサンプリング無し（greedy, `temp=0`）で再現性重視。
- 大語彙（例: 248k）の K-quant 埋め込みは、バックエンドの get_rows がその型に非対応の場合のみ F16 変換版を VRAM に持つ（実質ロスレス）。CUDA は K-quant/IQ 非対応のため変換が走る。Metal / CPU はネイティブ対応なのでコピー自体を作らない（メモリ節約）。VRAM が厳しい場合は `--embd-q8` で Q8_0 化（F16 比 約半分、わずかな量子化誤差あり）。
- オフロードの RAM 階層は host pinned メモリを使う。単一の `cudaHostAlloc` には上限（環境依存, 例 ~15.5GB）があるため、エキスパートは複数チャンクに分けて pin する。
- CUDA グラフ / Flash Attention は CUDA ビルドで既定有効。
- `--cache-profile` は CLI では実行のたびに上書き保存される（サーバーは読み込みのみ）。アクセスパターンの異なる実行（MTP と plain 等）を混ぜるとプロファイルが汚れるので、ベスト状態のファイルをコピーして固定する運用を推奨。
- ユニファイドメモリ（Apple Silicon）では SSD 階層のミスをスロット実メモリへ直接 `pread` する（ゼロコピー）。チャットテンプレートは ChatML 固定の最小実装（jinja 未対応; tool / 画像入力は未対応）。
