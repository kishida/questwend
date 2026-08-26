# Qwen3.8-Flash-Next (`qwen4exp`) — 参照キャプチャと検証データ

qwencpp に `qwen4exp` アーキテクチャを実装するための、llama.cpp 参照実装からの
キャプチャ置き場。Step-3.7 のときの `tests/step-ref/` と同じ役割・同じ手順。

## 対象

| | |
|---|---|
| モデル | `D:\dev\gguf\unsloth\Qwen3.8-Flash-Next-GGUF\Qwen3.8-Flash-Next-UD-IQ1_S-0000{1,2,3}-of-00003.gguf` |
| 参照実装 | llama.cpp PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) — `D:\dev\llama.cpp` のブランチ `pr-27742` |
| HF `model_type` | `qwen4_exp` / GGUF `general.architecture` = `qwen4exp` |

## ディレクトリ

| | |
|---|---|
| `dump_gguf.py` | GGUF ヘッダのみを読むダンパ。ダウンロード途中の `.part` でも動く（`gguf_dump.py` は全体を mmap して reshape するので落ちる） |
| `00-arch/` | ハイパーパラメータ・テンソル一覧・トークナイザ（**シャード 1 だけで取れるので取得済み**） |
| `03-tokenizer/` | `llama-tokenize` の出力 |
| `04-tensors/` | `llama-debug --tensor-filter` の中間テンソルダンプ |
| `05-logits/` | `llama-debug --save-logits` の logits |

## アーキテクチャ要約（`00-arch/gguf-kv.txt` の実測値）

- 48 層 / d_model 2560 / vocab 248320 / ctx 262144、MTP（nextn）なし
- MoE: 512 experts、top-10、`n_ff_exp` 640、sigmoid ゲート付き shared expert 640
- 3/4 の層が Gated DeltaNet、1/4 が full attention（`full_attention_interval = 4`）
  - GDN は Qwen3.5 と同一で、**出力ゲートだけ silu → sigmoid**（PR 側コメントで明言）
- **hyper-connection** `count = 4` / `low_rank = 320` → 残差ストリームが 4×2560 = 10240 幅
- **QSA sparse attention**: indexer 4 head × 128、`top_k = 2048`、`compress_ratios[il] = 4`（full-attn 層のみ）
  - キャッシュ済みトークンが `top_k + ratio - 1 = 2051` 未満なら **dense attention と数値的に完全一致**
- mRoPE `dimension_sections = [11, 11, 10, 0]`、`n_rot = 64` / `head_dim = 256`（partial）
- **PLE n-gram embedding**: 層 index **1** のみ。3-gram × 8 head/gram = 16 head × 160 次元

### トークン ID の落とし穴

`ple.eos_token_id = 248044` は `<|endoftext|>` で、**チャットの EOS（`248046` = `<|im_end|>`）とは別**。
PLE の n-gram 文脈リセットはこちらで起きる。`ple.image_token_id = 248056` = `<|image_pad|>`。
詳細は `00-arch/special-tokens.txt`。

## サイズ内訳（`00-arch/gguf-kv.txt` 末尾、1224 テンソル / 67.55 GiB）

| 区分 | サイズ | 型 |
|---|---|---|
| `per_layer_token_embd`（PLE テーブル） | **26.82 GiB** | IQ4_NL、320,001,536 行 × 160 |
| routed experts (`ffn_*_exps`) | **37.11 GiB** | IQ4_NL 21.09 / IQ1_S 10.38 / IQ2_XXS 5.64 |
| その他（attn / GDN / HC / shexp / embd / output） | **3.62 GiB** | Q5_K / Q8_0 / Q4_K / Q6_K / F32 / BF16 |

expert の 21.09 GiB が IQ4_NL なのは `n_ff_exp = 640` が 256 の倍数でなく、
`ffn_down_exps`（`ne[0] = 640`）に 256-block 量子化を使えないため。
expert 1 個 3 role 合計 1.55 MiB、top-10 × 48 層 = **742 MiB/token**（ヒット率 0% のとき）。

## 参照キャプチャ手順

`llama-debug`（`examples/debug`）を使う。`pr-27742` をチェックアウトしてビルドすること。

```bash
cmake --build D:/dev/llama.cpp/build --config Release --target llama-debug llama-completion llama-tokenize
```

既知の落とし穴（Step-3.7 のときに踏んだもの、そのまま当てはまる）:

- `--save-logits` を付けるとテンソルダンプ callback が**無効になる**（排他）。別々に実行する。
- `context_length = 262144` なので **`-c` 必須**。
- `common/debug.cpp` はフィルタを `"^" + pattern` と先頭アンカーする。
  `-0$` のような末尾指定は永久に一致しないので `.*(-(0|1|3)$|result_)` の形で書く。

### 追加の注意（このモデル固有）

- **層 1 が PLE 層**なのでテンソルフィルタには必ず `-1$` を含める（`ple_embd` / `ple_gate` /
  `ple_gated_value` / `ple_conv_out`）。
- QSA を dense と一致させたいプロンプトは **2051 トークン未満**にする。
  QSA そのものの検証には 2051 超が必要。
- PLE のランダム read は 16 行/token しかないので、数トークンの参照取得なら
  HDD でも一瞬で終わる（26.8 GiB のテーブル全体は読まれない）。
