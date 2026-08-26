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
| `make_tiny_model.py` | 同じ構造の**極小 qwen4exp を乱数生成**（8層 / d=128 / 7 MiB、全 F32）。`--no-ple` で PLE 抜き版 |
| `check_tiny.py` | 極小モデルを qw-cli に通し、記録済みの llama.cpp 参照と比較する |
| `check_weights.py` | GGUF の重みが壊れていないか、ロードせずに数秒で確認する |
| `top_logits.py` | `--dump-logits` の出力を要約（非有限の有無・分布の広さ・上位トークン） |
| `compare_logits.py` | `index: value` 形式の logits ダンプ 2 本を比較（最大差・argmax・top-k 順序） |
| `01-tiny/` | 極小モデルの参照 logits。`.gguf` は seed 固定で再生成できるのでコミットしない |

`QWEN_QSA_DEBUG=1` を付けると QSA の中間テンソル（raw キー / プール / q / スコア /
top-k / 生成したマスク）が stderr に出る。llama.cpp 側の
`llama-debug --tensor-filter '.*indexer_.*-<層>$'` と突き合わせられる。
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

## 極小モデルによる検証（67 GiB を待たずに済ませる）

実物は 67.55 GiB あり、HDD 上では 1 回のグラフ確認に数分かかる。そこで
`make_tiny_model.py` が**同じ構造の乱数モデル**（8層 / d=128 / hc=4 / 512→8 experts /
層1 に PLE / 全 F32、7 MiB）を書き出す。**同じファイルを llama.cpp `pr-27742` と
qwencpp の両方が読める**ので、これがそのまま数値の参照になる。

```bash
python tests/qwen4/check_tiny.py
```

`.gguf` はコミットしない（`.gitignore` の `*.gguf`）。生成は seed 固定なので
`check_tiny.py` が毎回作り直しても記録済みの参照と一致する。

### 現在の状態

`python tests/qwen4/check_tiny.py` は全ケース PASS。

| ケース | 内容 | 結果 |
|---|---|---|
| `dense-5tok` | hyper-connection / GDN / dense attention / MoE | 一致（span の 0.008%） |
| `ple-5tok` | PLE の n-gram ハッシュ・行 gather・dilated conv | 一致（0.008%） |
| `dense-5tok-ngram-off` | `--ngram off` が PLE 無しモデルを再現するか | 一致 |
| `qsa1-20tok` | QSA（indexer キャッシュ・ブロックスコア・top-k・マスク生成） | 一致（0.011%） |
| `qsa-20tok` | QSA（compress ratio 2、ブロック平均あり） | argmax 一致（下記の理由で数値一致は不可能） |
| `offload` / `offload-ssd` | expert-offload 経路（segA/segB）が広い残差と PLE を運べるか | 一致（0.169%） |
| `ple-gen` / `nople-gen` / `ple-ssd-gen` | デコード間の状態引き継ぎ | 一致（4 プロンプト × 8 トークン貪欲生成） |

### QSA が ratio 2 で厳密一致しない理由（実装バグではない）

予算は `indexer_top_k + compress_ratio - 1` セル。これは「完全なブロック群 + 不完全な
末尾」をちょうど覆う設計だが、**トークンがブロック境界に乗ると末尾が空になり**、予算が
「完全なブロック n 個 + 端数 1 セル」になる。その端数がブロックのどちらのメンバーになるかは
**同一スコアの並び替え順**で決まり、参照実装側でも定義されていない（しかも llama.cpp の
n_kv パディング量に依存する）。

`--qsa-ratio=1` の極小モデルはブロックが 1 トークン幅なので端数が存在せず、そこでは
**厳密に一致する**。ratio 2 のケースはブロック平均そのものを見るために残してあり、
argmax のみで判定している。

実物では `top_k=2048` / `ratio=4` なので端数は最大 3 セル / 2051 セル（0.15%）。

### 32 トークンを超えると GDN が分岐する（本移植とは無関係）

QSA を両側で無効にした極小モデルで測ると、プロンプト長 30 → 32 で最大差が
0.0003 → 0.088 に跳ぶ。原因は delta-net の実装差で、

- qwencpp は常に融合オペ `ggml_gated_delta_net`（逐次スキャン）を使う
- llama.cpp は既定で `build_delta_net`（グラフ構築、`build_delta_net_chunking` は `CS = 64`）を使い、
  長さに応じて逐次形とチャンク形を切り替える

数学的には等価だが丸めが違う。Qwen3.5 / 3.6 でも同じはずで（そちらは量子化誤差に埋もれて
見えない）、qwen4exp 固有の問題ではない。**QSA の検証プロンプトを 32 トークン未満に
してあるのはこのため。**

## `--ngram`: n-gram embedding のスイッチ

```
--ngram <mode>       disk（既定） / ram / off
--ngram-cache <MB>   disk モードの行キャッシュ（既定 256）
```

**どのモードでも GPU には載らない。** qwencpp は llama.cpp と違い、PLE テーブルを
ggml テンソルにしていない（`Model::ple_table()` はメタデータだけを持ち、どの
backend buffer にも入らない）。行番号はトークン ID だけから決まりモデルの計算に
依存しないので、ホスト側で pread → 逆量子化 → `[n_embd, T]` の F32 入力として
グラフに渡している。1 トークンあたり 16 行 × 90 バイト。

| モード | 常駐 | 律速 |
|---|---|---|
| `disk` | 行キャッシュのみ（既定 256 MB） | ランダム read の **IOPS**（帯域ではない） |
| `ram` | テーブル全体 26.8 GiB（量子化のまま） | なし |
| `off` | 0 | なし。モジュールごとスキップ |

`off` が意味を持つ理由: PLE は残差へのゲート付き加算ブランチなので、外しても
スタックは動く。そして 64 GB マシンでは **`off` にすると expert 37.1 GiB が RAM に
載る**（26.8 + 37.1 = 63.9 GiB では載らない）。品質への影響は未測定。

実装は [core/ngram_table.h](../../core/ngram_table.h) / [.cpp](../../core/ngram_table.cpp)。
prefill では全行番号がグラフ実行前に判明するので、miss をファイルオフセット順に
ソートしてから読む（HDD ではランダムシークが片方向スイープになる）。

### 実モデルでの動作（2026-08-27）

**Apple M3 Ultra / 512GB / Metal で `UD-Q4_K_XL`（4 シャード）が完走。**
`--ngram ram` / `--ngram disk` / `--ngram off` の 3 モードとも同一の出力で、
プロンプトのタイポ（"Framce"）を正しく France と解釈している。

```
backend: GPU [MTL0] Apple M3 Ultra
token_embd: q8_0 natively supported by backend get_rows (no fallback copy)
ngram table: 26.8 / 26.8 GiB
ngram: ram (27465 MB resident)
The capital of Framce is
<think>
The user is asking about the capital of France (they made a typo writing
"Framce" instead of "France"). The capital of France
```

n-gram テーブルは Q4_K_XL でも 26.8 GiB のまま。`ple_head_dim = 160` は 256 の倍数でないので
256-block の K-quant が使えず、32-block の IQ4_NL に落ちるため。

**3 モードで貪欲デコードの出力が一致した**のは `--ngram off` の品質影響についての
最初の実データだが、1 プロンプトでは何も結論できない。まともな測定はまだ。

Windows 側では IQ1_S で層 0〜28 まで健全に動作することを確認済み
（層 29 以降はダウンロード破損のため未評価。`check_weights.py` 参照）。

### まだ検証していない経路

**QSA（キャッシュ 2051 トークン超）は Metal 上で未実行。** それ未満では QSA は完全に
不活性（dense と数値一致）なので、上の実行は QSA を一度も通っていない。
`ggml_fill` / `ggml_set_rows` を使うので Metal 側のカーネル有無が最初の関門。

## ダウンロードの検証（先にやること）

```bash
python tests/qwen4/check_weights.py <先頭シャード>.gguf
```

RMSNorm の gamma は学習済みモデルなら必ず 1 付近にある。それだけを読む
（層あたり数 KB）ので数秒で終わり、破損した領域を層番号で指してくれる。

実際にこれで IQ1_S の破損を捕まえた: 層 29〜32 だけ gamma の中央値が 0.0001〜0.013、
`hc_*_inject` などに inf が混入していた。**切り詰めではなく途中のバイトが化けている**
種類の破損なので、サイズを見ても分からない。全ロードは数分かかるうえ、出てくるのは
「logits が全部 NaN」という原因の分からない結果だけ。

## 512GB Mac で動かす（全常駐）

Q4_K_XL は全部 RAM に載るので、オフロードを使わない**常駐パス**（`build_graph`）で走る。
これが極小モデルで最も検証されている経路でもある。

```bash
git clone <repo> qwencpp && cd qwencpp && git checkout qwen4
cmake -B build -DCMAKE_BUILD_TYPE=Release -DQW_METAL=ON
cmake --build build -j

python tests/qwen4/check_weights.py ~/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-*.gguf

./build/qw-cli -m ~/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-*.gguf --info
./build/qw-cli -m ~/models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-*.gguf \
    -p "The capital of France is" -n 32 --n-ctx 4096 --ngram ram --temp 0
```

`--ngram ram` は n-gram テーブル（量子化のまま、Q4_K_XL なら 30 GB 前後）を丸ごと
ホストメモリに置く。512GB なら素直にこれが速い。ディスクに残したいときは
`--ngram disk --ngram-cache 4096`、品質への影響を測りたいときは `--ngram off`。

うまくいかないときの切り分け:

| 症状 | 次の一手 |
|---|---|
| logits が NaN / 出力が同じトークンの連打 | `QWEN_NAN_CHECK=1`（オフロード経路のみ）と `check_weights.py` |
| 2051 トークンを超えると出力が崩れる | QSA。`.*indexer_.*` を `QWEN_QSA_DEBUG=1` と突き合わせる |
| 起動が遅い | 常駐パスは全重みを読む。Q4_K_XL なら 100 GB 超 |

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
