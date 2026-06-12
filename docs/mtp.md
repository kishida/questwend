# MTP (Multi-Token Prediction / nextn) — 実装と評価まとめ

Qwen3.5/3.6 系の一部モデルに含まれる **nextn ブロック**（MTP module）を使った
**自己推測デコード（self-speculative greedy decode）** を QuestWend に実装した記録。

- 関連コミット: `fc0378d`（MTP本体）, `fe7de96`（推測ループ `--mtp`）,
  `eb11017`（batched cache の router バグ修正）, `805d244`（offload/MoE-MTP対応）
- 関連コード: `core/runtime.cpp`（`build_mtp` / `mtp_draft` / `mtp_draft_cached` /
  `decode_verify` / `decode_verify_cached` / `generate_mtp`）

---

## 1. MTP とは

通常の transformer は「直前までの hidden」から **次の1トークン** を予測する。
MTP 付きモデルは末尾に **nextn ブロック**（block index = `n_layer - nextn_predict_layers`）
を持ち、メインスタックの最終 hidden `h` と「いま確定したトークン `t` の embedding」を
組み合わせて **`t` の *次の次* のトークン** を1手先読み（draft）できる。

```
h'      = eh_proj( concat( enorm(emb(t)), hnorm(h) ) )   // 埋め込みが先（重要）
block   = gated attention + FFN（このブロック専用の KV を持つ）
logits  = shared_head_norm -> lm_head（メインと共有）
```

これを使った **自己推測デコード**:

1. メインモデルで `x` を確定し、その hidden から MTP で次トークン `d` を draft。
2. メインモデルで `[x, d]` を **1回の2トークン forward** で verify。
   - `y = argmax(verify_logits[0])`（`x` の本当の次トークン）
   - `y == d` なら **draft 的中** → `x` と `d` の2トークンを一度に確定。
   - 外れたら `x` だけ確定し `d` は破棄（recurrent 状態はロールバック）。
3. 的中するほど「1回の verify forward で2トークン進む」ので main forward 回数が減る。

ロスレス保証: 受理は `argmax` 一致時のみなので、出力は **verify（=メイン）モデルの
greedy 出力と必ず一致**する。MTP は速度最適化であって品質は変えない。

---

## 2. 実装

### 2.1 非オフロード（VRAM常駐）パス
- `build_mtp`: nextn ブロックを1グラフで構築（dense FFN 前提）。
- `mtp_draft`: 上を `mtp_galloc` で1回 compute。
- `decode_verify`: `build_graph(n_tokens=2)` を `galloc` で compute、
  各位置の logits（`vL1/vL2`）と hidden（`vh1/vh2`）を取り出す。
- 小さい dense-MTP モデル（例: Qwen3.5-0.8B-MTP）で動作。

### 2.2 オフロード（expert offload / SSD 階層）パス — `805d244`
35B-A3B-MTP の **nextn ブロックは MoE**（256 experts + shared expert）で、
その experts も SSD 階層にある。メイン同様に **VRAM expert cache 経由のセグメント
実行** が必要になる。

| 関数 | 役割 |
|---|---|
| `decode_cached` | 末尾で main hidden を `mtp_hidden` にキャプチャ（draft 入力用） |
| `mtp_draft_cached` | nextn ブロックを `decode_cached` と同型にセグメント実行: seg0 `eh_proj` → segA gated attention + router → `ensure(L)` → segB cached-MoE + shared expert + shared head → logits |
| `decode_verify_cached` | 2トークン verify を `decode_cached_batch` の **verify モード**で実行。両位置の logits/hidden を取得。プールが `2*n_used` 未満なら単一トークン decode 2回にフォールバック |
| `mtp_draft` / `decode_verify` | `ecache` があれば上記 `*_cached` に分岐 |

前提となったバグ修正（`eb11017`）:
`ggml_argsort_top_k` は **ストライド付きビュー**を返すが、`ggml_backend_tensor_get`
はストライドを無視して連続コピーするため、マルチトークン（T>1）の router 選択が
列1以降で壊れていた。`ggml_cont` で連続化して解決。これにより `decode_cached_batch`
が正しくなり、2トークン verify が成立した。

---

## 3. 検証結果

### 3.1 正しさ
- **Qwen3.5-0.8B-MTP（dense, 非offload）**: コヒーレント出力、`--mtp` 動作。
- **Qwen3.6-35B-A3B-MTP（GDN MoE, SSD offload）**:
  - draft 受理率 **81–92%**、**~1.8 tok/forward**。
  - 出力は plain decode と **完全一致（ロスレス）**。
  - GDN の reject ロールバック（backup/restore states）も正常。

### 3.2 速度（Qwen3.6-35B-A3B-MTP, RTX 4060 Ti, `--vram-budget 15000 --experts-ssd`, 約50%常駐 / hit 82%）

| 方式 | 速度 | expert fetch 総アクセス |
|---|---|---|
| **MTP**（nextn を SSD offload, 旧実装） | 4.9–6.8 tok/s（受理81%, 1.81 tok/forward） | 60,744 |
| **MTP**（nextn を VRAM 常駐, `d40ec74`） | **7.0–7.7 tok/s** | 59,520 |
| **plain decode** | **8.8–9.0 tok/s** | 50,880 |

→ nextn ブロックを VRAM 常駐にすると MTP は大きく改善し、plain との差は
**約2倍 → 約15%** に縮まる。ただし依然 plain の方が速い。
HDD→SSD へ置き換えても結論は変わらない（ボトルネックは生のディスク速度ではない）。

### 3.2b 計算律速での比較（Qwen3.6-27B-MTP, dense+GDN, 全量 VRAM 常駐）

27B（`qwen35`, dense+GDN, 語彙248k）を Q3_K_S で **完全に GPU に載せて** 計測
（experts オフロード無し ＝ 計算律速）。`--mtp` の本領が出る領域。

| 方式 | 速度 |
|---|---|
| **MTP** (`--mtp`) | **12.1–13.7 tok/s**（受理79%, 1.79 tok/forward） |
| **plain decode** | 10.3–11.8 tok/s |

→ **計算律速では MTP が約 +16% 速い**。main forward 回数の削減がそのまま効く。
SSD 律速の 35B（§3.2）とは逆の結果で、**MTP の損得はメモリ階層で決まる**ことを示す。

> 補足: 248k 語彙 × n_embd 5120 の K-quant 埋め込みは get_rows 用の dequant コピーが
> 必要で、F32 だと 5GB（→16GB に載らず OOM）。**F16 コピー（2.5GB）に変更**して
> 27B Q3_K_S が 16GB に収まるようにした（`948d62e`）。

### 3.3 最適化: nextn ブロックを VRAM 常駐に（`d40ec74`）

nextn ブロックは「draft する1トークンごとに1回」走るだけなので、その 256 experts
（約464MB）を毎 draft で SSD から stream するのは純粋なオーバーヘッドだった。
`Model::is_offloaded_expert` で **block index >= n_main() の experts は offload
しない**（VRAM 常駐）。ExpertCache は main スタック（n_main 層）だけを管理し、
`mtp_draft` は常に単一グラフ `build_mtp`（per-draft のセグメント化・host 同期なし、
MoE FFN は `build_moe`）を使う。これで draft の SSD fetch とグラフ分割オーバーヘッドが
消え、4.9→7.7 tok/s に改善した。

`--mtp` を使わないときは nextn experts を VRAM 常駐にしても無駄なので、
`RuntimeConfig::use_mtp`（CLI `--mtp` で立つ）が false のときは nextn experts も
メインスタック同様にオフロードする（`8a24f4f`）。これで非 MTP 実行時の VRAM 浪費を回避。
（35B SSD, budget 15000: 非 --mtp でメインキャッシュ常駐 54%→56%。）

### 3.4 メモリ階層別まとめ（MTP の損得は階層で決まる）

experts の置き場所を変えて、MTP vs plain を横並びにしたもの。

| モデル / 階層 | MTP | plain | MTP/plain |
|---|---|---|---|
| 35B-A3B-MTP, **SSD**（`--experts-ssd`, 54%常駐） | 7.0–7.7 tok/s | 8.8–9.0 | **約15%遅い** |
| 35B-A3B-MTP, **RAM**（`--vram-budget` のみ, 54%常駐） | 9.5–10.1 tok/s | 10.9–11.5 | **約12%遅い** |
| 27B-MTP (dense), **全量 VRAM 常駐** | 12.1–13.7 tok/s | 10.3–11.8 | **+16%速い** |

観察:
- **RAM 退避は SSD より全体が速い**（plain 9→11.5、MTP 7→10）。RAM 帯域が SSD より広いため。
- しかし **MTP/plain の比はほぼ変わらない**（SSD 15% → RAM 12% 遅い）。MTP も plain も
  同じく fetch 速度の恩恵を受けるので、MTP の不利（verify が2位置分の expert を fetch）は
  相殺されない。**offload である限り（RAM/SSD 問わず）fetch 律速で MTP は負ける**。
- MTP が勝つのは **全量 VRAM 常駐＝計算律速**のときだけ（27B）。

> 注意: 35B を RAM 退避すると 19GB の experts を pinned（page-locked）メモリに確保できず
> （ホストのロック可能メモリ上限）、pageable メモリ経由の H2D になる。pin できれば RAM 退避は
> さらに速くなり得るが、MTP/plain の比は同様と見込まれる。

---

## 4. なぜ offload では遅いのか（重要な教訓）

自己推測 MTP は **「main forward の *回数*」を減らす** が、
**「expert fetch の *総量*」は減らさない**:

- 48トークン生成あたり
  - plain: 単一トークン forward **48回**
  - MTP: 2トークン verify forward **26回**（≈52トークン分の fetch）
        ＋ `mtp_draft` **約47回**（各々 MoE 1層の fetch + lm_head）
- `1.81 tok/forward` は **verify ステップだけ**を数えた指標で、draft の
  オーバーヘッドを隠している。

決定的だったのは **語彙 248,320 の巨大な lm_head**（約0.5GB）。
これが draft と verify の各列で毎回計算されるため、「draft は main より十分安い」
という自己推測の大前提が崩れる。さらに nextn ブロック自体が 256-expert MoE で
自前の SSD fetch を発生させる。

### MTP が勝てる条件
1. **計算律速**であること（experts が高常駐 = fetch が実質ゼロ）。
   → このとき main forward 回数の削減がそのまま効く。
2. **draft が main より十分安い**こと（層数が多く / 語彙が小さい / lm_head が軽い）。

19GB の experts は 16GB GPU に乗り切らず、35B では条件1に到達できない。
nextn ブロックを VRAM 常駐にする最適化（§3.3）で draft 側のコストはほぼ排除できたが、
**verify（2トークン）のメインスタック expert fetch** が残りのボトルネックで、これは
自己推測の本質上消せない。よって `--mtp` は **opt-in** のまま維持する。実装はロスレスで、
より大きな VRAM（experts が大半常駐）や語彙の小さいモデルでは速度メリットが出る。

---

## 4.5 マルチトークン draft（`--draft N`, `e5a5fbd`）

nextn ブロックは1個でも **ブロック出力 hidden を次 draft の入力にチェーン**することで
N トークン先読みできる。`[x, d_1..d_N]` を1回の (N+1) トークン verify で確認し、
**最長一致 prefix を受理**する（N=1 は従来の2トークン verify と完全一致）。

### 受理率（深さで減衰、27B 実測）
| draft 深さ | 受理率 | tok/forward |
|---|---|---|
| 1 | 80–88% | 1.79–1.87 |
| 2 | 70–77% | 2.40–2.59 |
| 3 | 65–66% | 2.89 |
| 4 | 56%   | – |

深いほど後段 draft は自分の(近似)出力 hidden に依存するため的中率が落ちる。

### クリーン実測（GPU 専有, n=128）
| モデル / 階層 | plain | draft=1 | draft=2 |
|---|---|---|---|
| 27B（dense+GDN, 全 VRAM＝計算律速） | 17.1 | 16.8 | **17.7** |
| 35B（MoE+GDN, RAM offload＝fetch律速） | **19.7** | 16.9 | 12.9 |

- **draft=2 > draft=1**（27B 17.7 vs 16.8）は unsloth の「draft=2 最良」と一致。
- ただし**最適化済みの plain が速い**ため正味ゲインは小さい:
  27B 計算律速で draft=2 が plain を **+3.5%** 上回る程度、35B offload では MTP が負ける
  （fetch 律速 + draft が深いほど追加 fetch で悪化）。
- MTP が伸びない要因（このモデル特有）: **語彙 248k の巨大 lm_head を draft ごとに計算**、
  **GDN 再帰状態の partial-accept 再デコード**（KV と違い部分ロールバック不可）。

> ~~既知の perf バグ: `--draft >= 3` は激遅化する~~ → **§4.6 の GDN チェックポイントで解消**。

---

## 4.6 高速化ラウンド2: llama.cpp 比較で見つけたオーバーヘッド除去

llama.cpp は同環境の 27B Q2 (ctx=2000) で plain 19 → MTP 28 tok/s（+47%）を出すが、
当時の本実装は 19 → 18 と**逆に遅かった**。`QWEN_PROF_MTP=1` でフェーズ分解した結果
（1サイクル = draft K 回 + verify 1回 = 2.26 tok）:

| フェーズ | 旧 (ms/cycle) | 新 (ms/cycle) | 対策 |
|---|---|---|---|
| draft ×2 | 12.7 | 11.0 | 永続グラフ + **専用 backend インスタンス**（CUDAグラフ枠を verify と分離）+ GPU argmax（248k logits 読み戻し→4B） |
| verify (3tok) | 71.0 | 65.3 | 永続グラフ化（毎回の graph 構築+alloc を排除）+ GPU argmax |
| settle (GDN巻き戻し) | 28.4 | **1.5** | **GDN 状態チェックポイント**（下記） |
| resync (MTP KV 書き直し) | 7.2 | **1.4** | **headless 化**（共有 lm_head ~1GB の matmul をスキップ。llama.cpp の catch-up decode が logits=0 で head を省くのと同じ） |

**GDN 状態チェックポイント**: verify グラフ内で `ggml_gated_delta_net` をトークン毎に
分割実行（逐次演算なので FLOPs 同一）し、各トークン後の conv/ssm 状態を checkpoint
バッファに `cpy`。partial accept 時は checkpoint[a] を `tensor_copy` で復元するだけ
（旧: 状態 restore + 受理トークンのフル再デコード = 1回 ~61ms）。
これが draft>=3 激遅バグの根治にもなった（reject 頻度が上がっても再デコードが無いため）。

**付随修正**: 埋め込み get_rows フォールバックを Q8_0 化すると 2.5GB → 1.35GB になり、
27B Q2 + MTP 一式が ctx=2000 でも VRAM 16GB に収まる
（以前は WDDM ページングで全体が劣化していた。VRAM 残量が成績を支配するので注意）。
※ Q4_K→Q8_0 はロスレスではない（plain 出力の品質劣化が観測された）ため、その後
**既定は F16 に戻し、Q8_0 は `--embd-q8` の opt-in** とした。VRAM が苦しいときだけ使う。

### 実測（27B-UD-Q2_K_XL, ctx=2000, RTX 4060 Ti, n=200, GPU 専有）

| 方式 | tok/s (prefill込) | 生成のみ換算 |
|---|---|---|
| plain | 18.2 | ~18.5 |
| `--mtp --draft 1` | 22.1 | – |
| `--mtp --draft 2` | **24.2** | **~28.5** |
| `--mtp --draft 3` | 24.6 | – |
| `--mtp --draft 4` | 24.0 | – |

- plain 比 **+33%**（生成のみでは llama.cpp の 28 tok/s と同水準）。draft 2〜3 が最良。
- 27B 非オフロードはロスレス確認済み（draft=1/2 で plain と出力完全一致）。
- 35B offload は従来どおり MTP 不利（fetch 律速）。なお offload では plain（単発
  経路）と MTP verify（バッチ経路）の浮動小数点順序差により、まれに near-tie の
  argmax が反転して出力が分岐し得る（旧実装から存在する既知の性質、品質劣化ではない）。
- 残る上限要因: draft 1回 ~5.4ms のうち ~3.7ms は **語彙 248k の共有 lm_head
  mat-vec（~1GB 読み）**で、draft の argmax に必須なため不可避（llama.cpp も同条件）。

---

## 4.7 高速化ラウンド3: オフロード verify の改善（48GB Mac / 122B が対象）

ターゲット: Qwen3.5-122B-A10B（qwen35moe, 49層, 語彙248k）を M4 Pro 48GB +
`--experts-ssd --vram-budget 36000` で実行。plain はこの時点で 15 tok/s
（§offload 最適化と埋め込みフォールバック廃止の効果。ヒット ~100% で GPU 計算律速 96%）、
MTP は 10 tok/s と逆転していた。`QWEN_PROF_MTP` の per-cycle 分解で対策:

| 対策 | 効果 (122B 実測, ms/cycle) |
|---|---|
| `decode_cached_batch` を segB(L)+segA(L+1) 融合（同期 2回/層→1回/層, `78a056c`） | verify 414 → 157 |
| GDN チェックポイントをバッチ verify に配線（`23c8b3d`） | settle 31.8 → 1.9（再デコード 0） ※ verify は per-token scan のカーネル増で +23ms と相殺気味 |
| ユニファイドメモリでゼロコピー SSD 読み（`84b5076`, 下記） | フェッチ実効 1.1 → 1.5 GB/s, 起動 prefetch ~5s 短縮 |

**ゼロコピー SSD 読み**: Metal の shared バッファはスロットプールの実メモリが
ホスト可視（`tensor->data` が実ポインタ）なので、SSD ミスを `pread()` で
スロットへ直接読む。staging vector・memcpy・stdio バッファが全部消える。
ホスト可視性は `mincore()` でプール毎に判定（private バッファは PAGEZERO 内の
未マップ仮想アドレスなので誤検出なし）。`QWEN_NO_DIRECT_FETCH=1` で旧経路。

### 結論（122B, M4 Pro 48GB, Q3_K_M, ウォームプロファイル）

| 方式 | tok/s | ヒット率 |
|---|---|---|
| plain | **15.1** | 100% |
| `--mtp --draft 2` | 11.7 | 97.6% |

**この構成では MTP は plain に勝てない**。理由は二重の構造的ハンデ:
1. nextn ブロック常駐（~1GB）の分スロットが減る（57% vs 60% 常駐）
2. verify は 3 トークン分のエキスパート union（最大24個/層）にアクセスするため
   working set が広がり、ミスが plain の 0 に対し ~90/cycle 発生する

メモリに余裕がある（常駐 100% にできる）モデルサイズなら §4.6 のとおり MTP が
+30% 出る。**「キャッシュがギリギリの SSD 階層では plain、全量／ほぼ全量常駐なら MTP」**
が運用指針。

**注意: `--cache-profile` は実行のたびに上書き保存される**（CLI の場合）。MTP 実行は
アクセスパターンが異なるため、plain 用のプロファイルが汚れてヒット率が落ちる。
ベスト状態のプロファイルをコピーして固定する運用を推奨。

---

## 4.8 バッチプレフィル（prefill の刷新）

従来の MTP prefill はトークン逐次だった: `decode({tok_i})`（メイン 1 トークン）+
`mtp_resync(tok_{i+1})`（nextn 1 トークン）× P 回。生成は速くても長いプロンプトで
立ち上がりが遅い、画像埋め込みを注入できない、という 2 つの問題があった。

現在は 2 段のバッチで処理する:

1. **メイン一括 forward** — `decode(prompt)` をそのまま使い、全トークンの最終 hidden
   （pre-output-norm）を一括キャプチャ（`bh_all`）。SSD オフロード時はチャンク分割の
   `decode_cached_batch` 経由（チャンク毎に hidden を追記）。画像 embedding override も
   この経路がそのまま処理する。
2. **nextn KV のバッチ構築** — `build_mtp` を T トークン化し、`(tok_{i+1}, h_i)` の組を
   512 トークンずつ 1 グラフで流して MTP KV を書く（headless、ロジット読み戻しなし）。
   画像スパンは nextn 側の `emb(tok)` にも splice する（draft 品質のため。出力の正しさは
   verify が保証）。

実測（27B-UD-Q2_K_XL, 322 トークンプロンプト, RTX 4060 Ti, ウォーム）:
prefill 込みの総時間 **17.6s → 1.7s（約 10 倍）**。出力・受理率は不変
（数値経路はむしろ plain のバッチ prefill と一致するようになった）。

SSD 階層の plain prefill も同じバッチチャンク経路がデフォルトになった
（35B + 画像 600 トークンで 7.7 → 11.3 tok/s。ヒット率が低い環境では SSD 帯域律速）。
旧挙動は `QWEN_MTP_NO_BATCH_PREFILL=1` / `QWEN_NO_BATCH_PREFILL=1` で戻せる。
副産物として **画像 + MTP の併用**（CLI / サーバーとも）が解禁された。

---

## 5. 使い方

```bash
# 自己推測デコード（MTP ブロックを持つモデルのみ）
qw-cli -m model-MTP.gguf -p "..." -n 128 --mtp [--vram-budget N --experts-ssd]

# マルチトークン draft（1 verify あたり N トークン先読み, 既定 1; 実用は 1〜2）
qw-cli -m model-MTP.gguf -p "..." -n 128 --mtp --draft 2

# draft 受理率の計測
QWEN_MTP_TEST=1 qw-cli -m model-MTP.gguf -p "..." -n 64

# 受理を強制的に無効化（= 純 verify 出力、ロスレス確認用）
QWEN_MTP_NOACCEPT=1 qw-cli -m model-MTP.gguf -p "..." -n 64 --mtp
```

---

## 6. まとめ

- dense / MoE / offload を含め **MTP 自己推測デコードを完全実装**し、ロスレス・
  高受理率（79–93%）を確認。
- **損得はメモリ階層で決まる**（クリーン環境 = GPU 専有で再計測）:
  - **計算律速（モデルが全量 VRAM 常駐, 27B）** → §4.6 の最適化後、draft=2 で
    **plain +33%**（18.2 → 24.2 tok/s, ctx=2000）。llama.cpp の MTP 同水準。
    ※それ以前は「概ね中立」だった（verify 毎回グラフ構築・GDN 再デコード・
    lm_head 読み戻しのオーバーヘッドが speculative ゲインを食っていた）。
  - **フェッチ律速（experts オフロード, 35B）** → MTP は遅い（plain 19.7 vs draft=1 16.9
    vs draft=2 12.9）。verify の追加 expert fetch が支配項で、自己推測の構造上消せない。
  - 残る固定費: 語彙 248k の巨大 lm_head を draft ごとに計算（draft の argmax に必須）。
- 付随する最適化:
  - nextn ブロックを VRAM 常駐（`d40ec74`）— offload 時の draft オーバーヘッド削減。
  - 非 MTP 時は nextn もオフロード（`8a24f4f`）— VRAM 浪費の回避。
  - 埋め込み get_rows フォールバックを F16 化（`948d62e`）→ さらに Q8_0 化（§4.6）。
- offload 環境で MTP 以外に効く方向: **常駐率向上・プリフェッチ強化
  （expert fetch そのものの削減）**。
