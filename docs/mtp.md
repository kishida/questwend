# MTP (Multi-Token Prediction / nextn) — 実装と評価まとめ

Qwen3.5/3.6 系の一部モデルに含まれる **nextn ブロック**（MTP module）を使った
**自己推測デコード（self-speculative greedy decode）** を qwencpp に実装した記録。

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

## 5. 使い方

```bash
# 自己推測デコード（MTP ブロックを持つモデルのみ）
infer -m model-MTP.gguf -p "..." -n 128 --mtp [--vram-budget N --experts-ssd]

# draft 受理率の計測
QWEN_MTP_TEST=1 infer -m model-MTP.gguf -p "..." -n 64

# 受理を強制的に無効化（= 純 verify 出力、ロスレス確認用）
QWEN_MTP_NOACCEPT=1 infer -m model-MTP.gguf -p "..." -n 64 --mtp
```

---

## 6. まとめ

- dense / MoE / offload を含め **MTP 自己推測デコードを完全実装**し、ロスレス・
  高受理率（79–93%）を確認。
- **損得はメモリ階層で決まる**:
  - **計算律速（モデルが全量 VRAM 常駐）** → MTP は速い。
    27B dense を全 VRAM で **+16%（10–12 → 12–14 tok/s）**。
  - **SSD フェッチ律速（experts オフロード）** → MTP は遅い。35B offload で
    plain の ~15%（nextn VRAM 常駐後）〜2倍（旧）遅い。verify の 2 位置 expert
    fetch が支配項で、これは自己推測の構造上消せない。
- 付随する最適化:
  - nextn ブロックを VRAM 常駐（`d40ec74`）— offload 時の draft オーバーヘッド削減。
  - 非 MTP 時は nextn もオフロード（`8a24f4f`）— VRAM 浪費の回避。
  - 埋め込み get_rows フォールバックを F16 化（`948d62e`）— 大語彙モデルの VRAM 半減。
- offload 環境で MTP 以外に効く方向: **常駐率向上・プリフェッチ強化
  （expert fetch そのものの削減）**。
