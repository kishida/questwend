# offload 高速化: Luce Spark 取り込み計画 (1→2→3)

## 背景 / 狙い
参考: Luce Spark (Apache2.0) https://github.com/Luce-Org/lucebox-hub
35B-A3B MoE を 16GB GPU で all-GPU の ~85% 速度（~100 tok/s）で回す。
3要素: ①calibrated placement ②bounded async cache ③one fused graph。
報告値（RTX3090, Q4_K_M, ctx4096, 全experts=host RAM 前提・SSDではない）:
  naive offload 66 (55%) → +calibration 81 (68%) → +cache+fused ~100 (85%) / all-GPU 119 (100%)
  cold-hit: uniform 36% → calibrated 7%

## 我々の現状マッピング（重要: 部品は既に有る）
- ① = `--cache-profile`（per-(layer,expert)頻度・save/load、server は読み込みのみ）
- ② = `ExpertCache`（LRU・型プール・slot remap）だが **pinnedでない & 計算とオーバーラップせず stall**
- ③ = `decode_cached_fast`（常駐前提の単一融合グラフ＋ミス時 decode_cached フォールバック）だが experimental 止まり
- slotインデックスは既に入力テンソル化済み: `p_slot_g/p_slot_u/p_slot_d`（配線は揃っている）

## graph break の正体（=今つぶす対象）
decode_cached / decode_cached_batch は1層を必ず分割:
  seg A(attn+router) → [ホスト同期: selected読み戻し→ensure()→同期fetch] → seg B(expert matmul)
この「同期fetch」が stall。Spark は同じ mul_mat_id+slot入力のまま、
**weight常駐更新をグラフ外で非同期化**するので形状不変→CUDA graph化→1サブミット。
「miss = stall でなく throughput低下」はこの非同期化の言い換え。

## 前提差（必ず意識）
Spark は experts を host RAM に置き RAM↔VRAM PCIe 転送が前提。
→ まず効かせるべきは **RAM offload 経路 `--vram-budget`（SSDではない）**。
   現状この経路は sched 任せ＝彼らの「naive 66 tok/s」相当。
SSD tier は別レジーム（転送自体が遅い）。低residencyの完全stallゼロ化は
Spark自身が "open work"（次token expert予測 recall ~53% 頭打ち）と認める最難関。
→ 高residencyでは素直に効く / 低residencyのミスは「小stall許容 or 既存フォールバック」。

---

## STEP 1: エキスパート host バッファを pinned memory 化
目的: DMA高速化 + STEP2の真の非同期化の前提。低コスト・独立。
対象: core/runtime.cpp の ExpertCache の host 側 expert ストレージ確保箇所
      （RAM tier: experts を CPU に置く `load_weights_split` / cpu_buft 経路、
        および fetch_slab の RAM 読み出し元バッファ）
手順:
  - host expert weight の確保を通常 malloc/ggml CPU buffer から
    cudaHostAlloc(…, cudaHostAllocDefault)（pinned）へ。
  - ggml では ggml_backend_cuda_host_buffer_type() を使うと pinned host buffer が取れる。
    → CPU側 expert buffer の buft をこれに差し替えるのが最小変更。
  - SSD tier（--experts-ssd）はファイル直読なので、pread先の中間バッファを pinned に。
検証:
  - `--vram-budget` 単独で 30B / 35B-MTP の出力不変（回帰）。
  - host→device コピーの所要時間ログを足して 1.5〜2x 短縮を確認。
gotcha: pinned は確保コスト高＆物理RAM固定。expert総量ぶん常時 pin で OOM 注意。
        プールぶん（常駐+ring分）だけ pin、コールド本体は通常RAMでも可。

## STEP 2: expert コピーを別CUDAストリームで計算とオーバーラップ
目的: seg A/B 境界の stall を消す（②の核心）。
対象: ExpertCache::ensure / fetch_slab / fetch_parallel（reserve_victim, QWEN_PREFETCH_THREADS）
      と decode_cached の seg A→seg B 同期点。
手順:
  - 専用 cudaStream_t（copy stream）を作る。ggml なら ggml_backend_cuda で
    別ストリームの backend を用意 or 直接 cudaMemcpyAsync(copy_stream)。
  - ensure() の「VRAMスロットへ書き込む」コピーを copy_stream の cudaMemcpyAsync に。
  - 計算ストリームは seg B 発行前に、その層で必要なスロットのコピー完了 event を待つ
    （cudaStreamWaitEvent）。ヒット分は待ち不要、ミス分のみ待つ → stall最小化。
  - 可能なら「層 L の compute 中に 層 L+1 の予測expertsを copy_stream で先読み」
    （ダブルバッファ）。予測元は cache_profile（STEP1 と独立に効く）。
検証:
  - 出力不変（回帰）。
  - nsight/簡易タイマで copy と compute の重なりを確認。decode tok/s 改善。
gotcha: スロットを書き換える前に、まだ使用中(compute未完)のスロットを evict しないこと。
        ring の over-allocation（空きスロット）で「コピー先 != 使用中」を保証。
        event 管理を slot 単位で持つ。

## STEP 3: decode_cached_fast（単一融合グラフ）を RAM offload の本線 decode に昇格
目的: 40個/token の per-layer graph を 1 fused graph に。full residency で all-GPU bit一致・同速。
対象: decode_cached_fast（既存スケルトン）, build_graph の融合, decode() ルーティング
手順:
  - decode_cached_fast を「ExpertCache の固定スロットを mul_mat_id で参照する単一グラフ」に整備。
    routed FFN を attention graph に畳み込む（build_graph と同型を1グラフで）。
  - 常駐は cache_profile で高める。スロット表（expert→slot）は STEP2 の非同期 ring で更新。
  - ミス処理:
      高residency: そのまま走らせ、コールドは copy_stream で埋め event 待ち（小コスト）。
      フォールバック: 既存 decode_cached（seg A/B）を保険として残す（QWEN env で切替）。
  - decode() の分岐で、RAM offload 時のデフォルトを fast path に。
検証:
  - full residency で all-GPU と argmax/bit 一致（彼らの spark/bench.py 相当の自前チェック）。
    → 既存 QWEN_GDN_TEST/比較スクリプトを流用。128/128 一致を目標。
  - 60〜90% residency で tok/s を測り naive(sched) 比の改善を確認。
gotcha: GDN(recurrent)層は state 更新があるので融合時の順序・state backup に注意
        （init_state_backup / backup_states は既存）。MoE層と GDN層の混在を1グラフで。
        CUDA graph 実キャプチャまでやるなら入力テンソルのアドレス固定が必要。

---

## 完了の目安
- STEP1+2: RAM offload decode が naive(sched) より明確に速い、出力不変。
- STEP3: full residency で all-GPU bit一致・同速、partial で ~85% クラス。

## 参考に直接読むべき箇所（後でクローンして）
- dflash の fused graph 構築（③）
- spark cache の async copy / pinned host（②①）
- 注意: 今は read-only fork でクローン不可。通常セッションで実施。

---

## 実測結果と判断（2026-06-09, RTX4060Ti 16GB, 35B-A3B-MTP Q4_K_M, --vram-budget 15000, 54%常駐/hit82%, RAM offload）

### プロファイル（前提の確認 → 妥当）
ExpertCache に fetch 計測を追加（`expert cache fetch:` 行）。
- decode 64tok 5.8s のうち **fetch 2.7s ≈ 43%**。実効 **2.3 GB/s**（pinned PCIe の ~1/5〜1/10）。
- 内訳: ミス約10500回 × 小スラブ(~0.6MB) の **同期 cudaMemcpy**。→ 帯域でなく
  **レイテンシ/呼び出しオーバーヘッド律速**だった（1 fetch ≈ 0.2ms）。

### STEP1 pinned（実装済み・採用）— 2段で完成
**1a. pinned staging（commit b912c98）**: 1スラブ分の有界 pinned buffer 経由に変更
（`stage_host()`）。fetch 2703→2217ms、11.0→12.3 tok/s（+11%）。ただし帯域 2.8GB/s
止まり（ソースが pageable のまま、bounce memcpy 残）。

**1b. pin失敗の真因究明＋チャンク pin（commit df12418）**:
- probe（cudaHostAlloc）の結果: **単一確保の上限 ≈ 15.5GB**。チャンク（512MB〜2GB）なら
  19GB 到達。→ **容量(64GB/36GB空き)ではなく「1個の巨大 page-locked 確保」の上限**が原因。
- `load_weights_split` を **expert を複数 host buffer（各≤8GB）に分割確保**へ変更
  → 全19GB を確実に pin（"in 3 pinned chunk(s)"、pin失敗警告も消滅）。
- ソースが pinned になったので fetch を **ソースから直接 H2D（bounce 省略）** に。
- 効果: **fetch 2703→1324ms（51%減）、2.3→4.7 GB/s（2倍）、
  decode 11.0→~14 tok/s（plain）/ 9.5→13.2 tok/s（MTP）**。出力不変。

### STEP2 async copy stream（**解禁**: pinned ソースが揃った → 次の候補）
- 1b で expert 本体が全 pinned になったため cudaMemcpyAsync の前提が満たされた。
- 残課題はレイテンシ（多数の小同期コピー）。専用 copy stream + event で seg A→B の
  ミス fetch を計算とオーバーラップすれば更に改善余地（ggml 抽象の外で CUDA 直叩きが要る）。

### STEP3 fused graph = decode_cached_fast（**不採用**: partial residency で逆に遅い）
- `QWEN_FASTCACHE` で実測: **9.2 tok/s（default 12.3 より遅い）**。出力は一致。
- 理由: 楽観的単一グラフ → 残差検証 → **ミス時に decode_cached へ全フォールバック**。
  hit82% でもレイヤ単位でほぼ毎トークン1ミスは起き、二度手間になる。
- near-100% 常駐（＝モデルがほぼ VRAM に乗る）でのみ有効。35B/16GB では到達不可。

### 総括
- **計画の前提（fetch がボトルネック）は正しい**。RAM offload で fetch ≈ 43%。
- **STEP1 完成（チャンク pin + 直接 H2D）で RAM offload decode 11→~14 tok/s（+25%）**。
  「pin できない」の真因は容量でなく単一確保上限（≈15.5GB）で、チャンク分割で解決。
- STEP3（fused graph）は partial residency で逆効果のため不採用。
- 次に効きそうな方向:
  1) **STEP2 async overlap**（pinned ソースが揃い解禁。copy stream + event）。
  2) ミス**回数**削減（calibrated preload=`--cache-profile`、常駐率↑）。
  3) per-layer のミスを 1回の大きな転送にまとめる（slot 連続化が要る）。