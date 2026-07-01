# prefill 高速化: layer-major 再構成 + 質量ベース expert pruning (2026-07-02)

環境: RTX 4060 Ti 16GB / Qwen3.6-35B-A3B-UD-Q4_K_M (40層 x 256 experts, 8 used) /
RAM オフロード `--vram-budget 13500`(注: 14000 はデスクトップ常駐アプリの VRAM ~1.9GB と衝突して
プール確保失敗)/ 常駐率 51% / プロンプト 5705 tok (ChatML)。

## 診断(QWEN_PREFILL_STATS=1 を decode_cached_batch に追加)

旧 prefill はチャンク = `min_slots/n_used` = 49 tok に分割し、チャンクごと層ごとに選択 expert の
和集合を ensure。実測:

- 総 fetch **151.6GB = 全 expert (18.7GB) の 8.1 周分**。fetch が prefill 時間の **73%**
  (チャンク間回帰: wall = 109ms + 0.218ms/MB → H2D 4.5GB/s = pinned 上限に張り付き)。
  → 帯域でなく**バイト数**が敵。
- 和集合は 49tok で 88.5/256、512tok で ~200、プロンプト全体で ~255(ほぼ全部)。
  隣接チャンク重複 ~80% あるが LRU が毎チャンク ~21% を取り直す。
- **チャンク拡大単独は逆効果**: チャンクの全層フットプリント(Σ union×3role)がプール総 slot
  (15.7k)を超えると、巡回アクセスに対し LRU が最悪化(次に使う層の expert を追い出す)して
  ほぼ全ミス。チャンク256 で 260GB/103 tok/s に悪化。64 が旧構造のスイートスポット(130 tok/s)。
- 層別: 序盤層 (L0-L2) の和集合が最大 (133-143)、終盤層 (L34-L37) は 63-75 と集中
  → pruning は終盤層ほど安全という設計指針。

## 対策1: layer-major prefill(ロスレス)

「チャンク外側・層内側」をやめ、**層外側**に再構成(decode_cached_batch 内、T > QWEN_SEGA_CHUNK
=256 で発動):

- seg A(attn/GDN + router)は層内で 256tok サブチャンクに分割して逐次実行。
  KV は `build_attn` に追加した kv_pos 引数で n_past+tc0 に連続書き込み、GDN state は
  サブチャンク逐次実行で自然に連鎖。数学的に全チャンク一括と等価。
- seg B(expert matmul)は**層ごとに全トークン一括**。1回の ensure の distinct expert 数が
  プール容量(`ExpertCache::capacity(layer)`)を超えると自壊(ensure 内で自分の slot を
  追い出す)ので、和集合上限でトークンをスライス(+ QWEN_SEGB_SLICE=1024 で MoE 活性の
  メモリ上限)。
- → **層あたり expert fetch はチャンクにつき和集合1回**。expert トラフィックはチャンク数に
  比例し、トークン数に依存しない。decode() の外側チャンク既定を 4096 に(QWEN_BATCH_CHUNK)。
- server: 非 time-slice 時の prefill チャンク 512→4096(QWEN_PF_CHUNK)。切断検出粒度は
  ~10s に粗くなるトレードオフ。

結果(出力品質は不変、-n 64 の要約で確認):

| 構成 | prefill | 総 fetch |
|---|---|---|
| 旧 (チャンク49) | 121.8 tok/s (46.6s) | 151.6GB |
| layer-major, チャンク4096 (既定) | **410.3 tok/s** | 33.4GB |
| layer-major, チャンク8192 (1チャンク) | **481.5 tok/s** | **17.4GB**(=和集合×1回の下限) |

fetch 下限到達後は ~2/3 が計算律速。

## 対策2: 集約質量 pruning(lossy、QWEN_PREFILL_PRUNE=eps)

層の seg A 完了後、チャンク全体の router 重みを expert 別に集約し、**非常駐 expert を質量の
小さい順に、落とした質量が層合計の eps に達するまで fetch スキップ**。落としたエントリは
weight 0 + トークン行内で重複しない kept expert の id(フィラー)に差し替え、残り重みを
トークンごとに再正規化。

| eps | prefill | 総 fetch | 層平均 drop 数 |
|---|---|---|---|
| 0 | 481.5 tok/s | 17.4GB | 0 |
| 0.02 | 509.4 | 11.3GB | 86/256 |
| 0.05 | 599.4 | 9.0GB | 120/256 |
| 0.10 | **643.6** | 7.5GB | 150/256 |

eps=0.10 でも greedy 64tok の要約は正常(**旧比 5.3 倍**)。落ちるのは質量下位のロングテール。
※ 品質評価は目視レベル。perplexity / 長文生成のループ検査は未実施。SSD tier(帯域が
細く fetch 律速が強い)では pruning の効きがさらに大きいはず(未計測)。

## SSD tier(--experts-ssd、RAM に expert を置かない)での計測

同一プロンプト・同一 VRAM 予算。予想どおり fetch 律速が強く、layer-major と pruning の効きは
RAM tier より大きい:

| 構成 | prefill | ミス数 | 出力 |
|---|---|---|---|
| 旧構造相当(チャンク49) | 53.4 tok/s (106.8s) | 273k | 正常 |
| layer-major 1チャンク | 292.8 tok/s (19.5s) | 34.0k | 正常 |
| + pruning eps=0.05 | 437.3 tok/s (13.0s) | 17.3k | 正常 |
| + pruning eps=0.10 | **469.7 tok/s (12.1s)** | 15.9k | 正常 |

**SSD 旧構造比 8.8 倍**。pruning 後は RAM tier(9.5-11.2s)にほぼ肉薄 = SSD ストリーミングが
ほぼ計算の影に隠れる水準。decode は 9-15 tok/s(SSD ミスあり、従来どおり)。
注: SSD の fetch バイト/時間が stats に載っていなかったため fetch_parallel に計上を追加済み
(この表のミス数は expert cache stats 行から)。

### ⚠ 上の表はページキャッシュ読みだった(ユーザー指摘: タスクマネージャのディスクが無反応)

RAM 64GB / GGUF 21.1GB なので、先行ランでファイル全体が Windows スタンバイリストに載っており、
「SSD」計測が実際はメモリ読みだった。`QWEN_SSD_DIRECT=1`(FILE_FLAG_NO_BUFFERING の direct I/O、
Windows のみ、expert_cache.cpp に追加)でページキャッシュをバイパスした**真の SSD** 計測:

| 構成 | キャッシュ済み | 真の SSD (direct) |
|---|---|---|
| 旧構造(チャンク49) | 53.4 tok/s | **15.4 tok/s**(161GB / 320s) |
| layer-major 1チャンク | 292.8 | **127.9**(17.4GB / 34.3s) |
| + pruning eps=0.10 | 469.7 | **199.5**(7.5GB / 20.0s) |
| decode 通常 | 13.3 | **3.1 tok/s** |
| decode 常駐限定 | 35.4 | **25.2 tok/s(8.1x)** |

- 実効読み速度 ~0.5GB/s(C: = CFD PG1VN 512GB NVMe Gen3、0.6MB スラブ×8スレッド同期読み)。
  QWEN_PREFETCH_THREADS=32 で 0.63GB/s / 151.9 tok/s(+24%、スケール鈍い=ドライブ律速)。
- 帯域が細いほどトラフィック削減の価値が上がる: 真の SSD では prefill **15.4→199.5 = 13倍**、
  decode は常駐化で **3.1→25.2 = 8倍**(ウォームアップ後はディスクに触らない)。
- モデルが RAM に収まる場合はページキャッシュが「タダの RAM tier」として実際に効く(それが
  前の表)。direct I/O の数字は「モデル ≫ RAM」レジームの見積もりとして読む。
### SSD 連続読み合体の顛末(実装済み・このマシンでは効果なし)

「0.6MB×255 のバラ読み → 層テンソルまるごとの連続読み」を実装して検証した結果:

1. 最初の実装は seg B スライス(1024tok×6)ごとの ensure が残り expert を疎に fetch し、
   gap マージで**層テンソルを最大6回 re-read(総読み 65GB)**していた。
   → **層の和集合を最初に1回の ensure でまとめて prefetch**(union ≤ capacity のとき)+
   **密度ガード**(欲しいスラブ/スパン < 50% の run は個別読みへ)で修正。runs 2213→363。
2. アップロード律速も発見: pageable ステージングからの同期 cudaMemcpy を
   **pinned ステージング + async DMA(バッファ再利用時のみ同期)** に変更(合体・個別両経路)。
3. それでも 0.44-0.48GB/s 止まり。`--bench-read`(生 unbuffered 読みベンチ、追加)で確定:
   **このドライブはファイル前半 1.87GB/s / 後半 0.15GB/s / ランダム 0.27GB/s**。
   C: が 97% 使用(空き 14.9GB)で、GGUF 後半が劣化領域に書かれている。**媒体律速**であり、
   バラ読み QD8(~0.5GB/s)をソフトでは超えられない。
4. 合体読みは健全なドライブ(シーケンシャル ≫ ランダム)でのみ有効なため
   **opt-in(QWEN_COALESCE=1)に降格**。和集合一括 prefetch と pinned async は無条件で有効。

**このマシンでの改善策(コードでなく運用)**: C: の空きを 50GB 以上確保してから GGUF を
書き直す(別ドライブへコピー→戻す)と前半並みの ~1.9GB/s に戻る可能性が高い。その状態なら
QWEN_COALESCE=1 で fetch ~10s 級(prefill ~300 tok/s)が狙える。
QWEN_COAL_DEBUG=1 で run ごとの read/upload 時間を表示。

## 落とし穴(再発防止)

1. **mul_mat_id は 1 トークンの id 行内に重複があると壊れる**(CUDA MMQ の expert 簿記が
   token あたり 1 マッチ前提 → illegal memory access)。id 差し替えは行内 distinct 必須。
2. **ensure() の distinct 数 > プール slot 数は無言で自壊**(自分の slot を evict)。
   旧チャンク上限 49 はこれの悲観的回避だった。capacity() + スライスで置換。
3. **生プロンプトの greedy 先頭トークンは数値的に拮抗しうる**(EOS vs <think>)。チャンク
   サイズ変更で argmax が反転しても破損ではない(旧 sched 経路でも反転を確認)。品質 A/B は
   ChatML で行うこと。
4. LRU + 巡回アクセス(層 0→39 の繰り返し)は「フットプリント ≤ プール」を超えた瞬間に
   崖(ほぼ全ミス)。中途半端なチャンク拡大は悪化する。

## decode 側: 常駐限定ルーティング(QWEN_RESIDENT_DECODE=1)

融合単一グラフ(旧 QWEN_FASTCACHE、ミス時全フォールバックで不採用だった 3a)を、router logits に
**常駐マスク**(非常駐 = -inf、`resmask_all` [n_exp, n_layer])を加算して復活させた。選択が構造的に
常駐のみになるため:

- 投機実行でなくなる → GDN state バックアップ / sel_all 読み戻し検証 / フォールバックを全省略。
- 常駐版数(miss+eviction カウント)が変わらない限り g2s/マスクの再アップロードもスキップ
  → トークンあたりの host 仕事は tok/pos/mask アップロードと logits 読み戻しのみ(実質 pure-GPU)。
- **ウォームアップ自動化**: 層の常駐数が QWEN_RESIDENT_MIN(既定32)未満の間はその層をマスクせず、
  ミス→フォールバックでキャッシュが温まる。全層が閾値に達した時点でマスク完成→固定。
  (「数トークン動かして温めてから expert を固定する」動的 pruning 案の実装形)
- **締め切り**: ルーティングが集中した終盤層は短いプロンプトだと常駐 32 種に永遠に届かず
  マスクが完成しない(サーバの短プロンプトで発覚、gen 21 tok/s のまま)。
  QWEN_RESIDENT_WARMUP(既定16)トークンの decode 後は floor を n_used に下げて
  「その時点の常駐で固定」する。修正後: 27 tok プロンプトで gen 51.0 tok/s・出力正常。
- **バックグラウンド補充(重要)**: 完全凍結は短プロンプト+コード生成で顕著に破綻する
  (ユーザー報告: Java コードが壊れ、コメント羅列に退化)。融合グラフ内でマスク前 logits の
  top-k(=本来選びたかった expert、`want_all`)も記録し、毎トークン読み戻して
  「欲しかったが不在」の expert を QWEN_RESIDENT_REFILL 個/トークン(既定: RAM 8 / SSD 2、
  0で無効)まで裏から取り寄せ、次トークンからマスクに合流させる。実際に使った expert は
  touch して LRU を保つ(補充の追い出しが古株に当たるように)。
  実測(短プロンプト → Java Swing コード 400 tok、greedy):
  | 補充 | gen | 出力 |
  |---|---|---|
  | なし (REFILL=0) | 54.8 tok/s | 破綻(コメント無限羅列) |
  | あり (REFILL=8) | **44.7 tok/s** | **正常なコード**(EDT 処理まで正しい) |
  通常 decode 20.8 tok/s 比でなお 2.1 倍。凍結ではなく「数トークン遅れで追従する常駐集合」になった。

計測(同一プロンプト、-n 256、greedy):

| tier | 通常 decode | 常駐限定 decode |
|---|---|---|
| RAM | 20.8 tok/s | **40.9 tok/s (2.0x)** |
| SSD | 13.3 tok/s | **35.4 tok/s (2.7x)** |

256 トークン生成でループ・破綻なし(目視)。lossy(router が常駐集合に制限される)なので
perplexity 級の定量評価は今後。

## 今後

- pruning / 常駐限定 decode の品質定量評価(perplexity、長文ループ検査)、層別 eps。
- 常駐限定 decode 中のバックグラウンド補充(「欲しかったが不在」の集計と copy stream 補充)。
- CUDA graph 実キャプチャ(マスク完成後は入力アドレス固定が可能)。
- server の切断検出をチャンク間 → 層間コールバックに(要 GDN state の整合設計)。
