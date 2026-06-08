# 指示記録 2026-06-07

1. ggml_backend_cuda_graph_compute のログが CLI 出力に混じるので外す。そのあと Phase B を実装し、`D:\dev\gguf\unsloth\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` を使って GPU メモリを 8GB に制限してテスト。
   → 対応: ggml-cuda.cu の CUDA graph 関連 GGML_LOG_DEBUG 3 箇所を削除（warmup complete / warmup reset / disabling due to GPU arch）。build\bin\Release の DLL が build\Release に自動コピーされない点に注意（手動コピー必要）。Phase B (CPU offload via sched) は 35B/8GB で動作、gen 8.2 tok/s。

2. 「では動的キャッシュを実装して」
   → Phase B v2: 動的 per-expert VRAM キャッシュを実装。

3. （方式選択への回答）「静的レイヤー配置でやるなら llama.cpp をおとなしくそのまま使えばいいので、動的 expert キャッシュで。このプロジェクトは expert キャッシュを実装することに意味がある。静的にやるとき、ssd にどう対応する？」
   → 静的配置は却下。動的 per-expert キャッシュを採用。SSD 階層(Phase C)の土台として、ミス時のフェッチ(ExpertCache::fetch_slab)を唯一のチョークポイントにし、ソースを CPU RAM → SSD pread に差し替え可能に設計。

## 実装結果（動的 expert キャッシュ）
- 新規: core/expert_cache.{h,cpp}（ExpertCache）。型シグネチャ(role+type+shape)ごとに VRAM スロットプール。LRU。
- runtime.cpp: decode_cached() を追加。decode を層ごとに分割実行（seg A: attention/GDN+router → selected をホスト読み戻し → ensure() でミス expert を CPU→VRAM コピー → seg B: cache スロットで mul_mat_id）。正確性は実ルーティングに従う。
- prefill は既存の CPU-sched パス維持、decode(1token) のみキャッシュ。
- バグ修正: (a) シグネチャに role を含めないと gate/up が同一プールに衝突 → role 必須。(b) n_slots < n_used で1層内スラッシュ→破損 → n_used で floor（予算より正確性優先）。
- 結果: 35B-A3B IQ2_XXS / 8GB budget で定常 14.9 tok/s・ヒット率 87%（ベースライン 8.2 tok/s から +82%）。出力はフル GPU リファレンスと完全一致。4GB の厳しい予算でも正確（6.2 tok/s, 全 streaming）。
- 既知の改善余地: F32 トークン埋め込み(約2GB)が weights_buf を圧迫し予算を食う → F16 化で半減可能（CUDA get_rows は F16 対応）。

4. 「Q3_K_S を4060側のVRAMぜんぶ使って速度測定」(`Qwen3.6-35B-A3B-UD-Q3_K_S.gguf`, 14.3GB)
   → --vram-budget 15500 で 12〜14.6 tok/s・ヒット率88%・エビクション0（ホットセット完全常駐, 83% resident, 約12.5GB VRAM使用）。

5. （質問）ホットセット完全常駐＝GPUだけで動いてる？
   → No と回答。真のソースは CPU RAM、VRAM は使用分のコピーのみ。未使用17%はCPUのまま。decode中もコールドミス時にH2D転送。計算自体は常にGPU（コピー後 matmul）。prefillは今もCPU experts。エビクション0なら新expertを引かなくなればヒット率100%に収束＝実質GPU単独。

6. 「キャッシュ利用をファイルに書き出して再利用すれば起動から速くなる？当初計画にあった」→ 軽量版プロファイル先読みを今実装。
   → 実装: ExpertCache に per-key アクセス頻度カウント + save_profile/load_prefetch。CLI `--cache-profile <file>`。終了時に (role,layer,expert,count) をテキスト保存、起動時に頻度降順でVRAMスロットを事前充填。
   → 効果(Q3/15500): 同一プロンプト=先読み13821→100%ヒット・21.5 tok/s（コールド14.6から+47%）。別プロンプトでも88→94.5%ヒット・17.2 tok/s（+18%, 普遍的ホットセットを捕捉）。
   → 設計メモ: プロファイル本命はSSD階層(Phase C)。CPU RAM階層では「再起動後の最初の生成のもたつき解消」が主効果。同一プロセス常駐ならメモリ上で既に温まるためファイル不要。fetch_slab を SSD pread 化すれば Phase C に直結。

7. （質問）llama.cpp は expert 8層CPU退避で37 tok/s。cache 100%ヒットでも遅いのはグラフ作り直しのせい？
   → 回答: グラフ再構築は症状。本質は「expertストリーミングが層ごとGPU↔CPU同期(router結果待ち)を強制→約82 submit+40同期/トークン→単一グラフ/CUDA Graph/融合を捨てている」。llama.cppは全expert常駐で1グラフ。
   → 「2を目指そう」= 楽観的シングルグラフを実装することに。

8. 楽観的シングルグラフ(option 2)を実装。
   → 設計: 永続グラフ1本。MoEは cacheプール + in-graphリマップ slot=get_rows(g2s[role,層], selected)。全層selectedを1本(sel_all)に集約し1回読み戻して常駐検証。全ヒット→採用。ミス→GDN状態復元しslow(decode_cached)へフォールバック。GDN状態backup/restore。
   → プリミティブ確認: get_rows I32 src→I32 dst可、cpy I32可、CUDA Graph無効化条件(3268-3280行)は非量子化 or ne[2]>max のみ→量子化cacheプール/ne[2]=1なら有効。
   → 実装: ExpertCache に resident()/slot_of(dense)/slot_of_row()/touch()/ensure_resident() 追加。runtime に decode_cached_fast + build_moe cache-fast分岐 + g2s_all/sel_all + 状態backup。CLI なし(QWEN_FASTCACHE=1 で opt-in)。
   → 結果: 正しい(出力リファレンス一致)が **slow path(21.5)を下回り未達**。ブロッカー2つ:
     (a) fast/slow数値発散: fast=persistent set_rows KV、slow=view-cpy KV でグラフ構造差→FP誤差→greedyが別トークン→別expert→残ミス1.5%。同一プロンプトwarmで100%にならず98.5%の原因。
     (b) all-or-nothingフォールバック: 1.5%ミスでも毎トークン≥1ミス→毎回slowへ二度手間→pure slowより遅い(warm 15.1 < slow 21.5)。
     (c) repair-rerun(ミスexpertだけfetchしfastグラフ再実行で単一グラフ維持)案は cudaGraphLaunchリプレイでillegal access クラッシュ(1トークン内の複数compute+weight poolへのtensor_set干渉)。
   → 判断: デフォルトを実績あるslow pathに戻し回帰回避(cache_fast_enabled=false, QWEN_FASTCACHE=1でopt-in)。fast pathは実験扱いで残置(slowフォールバック版、正しいが速度未達)。
   → 注意: 測定中にマシンが競合状態(3050が常時35%util/3GB、slow pathが21.7→7.2に劣化)。クリーンなfast 100%ヒット計測は未取得。
   → 次の選択肢: (1)発散除去(slowフォールバックもpersistent KVに合わせfast自己無撞着化→warm 100%ヒット狙う)+repair-rernのCUDA Graph回避、(2)fast棚上げしslow(21.5)を製品とする、(3)Phase C(SSD)へ。

9. 「vramに載らないQ4_K_XLを置いた。これで試す。データだけとってslowを残しssd利用にすすむ。このggufはssdにあるので基準にできる」(`C:\Users\naoki\Desktop\gguf\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, 20.8GB)
   → ベースライン取得(slow path, RAM版, --vram-budget 15500): experts 18760MB(RAM), 非expert GPU 2554MB, VRAM常駐53%(9980MB), 6プール。COLD 10.5 tok/s(87.3%hit), WARM(先読み14281) 17.9 tok/s(99.6%hit)。
   → Phase C(SSD階層)を実装:
     - Model::load_weights_ssd() : routed experts を一切RAMにロードせず非expert(+shexp)のみGPUへ。meta tensorのne/nbは保持。
     - Model::tensor_file_offset()/path() 追加。
     - ExpertCache に ssd フラグ: コンストラクタでGGUFをfopen+各(role,layer)のファイルオフセット precompute。fetch_slab を pread(_fseeki64+fread)に分岐。これが唯一のチョークポイント。
     - runtime: ssd_mode。load_weights_ssd使用、sched/cpu_backend無し。prefill(複数トークン)は sched不在のため decode_cached をトークン毎ループ。init_cache を sched||ssd_mode でゲート。
     - CLI `--experts-ssd` で opt-in。
   → 結果(SSD版): COLD 8.0 tok/s(87.2%hit, 16062 SSD pread), WARM(先読み15646 from disk) **14.8 tok/s(99.5%hit)**。
   → 評価: **SSD warm 14.8 は RAM warm 17.9 の約83%**。experts 18.3GBを一切RAMに持たずわずか17%ペナルティ。warmでは99.5%がVRAMヒットで稀なミスのみSSD pread。VRAMにもRAMにも載らない巨大MoEを実用速度で回せることを実証。
   → slow path維持・デフォルト不変。SSDはopt-in。

10. 「macで動いた(backend GPU [MTL0])。分割ggufに対応してる？してなかったらやって。単純に分割されているはず。このPCに分割ggufは無いので他でテスト」
    → 分割(sharded)GGUF対応を実装。llama.cpp規約 `<prefix>-NNNNN-of-MMMMM.gguf`(5桁0詰め, 1始まり)。第1シャードのパスを渡せば残りを自動探索。
    → Model変更:
      - discover_shards(path): `-of-` パターンを検出し全シャードパスを生成(非分割なら{path})。
      - Model::load: 全シャードのgguf metaを開き、統合meta_コンテキストに全テンソルをコピー、src_(name→{shard path, 絶対offset})を構築。KVは第1シャードのgguf_を保持、他shardのgguf/metaは解放。
      - src_導入: tensor_file_offset()/tensor_file() は src_ 参照。read_tensor_bytes(name,dst,nb,files): 該当shardファイルを(FILE*キャッシュ付きで)開きseek+read。
      - load_weights/load_weights_split/load_weights_ssd を read_tensor_bytes 経由に統一(単一/分割両対応)。
      - ExpertCache(SSD): foff_/fpath_ を per(role,layer)で保持、files_(path→FILE*)で遅延オープン。expertが別シャードでもOK。
    → 検証: 単一ファイルの全経路(0.6B load_weights / Q4 --experts-ssd)で回帰なし・正常出力。分割gguf実機テストはユーザが他環境で実施予定。
    → 使い方: 分割モデルは第1シャード(...-00001-of-NNNNN.gguf)を -m に渡す。`gguf: N shards, M tensors total` と表示。

11. 【マイルストーン報告】48GBのMacで Qwen3.5-122B-A10B が 6 tok/s で動作。
    → Phase Dターゲット(122B-A10B)達成。48GB RAMに載りきらない122Bを --experts-ssd(SSD streaming)+動的expertキャッシュ+プロファイル先読みで実用速度(6 tok/s)で実行。Metal(MTL0)バックエンド。
    → 一連の積み上げ(Phase B動的キャッシュ→profile先読み→Phase C SSD階層→分割gguf対応)が結実。

12. 「prefetch の非同期化とprefill高速化やってみて」
    → (A) 非同期プリフェッチ実装【採用】: ExpertCache::ensure(SSD) を「ヒット解決+ミスのvictim予約(reserve_victim)→複数スラブを並列pread(fetch_parallel, 8ワーカ各自のFILEハンドル)→直列H2D」に変更。QWEN_PREFETCH_THREADS(既定8)。
      効果(Q4 SSD cold): gen 4.7→8.8 tok/s(約1.9倍), prefill 2.1→3.9。ヒット率・ミス数不変(計算は同一、ディスク読みを並列化)。出力正常。
    → (B) batched prefill 実装【バグのため既定無効・QWEN_BATCHでopt-in】: decode_cached_batch でプロンプトをチャンク単位(min_slots/n_used, ≤256)に分割し1回の分割forwardで処理。ExpertCache::min_slots()追加。
      バグ: マルチトークン(T>1)で出力が壊れる。切り分け: T=1(chunk=1)は正しい、T≥2で破損。flash無効/CUDA融合無効でも破損。per-token seg Bでも破損(MoE無実)。attention/GDN/routerはbuild_graphと同一コードのはず。最終トークンの層0(GDN)出力が token-by-token と乖離(12.04 vs 10.36)。原因未特定(セグメント分割周りが濃厚だが未解決)。
      → 既定は実績ある token-by-token prefill に戻した(正しい)。これも非同期プリフェッチの恩恵で高速化(2.1→3.9)。batched は QWEN_BATCH で実験可能なまま残置。
    → CUDAで検証。Metal(ユーザ環境)では GDN系opのマルチトークン挙動が違う可能性ありだが未確認。
    → 次の課題: batched prefill の T>1 バグ特定(build_graph単一グラフ版と分割版の差分。GDN state/carryテンソル/gallocr再利用が容疑)。
