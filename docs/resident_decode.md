# 常駐限定デコード (resident decode) の仕組みとexpertのライフサイクル

`--resident-decode`(環境変数 `QWEN_RESIDENT_DECODE=1`)の動作仕様。
実装は [core/runtime.cpp](../core/runtime.cpp) の `decode_cached_fast` /
`build_router`、常駐管理は [core/expert_cache.cpp](../core/expert_cache.cpp)。
経緯・計測は [prefill_layer_major.md](../prefill_layer_major.md) を参照。

## 何をするものか

expertオフロード時のdecodeを「**VRAMキャッシュに常駐しているexpertだけで完結する
単一融合グラフ**」にする高速化。fetch待ち・ミス検証・GDN状態バックアップが消え、
トークンあたりのhost仕事はtoken/pos/maskアップロードとlogits読み戻しだけになる
(RAM階層で約2倍、SSD階層で約2.7倍)。**lossy**(近似)である点が本質的なトレードオフ。

## ルーティングの改変

routerのlogitsに層ごとの**常駐マスク**(非常駐expert = -inf)を加算してから
softmax / top-k する。効果:

- top-k は構造的に常駐expertしか選べない → キャッシュミスが起きない
- 本来のtop-kに非常駐が居た場合、**次点の常駐expertが代わりに選ばれ、
  重みは選ばれたk個で再正規化**される
  - 「抜けたexpertの寄与がゼロになる」のではなく「別のexpertが代弁する」近似
- expert は gate/up/down の3テンソル全てが常駐のときのみ「常駐」扱い

マスクと slot remap テーブルは miss+eviction カウント(スタンプ)が変わった
ときだけ再アップロードされる。定常状態ではアップロードなし。

## 3つの動作段階

1. **ウォームアップ(投機実行)** — 層の常駐expert数が `QWEN_RESIDENT_MIN`
   (既定32)未満の間、その層はマスクされない。実行は投機的:
   GDN状態をバックアップ → キャッシュ前提で1グラフ実行 → 選択expertの常駐を
   検証 → ミスなら巻き戻して低速パスで正しく再計算(このときミスした
   expertがfetchされキャッシュが温まる)。
2. **凍結** — fast-decodeが `QWEN_RESIDENT_WARMUP`(既定32)トークンを
   超えると閾値が n_used まで下がり、**その時点で常駐しているものがマスク
   として固定**される(「数トークン温めてピン留め」)。ルーティングが集中する
   終盤層が短いプロンプトで32種に届かない問題への締め切り。
3. **バックグラウンド補充** — グラフは**マスク前のrouterが本来選びたかった
   top-k(`want_all`)も記録**しており、ホストが毎トークン
   `QWEN_RESIDENT_REFILL` 個(既定: RAM 8 / SSD 4)を上限に
   「欲しいのに不在」のexpertを非同期ロードする。ロード済みは次トークン
   からマスクに合流する。実際に使ったexpertは毎トークンtouchされ、
   補充による追い出しが古株に当たるようにしている。

## expertがロード(常駐化)される経路の全体像

| 経路 | タイミング | 備考 |
|---|---|---|
| プロファイルprefetch | 起動時 | `--cache-profile` のホットexpert頻度を先読み |
| prefill | リクエスト処理 | バッチの選択expertを計算前にensure。ただし `--prefill-prune` 有効時はrouter質量の小さい非常駐expertをfetchせずゼロ重みフィラーで代替 |
| ウォームアップ中のミス | decode序盤 | 低速パスへのフォールバック時にfetch |
| バックグラウンド補充 | マスク凍結後 | 毎トークン REFILL 個まで、want_all 起点 |
| 追い出し | 常時 | プールはLRU。使用中expertはtouchで保護 |

## 品質特性と既知の劣化パターン

- 完全凍結(`REFILL=0`)は短プロンプト+コード生成で顕著に破綻する
  (実測: Javaコードがコメント羅列に退化)。既定の補充ありでは通常の
  会話・コードで目視破綻なし。
- **ドメイン切替に弱い**: 凍結パレットは「プロンプト+最初の~32トークン」で
  決まるため、生成の途中で話題の層が変わる(例: 説明文 → CSSの数値列)と、
  その文脈が要求するexpertがパレットに無いことがある。補充はSSD階層で
  毎トークン4個なので、48層×最大k個の不足が同時発生すると追いつくまで
  百トークン以上のラグがあり、その間**同じ文脈では毎回同じ代替が起きる**。
  ランダムなノイズではなく「特定パターンが全域で一貫して崩れる」形で現れる
  (実例: OpenCodeでのWeb画面生成時、スタイルシートの数値の後の `px` が
  全域で欠落)。
- MoEのルーティングはトークン(の隠れ状態)ごと・層ごとに決まるため、
  「数値の直後に単位を出す」のような局面は毎回ほぼ同じexpert集合を呼ぶ。
  そこが不在だと毎回同じ形で劣化する、という機構的な帰結。

### 症状が出たときの切り分け・対処

1. 同じプロンプトを `--resident-decode` なしで再実行 → 直れば本機能の近似が原因
   (llama.cpp同一GGUFとの比較なら量子化・モデル素の挙動も切り分けられる)
2. `--resident-refill` を増やす(SSDでも 8〜16)、`--resident-warmup` を伸ばす
3. 対象ドメインのテキストがプロンプトに含まれるようにする(prefillが温める)。
   その際 `--prefill-prune` は切る
4. 精度優先タスク(コード生成など)では `--resident-decode` を外す運用が確実

## 関連ノブ一覧

| ノブ | 既定 | 意味 |
|---|---|---|
| `--resident-decode` / `QWEN_RESIDENT_DECODE=1` | off | 常駐限定ルーティングdecode(lossy) |
| `QWEN_RESIDENT_MIN=N` | 32 | マスク発動に必要な層あたり常駐数 |
| `--resident-warmup` / `QWEN_RESIDENT_WARMUP=N` | 32 | このdecodeトークン数の後は常駐数を問わずマスク固定 |
| `--resident-refill` / `QWEN_RESIDENT_REFILL=N` | RAM 8 / SSD 4 | マスク中の補充expert数/トークン(0=完全凍結、非推奨) |
| `--prefill-prune <eps>` | off | prefillでの低質量非常駐expertのfetchスキップ(lossy) |
| `QWEN_FASTCACHE=1` | off | マスク無しの楽観単一グラフ版(全常駐前提+ミス時フォールバック) |
