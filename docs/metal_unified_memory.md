# Metal / ユニファイドメモリでの大型 MoE 実行（48GB Mac + 122B チューニング記録）

ターゲット: **Qwen3.5-122B-A10B**（qwen35moe, 49層, experts 48GB on disk, 語彙248k）を
**M4 Pro 48GB** で `--experts-ssd` により実行する。

```bash
infer -m Qwen3.5-122B-A10B-UD-Q3_K_M-00001-of-00003.gguf \
  -p "..." -n 100 --vram-budget 36000 --experts-ssd \
  --cache-profile q122b.prof --log-tokens-per-sec
```

## 結果サマリ

| 構成 | チューニング前 | 後 |
|---|---|---|
| Q3_K_M plain | 8.7 tok/s（ヒット96%） | **15.1 tok/s**（ヒット100%, GPU計算律速96%） |
| Q3_K_M `--mtp --draft 2` | 8.8 tok/s | 11.7 tok/s（※plainに負ける。[mtp.md §4.7](mtp.md) 参照） |
| Q4_K_M plain | – | 6 tok/s（ヒット80%台。スラブが大きく常駐率が不足） |

ボトルネックの変遷: ホストオーバーヘッド+ミスフェッチ律速（wall 115ms = GPU 20ms + その他 95ms）
→ **GPU 計算律速**（wall 66ms = GPU 64ms + ホスト 3ms）。
ここから先は ggml の Metal カーネル効率の世界で、QuestWend 側の伸びしろはほぼ無い。

## 効いた施策（コミット順）

1. **埋め込みフォールバックのバックエンド対応判定**（`255bda0`）
   CUDA は K-quant の get_rows 非対応だが **Metal はネイティブ対応**。
   `ggml_backend_supports_op` で実際に問い合わせ、対応していれば F16 変換コピーを
   作らない。122B（語彙248k）で **約2.4GB のユニファイドメモリが解放**され、
   そのまま expert キャッシュに回る（常駐 57%→60%、ヒット 96%→99.9%）。
   これが最大の寄与。

2. **GPU 重みバッファ解放漏れの修正**（`35e6035`）
   1 により embd フォールバックのエイリアス経由で偶然解放されていたバッファが
   リークし、exit 時に `GGML_ASSERT([rsets->data count] == 0)` で abort していた。
   所有権を明示（`weights_buf_owned`）して修正。

3. **バッチ verify の segB+segA 融合**（`78a056c`）
   Metal はコマンドバッファの submit/wait 往復が 1 回 0.2〜0.5ms と高く、
   レイヤーごとに 2 回同期するバッチ経路は 49 層で大きな損だった。
   decode_cached と同じ「segB(L)+segA(L+1) を 1 グラフ」に統一。

4. **ゼロコピー SSD 読み**（`84b5076`）
   Metal の shared バッファ（ユニファイドメモリ）はスロットプールの実メモリが
   ホスト可視なので、SSD ミスを `pread()` で**スロットに直接**読み込む。
   staging vector / memcpy / stdio バッファリングが全部消える。
   実効フェッチ 1.1 → 1.5 GB/s、起動時のプロファイル prefetch（~27GB）が約5秒短縮。
   ホスト可視性は `mincore()` でプール毎に1回判定（Metal private バッファの
   仮想プレースホルダは PAGEZERO 内 = 常に未マップなので誤検出しない）。
   `QWEN_NO_DIRECT_FETCH=1` で旧経路に戻せる。

5. **ggml ログフィルタ**（`da19ad4`, `7411c6c`）
   ggml のデフォルトログはレベル無視で全部出る。Metal はバッファ確保毎の DEBUG、
   ワーキングセット上限近くで確保毎の WARN、起動時 ~25 行の INFO バナーを出すため、
   DEBUG/INFO を破棄し連続する同一 WARN を1回に集約。`QWEN_GGML_DEBUG=1` で全表示。

## 運用上の注意

- **`--cache-profile` は CLI では実行のたびに上書き保存される**。
  アクセスパターンの異なる実行（MTP、別タスク）を挟むとプロファイルが汚れて
  ヒット率が落ちる。ベスト状態のファイルをコピーして固定するのを推奨
  （`infer-server` は読み込みのみで上書きしない）。
- `--vram-budget` は recommendedMaxWorkingSetSize（48GB機で ~38.6GB）から
  非エキスパート重み（~7GB）+ KV + 計算バッファを引いた範囲に収める。
  超えると確保毎に WARN が出る（ログフィルタで1回に集約される）。
- **量子化の選択がヒット率を支配する**: Q4_K_M はスラブが Q3_K_M の ~1.3 倍で
  同じ予算だと常駐スロットが減り、ヒット 80%台 → 6 tok/s まで落ちる。
  48GB 機で 122B はヒット ~100% にできる Q3_K_M 系が実用ライン。
- MTP はこのレジーム（キャッシュがギリギリの SSD 階層）では plain に勝てない。
  詳細は [mtp.md §4.7](mtp.md)。

## デバッグ / プロファイル用環境変数

[README](../README.md) の環境変数一覧を参照。このチューニングで主に使ったもの:

```bash
QWEN_PROF_DC=1   infer ...   # decode_cached の wall / GPU / host 分解（exit時に表示）
QWEN_PROF_MTP=1  infer ...   # MTP の per-cycle フェーズ分解（draft/verify/settle/resync）
QWEN_GGML_DEBUG=1 infer ...  # ggml の生ログ（確保毎の DEBUG 等）を全部表示
```
