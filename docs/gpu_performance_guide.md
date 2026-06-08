# ggml GPU推論エンジン 性能最適化ガイド

Qwen3/3.5実装での知見。ggmlを`add_subdirectory()`で取り込む実装全般に適用可能。

## 結果サマリ（RTX 4060 Ti）

| モデル | 最適化前 | 最適化後 | 主要因 |
|---|---|---|---|
| 0.6B dense | 52 tok/s | 196 tok/s | CUDA Graph |
| 0.8B GDN | 56 tok/s | 154 tok/s | CUDA Graph |
| 30B-A3B MoE | 39.5 tok/s | 91 tok/s | CUDA Graph + グラフ再利用 |

---

## 1. CUDA Graph 有効化（最重要・最大インパクト）

### 問題

ggmlを`add_subdirectory()`で取り込むと、**CUDA Graphがコンパイルから丸ごと除外される**。

- `GGML_CUDA_GRAPHS` CMakeオプションの既定値: **OFF**（コメント: "llama.cpp only"）
- これにより `GGML_CUDA_USE_GRAPHS` マクロが未定義 → `USE_CUDA_GRAPH` 未定義 → `#ifdef USE_CUDA_GRAPH` ブロックが全て除去される
- 結果: CUDA Graph機構が**コードごと存在しない**状態でビルドされる

### 対処

```cmake
# CMakeLists.txt
if(QWENCPP_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)  # ← 必須
endif()
```

### 効果

- decode compute: 14ms → 9.2ms（-35%）
- 30B MoE: 58 → 91 tok/s（+57%）
- 0.6B dense: 81 → 196 tok/s（+142%）

### 診断方法

CUDA Graphが実際に動いているか確認するには`ggml-cuda.cu`に計装を追加：

```cpp
// ggml_backend_cuda_graph_compute() 内
if (getenv("QWEN_CG_DEBUG")) {
    fprintf(stderr, "[cg] enabled=%d compatible=%d\n",
            use_cuda_graph, graph_compatible);
    fprintf(stderr, "[cg] props_changed=%d warmup=%d\n",
            properties_changed, graph_warmup_done);
}
```

正常動作時のログ遷移：
```
decode 1: [cg] props_changed=1 warmup=0   ← 無効（初回）
decode 2: [cg] props_changed=0 warmup=0   ← warmup開始
decode 3: [cg] props_changed=0 warmup=1   ← Graph再生開始 ✓
```

`enabled=0` しか出ない場合 → `USE_CUDA_GRAPH` 未定義。`GGML_CUDA_GRAPHS=ON` を確認。

---

## 2. グラフ再利用（Persistent Decode Graph）

### 目的

CUDA Graphはグラフトポロジーが安定していることが前提（warmup 2回必要）。  
decode毎にグラフを再構築すると：
- build/alloc オーバーヘッド ~2.5ms/token
- グラフトポロジーが変化 → CUDA Graph warmupが完了しない

### 実装: KV書き込みの動的化

```cpp
// グラフ構築時: KV書き込み位置を入力テンソル化
ggml_tensor * d_kvidx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
ggml_set_input(d_kvidx);

// K/V書き込みに ggml_set_rows を使用（固定offsetのviewの代わり）
ggml_build_forward_expand(gf, ggml_set_rows(ctx, k_cache[il], Kflat, d_kvidx));
ggml_build_forward_expand(gf, ggml_set_rows(ctx, v_cache[il], Vflat, d_kvidx));

// 実行時: 毎トークン更新
int64_t kvidx = n_past;
ggml_backend_tensor_set(d_kvidx, &kvidx, 0, sizeof(int64_t));
```

### KVバケット化

attention長 n_kv を固定値（KV_BUCKET単位）で切り上げ → n_kv変化によるグラフ再構築を削減：

```cpp
static const int KV_BUCKET = 32;
const int want_nkv = std::min(
    ((n_past + 1 + KV_BUCKET - 1) / KV_BUCKET) * KV_BUCKET,
    n_ctx
);
if (!dgf || want_nkv != d_nkv) {
    // グラフ再構築（32トークンごと）
}
// それ以外はグラフ再利用
```

**注意**: KVキャッシュは事前にゼロ初期化が必須。バケット内の未使用スロットがNaNになるとattentionが壊れる。

---

## 3. GPU固有バグ: V cache 転置格納

### 症状

単一トークンdecode（prefill後の最初のdecode）で出力が崩壊する。

### 原因

V cacheを `[n_ctx, n_embd_gqa]`（転置）で格納すると、  
書き込み時の `ne0 = n_embd_gqa_per_token = 1` になり、  
CUDA の strided copy がこのケースに非対応。

### 対処

V cacheを **非転置 `[n_embd_gqa, n_ctx]`** で格納し、**読み出し時に転置**：

```cpp
// allocation: K同様の非転置レイアウト
v_cache[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_gqa, n_ctx);

// write: contiguous（ne0 = n_embd_gqa = 大きい値）
ggml_tensor * v_dst = ggml_view_2d(ctx, v_cache[il], n_embd_gqa, n_tokens,
                                    n_embd_gqa * sizeof(float),
                                    n_past * n_embd_gqa * sizeof(float));
ggml_build_forward_expand(gf, ggml_cpy(ctx, Vflat, v_dst));

// read: transpose → reshape
ggml_tensor * Vslice = ggml_view_2d(ctx, v_cache[il], n_embd_gqa, n_kv, ...);
ggml_tensor * Vt = ggml_cont(ctx, ggml_transpose(ctx, Vslice));  // [n_kv, n_embd_gqa]
ggml_tensor * Vc = ggml_reshape_3d(ctx, Vt, n_kv, n_embd_head, n_head_kv);
```

---

## 4. GPU固有バグ: token_embd K-quant対応

### 症状

Q2_K / Q3_K_M / IQ* 量子化モデルで、ロード直後またはfirst decode時にクラッシュ。

### 原因

CUDA の `get_rows` が K-quant / IQ 型に未対応。

### 対処

モデルロード時に F32 dequantコピーを作成し、graphではそちらを使用：

```cpp
// model.cpp: load_weights()
ggml_tensor * te = tensor("token_embd.weight");
const bool need_f32_embd = te &&
    te->type != GGML_TYPE_F32 && te->type != GGML_TYPE_F16;

if (need_f32_embd) {
    // 別contextでF32テンソルを確保
    tok_embd_rows_ = ggml_new_tensor_2d(embd_ctx_, GGML_TYPE_F32, te->ne[0], te->ne[1]);
    embd_buf_ = ggml_backend_alloc_ctx_tensors(embd_ctx_, backend);

    // ロード時にdequant
    const int64_t ne = ggml_nelements(te);
    std::vector<float> f32buf(ne);
    ggml_get_type_traits(te->type)->to_float(quantized_data, f32buf.data(), ne);
    ggml_backend_tensor_set(tok_embd_rows_, f32buf.data(), 0, ne * sizeof(float));
}

// graph構築時: model.tok_embd_rows() を使用（元のtensor()の代わりに）
ggml_tensor * embd = ggml_get_rows(ctx, model.tok_embd_rows(), inp_tokens);
```

---

## 5. カーネル数削減：ggml-cuda 既存融合の把握

### 背景

30B MoEのdecode compute ~18ms の大部分は「大量の極小GPUカーネル」の逐次起動オーバーヘッド。  
CUDA Graph有効化前の診断で判明した内容。

### ggml-cuda に既に実装されている融合（確認済み）

| 処理 | 融合内容 | 確認方法 |
|---|---|---|
| MoE routing | `ggml_top_k` → `ggml_argsort` 融合（topk-moe） | ggml-cuda.cu の `GGML_OP_TOP_K` ハンドラ |
| FFN gate+up | SwiGLU / GLU の gate×up を1カーネル化 | `ggml_mul_mat` + `ggml_gelu` 融合 |
| add連鎖 | expert出力の加算チェーン（8 add → 1カーネル） | `ggml_add` の連鎖融合 |
| rms_norm+mul | normalization後のweight乗算 | `GGML_OP_RMS_NORM` + `GGML_OP_MUL` 融合 |

**診断方法**: `GGML_CUDA_GRAPHS=OFF` でビルドし、融合ON/OFFをビルドフラグで切り替えてcompute時間を比較。  
結果: 融合ON=17ms / OFF=26ms → 融合効果はあるが、カーネル数律速は残存。

### MoE集約GEMV化の試み（効果なし）

**アイデア**: expert出力集約（8 addカーネル）を `experts_out^T @ weights` のGEMV1本に置き換え  
→ カーネル数: 8 → 2（GEMV + reshape）

**結果**: 改善なし。**理由**: add連鎖は既にggml-cudaが融合済みのため、GEMVへの置き換えは無意味。  
「融合されているかどうか」を確認してから実装すべき典型的な失敗例。

### カーネル数が律速かどうかの診断手順

1. `QWEN_PROF=1` でcompute時間を計測
2. `QWEN_NO_REUSE=1 QWEN_NO_FLASH=1` でグラフ再利用・Flash Attentionを無効化した場合と比較
3. `nsys profile` (Nsight Systems) でカーネル起動数をカウント
4. compute時間が **グラフサイズ（ノード数）に比例** して増えるならカーネル起動オーバーヘッド律速

**30B MoEの場合**: decode 1トークンで約2400カーネル。CUDA Graph有効化でこれを1回のgraph replayに圧縮 → 効果大。

---

## 6. Flash Attention

### 実装

```cpp
if (use_flash) {
    // Q: [n_embd_head, n_tokens, n_head, 1]
    ggml_tensor * Qf = ggml_permute(ctx, Q, 0, 2, 1, 3);
    // K: [n_embd_head, n_kv, n_head_kv, 1]（非転置）
    // V: [n_embd_head, n_kv, n_head_kv, 1]（非転置）
    // mask: [n_kv, n_tokens] F16 contiguous
    ggml_tensor * r = ggml_flash_attn_ext(ctx, Qf, Kc, Vc, mask_f16, scale, 0.0f, 0.0f);
    return ggml_reshape_2d(ctx, r, n_embd_head * n_head, n_tokens);
}
```

### 制約

- **mask は F16・contiguous 必須**（F32不可: `GGML_ASSERT(mask->type == GGML_TYPE_F16)` でabort）
- head dim は 64 / 128 / 256 等の対応値のみ

### 効果

30B MoEで +12%（attention カーネル数削減）。

---

## 7. 性能計測フレームワーク

### Prefill / Decode 分離

```cpp
auto t0 = now();
runtime.decode(prompt_tokens);  // prefill
double prefill_s = elapsed(t0);
int prefill_n = prompt_tokens.size();

auto t1 = now();
int decode_count = 0;
while (!done) {
    runtime.decode({next_token});
    decode_count++;
}
double decode_s = elapsed(t1);

fprintf(stderr, "prefill: %.1f tok/s  decode: %.1f tok/s\n",
        prefill_n / prefill_s, decode_count / decode_s);
```

**知見**: 30B MoEではprefill 459 tok/s vs decode 39.5 tok/s（11.6倍差）= decodeの固定オーバーヘッドが支配的

### compute内訳計測（envゲート）

```cpp
// runtime.cpp decode() 内
const bool prof = getenv("QWEN_PROF") != nullptr;
auto tb = now();
// ... build graph ...
auto ta = now();
// ... alloc ...
auto ti = now();
// ... set inputs ...
auto tc = now();
ggml_backend_graph_compute(backend, gf);
auto tr = now();
// ... read logits ...
if (prof) {
    fprintf(stderr, "[prof] build=%.2f alloc=%.2f input=%.2f compute=%.2f read=%.2f ms\n",
            ms(tb,ta), ms(ta,ti), ms(ti,tc), ms(tc,tr), ms(tr,now()));
}
```

使用方法: `QWEN_PROF=1 ./infer -m model.gguf -p "..." -n 10`

---

## 8. 実装チェックリスト

- [ ] `set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)` を CMakeLists.txt に追加
- [ ] token_embd: K-quant モデルなら F32 dequantコピーを作成
- [ ] V cache: 非転置レイアウト `[n_embd_gqa, n_ctx]` で格納
- [ ] KVキャッシュ: 使用前にゼロ初期化（NaN回避）
- [ ] グラフ再利用: `ggml_set_rows` + I64 index入力でKV書き込みを動的化
- [ ] KVバケット化: n_kv変化の頻度を制御（推奨: 32トークン単位）
- [ ] Flash Attention: mask を F16 化
- [ ] CUDA Graph診断: `QWEN_CG_DEBUG` env で warmup進行を確認
- [ ] prefill/decode分離計測: `QWEN_PROF` env で compute内訳確認

---

## 9. 教訓

1. **ggml standalone はデフォルト最適化が無効**: `GGML_CUDA_GRAPHS` 等、llama.cppがデフォルトONにしているオプションは明示指定が必要
2. **GPU固有バグはCPUで再現しない**: V cache転置問題・K-quant get_rows問題はGPUパスのみ → GPU専用テストが必須
3. **CUDA Graphの前提**: グラフトポロジーの安定性（グラフ再利用）が必須。これなしではwarmupが完了しない
4. **「遅い」の前に「動いているか」**: compute時間が改善しない場合、最適化が実際にコンパイルされているか確認する
