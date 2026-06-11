# 画像入力（Qwen3-VL / mmproj）対応計画

検証済みの Java 参照実装（`qwenjava/Mmproj.java`, `VISION_PLAN.md`）を仕様書として、
qwencpp に画像入力を実装する。Java 版と違い ViT forward は **ggml グラフ（GPU）** で実装する
（Java CPU 版は ViT 19s → GPU なら 1s 未満の見込み）。

- ターゲット: `D:\dev\gguf\unsloth\Qwen3.5-0.8B-GGUF`（`Qwen3.5-0.8B-Q8_0.gguf` + `mmproj-F32.gguf`）
- テスト画像: `C:\Users\naoki\Desktop\09ptRcFn.jpg large.jpg`
- チャットテンプレートは jinja を vendor せず、**Qwen テンプレート等価ロジックを直接実装**する
  （qwencpp は Qwen 専用エンジンなので汎用 jinja は不要、という判断。nlohmann/json はサーバーで使用済み）

## アーキテクチャ要点（Java 版で検証済みの事実）

- mmproj は `arch=clip`, `clip.projector_type=qwen3vl_merger`。768×768 固定、patch 16
  → 48×48 パッチ → ViT（24層, LayerNorm+bias, GELU 2層FFN, qkv bias 付き, QK-norm なし）
  → post_ln → 2×2 spatial merge（並べ替えのみ）→ mm.0/mm.2 の GELU MLP → **576 トークン × LLM dim**
- patch_embd は `weight` と `weight.1`（動画フレーム用）を**両方適用して加算**
- 前処理: リサイズ 768×768、`(pixel/255 - 0.5) / 0.5` 正規化（ImageNet 正規化ではない）
- ViT 側 M-RoPE: NeoX ペア, n_dims = d_head/2, セクション0=py / セクション1=px,
  theta_scale = `10000^(-4/d_head)`, セクション毎に theta リセット
- LLM 側: `<|vision_start|>` + `<|image_pad|>`×576 + `<|vision_end|>` を展開し、
  `<|image_pad|>` 位置の埋め込みを ViT 出力で差し替える。
  **1D 連番位置 ID でも実用品質で動く**（Java Phase 5a で確認）。LLM 側 M-RoPE
  （`rope.dimension_sections`, qwen35 は [11,11,10,0]）は精度向上の後続改良。
- deepstack は全 false（不要）

## 実装フェーズ

### V0: チャットテンプレート刷新（vision の前提、tool の土台）
- `core/chat.h` を Qwen テンプレート等価に拡張:
  - system プロンプトの既定値、ツール定義の注入（`<tools>` JSON 形式）、
    `<|vision_start|>/<|image_pad|>/<|vision_end|>` プレースホルダ展開
- メッセージ構造を `content` 文字列から「テキスト/画像パーツの配列」に拡張

### V1: 画像デコード + 前処理
- `stb_image.h`（public domain, 単一ヘッダ）を vendor して JPEG/PNG デコード
- リサイズ（bilinear で 768×768、まずはアスペクト無視の単純リサイズ）+ 正規化
- 出力: `float[3*768*768]`（CHW）

### V2: mmproj ロード + ViT forward（ggml/GPU）
- `core/vision.{h,cpp}`: mmproj GGUF を読む（既存 gguf_util 流用; arch=clip 用に
  ハイパラ読みだけ別系統）
- ggml グラフで ViT forward を構築（LayerNorm/GELU/conv は ggml 標準オペ;
  patch embedding は im2col 済み入力への matmul として実装すると簡単）
- ViT M-RoPE は `ggml_rope_multi(GGML_ROPE_TYPE_VISION)` か、Java 版と同じ
  事前計算 cos/sin の乗算で実装
- 検証: Java 版と同一画像で `imageEmbeddings` の出力を突き合わせ（RMS / cos類似度）

### V3: LLM 統合（非オフロード経路から）
- `Runtime::decode` に「トークン列 + 位置別埋め込み差し替え」入力を追加
  （build_graph の get_rows 結果の該当列を入力テンソルで上書き、または
   inp_embd 入力に一本化）
- 位置 ID はまず 1D 連番（Phase 5a 相当）
- CLI: `--image <path>`（複数可）、`--mmproj <path>`（省略時はモデルと同じディレクトリの
  `mmproj-*.gguf` を自動発見）
- 検証: テスト画像で内容説明が返ること、Java 版と応答傾向が一致すること

### V4: サーバー対応
- `/v1/chat/completions` の `content: [{type:"text"},{type:"image_url"}]` 形式
  （base64 data URI）を受理
- ブラウザ UI に画像添付

### V5（後続・任意）
- LLM 側 M-RoPE（`rope.dimension_sections` を読み `ggml_rope_multi` に切替。
  テキストのみの場合は従来 rope と数値一致するため回帰なしで入れられる）
- 動的解像度（`image_grid_thw`、position embedding の bicubic 補間）
- tool calling（V0 のテンプレート基盤の上に、`<tool_call>` パース + OpenAI 形式変換）

## 検証方法

- V2 の数値検証は Java 版を oracle にする（同一画像 → 576×dim 出力比較）。
  Java 側は `CheckViTFull.java` 等の検証ハーネスが既にある
- V3 は 0.8B（全量 VRAM 常駐、CUDA/Metal 両方で軽い）で end-to-end
