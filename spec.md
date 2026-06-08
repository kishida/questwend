# MoE推論エンジン 仕様書

## 概要

Qwen3.5系MoEモデルを、コンシューマーグレードのハードウェアで実用的な速度で動かすための推論エンジン。Expert weightの3段階キャッシュ管理と統計ベースのプリフェッチにより、VRAMとRAMの制約を超えて大規模モデルを動作させることを目的とする。

---

## ターゲットモデル

| モデル | 総パラメータ | アクティブ | Expert数 | アーキテクチャ |
|--------|------------|----------|---------|--------------|
| Qwen3-8B | 8B | 8B | - | Dense、GDNなし |
| Qwen3.5-9B | 9B | 9B | - | Dense + GDN |
| Qwen3.5-35B-A3B | 35B | 3B | 256 | GDN + MoE |
| Qwen3.5-122B-A10B | 122B | 10B | 256 | GDN + MoE |

最終目標: Qwen3.5-122B-A10B Q3KM を 8tok/sec 程度で動作させる

---

## 対象ハードウェア

### 推論ターゲット
- VRAM: 16GB（シングル）、またはマルチGPU
- RAM: 32-40GB（実効）
- SSD: NVMe（7GB/s read）

### 対応プラットフォーム
- Linux / Windows（CUDA）
- macOS（Metal、Apple Silicon）

---

## マルチGPU

### GPU構成による自動レイアウト選択

GPU性能差に応じて最適な配置戦略を自動判定する。

```
判定基準:
  各GPUのFP16 TFLOPSを取得（cudaDevicePropから推定）
  min_score / max_score > 0.8 → 同等とみなす

同等性能GPU（例: 4060 × 2）→ LayerSplit
  GPU0: Layer 0-23  + 高頻度Expert
  GPU1: Layer 24-47 + 高頻度Expert
  層間のhidden state転送: ~8KB/token（無視できる）
  各GPUが独立してExpertを管理、GPU間Expert転送なし

性能差ありGPU（例: 4060 + 3050）→ ExpertPool
  GPU0(高性能): Attention/DeltaNet + 高頻度Expert（計算専用）
  GPU1(低性能): Expertプール専用（VRAMの拡張として使用）
  遅いGPUに計算させない
  GPU1のExpertが必要な場合はGPU0へ転送して計算
```

### GpuLayout

| レイアウト | 条件 | 説明 |
|-----------|------|------|
| Single | GPU1枚 | デフォルト |
| LayerSplit | 同等性能GPU複数 | レイヤー均等分割 |
| ExpertPool | 性能差ありGPU複数 | メイン + Expertプール |

### InferConfig への追加

```cpp
struct GpuConfig {
    int   device_id;
    GpuRole role = GpuRole::Auto;  // Auto / Primary / ExpertPool
};

// GpuLayout
enum class GpuLayout { Auto, Single, LayerSplit, ExpertPool };

struct InferConfig {
    // ...既存フィールド...
    std::vector<GpuConfig> gpus       = {{0}};      // デフォルトGPU0のみ
    GpuLayout              gpu_layout = GpuLayout::Auto;
};
```

---

## アーキテクチャ

### レイヤー構成

```
[CLI / Server]          ← Step0で完成、以降変更なし
      │
[Engine API]            ← 外部インターフェース（pimpl）
      │
[推論コア]
  ├── Tokenizer
  ├── Forward Pass
  │     ├── Attention（GQA + RoPE）
  │     ├── Gated DeltaNet（Qwen3.5系）
  │     ├── FFN（SwiGLU）
  │     └── MoE Dispatch
  ├── Expert Cache Manager  ← 本エンジンの核心
  ├── Sampler
  └── Stats Collector
      │
[バックエンド]
  ├── ggml-cuda（Linux/Windows）
  └── ggml-metal（macOS）
```

---

## Expert Cache Manager

### 3段キャッシュ構成

```
Stage 1: VRAM / Metal GPU
  容量: ~5-7GB（Attention常駐分を除く）
  対象: 現在計算中 + 高頻度expert
  転送コスト: ゼロ（Metal Unified Memory）
              / PCIe DMA（CUDA）

Stage 2: RAM
  容量: 実効20-32GB
  対象: 統計上位のexpert群
  転送: cudaMallocHost（CUDA pinned）
        malloc（Metal、Unified Memoryのため不要）

Stage 3: SSD
  容量: モデル全体（mmap）
  対象: 低頻度expert、コールドスタート
  転送: pread + スレッドプール（共通）
        io_uring（Linux限定の追加最適化）
```

### キャッシュ戦略

- **静的ピン留め**: 統計上位N expertをRAMに常駐
- **LRU**: VRAMスロットの動的管理
- **Markovプリフェッチ**: expert遷移確率から次tokenのexpertを予測、非同期転送
- **オンライン学習**: セッション中も統計を更新、指数移動平均で適応

### プラットフォーム差分

```cpp
void* alloc_host_memory(size_t size) {
#ifdef GGML_USE_CUDA
    void* ptr;
    cudaMallocHost(&ptr, size);  // pinned memory
    return ptr;
#else
    return malloc(size);          // Metal: Unified Memoryなので通常malloc
#endif
}
```

---

## Gated DeltaNet

Qwen3.5のハイブリッドアーキテクチャ（3:1比）：

```
12ブロック × 4層:
  3層: Gated DeltaNet → MoE
  1層: Gated Attention → MoE

更新則: S = g·S + k⊗β(v − g·S^T k)
出力:   o = q^T S
```

**KVキャッシュへの影響**:
- DeltaNet層（75%）はKVキャッシュ不要、固定サイズの状態行列のみ
- Attention層（25%）のみKVキャッシュ使用
- 結果としてVRAMのKVキャッシュ消費が約1/4に削減

**実装方針**: llama.cppのggml-metal / ggml-cudaの実装を流用

---

## Engine API

```cpp
// core/engine.h  ← CLIもServerもこれだけ使う

enum class GpuLayout { Auto, Single, LayerSplit, ExpertPool };

struct GpuConfig {
    int     device_id;
    GpuRole role = GpuRole::Auto;  // Auto / Primary / ExpertPool
};

struct InferConfig {
    std::string model_path;
    int   n_gpu_layers   = -1;   // -1=auto
    int   n_ctx          = 4096;
    int   n_expert_gpu   = -1;   // VRAMに常駐するexpert数（-1=auto）
    int   n_expert_ram   = -1;   // RAMに常駐するexpert数（-1=auto）
    std::string stats_path = ""; // 統計ファイルのパス

    // マルチGPU
    std::vector<GpuConfig> gpus       = {{0}};
    GpuLayout              gpu_layout = GpuLayout::Auto;
};

struct GenerateConfig {
    float temperature = 1.0f;
    float top_p       = 0.9f;
    int   max_tokens  = 512;
    bool  stream      = false;
};

struct Message {
    std::string role;     // "system" / "user" / "assistant"
    std::string content;
};

class Engine {
public:
    explicit Engine(const InferConfig&);
    ~Engine();

    void generate(
        const std::vector<Message>& messages,
        const GenerateConfig&,
        std::function<void(std::string_view token)> on_token,
        std::function<void()> on_done
    );

    ModelInfo info()  const;
    Stats     stats() const;  // expert統計、速度統計

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

---

## CLIツール

```bash
# 基本
./infer -m model.gguf -p "こんにちは"

# インタラクティブ
./infer -m model.gguf -i

# Expert統計付き
./infer -m model.gguf -p "test" --log-experts

# 速度計測
./infer -m model.gguf -p "test" --log-tokens-per-sec

# llama.cppと出力比較（開発用）
./infer -m model.gguf -p "test" --compare-ref /path/to/llama-cli

# Expert cache設定
./infer -m model.gguf -i \
  --n-expert-gpu 40 \
  --n-expert-ram 137 \
  --stats-path ./expert_stats.json

# マルチGPU（自動レイアウト判定）
./infer -m model.gguf -i --gpu-ids 0,1

# マルチGPU（レイアウト明示指定）
./infer -m model.gguf -i --gpu-ids 0,1 --gpu-layout layer-split
./infer -m model.gguf -i --gpu-ids 0,1 --gpu-layout expert-pool

# シングルGPU指定
./infer -m model.gguf -i --gpu-id 1
```

### オプション一覧

| オプション | デフォルト | 説明 |
|-----------|----------|------|
| `-m` | 必須 | モデルファイル（gguf） |
| `-p` | - | プロンプト（非インタラクティブ） |
| `-i` | - | インタラクティブモード |
| `--n-ctx` | 4096 | コンテキスト長 |
| `--n-expert-gpu` | auto | VRAMに常駐するexpert数。統計上位N個をGPUに固定、残りはRAM/SSDから動的ロード |
| `--n-expert-ram` | auto | RAMにピン留めするexpert数。統計上位N個を起動時にRAM展開。GPU常駐分と合算してカバレッジを決定 |
| `--stats-path` | - | expert統計ファイルのパス。起動時に読み込んでキャッシュ配置を最適化。未指定時はLRUのみ |
| `--gpu-id` | 0 | 使用するGPU（シングル指定） |
| `--gpu-ids` | 0 | 使用するGPUリスト（カンマ区切り） |
| `--gpu-layout` | auto | `auto` / `layer-split` / `expert-pool` |
| `--log-experts` | - | expert使用統計をリアルタイム表示 |
| `--log-tokens-per-sec` | - | 速度計測 |
| `--compare-ref` | - | 指定したllama-cliと出力を比較（開発用） |

---

## Serverツール

OpenAI互換API

```bash
./infer-server -m model.gguf --port 8080
```

### エンドポイント

| Method | Path | 説明 |
|--------|------|------|
| POST | /v1/chat/completions | メイン推論（streaming対応） |
| GET  | /v1/models | モデル情報 |
| GET  | /health | ヘルスチェック |

ストリーミングはSSE（Server-Sent Events）

---

## 統計収集

```json
{
  "expert_hit_count": {
    "layer_0": {"expert_3": 1523, "expert_7": 1401, ...},
    "layer_1": {...}
  },
  "transition": {
    "layer_0": {
      "expert_3": {"expert_7": 0.62, "expert_2": 0.28, ...}
    }
  },
  "session_tokens": 10000,
  "cache_hit_rate": {
    "gpu": 0.42,
    "ram": 0.51,
    "ssd": 0.07
  }
}
```

---

## ディレクトリ構成

```
project/
  core/
    engine.h / engine.cpp       ← 外部API
    model.h / model.cpp         ← モデルロード
    forward.h / forward.cpp     ← forward pass
    deltanet.h / deltanet.cpp   ← Gated DeltaNet
    cache.h / cache.cpp         ← Expert cache manager
    sampler.h / sampler.cpp     ← サンプリング
    stats.h / stats.cpp         ← 統計収集
    ssd.h / ssd.cpp             ← SSDバックエンド
  cli/
    main.cpp
  server/
    main.cpp
    handlers.cpp                ← OpenAI互換エンドポイント
  third_party/
    ggml/                       ← submodule
    cpp-httplib/                ← submodule
  CMakeLists.txt
```

---

## 実装ステップ

### Step 0: Qwen3-8B Dense（GDNなし）
- 目的: 推論コアとツールの完成
- 成果物: CLI・Server完成、llama.cppとの出力比較でコア正当性確認
- ツールはここで完成、以降変更なし

### Step 1: Qwen3.5-9B Dense
- 目的: Gated DeltaNet forward の追加
- 確認: KVキャッシュ削減の実測、Step0との速度比較

### Step 2: Qwen3.5-35B-A3B（SSDなし）
- 目的: Expert Cache Managerの基本動作確認
- RAM全量使用、SSD不使用
- expert使用分布の統計取得
- Markovプリフェッチの実装・効果測定

### Step 3: 35B-A3B RAM制限
- 目的: SSD tierの動作確認
- RAMを意図的に制限してSSDアクセスを発生させる
- プリフェッチ精度・ヒット率の測定
- 122Bでのヒット率予測

### Step 4: Qwen3.5-122B-A10B
- 目的: 最終目標
- Q3KM量子化
- 目標: 8tok/sec（楽観）、4-6tok/sec（現実的）

---

## 数値目標（122B Q3KM）

```
モデルサイズ:  ~53GB
Expert 1個:   ~175MB（概算、要実測）
256 experts:  ~45GB

VRAM 16GB配分:
  非Expert常駐:   8GB
  KVキャッシュ:   1GB（DeltaNet効果で削減済み）
  Expertスロット: 7GB → 約40 expert

RAM 32GB配分:
  Expert用:      24GB → 約137 expert
  合計カバー:    177/256 → 約69%

目標キャッシュヒット率: 70%以上（統計プリフェッチ込み）
目標速度: 8tok/sec
```
