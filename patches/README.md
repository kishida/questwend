# third_party/ggml に載っているパッチ

`third_party/ggml` は upstream の [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
の `ggml/` を vendor したもの（コミットは `third_party/ggml/GGML_VERSION`）に、
以下を適用した状態です。どちらも **qwencpp 固有**で、upstream に入る見込みは無いため
再 vendor のたびに再適用が必要です。

| # | パッチ | 内容 |
|---|---|---|
| 01 | `01-qwencpp-metal-mmid-blocked.patch` | Metal `mm_id_map0` を expert ブロック単位ループに変更。expert キャッシュのスロットプールは `ne02` が threadgroup のスレッド上限を超えるため、upstream の `GGML_ASSERT(ne02 <= max_threads)` では動かない。`QWEN_METAL_MMID_MV=1` で mat-vec 経路に強制切替する A/B スイッチも含む |
| 02 | `02-qwencpp-cuda-quiet-graph-logs.patch` | CUDA graph の `GGML_LOG_DEBUG` 3 行を削除（グラフキーごとに毎回出て自前のログを埋める） |

> サブ 1bit 量子化（`IQ1_XS` / `IQ1_XXS` / `IQ1_XXXS`）は **`iq1-narrow` ブランチ**に分けてあります。
> 本家 llama.cpp に取り込まれたら、そのブランチを捨てて通常の再 vendor で拾えば済みます。

## 再 vendor の手順

```bash
# 1. upstream を取得（例: ローカルの llama.cpp クローンから）
cd /path/to/llama.cpp && git fetch origin master

# 2. ggml/ を差し替え（古い vendor の残骸を消すため必ず rm -rf してから展開する）
cd /path/to/qwencpp
rm -rf third_party/ggml
(cd /path/to/llama.cpp && git archive origin/master ggml) | tar -x -C third_party
echo "Vendored from https://github.com/ggml-org/llama.cpp ggml/ at commit $(git -C /path/to/llama.cpp rev-parse origin/master)" \
  > third_party/ggml/GGML_VERSION

# 3. パッチを番号順に適用（リポジトリルートから、--directory に注意）
for p in patches/0*.patch; do git apply --directory=third_party -p1 "$p" || echo "CONFLICT: $p"; done
```

適用後は `patches/` を再生成して「upstream + パッチ = 現ツリー」を保つこと
（upstream の変更で当たらなくなった場合は手で当て直してから再生成）。

### 落とし穴

- **`git apply` はリポジトリ内のサブディレクトリから実行すると、パッチのパスを
  リポジトリルート基準で解釈し、カレント外のパスを"成功"扱いで黙って読み飛ばす**。
  `cd third_party && git apply -p1 …` は何もせず exit 0 になる。必ず
  **リポジトリルートから `--directory=third_party`** を使うこと（上記手順のとおり）。
- **`rm -rf third_party/ggml` を省いて上書き展開すると、upstream で削除された
  ファイルが残る**（`fattn-wmma-f16.cu` など）。ビルドは通ってしまうことがあるので厄介。
- ggml の DLL は `build/bin/<config>/` に出るが exe は `build/<config>/` に出る。
  exe の隣に古い DLL が残っていると**それが優先されて古い ggml で動く**（実際に踏んだ:
  ソースと合わない行番号の assert が出る）。ルート `CMakeLists.txt` の POST_BUILD
  コピーで同期しているので、手でコピーした DLL を置かないこと。
- upstream の API 変更は qwencpp 側に波及する。今回は `ggml_gated_delta_net` の
  state 入力が平坦な `[S*S*H_v, K, n_seqs]` から初期状態のみの 4D
  `[S, S, H_v, n_seqs]` になり、スナップショット数 `K` が明示引数になった。
