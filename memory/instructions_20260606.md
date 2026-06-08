# 指示記録 2026-06-06

1. Qwen3/3.5専用の実行エンジンを、ggmlを使ってllama.cppを参考につくりたい。やりたいことはspec.mdにまとめられている。まずは実装計画をたててまとめてほしい。llama.cppのソースは D:\dev\llama.cpp、使えるものはなるべく使いたい。Javaで実装したQwen3.5実行エンジンは C:\Users\naoki\Desktop\qwenjava（gated deltanet・トークナイザ・ggufリーダー実装済み）。
   → jissou_keikaku.md に実装計画を作成。

2. 流用することで飛ばせるステップはあるか。qwen3/next/3.5 の3アーキテクチャに対応したい。
   → 調査の結果：3アーキ(qwen3/qwen3next/qwen35)は全て llama.cpp の llm_build_delta_net_base を共有しGDN実装済み。GDN数値立ち上げ・量子化dequant・トークナイザ/サンプラ開発は流用で省略可。Step0-1は「3アーキ同時立ち上げ+ツール完成(Phase A)」に圧縮し、Expert Cache(Step2-4)に集中。
   確定方針：戦略B=B2ハイブリッド（llm_graph_context全体は移植せず、GDN/グラフ式は写経、MoE/mul_mat_id周りは自前ヘルパで制御）／Win+CUDAとmacOS+Metalを並行／トークナイザ・サンプラはllama.cpp移植。jissou_keikaku.md を圧縮版Step構成に更新。

3. ソースコードはこのフォルダで完結させたい。D:\dev\llama.cpp への依存をなくしたい。
   → ggml/ サブツリーのみ third_party/ggml に vendor（スナップショット、submodule不使用）。GDN特殊op(gated_delta_net/solve_tri/ssm_conv/cumsum/tri/softplus)はCUDA・Metalともggml/内で完結することを確認。D:\dev\llama.cppはビルド/実行依存にしない（写経の参照元と--compare-ref検証のみ）。jissou_keikaku.md §1.5 に依存方針を明記。

4. Phase Aまで実装する。
   → Phase 0完了 + Phase A qwen3 Denseスライス完成・検証済み。
   - ビルド環境：cmake/ninjaはVS2022同梱（C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\）。CUDA 12.8あり。現状CPUビルド。DLLはbuild\bin\Release\からbuild\Release\へコピーが必要。
   - 実装済：core/gguf_util, core/model(重みロード), core/runtime(qwen3 forward+KVキャッシュ+greedy), core/tokenizer(GPT2 BPE), cli/main.cpp。
   - 検証済：Qwen3-0.6B Q8_0/Q4_K_M で「The capital of France is」→「Paris...」正常生成、exit 0。
   - 注意点メモ：(a) qwencpp_coreはSTATIC指定必須(ggmlがBUILD_SHARED_LIBS=ON), (b) weights_bufはModelが所有しRuntimeで二重freeしない, (c) V cacheは転置格納しview_3dで参照(reshape不可=非contiguous)。
   - Phase A残り：qwen3next(GDN+MoE), サンプラ(temp/top-p), -i対話, Server(OpenAI/SSE), --compare-ref, CUDA有効化。

5. qwen35を実装する（重み: D:\dev\gguf\unsloth\Qwen3.5-0.8B-GGUF）。
   → 完成・検証済み。core/runtime.cpp に GDN層(融合op ggml_gated_delta_net + causal conv1d + recurrent state) と gated attention(gate sigmoid + partial RoPE n_rot=64 + post_attention_norm残差) を実装。HParamsに ssm_dt_rank/n_rot/full_attn_interval/nextn_predict_layers と is_recurrent() 追加。
   - qwen35テンソル名: attn_qkv(6144), attn_gate(2048=z), ssm_conv1d[4,6144], ssm_dt.bias[16], ssm_a[16], ssm_alpha/beta[1024,16], ssm_norm[128], ssm_out[2048,1024], post_attention_norm(ffn_norm相当)。attn層: attn_q[1024,4096]=query+gate per head(stride2), attn_k/v[1024,512]。
   - GDN次元(0.8B): d_state=128, n_group(H_k)=16, dt_rank(H_v)=16, d_inner=2048, conv_ch=6144。fused opが AR/chunk両対応で全移植不要。
   - 検証:「Once upon a time」→「...superatom...」高品質。Q8_0/Q4_K_M OK。
   - 残り: qwen3next(GDN+MoE), サンプラ, -i, Server, --compare-ref, CUDA。

6. まずはツールの実装を。
   → サンプラ・対話・サーバ完成。core/sampler.cpp(greedy/temp/top-k/top-p/seed), core/chat.h(ChatML), cli更新(--chat/-i/--temp/--top-p/--top-k/--seed/--log-tokens-per-sec), server/main.cpp(OpenAI互換 /health,/v1/models,/v1/chat/completions SSE+非SSE)。
   - 依存vendor: third_party/cpp-httplib(httplib.h + httplib.cpp ※split版なので.cppもビルド必須), third_party/nlohmann/json.hpp。infer-serverはws2_32リンク(WIN32)。
   - 検証: --chatでthinkingモード「Red/Green/Blue」正答。サーバSSEチャンク(OpenAI形式)正常、health/models OK。
   - 残り: qwen3next(GDN+MoE), --compare-ref, CUDA有効化。

8. GPU対応＋性能最適化。
   - GPUバックエンド実装(ggml_backend_dev_by_type GPU選択, RuntimeConfig.use_cuda既定true, --cpu強制)。CUDA 12.8でggml-cudaビルド(arch89/86)。GPU=RTX4060Ti(16GB)選択。
   - GPU固有バグ2件修正: (a)q2_K等のtoken_embdはCUDA get_rows非対応→F32にdequant(Model::tok_embd_rows), (b)転置V cacheのne0==1書き込みがCUDAで壊れる(単一decode崩壊)→V cache非転置格納、読み出し時に転置。
   - 性能: llama.cpp比でdense×1.2, MoE×2.7と差。計測→prefill 459 vs decode 39.5 tok/s(11.6倍)=decode固定オーバーヘッド支配。内訳: build1.4+alloc1.1+compute18.5ms。compute=約2400極小カーネルのGPU側逐次実行が主。
   - グラフ再利用実装(decode_reuse): persistentグラフ1回構築・再利用、KV書込はggml_set_rows+index入力(I64)で動的化、n_kvはKV_BUCKET=256でバケット化(再利用維持＋attention実長比例)。KVキャッシュゼロ初期化(mask位置NaN回避)。→30B MoE 39.5→54.5 tok/s(+38%)。build=0達成。
   - 判明: CUDA Graphは2GPでも無効化されない(arch条件のみ)。残compute律速はカーネル数(約2400個/トークン、特にMoEのrouter/mul_mat_id/集約)。
   - Flash Attention実装: build_attnにuse_flash経路(ggml_flash_attn_ext, q=[d,nt,nh]/k,v=[d,nkv,nhkv]非転置/mask F16)。maskを全てF16化(manual softmaxもF16対応)。use_flash=cfg.use_cuda(QWEN_NO_FLASHで無効)。head128(qwen3/moe)・head256(qwen35)とも動作。
   - 性能まとめ(GPU, RTX4060Ti): 0.6B dense 56→56.8; qwen35 0.8B GDN 56→94.9; 30B-A3B MoE 39.5→61.1 tok/s(+55%)。出力は全て正常、CPU経路も回帰なし。
   - 残差(30B 61 vs llama.cpp 100)はMoEカーネル数。今後: op融合/MoE集約のカーネル削減、またはバッチ。

9. 性能をさらに最適化（キャッシュ効果測定の前提として）。
   - 融合調査: ggml-cuda融合は既発火(topk-moe/gate-up-swiglu GLU/add連鎖/rms_norm+mul)。融合on/off=17/26ms→カーネルオーバーヘッド律速確定。
   - MoE集約GEMV化(experts^T@weights, 8→2カーネル)→効果なし(add連鎖は既融合)。
   - CUDA Graph検証: reuse compute15.8 vs rebuild14.1ms→CUDA Graphは効いていない(GPU側per-kernel律速、CPU起動削減が無効)。reuse利点=build/alloc削減のみ。env QWEN_NO_REUSE/QWEN_NO_FLASH/QWEN_PROFで切替可。
   - KV_BUCKET 256→32(runtime.cpp)。dense/GDNのattention無駄削減。
   - 最終(GPU bucket32): 0.6B ~81, qwen35 0.8B ~82, 30B-A3B MoE ~58 tok/s。全て正常出力。
   - 結論: attention(flash)・router(融合)・集約(融合)は最適化済み。残差はMoE mul_mat_id(Q2_K GEMV×3×48層)のカーネル数/効率でggml内部領域。キャッシュ効果測定に進める状態。

10. サイドチャットの方針でCUDA Graph診断（デバッグビルドで計装）。
   - ggml-cuda.cu に env(QWEN_CG_DEBUG)計装3箇所(check_compability disable理由, メイン判定enabled/compatible/props_changed)+入口cg.logファイル直書き。
   - 【真因】[cg]ログで「compute関数は呼ばれるが #ifdef USE_CUDA_GRAPH 内のenabled行が出ない」→ USE_CUDA_GRAPH未定義と判明。common.cuh:1203 で GGML_CUDA_USE_GRAPHS 依存、それは CMake option GGML_CUDA_GRAPHS(既定OFF, "llama.cpp only")依存。**CUDA Graphがビルドから丸ごと欠落していた**(先の"GPU律速で効かない"は誤り)。
   - 対処: CMakeLists.txt で QWENCPP_CUDA時に set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)。フル再コンパイル(全.cu)。
   - 検証: [cg]ログで props_changed 1→0→warmup=1(CUDA Graph再生開始)を確認。decode compute 14→9.2ms。
   - 【最終性能 GPU RTX4060Ti, CUDA Graph有効】0.6B dense ~196, qwen35 0.8B GDN ~154, 30B-A3B MoE ~91 tok/s(元39.5)。llama.cpp(~100)にほぼ到達。出力全正常・CPU回帰なし。
   - 計装(env QWEN_CG_DEBUG の[cg]print)はggml-cuda.cuに残置(env無時ノーオーバーヘッド)。cg.log入口プローブは撤去。
   - 教訓: ggmlを add_subdirectory で取り込む場合、CUDA Graph等 llama.cpp側で有効化される最適化は明示ONが必要。

7. qwen3moe、qwen35moeの実装を。
   → 両方完成・検証済み。core/runtime.cpp に build_moe(softmax gating + ggml_argsort_top_k + ggml_mul_mat_id ×gate/up/down + 正規化weight + expert集約) を実装し is_moe()時にdense FFNと差し替え。shared expert(qwen35moe: ffn_*_shexp を sigmoid gate付きSwiGLUで加算)対応。HParamsに expert_weights_scale 追加。
   - MoEテンソル名: ffn_gate_inp[n_embd,n_expert](router), ffn_{gate,up}_exps[n_embd,ff_exp,n_expert], ffn_down_exps[ff_exp,n_embd,n_expert]。shared: ffn_{gate,up,down}_shexp + ffn_gate_inp_shexp[n_embd]。
   - GDNのH_k≠H_v(qwen35moe: n_group=16/dt_rank=32)は融合opが内部GQA処理(repeat不要)。
   - 検証(D:\dev\gguf): qwen3moe=Qwen3-30B-A3B-Instruct-2507 Q2_K(10.5GB)→「Paris」6.7tok/s; qwen35moe=Qwen3.6-35B-A3B-UD Q3_K_S(14.3GB)→「Paris, a city renowned...」5.7tok/s。CPU。
   - ※mul_mat_id がExpert Cache Managerの接続点。対応アーキ4種完成(qwen3/qwen35/qwen3moe/qwen35moe)、残りqwen3next。
   - 注意: 30B/35Bロードは10-15GB RAM要、実行中はinfer.exeロックされ再ビルド不可(先にStop-Process)。

8. GPU対応やって（残タスクの扱いが軽すぎたと指摘あり）。
   → GPU(CUDA)バックエンド有効化・全アーキでCPU一致・検証完了。
   - 実装: core/runtime.cpp で ggml_backend_dev_by_type(GPU)→dev_init、無ければCPU。RuntimeConfig.use_cuda既定true、CLI/Serverに--cpu。CMake -DQWENCPP_CUDA=ON で ggml-cuda ビルド(CUDA 12.8, arch 89/86=4060Ti/3050)。
   - ビルド注意: ggml-cudaのカーネルコンパイルは重い(mmq/fattn等で5-15分)。バックグラウンドタスクは途中で止まることがある→明示timeout 600000で実行。DLLは build\bin\Release\*.dll を build\Release\ へコピー。
   - **GPU固有バグ2件を修正(重要)**:
     (1) CUDAの get_rows が q2_K/K-quant未対応 → Model::load_weights で token_embd を to_float でF32化(embd_ctx_/embd_buf_/tok_embd_rows_)、graphは model.tok_embd_rows() を使用。
     (2) 転置V cache(旧[n_ctx,n_embd_gqa])の ne0==1 cpy書き込みがCUDAで壊れる→単一トークンdecodeで出力崩壊。V cacheを非転置[n_embd_gqa,n_ctx]に変更しK同様contiguous書き込み、読み出し時に transpose+cont+reshape。
   - デバッグ手法: CLIに一時CMP:診断(CPU/GPU並走でargmax軌跡比較、CMP_REPREFILL/CMP_SPLIT env)を入れて、prefill OK・n_tokens=1 decodeで乖離・cross-graph KVは無実、と切り分け→転置V書き込みを特定。診断コードは除去済み。
   - 速度(GPU=4060Ti, 全てCPUと一致): 0.6B 52, 0.8B GDN 56, 30B-A3B MoE Q2_K 36.5 tok/s(CPU比5.4x)。
   - Phase A残: qwen3next, --compare-ref(llama-cli), マルチGPU自動レイアウト。その後Step2 Expert Cache。
