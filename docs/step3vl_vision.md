# step3vl vision tower (Step-3.7-Flash mmproj)

Design notes for porting the Step-3.7-Flash vision tower. Everything here is read
off the mmproj GGUF and llama.cpp's `tools/mtmd/models/step3vl.cpp`, not inferred
from the HF config -- two of the obvious inferences turned out to be wrong.

## What the mmproj contains

`mmproj-step3.7-flash-f16.gguf`, 3.97 GB, 667 tensors.

| key | value |
|---|---|
| `clip.projector_type` | `step3vl` |
| `clip.vision.block_count` | 47 |
| `clip.vision.image_size` / `patch_size` | 728 / 14 -> 52x52 patches |
| `clip.vision.embedding_length` | 1536 |
| `clip.vision.feed_forward_length` | 8960 |
| `clip.vision.attention.head_count` | 16 (head dim 96) |
| `clip.vision.projection_dim` | 4096 (= the LLM's n_embd) |
| `clip.vision.projector.scale_factor` | 4 (two stride-2 convolutions) |
| `clip.vision.preproc_image_size` | 3024 |
| `clip.vision.image_mean` / `image_std` | CLIP's, stored in the file |

Tensors: `v.patch_embd` [14,14,3,1536], `v.pre_ln`, `v.position_embd`
[1536, **2704**] (= 52x52, no CLS token), 47 blocks of fused
`attn_qkv` [1536,4608] / `attn_out` / `ln1` / `ln2` / **`ls1`,`ls2`** / `ffn_up`
[1536,8960] / `ffn_down`, and no `post_ln`.

Projector: `mm.0.weight` [3,3,1536,3072], `mm.1.weight` [3,3,3072,6144],
`mm.model.fc.weight` [6144,4096].

## Two corrections to the obvious reading

**There is a 2D RoPE, on top of the learned position embedding.** The learned
`v.position_embd` is added to the patch embedding *and* every layer applies a 2D
rotary to Q/K (`build_rope_2d(cur, pos_w, pos_h, rope_theta, false)`), with
`rope_theta = 10000` hard-coded in llama.cpp -- it is not in the GGUF. Reading
only the tensor list suggests a plain learned-position ViT, which would be wrong.

**No activation between the two projector convolutions.** conv -> add bias ->
conv -> add bias -> flatten -> linear. An activation in between is the natural
guess and would be wrong.

52 -> 26 -> 13 with k=3 s=2 p=1 each, so 13x13 = 169 = the model's
`image_token_len`, which is the check that the strides are right.

## Where this differs from the Qwen3-VL tower already in core/vision.cpp

| | Qwen3-VL (existing) | step3vl |
|---|---|---|
| position | vision M-RoPE only | learned embedding **and** 2D RoPE |
| norms | `post_ln` after blocks | `pre_ln` before blocks, no post |
| residual | plain | **LayerScale** (`ls1`, `ls2`) |
| token order | 2x2-block contiguous (reordered on load) | plain row-major |
| merge | reshape 2x2 -> 2-layer MLP | two stride-2 conv2d -> linear |

The ViT block loop, the fused-QKV attention and the weight-loading scaffolding
carry over. The position handling, the projector and the token ordering do not.

## Open question

llama.cpp leaves `ffn_op` at its default `FFN_GELU` for step3vl, while the HF
config says the encoder's activation is `quick_gelu` (and `FFN_GELU_QUICK`
exists in the same enum). Matching llama.cpp is the right call for a port that
is verified against llama.cpp, but the discrepancy is worth resolving against
the reference implementation before trusting image quality.

## Scope

Start with a single 728x728 image and no tiling. `preproc_image_size = 3024`
means large images are sliced on a fixed window grid (`patch_token_len = 81`
= 9x9 in the HF config suggests the tile path), and that preprocessing is a
separate piece of work from the tower itself.
