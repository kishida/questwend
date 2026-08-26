#!/usr/bin/env python3
"""Generate a tiny randomly-initialised `qwen4exp` GGUF.

The real Qwen3.8-Flash-Next is 67.55 GiB, which makes every graph-shape
iteration a multi-minute read off a spinning disk. This writes a model with the
same *structure* -- hyper-connections, the 3-in-4 Gated DeltaNet pattern, the
QSA indexer, the MoE with a gated shared expert, and the layer-1 PLE n-gram
table -- at a size that loads instantly, in F32 so no quantisation type is in
the way.

Both engines read the same file, so llama.cpp `pr-27742` on it is a real
numerical reference for qwencpp: any shape or wiring mistake shows up here
rather than after a 67 GiB load.

    python tests/qwen4/make_tiny_model.py tests/qwen4/01-tiny/tiny-qwen4exp.gguf

--qsa-ratio=1 makes every block one token wide. That removes the one thing about
QSA the reference leaves undefined: when a token sits on a block boundary its
budget covers a whole number of blocks plus one leftover cell, and which member
of the last block that is comes down to how the sort breaks a tie. At ratio 1
there are no members to tie, so the two engines must agree exactly.

Pass --no-ple to leave the ple.* keys and tensors out entirely. Both engines
treat that as "this model has no PLE", which isolates hyper-connections, GDN,
attention and the MoE from the n-gram module while it is being brought up --
and is also what a --ngram off run has to reproduce numerically.

Needs llama.cpp's gguf-py (LLAMA_CPP=D:/dev/llama.cpp by default).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.environ.get('LLAMA_CPP', 'D:/dev/llama.cpp'), 'gguf-py'))
from gguf import GGUFWriter, GGUFValueType   # noqa: E402

ARCH = 'qwen4exp'

# --- geometry -------------------------------------------------------------
# Chosen so every constraint the architecture imposes still bites:
#   ple_head_dim * ple_n_heads == n_embd   (the gather feeds ple_key directly)
#   head_v_dim == ssm_d_state              (GDN q/k/v share the state width)
#   sum(rope_sections) * 2 == n_rot        (partial rope, as in the real model)
N_LAYER   = 8       # full attention at 3 and 7, GDN elsewhere; PLE on 1
N_EMBD    = 128
N_HEAD    = 4
N_HEAD_KV = 1
HEAD_DIM  = 32
N_ROT     = 16
ROPE_SEC  = [3, 3, 2, 0]

HC        = 4
HC_RANK   = 16
HC_DIM    = HC * N_EMBD

S         = 16      # ssm_d_state, also the GDN head width
H_K       = 2       # ssm_n_group
H_V       = 4       # ssm_dt_rank
D_INNER   = S * H_V
CONV_CH   = 2 * S * H_K + D_INNER
CONV_K    = 4

N_EXPERT  = 8
N_USED    = 2
FF_EXP    = 32

# 8 rather than 2: the block score is a sum of relu'd per-head dot products, so
# with few heads a random model scores many blocks at exactly 0. Ties there are
# broken by the sort, which is implementation-defined and depends on how far the
# reference pads its cache -- it would make the check disagree over nothing.
IDX_HEAD  = 8
IDX_DIM   = 16
IDX_TOPK  = 8
RATIO     = 2

PLE_LAYER  = 1
NGRAM      = 3
PER_GRAM   = 2
PLE_HEADS  = (NGRAM - 1) * PER_GRAM
PLE_DIM    = N_EMBD // PLE_HEADS
PLE_VOCAB  = [1009, 1013, 1019, 1021]      # per head, deliberately unequal
PLE_MULT   = [23703573157769, 20109073645365, 8052911324071]   # as in the real file
PLE_CONV_K = 4

N_BYTE    = 256                 # a GPT-2 byte-level alphabet, so ASCII maps 1:1
N_VOCAB   = N_BYTE + 2
PLE_EOS   = N_BYTE              # segment boundary, as ple.eos_token_id is in the real file
EOS       = N_BYTE + 1

rng = np.random.default_rng(20260827)


def t(*shape):
    """A weight in ggml's ne order, written as the numpy shape GGUFWriter wants.

    ggml names a 2-D weight [n_in, n_out]; numpy holds it transposed, so call
    sites can stay in ggml order and this reverses it.
    """
    scale = 1.0 / np.sqrt(shape[0])
    return (rng.standard_normal(tuple(reversed(shape))) * scale).astype(np.float32)


def ones(n):
    return np.ones(n, dtype=np.float32)


def is_recurrent(il):
    return (il + 1) % 4 != 0


def byte_alphabet():
    """GPT-2's byte-to-unicode alphabet: 256 single-character tokens.

    Every ASCII byte becomes exactly one token, so a test prompt tokenises the
    same way in both engines without depending on a merge table.
    """
    printable = (list(range(ord('!'), ord('~') + 1)) +
                 list(range(0xa1, 0xac + 1)) + list(range(0xae, 0xff + 1)))
    chars, n = list(printable), 0
    out = []
    for b in range(256):
        if b in printable:
            out.append(chr(b))
        else:
            out.append(chr(256 + n))
            n += 1
    del chars
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    with_ple = '--no-ple' not in sys.argv
    # compress_ratios all zero = dense attention everywhere, which isolates the
    # rest of the stack from QSA at any prompt length
    ratio = 0 if '--no-qsa' in sys.argv else RATIO
    for a in sys.argv[1:]:
        if a.startswith('--qsa-ratio='):
            ratio = int(a.split('=', 1)[1])
    out = args[0] if args else 'tiny-qwen4exp.gguf'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    w = GGUFWriter(out, ARCH)

    w.add_name('Tiny qwen4exp')
    w.add_block_count(N_LAYER)
    w.add_context_length(4096)
    w.add_embedding_length(N_EMBD)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_rope_dimension_count(N_ROT)
    w.add_rope_dimension_sections(ROPE_SEC)
    w.add_rope_freq_base(10000000.0)
    w.add_layer_norm_rms_eps(1e-6)

    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_USED)
    w.add_expert_feed_forward_length(FF_EXP)
    w.add_expert_shared_feed_forward_length(FF_EXP)

    w.add_ssm_conv_kernel(CONV_K)
    w.add_ssm_state_size(S)
    w.add_ssm_group_count(H_K)
    w.add_ssm_time_step_rank(H_V)
    w.add_ssm_inner_size(D_INNER)
    w.add_key_value(ARCH + '.full_attention_interval', 4, GGUFValueType.UINT32)

    w.add_key_value(ARCH + '.hyper_connection.count', HC, GGUFValueType.UINT32)
    w.add_key_value(ARCH + '.hyper_connection.low_rank', HC_RANK, GGUFValueType.UINT32)

    w.add_key_value(ARCH + '.attention.indexer.head_count', IDX_HEAD, GGUFValueType.UINT32)
    w.add_key_value(ARCH + '.attention.indexer.key_length', IDX_DIM, GGUFValueType.UINT32)
    w.add_key_value(ARCH + '.attention.indexer.top_k', IDX_TOPK, GGUFValueType.UINT32)
    w.add_key_value(ARCH + '.attention.compress_ratios',
                    [0 if is_recurrent(i) else ratio for i in range(N_LAYER)],
                    GGUFValueType.ARRAY, sub_type=GGUFValueType.INT32)

    offsets, acc = [], 0
    for v in PLE_VOCAB:
        offsets.append(acc)
        acc += v
    ple_rows = acc

    if with_ple:
        w.add_key_value(ARCH + '.ple.layers', [PLE_LAYER], GGUFValueType.ARRAY,
                        sub_type=GGUFValueType.INT32)
        w.add_key_value(ARCH + '.ple.ngram_size', NGRAM, GGUFValueType.UINT32)
        w.add_key_value(ARCH + '.ple.heads_per_ngram', PER_GRAM, GGUFValueType.UINT32)
        w.add_key_value(ARCH + '.ple.conv_kernel', PLE_CONV_K, GGUFValueType.UINT32)
        w.add_key_value(ARCH + '.ple.eos_token_id', PLE_EOS, GGUFValueType.UINT32)
        w.add_key_value(ARCH + '.embedding_length_per_layer_input', PLE_DIM, GGUFValueType.UINT32)
        w.add_key_value(ARCH + '.ple.layer_multipliers', PLE_MULT,
                        GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
        w.add_key_value(ARCH + '.ple.head_offsets', offsets,
                        GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
        w.add_key_value(ARCH + '.ple.head_vocab_sizes', PLE_VOCAB,
                        GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)

    # A byte-level alphabet plus one merge that matches nothing in ASCII text:
    # llama.cpp refuses a BPE vocabulary with no merge table at all, and a merge
    # that never fires keeps tokenisation a plain byte split.
    alphabet = byte_alphabet()
    w.add_tokenizer_model('gpt2')
    w.add_tokenizer_pre('default')
    w.add_token_list(alphabet + ['<|ple_eos|>', '<|endoftext|>'])
    w.add_token_types([1] * N_BYTE + [3, 3])
    w.add_token_merges([alphabet[0] + ' ' + alphabet[1]])
    w.add_bos_token_id(PLE_EOS)
    w.add_eos_token_id(EOS)
    w.add_pad_token_id(PLE_EOS)
    w.add_add_bos_token(False)

    # ---- tensors ----
    w.add_tensor('token_embd.weight', t(N_EMBD, N_VOCAB))
    w.add_tensor('output.weight',     t(N_EMBD, N_VOCAB))

    w.add_tensor('output_hc_norm.weight', ones(HC_DIM))
    w.add_tensor('output_hc_down.weight', t(HC_DIM, HC_RANK))
    w.add_tensor('output_hc_up.weight',   t(HC_RANK, HC_DIM))

    if with_ple:
        w.add_tensor('per_layer_token_embd.weight', t(PLE_DIM, ple_rows))

    for il in range(N_LAYER):
        p = 'blk.' + str(il) + '.'
        for mod in ('attn', 'ffn'):
            w.add_tensor(p + 'hc_' + mod + '_norm.weight',   ones(HC_DIM))
            w.add_tensor(p + 'hc_' + mod + '_down.weight',   t(HC_DIM, HC_RANK))
            w.add_tensor(p + 'hc_' + mod + '_up.weight',     t(HC_RANK, HC_DIM))
            w.add_tensor(p + 'hc_' + mod + '_inject.weight', t(HC_DIM, HC))

        if is_recurrent(il):
            w.add_tensor(p + 'attn_qkv.weight',   t(N_EMBD, CONV_CH))
            w.add_tensor(p + 'attn_gate.weight',  t(N_EMBD, D_INNER))
            w.add_tensor(p + 'ssm_conv1d.weight', t(CONV_K, CONV_CH))
            w.add_tensor(p + 'ssm_dt.bias',       rng.standard_normal(H_V).astype(np.float32))
            # -A_log.exp(): strictly negative, as the reference stores it
            w.add_tensor(p + 'ssm_a',             -np.abs(rng.standard_normal(H_V)).astype(np.float32))
            w.add_tensor(p + 'ssm_beta.weight',   t(N_EMBD, H_V))
            w.add_tensor(p + 'ssm_alpha.weight',  t(N_EMBD, H_V))
            w.add_tensor(p + 'ssm_norm.weight',   ones(S))
            w.add_tensor(p + 'ssm_out.weight',    t(D_INNER, N_EMBD))
        else:
            # attn_q holds [q|gate] interleaved per head, hence the doubling
            w.add_tensor(p + 'attn_q.weight',      t(N_EMBD, HEAD_DIM * N_HEAD * 2))
            w.add_tensor(p + 'attn_k.weight',      t(N_EMBD, HEAD_DIM * N_HEAD_KV))
            w.add_tensor(p + 'attn_v.weight',      t(N_EMBD, HEAD_DIM * N_HEAD_KV))
            w.add_tensor(p + 'attn_output.weight', t(HEAD_DIM * N_HEAD, N_EMBD))
            w.add_tensor(p + 'attn_q_norm.weight', ones(HEAD_DIM))
            w.add_tensor(p + 'attn_k_norm.weight', ones(HEAD_DIM))
            w.add_tensor(p + 'indexer.q_proj.weight', t(N_EMBD, IDX_HEAD * IDX_DIM))
            w.add_tensor(p + 'indexer.k_proj.weight', t(N_EMBD, IDX_DIM))
            w.add_tensor(p + 'indexer.q_norm.weight', ones(IDX_DIM))
            w.add_tensor(p + 'indexer.k_norm.weight', ones(IDX_DIM))

        if with_ple and il == PLE_LAYER:
            w.add_tensor(p + 'ple_key.weight',        t(N_EMBD, HC_DIM))
            w.add_tensor(p + 'ple_value.weight',      t(N_EMBD, N_EMBD))
            w.add_tensor(p + 'ple_norm_key.weight',   ones(HC_DIM))
            w.add_tensor(p + 'ple_norm_query.weight', ones(HC_DIM))
            w.add_tensor(p + 'ple_norm_conv.weight',  ones(HC_DIM))
            w.add_tensor(p + 'ple_conv1d.weight',     t(PLE_CONV_K, HC_DIM))

        w.add_tensor(p + 'ffn_gate_inp.weight',  t(N_EMBD, N_EXPERT))
        w.add_tensor(p + 'ffn_gate_exps.weight', t(N_EMBD, FF_EXP, N_EXPERT))
        w.add_tensor(p + 'ffn_up_exps.weight',   t(N_EMBD, FF_EXP, N_EXPERT))
        w.add_tensor(p + 'ffn_down_exps.weight', t(FF_EXP, N_EMBD, N_EXPERT))

        w.add_tensor(p + 'ffn_gate_inp_shexp.weight', t(N_EMBD, 1).reshape(N_EMBD))
        w.add_tensor(p + 'ffn_gate_shexp.weight',     t(N_EMBD, FF_EXP))
        w.add_tensor(p + 'ffn_up_shexp.weight',       t(N_EMBD, FF_EXP))
        w.add_tensor(p + 'ffn_down_shexp.weight',     t(FF_EXP, N_EMBD))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(out + ': ' + str(round(os.path.getsize(out) / 2 ** 20, 1)) + ' MiB, ' +
          str(N_LAYER) + ' layers, ' +
          ('PLE table ' + str(ple_rows) + ' rows' if with_ple else 'no PLE'))


if __name__ == '__main__':
    main()
