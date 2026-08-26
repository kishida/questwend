#!/usr/bin/env python3
"""Sanity-check a qwen4exp GGUF's weights without loading it.

A truncated download is easy to spot; a download that silently wrote the wrong
bytes into part of a shard is not. That happened here: an IQ1_S copy of
Qwen3.8-Flash-Next produced nothing but NaN logits, and the cause was three
layers of one shard holding garbage while the rest of the file was fine.

The test is that RMSNorm gammas sit near 1 in a trained model, always. Reading
just those -- a few kilobytes per layer -- is enough to find a damaged region,
and it costs seconds rather than the minutes a full load takes. Non-finite
values anywhere in the small float tensors are reported too.

    python tests/qwen4/check_weights.py <first-shard.gguf>

Exits non-zero if anything looks wrong, naming the layers so the shard holding
them can be re-fetched on its own.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_gguf import read_header, nbytes, shard_paths, GGML_TYPE   # noqa: E402

# Gammas live near 1. The band is wide because deep layers legitimately drift
# (2.3 was observed in a healthy one); the damaged layers came in at 0.0001 to
# 0.013, so there is more than an order of magnitude of margin either way.
GAMMA_LO, GAMMA_HI = 0.2, 5.0
NORM_SUFFIXES = ('ssm_norm.weight', 'hc_attn_norm.weight', 'hc_ffn_norm.weight',
                 'attn_q_norm.weight', 'attn_k_norm.weight',
                 'ple_norm_key.weight', 'ple_norm_query.weight',
                 'ple_norm_conv.weight', 'indexer.q_norm.weight',
                 'indexer.k_norm.weight', 'output_hc_norm.weight')

# reading a whole expert tensor would defeat the point; these are all small
MAX_ELEMS = 4_000_000


def as_f32(raw, tname):
    if tname == 'F32':
        return np.frombuffer(raw, dtype=np.float32)
    if tname == 'F16':
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if tname == 'BF16':
        return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return None


def data_start(path, tensors):
    """Byte offset of the tensor data section.

    Tensor offsets in the header are relative to it, and the file ends at the
    last tensor, so the difference is the header's own size.
    """
    return os.path.getsize(path) - max(off + nbytes(ne, tt)
                                       for _n, ne, tt, off in tensors)


def layer_of(name):
    m = re.match(r'blk\.(\d+)\.', name)
    return int(m.group(1)) if m else -1


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)

    bad_layers, n_checked, n_nonfinite = set(), 0, 0
    for path in shard_paths(sys.argv[1]):
        _, tensors = read_header(path)
        if not tensors:
            continue
        start = data_start(path, tensors)
        with open(path, 'rb') as f:
            for name, ne, tt, off in tensors:
                tname = GGML_TYPE.get(tt, ('?',))[0]
                n = 1
                for d in ne:
                    n *= d
                if n > MAX_ELEMS or tname not in ('F32', 'F16', 'BF16'):
                    continue
                f.seek(start + off)
                a = as_f32(f.read(nbytes(ne, tt)), tname)
                finite = a[np.isfinite(a)]

                if finite.size != a.size:
                    n_nonfinite += 1
                    bad_layers.add(layer_of(name))
                    print(f'  {name}: {a.size - finite.size} non-finite of {a.size}')

                if name.endswith(NORM_SUFFIXES) and finite.size:
                    n_checked += 1
                    med = float(np.median(np.abs(finite)))
                    if not GAMMA_LO < med < GAMMA_HI:
                        bad_layers.add(layer_of(name))
                        print(f'  {name}: median |gamma| = {med:.5f}, expected ~1')

    print(f'\nchecked {n_checked} norm weights; '
          f'{n_nonfinite} tensors held non-finite values')
    if bad_layers:
        named = sorted(l for l in bad_layers if l >= 0)
        print('DAMAGED. layers: ' + (', '.join(map(str, named)) or '(non-layer tensors)'))
        print('Re-fetch the shard holding them; the others are fine.')
        return 1
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
