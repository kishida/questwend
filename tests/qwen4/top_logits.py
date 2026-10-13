#!/usr/bin/env python3
"""Summarise one logit dump: is it healthy, and what did it want to say?

`--dump-logits` writes a line per vocabulary entry, which is 248320 lines for
Qwen3.8-Flash-Next. This prints the handful that matter -- whether anything is
non-finite, how wide the distribution is, and the top few tokens with their text
-- which is usually enough to tell a broken forward pass from a model that
simply decided to stop.

    python tests/qwen4/top_logits.py <dump.txt> [<model.gguf>] [--top=10]

Passing the GGUF resolves token ids to text; without it the ids are printed
bare. Only the tokenizer metadata is read, so any shard of a split model works
and nothing large is loaded.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_gguf import read_header   # noqa: E402


def load_vocab(path):
    kv, _ = read_header(path)
    d = {k: v for k, _t, v, _et in kv}
    return d.get('tokenizer.ggml.tokens'), d


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    top = 10
    for a in sys.argv[1:]:
        if a.startswith('--top='):
            top = int(a.split('=', 1)[1])
    if not args:
        raise SystemExit(__doc__)

    vals = {}
    for line in open(args[0], encoding='utf-8'):
        line = line.strip()
        if ':' not in line:
            continue
        i, v = line.split(':', 1)
        try:
            vals[int(i)] = float(v)
        except ValueError:
            continue

    finite = {i: v for i, v in vals.items() if math.isfinite(v)}
    print(f'{len(vals)} logits, {len(vals) - len(finite)} non-finite')
    if not finite:
        print('ALL NON-FINITE -- the forward pass is broken, not the sampling')
        return 1

    lo, hi = min(finite.values()), max(finite.values())
    print(f'range {lo:.4f} .. {hi:.4f}  (span {hi - lo:.4f})')
    if hi - lo < 1e-3:
        print('the distribution is flat -- every token is equally likely, '
              'which is what a dead forward pass looks like')

    toks, meta = (None, {})
    if len(args) > 1:
        toks, meta = load_vocab(args[1])

    special = {}
    for key in ('tokenizer.ggml.eos_token_id', 'tokenizer.ggml.bos_token_id',
                'qwen4exp.ple.eos_token_id'):
        if key in meta:
            special[meta[key]] = key.rsplit('.', 2)[-2] + ' id'

    print(f'\ntop {top}:')
    for i, v in sorted(finite.items(), key=lambda kv: -kv[1])[:top]:
        text = repr(toks[i]) if toks and i < len(toks) else ''
        tag = '   <-- ' + special[i] if i in special else ''
        print(f'  {i:7d}  {v:9.4f}  {text}{tag}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
