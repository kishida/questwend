#!/usr/bin/env python3
"""Compare two "index: value" logit dumps.

Reads the format both `llama-debug --save-logits` and `qw-cli --dump-logits`
write, and reports the worst absolute/relative gap plus whether the argmax and
the top-k ordering agree. Ordering is what actually matters for generation, so
it is reported separately from the raw numbers.

    python tests/qwen4/compare_logits.py <reference.txt> <ours.txt> [--top=10] [--tol=1e-3]

--tol is the pass threshold as a fraction of the logit span. 1e-3 is what an
f32 CPU graph reproducing another f32 CPU graph looks like; a GPU run against a
CPU reference needs more room (F16 KV cache, different kernels) and 5e-3 is the
measured figure there.
"""
import sys


def load(path):
    out = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        i, v = line.split(':', 1)
        try:
            out[int(i)] = float(v)
        except ValueError:
            continue
    return [out[i] for i in sorted(out)]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    top, tol = 10, 1e-3
    for a in sys.argv[1:]:
        if a.startswith('--top='):
            top = int(a.split('=', 1)[1])
        elif a.startswith('--tol='):
            tol = float(a.split('=', 1)[1])
    if len(args) < 2:
        raise SystemExit(__doc__)

    ref, ours = load(args[0]), load(args[1])
    if len(ref) != len(ours):
        raise SystemExit(f'length mismatch: {len(ref)} vs {len(ours)}')

    diffs = [abs(a - b) for a, b in zip(ref, ours)]
    worst = max(range(len(diffs)), key=lambda i: diffs[i])
    span = max(ref) - min(ref)

    order_ref = sorted(range(len(ref)), key=lambda i: -ref[i])
    order_our = sorted(range(len(ours)), key=lambda i: -ours[i])

    print(f'n            = {len(ref)}')
    print(f'max |diff|   = {diffs[worst]:.6g}  at index {worst} '
          f'(ref {ref[worst]:.6g}, ours {ours[worst]:.6g})')
    print(f'mean |diff|  = {sum(diffs) / len(diffs):.6g}')
    print(f'logit span   = {span:.6g}   -> max diff is {diffs[worst] / span:.3%} of span')
    print(f'argmax       = {order_ref[0]} vs {order_our[0]}  '
          f'{"MATCH" if order_ref[0] == order_our[0] else "MISMATCH"}')
    same = order_ref[:top] == order_our[:top]
    print(f'top-{top} order  = {"MATCH" if same else "MISMATCH"}')
    if not same:
        print(f'  ref  {order_ref[:top]}')
        print(f'  ours {order_our[:top]}')

    sys.exit(0 if diffs[worst] < tol * max(span, 1.0) else 1)


if __name__ == '__main__':
    main()
