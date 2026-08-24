#!/usr/bin/env python3
"""Diff a qwencpp QWEN_DUMP_LAYERS capture against the llama.cpp reference.

    python cmp.py 04-tensors.log 10-qw-tensors.log

Both sides print one whole-tensor sum per named intermediate, and the names
match (qwencpp's dbg() copies llama.cpp's cb() labels). The first genuine
divergence is what the implementation got wrong -- every later tensor inherits
it -- so read the top MISMATCH line and ignore the cascade below it.

Caveat: a sum that sits near zero (heavy cancellation across the tensor) has a
huge *relative* error for a negligible absolute one. Read the abs column before
believing a MISMATCH on a small sum.
"""
import argparse, io, re, sys

# Backslash-free patterns on purpose: these regexes survive being pasted around.
REF = re.compile("common_debug_cb_eval:[ ]+([^ ]+) = [(]([A-Za-z0-9_]+)[)]")
SUM = re.compile("sum = (-?[0-9.eE+]+)")
QW  = re.compile("qw_dump[^:]*:[ ]+([^ ]+) = [{]([^}]*)[}] sum = (-?[0-9.eE+]+)")


def parse_ref(path):
    """llama.cpp: a header line names the tensor, a later line carries its sum."""
    out, name = [], None
    for line in io.open(path, encoding="utf-8", errors="replace"):
        m = REF.search(line)
        if m:
            name = m.group(1)
            continue
        if name:
            m = SUM.search(line)
            if m:
                out.append((name, float(m.group(1))))
                name = None
    return out


def parse_qw(path):
    out = []
    for line in io.open(path, encoding="utf-8", errors="replace"):
        m = QW.search(line)
        if m:
            out.append((m.group(1), float(m.group(3))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="llama-debug capture, e.g. 04-tensors.log")
    ap.add_argument("qw", help="qw-cli QWEN_DUMP_LAYERS capture")
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="relative tolerance (default 0.02; the two run different "
                         "kernels, so bit equality is not expected)")
    a = ap.parse_args()

    # Either side may be a llama-debug capture or a qw_dump capture, so that two
    # qwencpp runs (resident vs offload, Metal vs CPU) can be diffed the same way.
    def load(path):
        r = parse_ref(path)
        return r if r else parse_qw(path)

    ref = load(a.ref)
    qw = dict(load(a.qw))
    if not ref:
        sys.exit("no reference tensors parsed from " + a.ref)
    if not qw:
        sys.exit("no qw_dump lines parsed from " + a.qw)

    ok = miss = bad = 0
    first_bad = None
    for name, rv in ref:
        if name not in qw:
            miss += 1
            continue
        qv = qw[name]
        rel = abs(rv - qv) / max(abs(rv), abs(qv), 1e-6)
        if rel <= a.tol:
            ok += 1
            continue
        bad += 1
        if first_bad is None:
            first_bad = name
        print("MISMATCH %-26s ref %15.6f  qw %15.6f  rel %7.4f  abs %12.6f"
              % (name, rv, qv, rel, abs(rv - qv)))

    print()
    print("matched %d, mismatched %d, missing from qw %d (of %d reference tensors)"
          % (ok, bad, miss, len(ref)))
    if first_bad:
        print("first divergence: %s  <- everything after it inherits this" % first_bad)
    seen = set(n for n, _ in ref)
    only = sorted(n for n in qw if n not in seen)
    if only:
        print("only in qw (no reference): " + ", ".join(only[:12]))


if __name__ == "__main__":
    main()
