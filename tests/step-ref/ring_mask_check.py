#!/usr/bin/env python3
"""Prove the SWA ring mask keeps the same key positions as a full KV cache.

    python ring_mask_check.py

Both formulas are transcribed from Runtime::Impl::fill_masks. The ring stores a
position at row `pos % R`, so a row's absolute position is only recoverable
modulo R: it is reconstructed as the newest position at or before the last one
the batch writes that lands on that row. This checks that the resulting key set
is exactly the causal window, for every query of every batch.

A runtime A/B cannot settle this on its own -- the two paths accumulate in a
different order, so they never agree bit for bit and a small logic error hides
inside the float noise. This does settle it, and it costs nothing to run.
"""


def full_keep(a, W, n_kv):
    """Rows a full cache leaves unmasked for a query at absolute position a."""
    return {j for j in range(n_kv) if j <= a and a - j < W}


def ring_keep(a, last, R, W):
    """Positions the ring leaves unmasked, given the batch's last position."""
    keep = set()
    for r in range(R):
        p = last - ((last - r) % R + R) % R
        if p >= 0 and p <= a and a - p < W:
            keep.add(p)
    return keep


def check(W, R, total, chunk, label, expect_ok=True):
    bad, pos = 0, 0
    while pos < total:
        n = min(chunk, total - pos)
        last = pos + n - 1
        for i in range(n):
            a = pos + i
            f, g = full_keep(a, W, total), ring_keep(a, last, R, W)
            if f != g:
                bad += 1
                if bad == 1:
                    print("    first at pos0=%d a=%d  only_full=%s only_ring=%s"
                          % (pos, a, sorted(f - g)[:6], sorted(g - f)[:6]))
        pos += n
    ok = (bad == 0)
    status = "OK" if ok else "%d BAD" % bad
    flag = "" if ok == expect_ok else "   <-- UNEXPECTED"
    print("%-38s %s%s" % (label, status, flag))
    return ok == expect_ok


def main():
    cases = [
        # (window, ring, total, chunk, label, expect_ok)
        (512, 1024, 8192, 512, "W=512 R=1024 chunk=512  default",     True),
        (512,  768, 8192, 256, "W=512 R=768  chunk=256  offload",     True),
        (512,  576, 8192,  64, "W=512 R=576  chunk=64   tight",       True),
        (512,  576,  640,  64, "W=512 R=576  chunk=64   the A/B",     True),
        (4,      6,   40,   2, "W=4   R=6    chunk=2    tiny",        True),
        (4,      5,   40,   1, "W=4   R=5    chunk=1    minimal",     True),
        # R < W + chunk: the runtime refuses this, and here is why
        (4,      4,   40,   2, "W=4   R=4    chunk=2    UNDERSIZED",  False),
        (512,  512, 2048, 256, "W=512 R=512  chunk=256  UNDERSIZED",  False),
    ]
    if all(check(*c) for c in cases):
        print("\nall cases behaved as expected")
        return 0
    print("\nSOME CASE BEHAVED UNEXPECTEDLY")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
