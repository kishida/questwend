#!/usr/bin/env python3
"""Dump a (possibly sharded, possibly still-downloading) GGUF's metadata and
tensor manifest without loading any weight data.

llama.cpp's gguf_dump.py memory-maps the whole file and reshapes every tensor,
so it dies on a .part file. This walks the header only, which is all that is
needed to see hparams, the tokenizer and the per-tensor types/sizes -- and it
works on a shard that is still being written.

    python tests/qwen4/dump_gguf.py <first-shard.gguf> [--tensors] [--kv]

Shards are discovered from the -0000N-of-0000M suffix; a shard that is still
downloading is picked up under its `downloading_<name>.part` name too.
"""
import collections, glob, os, re, struct, sys

# (name, bytes per block, elements per block)
GGML_TYPE = {
    0:('F32',4,1), 1:('F16',2,1), 2:('Q4_0',18,32), 3:('Q4_1',20,32),
    6:('Q5_0',22,32), 7:('Q5_1',24,32), 8:('Q8_0',34,32), 9:('Q8_1',36,32),
    10:('Q2_K',84,256), 11:('Q3_K',110,256), 12:('Q4_K',144,256),
    13:('Q5_K',176,256), 14:('Q6_K',210,256), 15:('Q8_K',292,256),
    16:('IQ2_XXS',66,256), 17:('IQ2_XS',74,256), 18:('IQ3_XXS',98,256),
    19:('IQ1_S',50,256), 20:('IQ4_NL',18,32), 21:('IQ3_S',110,256),
    22:('IQ2_S',82,256), 23:('IQ4_XS',136,256), 24:('I8',1,1), 25:('I16',2,1),
    26:('I32',4,1), 27:('I64',8,1), 28:('F64',8,1), 29:('IQ1_M',56,256),
    30:('BF16',2,1),
}

_SCALAR = {0:('B',1), 1:('b',1), 2:('H',2), 3:('h',2), 4:('I',4), 5:('i',4),
           6:('f',4), 7:('?',1), 10:('Q',8), 11:('q',8), 12:('d',8)}


def read_header(path):
    """Walk the header only.

    -> (kv    list of (key, wire_type, value, elem_type),
        tensor list of (name, ne, ggml_type, offset))

    `elem_type` is the array's element type, or None when the key is not an
    array. Rewriting a KV block needs it: several arrays here hold small values
    in a wide type (`ple.head_offsets` starts at 0 but is UINT64) and one holds
    a large-looking value in a narrow one, so it cannot be inferred from the
    values."""
    f = open(path, 'rb')
    u32 = lambda: struct.unpack('<I', f.read(4))[0]
    u64 = lambda: struct.unpack('<Q', f.read(8))[0]

    def string():
        return f.read(u64()).decode('utf-8', 'replace')

    def value(t):
        """-> (value, elem_type); elem_type is None unless t is ARRAY."""
        if t == 8:
            return string(), None
        if t == 9:
            et, n = u32(), u64()
            return [value(et)[0] for _ in range(n)], et
        fmt, size = _SCALAR[t]
        return struct.unpack('<' + fmt, f.read(size))[0], None

    magic = f.read(4)
    if magic != b'GGUF':
        raise SystemExit(f'{path}: not a GGUF file (magic {magic!r})')
    u32()                                   # version
    n_tensor, n_kv = u64(), u64()

    kv = []
    for _ in range(n_kv):
        k = string(); t = u32(); v, et = value(t); kv.append((k, t, v, et))

    tensors = []
    for _ in range(n_tensor):
        name = string()
        ne = [u64() for _ in range(u32())]
        tensors.append((name, ne, u32(), u64()))
    f.close()
    return kv, tensors


def nbytes(ne, tt):
    _, blk_bytes, blk_elems = GGML_TYPE.get(tt, ('?', 0, 1))
    n = 1
    for d in ne:
        n *= d
    return n // blk_elems * blk_bytes


def shard_paths(first):
    """Every shard of a split model, .part names included."""
    m = re.match(r'(.*)-(\d{5})-of-(\d{5})\.gguf$', os.path.basename(first))
    if not m:
        return [first]
    stem, _, total = m.group(1), m.group(2), int(m.group(3))
    d = os.path.dirname(first) or '.'
    out = []
    for i in range(1, total + 1):
        name = f'{stem}-{i:05d}-of-{total:05d}.gguf'
        for cand in (os.path.join(d, name),
                     os.path.join(d, f'downloading_{name}.part')):
            if os.path.exists(cand):
                out.append(cand)
                break
        else:
            print(f'# missing shard {i}: {name}', file=sys.stderr)
    return out


def fmt(v, limit=6):
    if isinstance(v, list):
        head = ', '.join(repr(x)[:40] for x in v[:limit])
        return f'[{head}{", ..." if len(v) > limit else ""}]  (n={len(v)})'
    if isinstance(v, str) and len(v) > 200:
        return repr(v[:200]) + f' ... ({len(v)} chars)'
    return repr(v)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = {a for a in sys.argv[1:] if a.startswith('--')}
    if not args:
        raise SystemExit(__doc__)
    paths = shard_paths(args[0])

    if '--tensors' not in flags or '--kv' in flags:
        kv, _ = read_header(paths[0])
        print(f'==== {len(kv)} KV pairs ({os.path.basename(paths[0])}) ====')
        for k, _t, v, _et in kv:
            print(f'{k} = {fmt(v)}')
        print()

    rows, total, by_cat, by_cat_type = [], 0, collections.Counter(), collections.Counter()
    for p in paths:
        _, tensors = read_header(p)
        shard_total = 0
        for name, ne, tt, _off in tensors:
            nb = nbytes(ne, tt)
            tname = GGML_TYPE.get(tt, (f'TYPE_{tt}',))[0]
            rows.append((name, ne, tname, nb))
            total += nb
            shard_total += nb
            cat = ('ple'   if 'per_layer_token_embd' in name else
                   'exps'  if '_exps' in name else 'other')
            by_cat[cat] += nb
            by_cat_type[(cat, tname)] += nb
        print(f'# {os.path.basename(p)}: {len(tensors)} tensors, '
              f'{shard_total / 2**30:.2f} GiB', file=sys.stderr)

    if '--tensors' in flags:
        print('==== tensors ====')
        for name, ne, tname, nb in rows:
            print(f'{name}\t{"x".join(map(str, ne))}\t{tname}\t{nb}')
        print()

    print('==== size by category ====')
    for cat, nb in by_cat.most_common():
        print(f'{cat:8s} {nb / 2**30:8.2f} GiB')
        for (c, t), v in sorted(by_cat_type.items(), key=lambda x: -x[1]):
            if c == cat:
                print(f'    {t:10s} {v / 2**30:8.2f} GiB')
    print(f'{"TOTAL":8s} {total / 2**30:8.2f} GiB   ({len(rows)} tensors)')


if __name__ == '__main__':
    main()
