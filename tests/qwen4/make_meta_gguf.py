#!/usr/bin/env python3
"""Build a metadata-only GGUF from the first shard of a split model.

Shard 1 of a split GGUF carries every KV pair but no tensor data, so it is
complete long before the rest of the download is. Copying its metadata into a
standalone file (split.* keys dropped, so the loader does not go looking for
shards 2..N) gives something `qw-cli --info` can open, which is enough to
exercise hparam parsing without the 67 GiB of weights.

    python tests/qwen4/make_meta_gguf.py <shard-00001-of-000NN.gguf> <out.gguf>

Needs llama.cpp's gguf-py on the path (LLAMA_CPP=D:/dev/llama.cpp by default).
"""
import os, sys

sys.path.insert(0, os.path.join(os.environ.get('LLAMA_CPP', 'D:/dev/llama.cpp'), 'gguf-py'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gguf import GGUFWriter, GGUFValueType          # noqa: E402
from dump_gguf import read_header                   # noqa: E402

# GGUF wire type -> gguf-py enum. Same numbering; kept explicit so a mismatch
# fails loudly rather than writing a wrongly typed key.
WIRE = {t.value: t for t in GGUFValueType}

SKIP = ('split.no', 'split.count', 'split.tensors.count')


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]

    kv, tensors = read_header(src)
    if tensors:
        print(f'note: {src} carries {len(tensors)} tensors; only KV is copied',
              file=sys.stderr)

    arch = next(v for k, _t, v, _et in kv if k == 'general.architecture')
    w = GGUFWriter(dst, arch)
    n = 0
    for key, wire_t, val, elem_t in kv:
        if key in SKIP or key == 'general.architecture':
            continue
        t = WIRE[wire_t]
        if t == GGUFValueType.ARRAY:
            w.add_key_value(key, val, t, sub_type=WIRE[elem_t])
        else:
            w.add_key_value(key, val, t)
        n += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.close()
    print(f'{dst}: {n} KV pairs, arch={arch}')


if __name__ == '__main__':
    main()
