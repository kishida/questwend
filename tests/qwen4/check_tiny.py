#!/usr/bin/env python3
"""Run the tiny qwen4exp model through qw-cli and diff it against the recorded
llama.cpp references.

    python tests/qwen4/check_tiny.py [--rebuild-ref]

The .gguf files are not in the repo (.gitignore excludes *.gguf); they are
regenerated here from make_tiny_model.py, whose RNG is seeded, so the same
bytes come out every time and the recorded references stay valid.

--rebuild-ref re-runs llama.cpp to regenerate the references, which is only
needed after changing the generator or moving to a newer PR revision.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TINY = os.path.join(HERE, '01-tiny')

QW_CLI = os.environ.get('QW_CLI', os.path.join(ROOT, 'build-cpu', 'Release', 'qw-cli.exe'))
LLAMA_DEBUG = os.environ.get(
    'LLAMA_DEBUG',
    os.path.join(os.environ.get('LLAMA_CPP', 'D:/dev/llama.cpp'),
                 'build', 'bin', 'Release', 'llama-debug.exe'))

# (reference dir, prompt, model variant, what a mismatch would mean)
CASES = [
    ('dense-5tok', 'Hello', 'nople',
     'hyper-connections / GDN / dense attention / MoE'),
    # 36 tokens is past indexer_top_k + compress_ratio - 1 = 9, so llama.cpp
    # runs QSA here and qwencpp still runs dense. Expected to differ until QSA
    # lands; kept so the day it lands the check flips to a pass on its own.
    ('qsa-36tok', 'Hello world, this is a longer prompt', 'nople', 'QSA'),
]


def gguf(variant):
    return os.path.join(TINY, 'tiny-qwen4exp-nople.gguf' if variant == 'nople'
                        else 'tiny-qwen4exp.gguf')


def generate_models():
    for variant, extra in (('ple', []), ('nople', ['--no-ple'])):
        subprocess.run([sys.executable, os.path.join(HERE, 'make_tiny_model.py'),
                        gguf(variant)] + extra, check=True)


def rebuild_ref():
    for ref, prompt, variant, _why in CASES:
        out = os.path.join(TINY, 'ref', ref)
        os.makedirs(out, exist_ok=True)
        subprocess.run([LLAMA_DEBUG, '-m', gguf(variant), '-p', prompt,
                        '-c', '256', '-n', '1', '-ngl', '0', '--save-logits',
                        '--logits-output-dir', out], check=True)


def main():
    generate_models()
    if '--rebuild-ref' in sys.argv:
        rebuild_ref()

    failures = []
    for ref, prompt, variant, why in CASES:
        ref_dir = os.path.join(TINY, 'ref', ref)
        ref_txt = next(f for f in sorted(os.listdir(ref_dir))
                       if f.endswith('.txt') and not f.endswith('-prompt.txt'))
        ours = os.path.join(TINY, 'qwencpp-' + ref + '.txt')

        # qw-cli narrates the load on stderr; the diff below is the output that
        # matters, so keep it out of the way
        subprocess.run([QW_CLI, '-m', gguf(variant), '-p', prompt, '-n', '1',
                        '--cpu', '--dump-logits', ours],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # flush: the diff runs as a child process and would otherwise print first
        print('==== ' + ref + '  (' + why + ') ====', flush=True)
        rc = subprocess.run([sys.executable, os.path.join(HERE, 'compare_logits.py'),
                             os.path.join(ref_dir, ref_txt), ours]).returncode
        if rc != 0:
            failures.append(ref)
        print()

    if failures:
        print('mismatched: ' + ', '.join(failures))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
