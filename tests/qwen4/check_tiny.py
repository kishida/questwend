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
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TINY = os.path.join(HERE, '01-tiny')

QW_CLI = os.environ.get('QW_CLI', os.path.join(ROOT, 'build-cpu', 'Release', 'qw-cli.exe'))
# The expert-offload paths only engage on a CUDA build (use_expert_offload is
# gated on it), so those cases are skipped when there is no such binary.
QW_CLI_CUDA = os.environ.get('QW_CLI_CUDA',
                             os.path.join(ROOT, 'build', 'Release', 'qw-cli.exe'))
LLAMA_DEBUG = os.environ.get(
    'LLAMA_DEBUG',
    os.path.join(os.environ.get('LLAMA_CPP', 'D:/dev/llama.cpp'),
                 'build', 'bin', 'Release', 'llama-debug.exe'))
LLAMA_COMPLETION = os.environ.get(
    'LLAMA_COMPLETION',
    os.path.join(os.environ.get('LLAMA_CPP', 'D:/dev/llama.cpp'),
                 'build', 'bin', 'Release', 'llama-completion.exe'))

# (reference dir, prompt, model variant, extra qw-cli args, what a mismatch means)
CASES = [
    ('dense-5tok', 'Hello', 'nople', [],
     'hyper-connections / GDN / dense attention / MoE'),
    ('ple-5tok', 'Hello', 'ple', [],
     'the PLE n-gram hash, gather and dilated conv'),
    # --ngram off has to reproduce the no-PLE model exactly: same stack, same
    # weights, minus one gated additive branch.
    ('dense-5tok', 'Hello', 'nople', ['--ngram', 'off'],
     '--ngram off on a model that has no PLE at all'),
    # 36 tokens is past indexer_top_k + compress_ratio - 1 = 9, so llama.cpp
    # runs QSA here and qwencpp still runs dense. Expected to differ until QSA
    # lands; kept so the day it lands the check flips to a pass on its own.
    ('qsa-36tok', 'Hello world, this is a longer prompt', 'nople', [], 'QSA'),
]

# Logits only cover the prompt's last position, which says nothing about what
# is carried between calls -- the GDN conv/ssm state, the PLE conv history and
# the n-gram token window. Generating greedily for several steps does.
#
# Greedy decoding is chaotic: the two engines agree to ~0.02% of the logit
# span, which is enough to flip a near-tie, and one flip changes every token
# after it. So the bar is "everything but possibly the last step", over several
# prompts -- a real wiring bug diverges at the first step, not the last. Two of
# the eight runs below do sit on such a tie at the final step.
#
# Compared as bytes rather than token ids because that is what both CLIs print;
# a byte-level token can decode to more than one byte, so the totals vary.
GEN_N = 8
GEN_PROMPTS = ['Hello', 'Hi', 'Test', 'abc']

# The expert-offload paths build their own segmented graphs, so they need their
# own coverage: the wide residual and the PLE branch have to survive being split
# across seg A / seg B and carried in persistent buffers between them.
#
# QWEN_NO_FLASH because the CUDA flash-attention kernel does not take this
# model's 32-wide heads (the real model's are 256); it is unrelated to qwen4exp.
# The looser tolerance is GPU-vs-CPU numerics, not the offload path -- the
# resident GPU run lands on exactly the same numbers.
OFFLOAD_CASES = [
    ('ple-5tok', 'Hello', 'ple', ['--gpus', '0', '--vram-budget', '2'],
     'seg A / seg B carrying the wide residual and the n-gram branch'),
    ('ple-5tok', 'Hello', 'ple', ['--gpus', '0', '--vram-budget', '2', '--experts-ssd'],
     'the same with experts streamed from the file'),
]
OFFLOAD_TOL = '--tol=5e-3'


def gguf(variant):
    return os.path.join(TINY, 'tiny-qwen4exp-nople.gguf' if variant == 'nople'
                        else 'tiny-qwen4exp.gguf')


def generate_models():
    for variant, extra in (('ple', []), ('nople', ['--no-ple'])):
        subprocess.run([sys.executable, os.path.join(HERE, 'make_tiny_model.py'),
                        gguf(variant)] + extra, check=True)


def rebuild_ref():
    seen = set()
    for ref, prompt, variant, _args, _why in CASES:
        if ref in seen:
            continue
        seen.add(ref)
        out = os.path.join(TINY, 'ref', ref)
        os.makedirs(out, exist_ok=True)
        subprocess.run([LLAMA_DEBUG, '-m', gguf(variant), '-p', prompt,
                        '-c', '256', '-n', '1', '-ngl', '0', '--save-logits',
                        '--logits-output-dir', out], check=True)


def generation(exe_args, env=None):
    r = subprocess.run(exe_args, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL, env=env)
    return r.stdout


ANSI = re.compile(rb'\x1b\[[0-9;]*m')


def check_generation():
    """Greedy generation must follow the reference across several decode steps."""
    # --experts-ssd sends every generated token through decode_cached (the
    # segmented single-token path), which is the one place the hyper-connection
    # carries and the per-token n-gram gather have to survive a graph boundary.
    runs = [('ple', QW_CLI, ['--cpu'], None),
            ('nople', QW_CLI, ['--cpu'], None)]
    if os.path.exists(QW_CLI_CUDA):
        runs.append(('ple', QW_CLI_CUDA,
                     ['--gpus', '0', '--vram-budget', '2', '--experts-ssd'],
                     dict(os.environ, QWEN_NO_FLASH='1')))

    failures = []
    for variant, exe, extra, env in runs:
        label = variant + ('-ssd' if '--experts-ssd' in extra else '') + '-gen'
        print('==== ' + label + '  (state carried between decode calls) ====',
              flush=True)
        for prompt in GEN_PROMPTS:
            ref = generation([LLAMA_COMPLETION, '-m', gguf(variant), '-p', prompt,
                              '-c', '256', '-n', str(GEN_N), '-ngl', '0',
                              '--temp', '0', '--no-warmup'])
            ours = generation([exe, '-m', gguf(variant), '-p', prompt,
                               '-n', str(GEN_N), '--temp', '0'] + extra, env=env)
            # llama-completion colours the echoed prompt; qw-cli does not
            tail = lambda b: ANSI.sub(b'', b).split(prompt.encode())[-1].strip()
            a, b = tail(ref), tail(ours)
            common = 0
            while common < min(len(a), len(b)) and a[common] == b[common]:
                common += 1
            ok = common >= min(len(a), len(b)) - 1
            print('  %-8s %s  %d/%d bytes agree' %
                  (prompt, 'ok  ' if ok else 'FAIL', common, len(a)))
            if not ok:
                print('    ref  ' + repr(a))
                print('    ours ' + repr(b))
                failures.append(label + ':' + prompt)
        print()
    return failures


def check_offload():
    """The offload paths reach the same numbers as the resident graph."""
    if not os.path.exists(QW_CLI_CUDA):
        print('==== offload  (skipped: no CUDA build at ' + QW_CLI_CUDA + ') ====')
        print()
        return []

    env = dict(os.environ, QWEN_NO_FLASH='1')
    failures = []
    for ref, prompt, variant, extra, why in OFFLOAD_CASES:
        ref_dir = os.path.join(TINY, 'ref', ref)
        ref_txt = next(f for f in sorted(os.listdir(ref_dir))
                       if f.endswith('.txt') and not f.endswith('-prompt.txt'))
        name = 'offload' + ('-ssd' if '--experts-ssd' in extra else '')
        ours = os.path.join(TINY, 'qwencpp-' + name + '.txt')
        subprocess.run([QW_CLI_CUDA, '-m', gguf(variant), '-p', prompt, '-n', '1',
                        '--dump-logits', ours] + extra, check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('==== ' + name + '  (' + why + ') ====', flush=True)
        rc = subprocess.run([sys.executable, os.path.join(HERE, 'compare_logits.py'),
                             os.path.join(ref_dir, ref_txt), ours,
                             OFFLOAD_TOL]).returncode
        if rc != 0:
            failures.append(name)
        print()
    return failures


def main():
    generate_models()
    if '--rebuild-ref' in sys.argv:
        rebuild_ref()

    failures = []
    for ref, prompt, variant, extra, why in CASES:
        ref_dir = os.path.join(TINY, 'ref', ref)
        ref_txt = next(f for f in sorted(os.listdir(ref_dir))
                       if f.endswith('.txt') and not f.endswith('-prompt.txt'))
        name = ref + ('-' + '-'.join(a.lstrip('-') for a in extra) if extra else '')
        ours = os.path.join(TINY, 'qwencpp-' + name + '.txt')

        # qw-cli narrates the load on stderr; the diff below is the output that
        # matters, so keep it out of the way
        subprocess.run([QW_CLI, '-m', gguf(variant), '-p', prompt, '-n', '1',
                        '--cpu', '--dump-logits', ours] + extra,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # flush: the diff runs as a child process and would otherwise print first
        print('==== ' + name + '  (' + why + ') ====', flush=True)
        rc = subprocess.run([sys.executable, os.path.join(HERE, 'compare_logits.py'),
                             os.path.join(ref_dir, ref_txt), ours]).returncode
        if rc != 0:
            failures.append(name)
        print()

    failures += check_offload()
    failures += check_generation()

    if failures:
        print('mismatched: ' + ', '.join(failures))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
