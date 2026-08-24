# step35 reference capture (llama.cpp oracle)

Golden data for the Step-3.7-Flash (`step35`) port, captured with a stock
llama.cpp build so the qwencpp graph can be diffed against a known-good one.

- llama.cpp `dd1ea5243` (b10355), Apple M3 Ultra / 512 GB, Metal
- model: `unsloth/Step-3.7-Flash-GGUF` **UD-IQ1_M** (52.85 GiB, fully resident)
- prompt for the tensor/logit captures: `The capital of France is`
  -> 6 tokens `[0, 671, 6102, 294, 8760, 344]`
- every run uses `-ngl 99 -c 4096 -fa off --no-warmup`; `-fa off` keeps the
  attention path on the soft_max branch so the numbers are comparable

| file | what it pins down |
|---|---|
| `01-sanity.log` | the oracle answers "Paris"; full `print_info` hparams |
| `02-chat.log` | the ChatML prompt as real token ids (roles are plain BPE text; `<think>`=128798) |
| `03-tokens.txt` + `tok-test.txt` | deepseek-v3 pre-tokenizer behaviour (3-digit groups, CJK class, punctuation runs) |
| `04-tensors.log` | 129 tensors for layers 0 / 3 / 44 plus `result_*`, each with a whole-tensor `sum` |
| `05-logits/` | final logits (`.bin` = raw f32 x 128896) and the prompt's token ids |
| `06-mtp.log` | the separate MTP GGUF loads as a *draft model*, not a patch on the trunk |
| `07/08/09-*.log` | `llama-bench` throughput (see the caveat below) |

## Using `04-tensors.log`

Each block prints the tensor's name, op, input shapes, a 3x3x3 corner sample and
`sum` over every element. **The `sum` is the comparison handle**: one scalar per
tensor, so a mismatching layer can be bisected without dumping full tensors.

The filter that produced it is `--tensor-filter '.*(-(0|3|44)$|result_)'`.
`common/debug.cpp` prepends `^` to whatever you pass, so a suffix-only pattern
like `-3$` matches nothing -- it becomes `^-3$`.

`--save-logits` disables the tensor callback entirely, so `04` and `05` have to
be separate runs.

## Comparing a qwencpp run

`QWEN_DUMP_LAYERS=0,3,44` makes qw-cli pin the same intermediates and print the
same labels, so `cmp.py` diffs the two mechanically (it reads either format on
either side, so two qwencpp runs can also be compared against each other):

```
python cmp.py 04-tensors.log 13-qw-offload.log
```

`13-qw-offload.log` is the offload path (`--vram-budget 20 --experts-ssd`, CPU
backend) against the same GGUF. Two facts worth keeping:

- Re-running it at a different budget (29% vs ~10% residency) gives **62/62
  identical sums**. The expert cache decides where a weight comes from, never
  what it is, and this is the check that says so.
- Its remaining deltas are no larger than those between the Metal and CPU
  backends running the *same* qwencpp code, so they are kernel noise rather than
  a fault in the offload graphs. Layer 0 agrees to 8e-7.

## SWA ring A/B

`QWEN_SWA_RING=0` turns the sliding-window KV ring off, which is how the ring is
checked: same prompt, ring on and off, all tensor sums compared.

```
QWEN_SEGA_CHUNK=64 QWEN_PREFILL_CHUNK=64 QWEN_SWA_RING=1 QWEN_DUMP_LAYERS=0,3 qw-cli -m <step35> -p "$(cat wrap-prompt.txt)" -n 1   --n-ctx 2048 --vram-budget 20 --experts-ssd | grep qw_dump > wrap-1.log
```

**Confirm the ring size in the log before trusting a run.** It is
`n_swa + max(QWEN_PREFILL_CHUNK, QWEN_SEGA_CHUNK)`, so both knobs have to be set
to shrink it, and the prompt has to be longer than the result or the ring never
wraps and the run proves nothing:

```
SWA KV ring: 576 rows on 33 sliding layers (12 full at 2048) = 0.18 GB instead of 0.38 GB
```

`wrap-prompt.txt` is 640 tokens, which wraps a 576-row ring.

The two runs are not bit-identical and cannot be: a ring layer's attention reads
the whole ring while a full one reads n_kv columns, so the softmax and the KQ
matmul accumulate in a different order. What must hold is that the ring keeps the
same *key positions*, and `ring_mask_check.py` proves that directly against the
same formulas the runtime uses -- including that an undersized ring fails, which
is what the runtime's own size check exists to prevent.

## Caveats

- **`07-bench.log` is a cold-cache artifact** (tg 17.64 t/s). The identical
  configuration re-run warm is in `08-fa.log` at 51.35 t/s. Always warm the page
  cache before benching a 50 GB+ model on Metal.
- **These captures do not exercise SWA.** A 6-token prompt fits inside the
  512-token window, so `kq-0` (full attention) and `kq-3` (sliding) have the same
  shape. Verifying the band mask needs a >512-token prompt.
