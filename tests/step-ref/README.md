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

## Caveats

- **`07-bench.log` is a cold-cache artifact** (tg 17.64 t/s). The identical
  configuration re-run warm is in `08-fa.log` at 51.35 t/s. Always warm the page
  cache before benching a 50 GB+ model on Metal.
- **These captures do not exercise SWA.** A 6-token prompt fits inside the
  512-token window, so `kq-0` (full attention) and `kq-3` (sliding) have the same
  shape. Verifying the band mask needs a >512-token prompt.
