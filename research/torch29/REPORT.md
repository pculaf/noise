# PR29 PyTorch experiment

This is research code, not a validated record submission. The previous validated
19.50-second result is unchanged at `research/readout-kernel`, commit `d3145e6`,
and is backed up at https://github.com/pculaf/noise/tree/research/readout-kernel.

## What was tested

The reference is Robert T. Lange's PR29, revision
`866615d3a0332219b3d62ef8eaadc9c9d905d2fe`. Its polynomial attention moments,
cubic readout expansion, head exchange, sharded ANVIL and multi-update execution
are credited prior work; see PROVENANCE.md. The Torch initializer is the
benchmark's seed-0 initializer, not the JAX RNG's bitwise initial state.

Changes beyond the direct formula port:

- Reuse moments for the two raw coordinates that touch the same RoPE pair.
  Each coordinate retains its own normalization-change and rotation coefficients.
  This removes duplicated accumulation without merging their perturbations.
- Batch independent layers with the same attention window into the kernel's
  leading batch dimension.
- Evaluate the readout loss difference directly with log1p and a tanh-difference
  identity, avoiding subtraction of complete cross-entropy losses.

## Findings

Tests ran remotely on Modal H100 GPUs. No local numerical model execution.
Component timings use warmed CUDA graph replay; they are not whole-training
times, and the reference is our PyTorch port of the PR29 formulas, not the
author's JAX executable or the maintainer's unpublished PyTorch implementation.

- Explicit finite-perturbation checks on small tensors: the BF16 moment port's
  Q/K errors were about 0.38% relative; value errors about 0.30–0.33%.
- On identical compiled inputs and tile sizes, paired-coordinate reuse was
  bit-identical to the unpaired port in the component comparisons.
- Stable readout differences reduced the initial-scale relative error against
  explicit float64 forward differences from about 0.00418 to 0.0000104.
  This is an accuracy result; improved convergence has not been established.
- A fused six-product readout kernel was slower and is not used by the trainer.
- Applying moment coefficients inside the tiled loop was slower than applying
  them after accumulation and is not used by the trainer.
- Single-GPU training and 15-update graph replay passed, with a maximum loss
  difference of about 0.00073 against eager execution on the repeated diagnostic
  batch. This is not a convergence experiment.
- The first eight-GPU attempt stopped on a nonfinite loss in the first group of
  updates. Its preceding relative-error comparison alone was insufficient:
  NaN checks were missing from that comparison. Subsequent diagnostics explicitly
  check finite estimates and weights before capture.

Whole attention-credit component across all eight layers, on one H100 at one
rank's PR29 workload shape (`torch29-batch-11`):

| Context | Global sequences | Direct formula port | Paired reuse | Paired + layer batching |
| --- | --- | --- | --- | --- |
| 1024 | 8 | 5.246 ms | 3.286 ms | 2.553 ms |
| 2048 | 4 | 7.664 ms | 4.802 ms | 3.814 ms |

Paired reuse is bit-identical here. Layer batching changes the compiler's
reduction layout: relative errors are 0.0000239 and 0.0000290, respectively.
The combined component speedups are 2.055x and 2.009x. This does **not** establish
a speedup over the JAX executable, nor over the maintainer's unpublished port.
In particular, the original PR's profiling reports faster component timings
than this initial direct Torch port. An end-to-end qualifying run is still needed.

## Final distributed outcome and stop

The queued retry (`torch29-distributed-10`) was cancelled with no running
containers at the user's request. Its replacement (`torch29-distributed-12`)
passed finite-gradient and finite-weight warmup checks in both context phases.
The real-batch paired/unpaired estimate comparison had maximum relative error
6.02e-8. It completed all 975 updates, then failed the final check that all eight
ranks have bit-identical weights. The magnitude and cause of the discrepancy
have not been established. No validation loss or final checkpoint was written:
the harness intentionally raises before those operations on a failed check.

The last progress report was update 915 at 8.331 seconds, training loss 4.20123.
This is not validation loss. The full elapsed time was not saved because the
check failed before the result write. It was already slower than PR29's 6.76
seconds. This run used paired reuse and stable readout, **not** layer batching,
which was assessed separately in the component benchmark above.

No more experiments were launched after the user requested stopping upon
completion of this evaluation (apart from the explicitly requested replacement
of the unallocated retry). There is no qualifying new record and no PR was
opened. Keep the validated 19.50-second branch separate from this research port.

New-round budget: USD 3.505 measured function-runtime estimate; USD 6.643 with
conservative startup/build allowances, within the USD 9 internal ceiling used
for the user's EUR 10 cap. These are estimates, not a Modal billing invoice.
Source snapshots, full console logs and per-rank warmup diagnostics remain in
the local `results/` folders; compact outcomes are in `campaign_summary.json`.

PR29's 6.21-second alternative is a reported mean over seeds 42–46 with increased
rates and Drift Leap. Its author reports seed-0 losses above 4.2 and did not
submit it as the qualifying record. It is useful evidence of performance
potential, not a seed-0 record that this experiment has beaten.

## Reading the code

- `moments.py`: attributed PyTorch formula reference and stable readout variant.
- `attention_moments.py`: Triton moment kernels, with optional coordinate reuse.
- `torch_train.py`: experimental trainer; `distributed_worker.py`: run harness.
- `components_worker.py`, `screen_worker.py`, `batch_worker.py`: component checks.
- `diagnostic_worker.py`: finite-estimate and eager/graph checks.
- `readout_moments.py`: unsuccessful fused readout candidate, retained for evidence.
- `results/` and `budget.json`: local run artifacts and conservative reservations.
