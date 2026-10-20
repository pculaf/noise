# Sources and scope

- Baseline preserved and pushed: `research/readout-kernel`, `d3145e6` (19.50 s, val 4.178345).
- New work: `research/torch-moments`; no changes to the validated branch.
- Robert T. Lange's [PR #29](https://github.com/qlabs-eng/noise/pull/29), revision `866615d3a0332219b3d62ef8eaadc9c9d905d2fe`: attention moments, cubic readout expansion, head sharding, batched/sharded ANVIL, multi-update dispatch. Ported formulas must be credited to this work. Its MIT source snapshot is under `reference/`.
- [PR #28](https://github.com/qlabs-eng/noise/pull/28): structured initialization, a later separate experiment.
- [PR #27](https://github.com/qlabs-eng/noise/pull/27): final weight averaging, a later separate experiment.

First verify isolated formulas against explicit forward perturbations. Then measure components and full updates on Modal. No numerical model execution locally. Do not claim a faithful full port, convergence, speedup, or new record until measured.

New GPU round authorized: maximum EUR 10. Use a conservative USD 9 ceiling with startup-inclusive reservations, no automatic retries, and bounded runtimes. The previous round is separate.
