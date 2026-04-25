# STATUS.md

## Purpose

- Compare a dynamic-topology network GA against Unit 1 on the same direct CartPole reward task.

## Contents

- `code/dynamic_cartpole_ga.py`: standalone dynamic graph-network GA for `CartPole-v1`.
- `code/plot_metrics.py`: regenerates the reward/topology curve.
- `plot/`: metrics and curve artifacts.
- `model/`: best evolved dynamic genome.

## Port Notes

- Source idea: `../ai_research/common/ne/popu/nets/dynamic`.
- This unit keeps the dynamic topology behavior: nodes can grow, prune, rewire, and mutate weights.
- It intentionally does not port the old Hydra, `common/`, Welford standardization, or batched tensor framework.
- Target task matches Unit 1: direct reward optimization on `CartPole-v1`, max 500 steps.

## Verification

- Verified on April 23, 2026 using `C:\Users\Max\venv\Scripts\python.exe` on the AMD Radeon RX 7800 XT machine.
- Command: `python .\dynamic_cartpole_ga.py --generations 100 --population-size 64 --elite-count 8 --episodes-per-individual 3`
- Result: best reward was `500.00` from generation 0 through generation 99.
- Final generation: `best=500.00`, `mean=446.02`, `median=500.00`, `mean_hidden=1.188`, `max_hidden=2`.
- Best mean generation: generation 92 with `mean=491.93`, `median=500.00`, `mean_hidden=1.109`, `max_hidden=2`.
- Runtime: `243.222s` for 100 generations.
- Artifacts: `plot/metrics.csv`, `plot/curve.svg`, and `model/best_genome.json`.
- Comparison to Unit 1: Unit 1 reached best reward `500.00` by generation 10 and ended with mean `463.29`; this dynamic unit found a perfect individual immediately and reached a higher peak mean, but ended with a slightly lower final mean.

## Next Steps

- Decide whether to run multiple seeds to estimate robustness, since this single seed solved CartPole unusually early.
