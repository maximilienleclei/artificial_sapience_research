# STATUS.md

## Purpose

- Test whether Unit 8's Torch CartPole engine matches the recorded human CartPole transitions in Unit 4.

## Contents

- `code/replay_human_cartpole.py`: offline replay/parity check against Unit 4 CartPole data.
- `plot/`: parity summary table and error distribution plot.

## Verification

- Verified on April 23, 2026 using `C:\Users\Max\venv\Scripts\python.exe` on the AMD Radeon RX 7800 XT machine with a hard timeout.
- One-step replay against Unit 4 CartPole data matched extremely closely:
  - `float64` max next-state error: about `2.26e-07` to `2.29e-07`
  - `float32` max next-state error: about `2.38e-07`
  - `float16` max next-state error: `1.953e-03`
- One-step reward match was `100%` for both participants in all tested dtypes.
- Full-episode free replay under recorded action sequences showed state drift over long horizons, especially on `sub01`, but episode returns still matched exactly for every recorded CartPole episode in `float64`, `float32`, and `float16`.
- First `sub01` float64 divergence above `1e-4` happened at episode 11, step 142, with max state error `1.03e-04`; it was not a done mismatch.
- Artifacts: `plot/summary.csv`, `plot/rollout_summary.csv`, `plot/sub01_first_divergence.json`, and `plot/human_cartpole_engine_parity.svg`.

## Next Steps

- Treat Unit 8's engine as behaviorally good enough for CartPole reward work, with `float32` the practical default and `float64` the parity reference.
