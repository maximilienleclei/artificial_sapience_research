# STATUS.md

## Purpose

- Port the adversarial-generation framework onto the GPU-native CartPole environment path.

## Contents

- `code/gpu_cartpole_adv_gen.py`: minimal static/dynamic adversarial-generation runner.
- `code/plot_adv_metrics.py`: curve generation for adversarial metrics.
- `plot/`: adversarial metric artifacts.
- `model/`: best saved adversarial individual metadata.

## Port Notes

- Source idea: old `common/ne/eval/adv_gen.py` and `common/ne/popu/adv_gen.py`.
- This port uses Unit 8 Torch CartPole physics and the Unit 3 PPO checkpoint as the target agent.
- It intentionally avoids the old Hydra/shared framework and keeps only the core co-evolution logic.

## Verification

- Static adversarial-generation smoke run verified on April 23, 2026 using `C:\Users\Max\venv\Scripts\python.exe` on the AMD Radeon RX 7800 XT machine.
- Command: `python .\gpu_cartpole_adv_gen.py --population-kind static --device cuda --time-budget-s 8`, with a 10-second external hard timeout.
- The run produced flushed metrics through generation 14 before the hard timeout boundary.
- Best total fitness reached `1.2090`; best mean total fitness reached `0.8964` at elapsed `7.647s`.
- Mean generator environment reward increased from `16.02` at generation 0 to a peak saved value of `63.56` at generation 13.
- Mean generator fitness and discriminator fitness both rose during the smoke run, which is enough to confirm the minimal framework is live.
- Artifacts: `plot/adv_metrics.csv`, `plot/adv_curve.svg`, and `model/adv_best.json`.

## Next Steps

- Try the dynamic population on the same timed budget.
- Decide whether to keep using the PPO checkpoint target or to move to a fully GPU-native target policy path later.
