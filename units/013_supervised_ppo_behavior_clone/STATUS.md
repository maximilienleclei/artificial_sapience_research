# STATUS.md

## Purpose

- Establish the plain supervised-learning baseline for PPO behavior cloning before comparing against GA-based methods.

## Contents

- `code/train_and_eval_clone.py`: bounded PPO dataset collection, supervised training, and closed-loop benchmark evaluation.
- `code/plot_clone_results.py`: training/evaluation figure generation.
- `data/`: saved PPO training dataset.
- `model/`: trained clone checkpoint and metrics JSON.
- `plot/`: benchmark comparison outputs.

## Notes

- This is the baseline that the upcoming GA behavior-matching unit should beat on the same closed-loop benchmark.
- The supervised clone should report both training and validation action accuracy before the closed-loop benchmark comparison.

## Verification

- Initial bounded smoke run on April 23, 2026 used `C:\Users\Max\venv\Scripts\python.exe` on the AMD GPU machine.
- Saved smoke result:
  - `1946` dataset steps collected from `4` PPO rollout episodes
  - train/val split `1557 / 389`
  - final train accuracy `0.9395`
  - best validation accuracy `0.9769`
  - closed-loop clone benchmark completed `3` Unit 12 seeds
  - clone return mean/std `469.0 / 43.84`
  - clone action=`1` mean rate `0.4996`
  - clone action-switch-rate mean `0.6248`
  - absolute delta vs PPO benchmark:
    - return mean `31.0`
    - action=`1` mean rate `0.00041`
    - action-switch-rate mean `0.0425`

## Artifacts

- `data/ppo_train_dataset.csv`
- `model/clone_policy.pt`
- `model/metrics.json`
- `plot/benchmark_comparison.json`
- `plot/episode_metrics.csv`
- `plot/clone_results.svg`

## Next Steps

- Increase the supervised data budget and benchmark completion count to get a more stable baseline.
- Compare this baseline against GA behavior-matching on the same Unit 12 seeds and metrics.
