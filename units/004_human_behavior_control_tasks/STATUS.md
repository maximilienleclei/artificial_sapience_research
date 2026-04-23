# STATUS.md

## Purpose

- Preserve human behavior control-task data for future imitation and comparison work.

## Contents

- Two participant datasets for Acrobot, CartPole, LunarLander, and MountainCar.
- Original scripts: `collect.py`, `data_analysis.py`, and `replay.py`.
- Existing summary plot: `plot/training_sessions_plot.png`.

## Verification

- Imported from `../ai_research/data/human_behaviour_control_tasks` on April 22, 2026.
- Plot regeneration verified on April 23, 2026 using `C:\Users\Max\venv\Scripts\python.exe` from the `data/` directory with `MPLBACKEND=Agg`.
- The script reported 79,581 total recorded steps across 8 participant/task files and regenerated `plot/training_sessions_plot.png`.
- The default `python` on this machine does not have `matplotlib`; use the documented venv or install plotting dependencies before rerunning.
- Not yet rerun or adapted to the current active codebase.

## Next Steps

- Decide whether the next active experiment should imitate human behavior from these recordings.
- If used, rewrite the loader/evaluator for the current repo rather than preserving the old framework shape.
