# BACKGROUND_RUNS.md

Background runs are allowed in this repo when they improve responsiveness, but they must follow a strict tracking pattern.

## Goals

- keep chat responsive while longer optimization continues
- avoid visible terminals or desktop popups
- avoid confusion about which files came from which run
- make progress easy to inspect and easy to stop

## Required Pattern

For any background run:

1. Give it a unique run name.
2. Write all in-progress outputs to a dedicated run directory first.
3. Write a machine-readable `run_status.json` file for that run.
4. Record:
   - unit
   - run name
   - command
   - working directory
   - machine / device
   - start time
   - scheduled stop deadline
   - main output paths
   - whether the run is `running`, `completed`, `stopped`, or `failed`
5. Keep canonical unit artifacts unchanged until the run completes and is verified.

## Preferred Layout

Within a unit, prefer a structure like:

```text
model/runs/<run_name>/
plot/runs/<run_name>/
run_status.json
```

If both model and plot artifacts are produced, record both directories in `run_status.json`.

## Progress Tracking

- Append metrics incrementally inside the run-specific directory.
- Status checks should read the run status file and the latest metrics file, not guess from timestamps alone.
- If there is an active background run, check it before starting another one unless the user explicitly wants parallel runs.

## Stop Rules

- Every background run must have a predetermined stop deadline.
- The run must be stoppable using the information recorded in `run_status.json`.
- Do not rely on memory or "the latest python process" to stop a run.

## Promotion Rule

Only after a background run finishes and the outputs are verified:

- copy or rename the chosen artifacts into the unit's canonical `model/` and `plot/` locations
- update the unit `STATUS.md`
- then commit, if git state allows it
