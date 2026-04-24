# STATUS.md

## Purpose

- Archive the shared executable runtime/workflow code used by later units for invisible background runs and live status inspection.

## Contents

- `code/start_background_run.py`: launch a tracked invisible background run.
- `code/check_background_run.py`: inspect the latest run status and logs for a unit.
- `code/background_worker.py`: detached worker that manages the bounded run lifecycle.
- PowerShell wrappers in `code/` for the same background-run workflow.

## Notes

- This unit precedes the PPO-clone convergence workflow units because those units depend on this executable runtime layer.
- Repo-root docs may stay shared, but mutable executable code should not live outside numbered units.

## Verification

- Introduced on April 23, 2026 from the previously shared repo-root `tools/` directory.
- Validated on Windows using `pythonw.exe` for the detached launcher and `python.exe` for the actual experiment process.

## Next Steps

- Keep later workflow changes versioned here instead of reintroducing mutable repo-root executable code.
