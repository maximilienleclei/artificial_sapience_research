# 013_experiment_runtime_tooling

Runtime tooling unit for invisible background execution and status polling.

This unit owns the executable workflow code that later units use for:

- launching invisible bounded background runs
- writing machine-readable run status files
- checking live run progress without blocking chat

It exists so this executable workflow layer is archived as a numbered unit rather than evolving as mutable repo-root code.
