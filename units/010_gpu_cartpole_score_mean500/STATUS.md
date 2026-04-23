# STATUS.md

## Purpose

- Try to achieve perfect population mean reward `500.00` on the direct CartPole score task in the new GPU-native environment, first for static networks and then for dynamic networks.

## Planned Scope

- Static network score optimization on the GPU-native CartPole engine.
- Dynamic network score optimization on the GPU-native CartPole engine.
- Comparison under matched wall-clock budgets, not matched generations.

## Constraints

- Use explicit wall-clock budgets with internal deadlines and external hard timeouts.
- Report reward and throughput in wall-clock terms.
- The first goal is population mean `500.00`, not just best individual `500.00`.

## Verification

- Not started yet.

## Next Steps

- Build the static-network score runner on top of the verified Unit 8 Torch CartPole physics.
- Use a short timed smoke run before any larger search.
