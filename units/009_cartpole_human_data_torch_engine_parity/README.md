# 009_cartpole_human_data_torch_engine_parity

Replay Unit 4 human CartPole transitions through the Torch CartPole engine from Unit 8.

This unit does not train anything. It checks whether Unit 8's ported CartPole physics reproduces the recorded `next_observation`, `reward`, and `done` fields in Unit 4's human CartPole data.

Run from `code/`:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\replay_human_cartpole.py --device cuda
```

Findings:

- one-step CartPole replay against Unit 4 was extremely close in `float64` and `float32`
- long-horizon state trajectories can drift, but final episode returns still matched exactly on all recorded CartPole episodes
- `float16` preserved returns here, but `float32` is the safer default
