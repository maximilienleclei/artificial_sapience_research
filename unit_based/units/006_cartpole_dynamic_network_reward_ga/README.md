# 006_cartpole_dynamic_network_reward_ga

Dynamic-topology network genetic algorithm optimizing CartPole reward directly.

This is a minimal port of the old dynamic-network idea from `../ai_research/common/ne/popu/nets/dynamic`, targeted at the same task as Unit 1 without bringing over the old Hydra/shared-framework stack.

Run from `code/`:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\dynamic_cartpole_ga.py --generations 100 --population-size 64
```

Regenerate the plot:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\plot_metrics.py
```

Verified 100-generation result: best reward stayed at `500.00`; final mean reward was `446.02`; peak mean reward was `491.93`.
