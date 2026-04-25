# 011_gpu_cartpole_adversarial_generation

Minimal port of the old adversarial-generation framework onto the GPU-native CartPole environment path.

This keeps the useful core:

- a dual-head population with action outputs plus one discriminator output
- target-agent rollouts from the existing Unit 3 PPO checkpoint
- generator fitness from discriminator scores on generated trajectories
- discriminator fitness from target-vs-generated separation

Run a short static smoke test:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\gpu_cartpole_adv_gen.py --population-kind static --device cuda --time-budget-s 8
```

Current static smoke result:

- best total fitness `1.2090`
- best mean total fitness `0.8964`
- mean generator environment reward reached `63.56`
