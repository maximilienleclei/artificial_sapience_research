from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

UNIT8_CODE = Path(__file__).resolve().parents[2] / "008_torch_cartpole_physics_parity" / "code"
if str(UNIT8_CODE) not in sys.path:
    sys.path.insert(0, str(UNIT8_CODE))

from torch_cartpole import step_cartpole  # noqa: E402

MAX_INCOMING = 3


class StaticAdvPopulation:
    def __init__(self, ih: torch.Tensor, hb: torch.Tensor, ho: torch.Tensor, ob: torch.Tensor, device: torch.device):
        self.ih = ih
        self.hb = hb
        self.ho = ho
        self.ob = ob
        self.device = device

    @classmethod
    def random(cls, pop: int, hidden: int, device: torch.device, seed: int) -> "StaticAdvPopulation":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        scale = 0.5
        dtype = torch.float32
        return cls(
            torch.randn(pop, 4, hidden, generator=g, device=device, dtype=dtype) * scale,
            torch.randn(pop, 1, hidden, generator=g, device=device, dtype=dtype) * scale,
            torch.randn(pop, hidden, 3, generator=g, device=device, dtype=dtype) * scale,
            torch.randn(pop, 1, 3, generator=g, device=device, dtype=dtype) * scale,
            device,
        )

    @property
    def population_size(self) -> int:
        return int(self.ih.shape[0])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.unsqueeze(1)
        hidden = torch.tanh(torch.bmm(x, self.ih) + self.hb)
        return torch.bmm(hidden, self.ho) + self.ob

    def get_actions(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs).squeeze(1)[:, :2].argmax(dim=1)

    def get_discrimination(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(obs).squeeze(1)[:, 2])

    def select(self, idx: torch.Tensor) -> "StaticAdvPopulation":
        return StaticAdvPopulation(self.ih[idx].clone(), self.hb[idx].clone(), self.ho[idx].clone(), self.ob[idx].clone(), self.device)

    def reproduce(self, elite_idx: torch.Tensor, mutation_std: float, seed: int) -> "StaticAdvPopulation":
        elites = self.select(elite_idx)
        elite_count = int(elite_idx.shape[0])
        if elite_count == self.population_size:
            return elites
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        child_count = self.population_size - elite_count
        parent = torch.randint(0, elite_count, (child_count,), generator=g, device=self.device)
        def mutate(x: torch.Tensor) -> torch.Tensor:
            return x[parent].clone() + torch.randn(x[parent].shape, generator=g, device=self.device) * mutation_std
        return StaticAdvPopulation(
            torch.cat([elites.ih, mutate(elites.ih)], 0),
            torch.cat([elites.hb, mutate(elites.hb)], 0),
            torch.cat([elites.ho, mutate(elites.ho)], 0),
            torch.cat([elites.ob, mutate(elites.ob)], 0),
            self.device,
        )


class DynamicAdvPopulation:
    def __init__(self, sources: torch.Tensor, weights: torch.Tensor, active: torch.Tensor, passes: int, device: torch.device):
        self.sources = sources
        self.weights = weights
        self.active = active
        self.input_size = 4
        self.output_size = 3
        self.passes = passes
        self.device = device

    @classmethod
    def random(cls, pop: int, max_hidden: int, passes: int, device: torch.device, seed: int, initial_scale: float) -> "DynamicAdvPopulation":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        mutable = 3 + max_hidden
        sources = torch.randint(0, 4, (pop, mutable, MAX_INCOMING), generator=g, device=device)
        weights = torch.randn(pop, mutable, MAX_INCOMING, generator=g, device=device) * initial_scale
        active = torch.zeros(pop, mutable, dtype=torch.bool, device=device)
        active[:, :3] = True
        return cls(sources, weights, active, passes, device)

    @property
    def population_size(self) -> int:
        return int(self.sources.shape[0])

    @property
    def mutable_count(self) -> int:
        return int(self.sources.shape[1])

    @property
    def total_nodes(self) -> int:
        return self.input_size + self.mutable_count

    def hidden_counts(self) -> torch.Tensor:
        return self.active[:, self.output_size :].sum(dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        values = torch.zeros(obs.shape[0], self.total_nodes, device=self.device)
        values[:, :4] = obs
        idx = self.input_size + torch.arange(self.mutable_count, device=self.device)
        for _ in range(self.passes):
            prev = values
            gathered = prev.gather(1, self.sources.reshape(obs.shape[0], -1)).reshape(obs.shape[0], self.mutable_count, MAX_INCOMING)
            out = torch.tanh((gathered * self.weights).sum(dim=2))
            out = torch.where(self.active, out, torch.zeros_like(out))
            values = values.clone()
            values[:, idx] = out
        return values[:, 4:7]

    def get_actions(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)[:, :2].argmax(dim=1)

    def get_discrimination(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(obs)[:, 2])

    def select(self, idx: torch.Tensor) -> "DynamicAdvPopulation":
        return DynamicAdvPopulation(self.sources[idx].clone(), self.weights[idx].clone(), self.active[idx].clone(), self.passes, self.device)

    def reproduce(self, elite_idx: torch.Tensor, mutation_std: float, seed: int, grow_probability: float, prune_probability: float, rewire_probability: float) -> "DynamicAdvPopulation":
        elites = self.select(elite_idx)
        elite_count = int(elite_idx.shape[0])
        child_count = self.population_size - elite_count
        if child_count <= 0:
            return elites
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        parent = torch.randint(0, elite_count, (child_count,), generator=g, device=self.device)
        child_sources = elites.sources[parent].clone()
        child_weights = elites.weights[parent].clone()
        child_active = elites.active[parent].clone()
        child_weights += torch.randn(child_weights.shape, generator=g, device=self.device) * mutation_std
        rewire_mask = torch.rand(child_sources.shape, generator=g, device=self.device) < rewire_probability
        rand_sources = torch.randint(0, self.total_nodes, child_sources.shape, generator=g, device=self.device)
        child_sources = torch.where(rewire_mask, rand_sources, child_sources)
        grow_draw = torch.rand(child_count, generator=g, device=self.device)
        for i in torch.nonzero(grow_draw < grow_probability, as_tuple=False).flatten().tolist():
            inactive = torch.nonzero(~child_active[i, self.output_size :], as_tuple=False).flatten()
            if inactive.numel():
                slot = self.output_size + int(inactive[0].item())
                child_active[i, slot] = True
        prune_draw = torch.rand(child_count, generator=g, device=self.device)
        for i in torch.nonzero(prune_draw < prune_probability, as_tuple=False).flatten().tolist():
            active_hidden = torch.nonzero(child_active[i, self.output_size :], as_tuple=False).flatten()
            if active_hidden.numel():
                slot = self.output_size + int(active_hidden[0].item())
                child_active[i, slot] = False
        return DynamicAdvPopulation(
            torch.cat([elites.sources, child_sources], 0),
            torch.cat([elites.weights, child_weights], 0),
            torch.cat([elites.active, child_active], 0),
            self.passes,
            self.device,
        )


def rollout_generator(population, max_steps: int, seed: int, deadline: float) -> tuple[list[torch.Tensor], torch.Tensor]:
    pop = population.population_size
    g = torch.Generator(device=population.device)
    g.manual_seed(seed)
    state = torch.empty(pop, 4, device=population.device).uniform_(-0.05, 0.05, generator=g)
    done = torch.zeros(pop, dtype=torch.bool, device=population.device)
    rewards = torch.zeros(pop, device=population.device)
    traj = []
    for _ in range(max_steps):
        if time.perf_counter() >= deadline:
            break
        traj.append(state.clone())
        action = population.get_actions(state)
        state, reward, terminated = step_cartpole(state, action)
        rewards += reward * (~done).float()
        done |= terminated
    return traj, rewards


def rollout_target(target_agent: PPO, pop: int, max_steps: int, seed: int, device: torch.device, deadline: float) -> tuple[list[torch.Tensor], torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    state = torch.empty(pop, 4, device=device).uniform_(-0.05, 0.05, generator=g)
    done = torch.zeros(pop, dtype=torch.bool, device=device)
    rewards = torch.zeros(pop, device=device)
    traj = []
    for _ in range(max_steps):
        if time.perf_counter() >= deadline:
            break
        traj.append(state.clone())
        actions, _ = target_agent.predict(state.detach().cpu().numpy(), deterministic=True)
        action_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
        state, reward, terminated = step_cartpole(state, action_t)
        rewards += reward * (~done).float()
        done |= terminated
    return traj, rewards


def evaluate(population, target_agent: PPO, max_steps: int, num_disc_samples: int, seed: int, deadline: float):
    pop = population.population_size
    traj_g, env_rewards = rollout_generator(population, max_steps, seed, deadline)
    disc_to_gen = torch.randint(0, pop, (pop, num_disc_samples), device=population.device)
    fitness_G = torch.zeros(pop, device=population.device)
    D_x_G_sum = torch.zeros(pop, device=population.device)
    gen_eval_counts = torch.zeros(pop, device=population.device)
    for k in range(num_disc_samples):
        gen_idx = disc_to_gen[:, k]
        sample_sum = torch.zeros(pop, device=population.device)
        for obs in traj_g:
            scores = population.get_discrimination(obs[gen_idx])
            fitness_G.scatter_add_(0, gen_idx, scores)
            sample_sum += scores
        ones = torch.ones(pop, device=population.device)
        gen_eval_counts.scatter_add_(0, gen_idx, ones)
        D_x_G_sum += sample_sum / max(1, len(traj_g))
    fitness_G = fitness_G / (gen_eval_counts * max(1, len(traj_g))).clamp(min=1)

    traj_t, target_rewards = rollout_target(target_agent, pop, max_steps, seed + 999, population.device, deadline)
    fitness_D = torch.zeros(pop, device=population.device)
    for obs in traj_t:
        fitness_D += population.get_discrimination(obs)
    fitness_D = fitness_D / max(1, len(traj_t))
    fitness_D = fitness_D - (D_x_G_sum / num_disc_samples)
    total = fitness_G + fitness_D
    return (
        total.detach().cpu().numpy(),
        fitness_G.detach().cpu().numpy(),
        fitness_D.detach().cpu().numpy(),
        env_rewards.detach().cpu().numpy(),
        target_rewards.detach().cpu().numpy(),
        population.hidden_counts().detach().cpu().numpy() if hasattr(population, "hidden_counts") else None,
    )


def write_metrics(path: Path, history: list[dict]) -> None:
    if not history:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def run(args: argparse.Namespace) -> None:
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    metrics_path = Path(args.metrics_path)
    best_path = Path(args.best_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    target_agent = PPO.load(args.target_agent_path)
    if args.population_kind == "static":
        population = StaticAdvPopulation.random(args.population_size, args.hidden_size, device, args.seed)
    else:
        population = DynamicAdvPopulation.random(
            args.population_size, args.max_hidden, args.passes, device, args.seed, args.initial_scale
        )

    start = time.perf_counter()
    deadline = start + args.time_budget_s
    history = []
    generation = 0
    best_json = None
    while generation < args.generations and time.perf_counter() < deadline:
        fitness, fitness_g, fitness_d, env_r, target_r, hidden = evaluate(
            population, target_agent, args.max_steps, args.num_disc_samples, args.seed + generation * 10000, deadline
        )
        ft = torch.as_tensor(fitness, dtype=torch.float32, device=device)
        elite_fit, elite_idx = torch.topk(ft, k=args.elite_count, largest=True, sorted=True)
        row = {
            "generation": generation,
            "best": round(float(np.max(fitness)), 4),
            "mean": round(float(np.mean(fitness)), 4),
            "mean_fitness_G": round(float(np.mean(fitness_g)), 4),
            "mean_fitness_D": round(float(np.mean(fitness_d)), 4),
            "mean_env_reward": round(float(np.mean(env_r)), 4),
            "mean_target_reward": round(float(np.mean(target_r)), 4),
            "elapsed_s": round(time.perf_counter() - start, 3),
            "time_budget_s": args.time_budget_s,
            "device": str(device),
            "population_kind": args.population_kind,
        }
        if hidden is not None:
            row["mean_hidden"] = round(float(np.mean(hidden)), 3)
        history.append(row)
        best_idx = int(elite_idx[0].item())
        best_json = {"fitness": float(elite_fit[0].item()), "population_kind": args.population_kind}
        write_metrics(metrics_path, history)
        best_path.write_text(json.dumps(best_json, indent=2) + "\n")
        print(
            "generation={generation:03d} best={best:.2f} mean={mean:.2f} mean_env_reward={mean_env_reward:.2f} "
            "mean_fitness_G={mean_fitness_G:.3f} mean_fitness_D={mean_fitness_D:.3f} elapsed_s={elapsed_s:.3f}".format(**row)
        )
        if time.perf_counter() >= deadline:
            break
        if args.population_kind == "static":
            population = population.reproduce(elite_idx, args.mutation_std, args.seed + generation + 1)
        else:
            population = population.reproduce(
                elite_idx, args.mutation_std, args.seed + generation + 1, args.grow_probability, args.prune_probability, args.rewire_probability
            )
        generation += 1
    if not history:
        raise RuntimeError("Time budget expired before completing one generation.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population-kind", default="static", choices=["static", "dynamic"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--time-budget-s", type=float, default=8.0)
    parser.add_argument("--generations", type=int, default=100000)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-disc-samples", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--mutation-std", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--max-hidden", type=int, default=24)
    parser.add_argument("--initial-scale", type=float, default=0.5)
    parser.add_argument("--grow-probability", type=float, default=0.25)
    parser.add_argument("--prune-probability", type=float, default=0.03)
    parser.add_argument("--rewire-probability", type=float, default=0.02)
    parser.add_argument(
        "--target-agent-path",
        default=str(
            Path(__file__).resolve().parents[2]
            / "003_cartpole_sb3_ppo_action_imitation"
            / "model"
            / "ppo-CartPole-v1.zip"
        ),
    )
    parser.add_argument("--metrics-path", default="../plot/adv_metrics.csv")
    parser.add_argument("--best-path", default="../model/adv_best.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
