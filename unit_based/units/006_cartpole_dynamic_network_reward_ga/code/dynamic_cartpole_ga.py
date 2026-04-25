from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np


MAX_INCOMING = 3


@dataclass
class DynamicGenome:
    input_size: int
    output_size: int
    sources: list[list[int]]
    weights: list[np.ndarray]
    hidden_count: int = 0
    passes: int = 3

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        input_size: int,
        output_size: int,
        initial_scale: float,
        passes: int,
    ) -> DynamicGenome:
        sources = []
        weights = []
        for _ in range(output_size):
            src = rng.choice(input_size, size=MAX_INCOMING, replace=True).astype(int).tolist()
            sources.append(src)
            weights.append(rng.normal(0.0, initial_scale, size=MAX_INCOMING))
        return cls(input_size, output_size, sources, weights, hidden_count=0, passes=passes)

    @property
    def mutable_count(self) -> int:
        return self.output_size + self.hidden_count

    @property
    def total_nodes(self) -> int:
        return self.input_size + self.mutable_count

    def clone(self) -> DynamicGenome:
        return DynamicGenome(
            input_size=self.input_size,
            output_size=self.output_size,
            sources=[src.copy() for src in self.sources],
            weights=[w.copy() for w in self.weights],
            hidden_count=self.hidden_count,
            passes=self.passes,
        )

    def act(self, observation: np.ndarray) -> int:
        values = np.zeros(self.total_nodes, dtype=np.float64)
        values[: self.input_size] = observation
        mutable_start = self.input_size
        for _ in range(self.passes):
            previous = values.copy()
            for idx, (src, weight) in enumerate(zip(self.sources, self.weights, strict=True)):
                values[mutable_start + idx] = np.tanh(float(previous[src] @ weight))
        output_values = values[self.input_size : self.input_size + self.output_size]
        return int(np.argmax(output_values))

    def mutate(
        self,
        rng: np.random.Generator,
        mutation_std: float,
        grow_probability: float,
        prune_probability: float,
        rewire_probability: float,
        max_hidden: int,
    ) -> DynamicGenome:
        child = self.clone()
        child.weights = [w + rng.normal(0.0, mutation_std, size=w.shape) for w in child.weights]

        for node_idx, src in enumerate(child.sources):
            for slot in range(MAX_INCOMING):
                if rng.random() < rewire_probability:
                    src[slot] = int(rng.integers(0, child.total_nodes))
                    if src[slot] == child.input_size + node_idx:
                        src[slot] = int(rng.integers(0, child.input_size))

        if child.hidden_count < max_hidden and rng.random() < grow_probability:
            child._grow_node(rng, initial_scale=mutation_std)

        if child.hidden_count > 0 and rng.random() < prune_probability:
            child._prune_hidden_node(rng)

        child._sanitize_sources(rng)
        return child

    def _grow_node(self, rng: np.random.Generator, initial_scale: float) -> None:
        new_node_index = self.total_nodes
        new_sources = rng.choice(self.total_nodes, size=MAX_INCOMING, replace=True).astype(int).tolist()
        new_weights = rng.normal(0.0, initial_scale, size=MAX_INCOMING)
        self.sources.append(new_sources)
        self.weights.append(new_weights)
        self.hidden_count += 1

        target_idx = int(rng.integers(0, self.mutable_count - 1))
        slot = int(rng.integers(0, MAX_INCOMING))
        self.sources[target_idx][slot] = new_node_index
        self.weights[target_idx][slot] = rng.normal(0.0, initial_scale)

    def _prune_hidden_node(self, rng: np.random.Generator) -> None:
        hidden_idx = self.output_size + int(rng.integers(0, self.hidden_count))
        hidden_node_index = self.input_size + hidden_idx
        replacement_sources = [self._remap_source_after_prune(value, hidden_node_index) for value in self.sources[hidden_idx]]
        replacement_sources = [value for value in replacement_sources if value is not None]
        if not replacement_sources:
            replacement_sources = list(range(self.input_size))
        self.sources.pop(hidden_idx)
        self.weights.pop(hidden_idx)
        self.hidden_count -= 1

        for src in self.sources:
            for slot, value in enumerate(src):
                if value == hidden_node_index:
                    src[slot] = int(rng.choice(replacement_sources))
                elif value > hidden_node_index:
                    src[slot] = value - 1

    def _remap_source_after_prune(self, value: int, hidden_node_index: int) -> int | None:
        if value == hidden_node_index:
            return None
        if value > hidden_node_index:
            return value - 1
        return value

    def _sanitize_sources(self, rng: np.random.Generator) -> None:
        total_nodes = self.total_nodes
        for node_idx, src in enumerate(self.sources):
            self_index = self.input_size + node_idx
            for slot, value in enumerate(src):
                if value < 0 or value >= total_nodes or value == self_index:
                    src[slot] = int(rng.integers(0, self.input_size))


@dataclass(slots=True)
class ScoredGenome:
    fitness: float
    genome: DynamicGenome


def evaluate_genome(
    env: gym.Env,
    genome: DynamicGenome,
    episodes: int,
    max_steps: int,
    rng: np.random.Generator,
) -> float:
    returns = []
    for _ in range(episodes):
        observation, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        total_reward = 0.0
        for _ in range(max_steps):
            action = genome.act(np.asarray(observation, dtype=np.float64))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        returns.append(total_reward)
    return float(np.mean(returns))


def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    env = gym.make("CartPole-v1")
    metrics_path = Path(args.metrics_path)
    best_path = Path(args.best_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        input_size = int(env.observation_space.shape[0])
        output_size = int(env.action_space.n)
        population = [
            DynamicGenome.random(rng, input_size, output_size, args.initial_scale, args.passes)
            for _ in range(args.population_size)
        ]
        history = []
        start = time.perf_counter()

        for generation in range(args.generations):
            scored = [
                ScoredGenome(
                    fitness=evaluate_genome(env, genome, args.episodes_per_individual, args.max_steps, rng),
                    genome=genome,
                )
                for genome in population
            ]
            scored.sort(key=lambda item: item.fitness, reverse=True)
            elites = scored[: args.elite_count]
            best = elites[0].fitness
            fitness_values = [item.fitness for item in scored]
            hidden_counts = [item.genome.hidden_count for item in scored]
            row = {
                "generation": generation,
                "best": round(best, 4),
                "mean": round(float(np.mean(fitness_values)), 4),
                "median": round(float(np.median(fitness_values)), 4),
                "mean_hidden": round(float(np.mean(hidden_counts)), 3),
                "max_hidden": int(np.max(hidden_counts)),
                "elapsed_s": round(time.perf_counter() - start, 3),
            }
            history.append(row)
            print(
                "generation={generation:03d} best={best:.2f} mean={mean:.2f} "
                "mean_hidden={mean_hidden:.2f} max_hidden={max_hidden}".format(**row)
            )

            next_population = [elite.genome.clone() for elite in elites]
            while len(next_population) < args.population_size:
                parent = elites[int(rng.integers(0, len(elites)))].genome
                next_population.append(
                    parent.mutate(
                        rng,
                        mutation_std=args.mutation_std,
                        grow_probability=args.grow_probability,
                        prune_probability=args.prune_probability,
                        rewire_probability=args.rewire_probability,
                        max_hidden=args.max_hidden,
                    )
                )
            population = next_population

        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        best_genome = scored[0].genome
        best_path.write_text(
            json.dumps(
                {
                    "fitness": scored[0].fitness,
                    "hidden_count": best_genome.hidden_count,
                    "passes": best_genome.passes,
                    "sources": best_genome.sources,
                    "weights": [w.tolist() for w in best_genome.weights],
                },
                indent=2,
            )
            + "\n"
        )
        return history[-1]
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--episodes-per-individual", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--mutation-std", type=float, default=0.08)
    parser.add_argument("--initial-scale", type=float, default=0.5)
    parser.add_argument("--grow-probability", type=float, default=0.25)
    parser.add_argument("--prune-probability", type=float, default=0.03)
    parser.add_argument("--rewire-probability", type=float, default=0.02)
    parser.add_argument("--max-hidden", type=int, default=24)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--metrics-path", default="../plot/metrics.csv")
    parser.add_argument("--best-path", default="../model/best_genome.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
