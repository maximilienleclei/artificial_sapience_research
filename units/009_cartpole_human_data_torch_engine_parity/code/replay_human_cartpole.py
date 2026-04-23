from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

UNIT8_CODE = Path(__file__).resolve().parents[2] / "008_torch_cartpole_physics_parity" / "code"
if str(UNIT8_CODE) not in sys.path:
    sys.path.insert(0, str(UNIT8_CODE))

from torch_cartpole import step_cartpole  # noqa: E402


def load_steps(path: Path) -> list[dict]:
    episodes = json.loads(path.read_text())
    steps: list[dict] = []
    for episode in episodes:
        steps.extend(episode["steps"])
    return steps


def load_episodes(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def evaluate_steps(steps: list[dict], device: torch.device, dtype: torch.dtype) -> dict:
    obs = torch.as_tensor([step["observation"] for step in steps], dtype=dtype, device=device)
    actions = torch.as_tensor([step["action"] for step in steps], dtype=torch.int64, device=device)
    recorded_next = torch.as_tensor([step["next_observation"] for step in steps], dtype=dtype, device=device)
    recorded_done = torch.as_tensor([step["done"] for step in steps], dtype=torch.bool, device=device)
    recorded_reward = torch.as_tensor([step["reward"] for step in steps], dtype=dtype, device=device)

    with torch.no_grad():
        predicted_next, predicted_reward, predicted_done = step_cartpole(obs, actions)

    abs_err = (predicted_next - recorded_next).abs()
    max_abs_err = abs_err.max(dim=1).values

    return {
        "count": len(steps),
        "max_abs_error": float(max_abs_err.max().item()),
        "mean_abs_error": float(abs_err.mean().item()),
        "median_max_abs_error": float(max_abs_err.median().item()),
        "done_match_rate": float((predicted_done == recorded_done).float().mean().item()),
        "reward_match_rate": float((predicted_reward == recorded_reward).float().mean().item()),
        "per_step_max_abs_error": max_abs_err.detach().cpu().to(torch.float64).numpy(),
    }


def rollout_episode(episode: dict, device: torch.device, dtype: torch.dtype) -> dict:
    steps = episode["steps"]
    state = torch.as_tensor(steps[0]["observation"], dtype=dtype, device=device).unsqueeze(0)
    actions = [step["action"] for step in steps]
    recorded_next = np.asarray([step["next_observation"] for step in steps], dtype=np.float64)
    recorded_done = np.asarray([step["done"] for step in steps], dtype=bool)
    recorded_return = float(sum(step["reward"] for step in steps))

    predicted_states = []
    predicted_done = []
    predicted_rewards = []
    with torch.no_grad():
        for action in actions:
            action_tensor = torch.tensor([action], dtype=torch.int64, device=device)
            state, reward, done = step_cartpole(state, action_tensor)
            predicted_states.append(state.squeeze(0).detach().cpu().to(torch.float64).numpy())
            predicted_done.append(bool(done.item()))
            predicted_rewards.append(float(reward.item()))

    predicted_states_np = np.stack(predicted_states)
    abs_err = np.abs(predicted_states_np - recorded_next)
    per_step_err = abs_err.max(axis=1)
    done_match = float(np.mean(np.asarray(predicted_done) == recorded_done))
    predicted_return = float(sum(predicted_rewards))
    return {
        "length": len(steps),
        "max_abs_error": float(per_step_err.max()),
        "final_abs_error": float(per_step_err[-1]),
        "mean_abs_error": float(abs_err.mean()),
        "done_match_rate": done_match,
        "recorded_return": recorded_return,
        "predicted_return": predicted_return,
        "return_abs_error": abs(predicted_return - recorded_return),
    }


def evaluate_episode_rollouts(episodes: list[dict], device: torch.device, dtype: torch.dtype) -> dict:
    rows = [rollout_episode(ep, device, dtype) for ep in episodes]
    return {
        "episode_count": len(rows),
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "max_final_abs_error": max(row["final_abs_error"] for row in rows),
        "mean_abs_error": float(np.mean([row["mean_abs_error"] for row in rows])),
        "mean_done_match_rate": float(np.mean([row["done_match_rate"] for row in rows])),
        "mean_return_abs_error": float(np.mean([row["return_abs_error"] for row in rows])),
        "max_return_abs_error": float(np.max([row["return_abs_error"] for row in rows])),
        "exact_return_match_rate": float(np.mean([row["return_abs_error"] == 0.0 for row in rows])),
    }


def first_divergence(
    episodes: list[dict],
    device: torch.device,
    dtype: torch.dtype,
    error_threshold: float,
) -> dict | None:
    for episode in episodes:
        steps = episode["steps"]
        state = torch.as_tensor(steps[0]["observation"], dtype=dtype, device=device).unsqueeze(0)
        with torch.no_grad():
            for step_idx, step in enumerate(steps):
                action_tensor = torch.tensor([step["action"]], dtype=torch.int64, device=device)
                predicted_next, _, predicted_done = step_cartpole(state, action_tensor)
                predicted_next_np = predicted_next.squeeze(0).detach().cpu().to(torch.float64).numpy()
                recorded_next = np.asarray(step["next_observation"], dtype=np.float64)
                per_dim_err = np.abs(predicted_next_np - recorded_next)
                max_err = float(per_dim_err.max())
                done_match = bool(bool(predicted_done.item()) == bool(step["done"]))
                if max_err > error_threshold or not done_match:
                    return {
                        "episode_id": episode["episode_id"],
                        "step_index": step_idx,
                        "recorded_observation": step["observation"],
                        "action": step["action"],
                        "recorded_next_observation": step["next_observation"],
                        "predicted_next_observation": predicted_next_np.tolist(),
                        "per_dim_abs_error": per_dim_err.tolist(),
                        "max_abs_error": max_err,
                        "recorded_done": bool(step["done"]),
                        "predicted_done": bool(predicted_done.item()),
                    }
                state = predicted_next
    return None


def main(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    plot_dir = Path("../plot")
    plot_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    datasets = [
        ("sub01", load_steps(data_dir / "sub01_data_cartpole.json")),
        ("sub02", load_steps(data_dir / "sub02_data_cartpole.json")),
    ]
    episode_sets = [
        ("sub01", load_episodes(data_dir / "sub01_data_cartpole.json")),
        ("sub02", load_episodes(data_dir / "sub02_data_cartpole.json")),
    ]

    start = time.perf_counter()
    summary_rows = []
    histograms: dict[str, np.ndarray] = {}
    for name, steps in datasets:
        result64 = evaluate_steps(steps, device=device, dtype=torch.float64)
        result32 = evaluate_steps(steps, device=device, dtype=torch.float32)
        result16 = evaluate_steps(steps, device=device, dtype=torch.float16)
        histograms[f"{name}_float64"] = result64["per_step_max_abs_error"]
        histograms[f"{name}_float32"] = result32["per_step_max_abs_error"]
        histograms[f"{name}_float16"] = result16["per_step_max_abs_error"]
        summary_rows.extend(
            [
                {
                    "dataset": name,
                    "candidate": "torch_gpu_float64",
                    "count": result64["count"],
                    "max_abs_error": result64["max_abs_error"],
                    "mean_abs_error": result64["mean_abs_error"],
                    "median_max_abs_error": result64["median_max_abs_error"],
                    "done_match_rate": result64["done_match_rate"],
                    "reward_match_rate": result64["reward_match_rate"],
                },
                {
                    "dataset": name,
                    "candidate": "torch_gpu_float32",
                    "count": result32["count"],
                    "max_abs_error": result32["max_abs_error"],
                    "mean_abs_error": result32["mean_abs_error"],
                    "median_max_abs_error": result32["median_max_abs_error"],
                    "done_match_rate": result32["done_match_rate"],
                    "reward_match_rate": result32["reward_match_rate"],
                },
                {
                    "dataset": name,
                    "candidate": "torch_gpu_float16",
                    "count": result16["count"],
                    "max_abs_error": result16["max_abs_error"],
                    "mean_abs_error": result16["mean_abs_error"],
                    "median_max_abs_error": result16["median_max_abs_error"],
                    "done_match_rate": result16["done_match_rate"],
                    "reward_match_rate": result16["reward_match_rate"],
                },
            ]
        )

    metrics_path = plot_dir / "summary.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    rollout_rows = []
    for name, episodes in episode_sets:
        result64 = evaluate_episode_rollouts(episodes, device=device, dtype=torch.float64)
        result32 = evaluate_episode_rollouts(episodes, device=device, dtype=torch.float32)
        result16 = evaluate_episode_rollouts(episodes, device=device, dtype=torch.float16)
        rollout_rows.extend(
            [
                {"dataset": name, "candidate": "torch_gpu_float64", **result64},
                {"dataset": name, "candidate": "torch_gpu_float32", **result32},
                {"dataset": name, "candidate": "torch_gpu_float16", **result16},
            ]
        )

    rollout_path = plot_dir / "rollout_summary.csv"
    with rollout_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rollout_rows[0].keys()))
        writer.writeheader()
        writer.writerows(rollout_rows)

    divergence = first_divergence(
        episode_sets[0][1],
        device=device,
        dtype=torch.float64,
        error_threshold=args.divergence_threshold,
    )
    divergence_path = plot_dir / "sub01_first_divergence.json"
    divergence_path.write_text(json.dumps(divergence, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, subject in zip(axes, ["sub01", "sub02"], strict=True):
        ax.hist(histograms[f"{subject}_float64"], bins=40, alpha=0.7, label="float64")
        ax.hist(histograms[f"{subject}_float32"], bins=40, alpha=0.7, label="float32")
        ax.hist(histograms[f"{subject}_float16"], bins=40, alpha=0.7, label="float16")
        ax.set_title(subject)
        ax.set_xlabel("per-step max abs state error")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
    axes[0].set_ylabel("transition count")
    axes[1].legend()
    fig.suptitle("Unit 4 Human CartPole Data vs Torch CartPole Engine")
    fig.tight_layout()
    plot_path = plot_dir / "human_cartpole_engine_parity.svg"
    fig.savefig(plot_path)

    print(f"Saved {metrics_path}")
    print(f"Saved {rollout_path}")
    print(f"Saved {divergence_path}")
    print(f"Saved {plot_path}")
    print(f"elapsed_s={time.perf_counter() - start:.3f}")
    for row in summary_rows:
        print(
            "{dataset} {candidate}: max_abs_error={max_abs_error:.3e} "
            "mean_abs_error={mean_abs_error:.3e} done_match_rate={done_match_rate:.3f} "
            "reward_match_rate={reward_match_rate:.3f}".format(**row)
        )
    for row in rollout_rows:
        print(
            "{dataset} {candidate} rollout: max_abs_error={max_abs_error:.3e} "
            "max_final_abs_error={max_final_abs_error:.3e} mean_abs_error={mean_abs_error:.3e} "
            "mean_done_match_rate={mean_done_match_rate:.3f} "
            "mean_return_abs_error={mean_return_abs_error:.3f} "
            "max_return_abs_error={max_return_abs_error:.3f} "
            "exact_return_match_rate={exact_return_match_rate:.3f}".format(**row)
        )
    if divergence is not None:
        print(
            "sub01 first_divergence: episode_id={episode_id} step_index={step_index} "
            "max_abs_error={max_abs_error:.3e} recorded_done={recorded_done} predicted_done={predicted_done}".format(
                **divergence
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[2] / "004_human_behavior_control_tasks" / "data"),
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--divergence-threshold", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
