from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    df = pd.read_csv("../plot/adv_metrics.csv")
    out = Path("../plot/adv_curve.svg")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["elapsed_s"], df["mean"], label="mean total fitness", color="#32746d")
    ax.plot(df["elapsed_s"], df["mean_fitness_G"], label="mean generator fitness", color="#b45f06")
    ax.plot(df["elapsed_s"], df["mean_fitness_D"], label="mean discriminator fitness", color="#4c5f99")
    ax.plot(df["elapsed_s"], df["mean_env_reward"], label="mean env reward", color="#7f8c8d")
    ax.set_title("Unit 11 GPU CartPole Adversarial Generation")
    ax.set_xlabel("elapsed seconds")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
