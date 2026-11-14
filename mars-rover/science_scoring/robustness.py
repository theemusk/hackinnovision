"""Monte Carlo robustness check: how stable is the ranking?

Each trial (1) perturbs the weights around 0.4 / 0.4 / 0.2 and
(2) adds noise to every input, with noise size proportional to that
target's own uncertainty. Then it re-ranks all targets.
Outputs: outputs/rank_stability.csv and outputs/rank_stability.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from science import load_targets, OUT_DIR, W_ICE, W_SCI, W_SAFE

N_TRIALS = 2000
SEED = 42


def run_monte_carlo(n_trials=N_TRIALS, seed=SEED):
    rng = np.random.default_rng(seed)
    df = load_targets()
    n = len(df)
    ice = df["ice_probability"].to_numpy()
    sci = df["scientific_value"].to_numpy()
    risk = df["terrain_risk"].to_numpy()
    sd = 0.15 * df["uncertainty"].to_numpy()

    weights = rng.dirichlet(np.array([W_ICE, W_SCI, W_SAFE]) * 50, size=n_trials)
    ranks = np.zeros((n_trials, n), dtype=int)
    for i in range(n_trials):
        ice_i = np.clip(ice + rng.normal(0, sd), 0, 1)
        sci_i = np.clip(sci + rng.normal(0, sd), 0, 1)
        risk_i = np.clip(risk + rng.normal(0, sd), 0, 1)
        w = weights[i]
        score = 100 * (w[0] * ice_i + w[1] * sci_i + w[2] * (1 - risk_i))
        order = np.argsort(-score)
        ranks[i, order] = np.arange(1, n + 1)

    out = pd.DataFrame({
        "target_id": df["target_id"],
        "prob_rank_1_pct": (ranks == 1).mean(axis=0) * 100,
        "prob_top_3_pct": (ranks <= 3).mean(axis=0) * 100,
        "mean_rank": ranks.mean(axis=0),
        "rank_5th_pct": np.percentile(ranks, 5, axis=0),
        "rank_95th_pct": np.percentile(ranks, 95, axis=0),
    }).sort_values("mean_rank").round(2).reset_index(drop=True)
    return out


def main():
    OUT_DIR.mkdir(exist_ok=True)
    res = run_monte_carlo()
    res.to_csv(OUT_DIR / "rank_stability.csv", index=False)
    print(res.to_string(index=False))

    top = res.head(6)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(top["target_id"], top["prob_top_3_pct"], color="#c1440e")
    ax.bar_label(bars, fmt="%.0f%%", padding=2)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Target ID")
    ax.set_ylabel(f"% of {N_TRIALS} trials ranked in top 3")
    ax.set_title("Ranking Stability under Weight and Input Uncertainty")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rank_stability.png", dpi=150)
    plt.close(fig)
    best = res.iloc[0]
    print(f"\n{best.target_id} ranked #1 in {best.prob_rank_1_pct:.1f}% of trials.")


if __name__ == "__main__":
    main()