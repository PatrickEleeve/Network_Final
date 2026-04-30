"""Variation 4: Community structure - SBM echo chambers vs ER random mixing.

This variation checks whether community structure amplifies the Fig. 4
selective-exposure mechanism. It compares a two-block directed SBM against a
directed ER graph with similar mean degree, using five independent graph,
attribute, and dynamics seeds.

Design:
    1. SBM (2 blocks, 500 each): p_in=0.05, p_out=0.01
       - Community-aligned Q: Q=+1 mostly in block 0, Q=-1 mostly in block 1
       - Community-uncorrelated Q: random Q across blocks
    2. ER (n=1000, p=0.03): comparable mean degree
       - Same Q assignments within each seed batch

All cases use selective exposure:
    Q=+1: Z ~ -1 + 2 Beta(8,1)
    Q=-1: Z ~ -1 + 2 Beta(1,8)
with c=0.50, d=0.45 (memory case, same as Fig. 4).

Run:
    python3 scripts/variation4_community.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dynamics import run_to_stationarity
from src.graph_construction import directed_er, directed_sbm
from src.validation import empirical_moments

# Parameters
N = 1000
N_BLOCK = 500
N_ITER = 200
SEED = 42
N_SEEDS = 5
SEED_STEP = 1000
SCENARIO = "fig4"

C = 0.50
D = 0.45

P_ER = 0.03
P_IN = 0.05
P_OUT = 0.01

FIG_PATH = Path(__file__).resolve().parents[1] / "figures" / "variation4_community.png"

PAPER = dict(var=0.1484, mean_plus=+0.3684, mean_minus=-0.3684, var_cond=0.0095)


def build_sbm(seed: int):
    p_matrix = np.array([[P_IN, P_OUT], [P_OUT, P_IN]], dtype=np.float64)
    A = directed_sbm([N_BLOCK, N_BLOCK], p_matrix, C, seed=seed)
    block_id = np.zeros(N, dtype=np.int64)
    block_id[N_BLOCK:] = 1
    return A, block_id


def build_er(seed: int):
    return directed_er(N, P_ER, C, seed=seed)


def assign_Q_community(block_id: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = np.where(block_id == 0, +1.0, -1.0)
    flip = rng.random(N) < 0.02
    Q[flip] *= -1.0
    return Q


def assign_Q_random(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0]), size=N)


def mean_se(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size))


def run_case(label: str, A, Q: np.ndarray, seed_dyn: int) -> dict:
    S = np.zeros(N, dtype=np.int8)
    out = run_to_stationarity(
        A, Q, S, c=C, d=D, scenario=SCENARIO, n_iter=N_ITER, seed=seed_dyn
    )
    stats = empirical_moments(out["R"], Q)
    stats["label"] = label
    stats["q_gap"] = stats["mean_Q_plus"] - stats["mean_Q_minus"]
    return stats


def aggregate(rows: list[dict]) -> dict:
    out = {"label": rows[0]["label"], "rows": rows}
    for key in ["var", "mean_Q_plus", "mean_Q_minus", "q_gap"]:
        mean, se = mean_se([r[key] for r in rows])
        out[f"{key}_mean"] = mean
        out[f"{key}_se"] = se
    return out


def print_summary(results: list[dict]) -> None:
    print("\n" + "-" * 98)
    print(
        f"{'Case':>22}  {'Var(R*)':>15}  {'E[R|+]':>15}  "
        f"{'E[R|-]':>15}  {'Q gap':>15}"
    )
    print("-" * 98)
    for r in results:
        print(
            f"{r['label']:>22}  "
            f"{r['var_mean']:8.4f} +/- {r['var_se']:<6.4f}  "
            f"{r['mean_Q_plus_mean']:8.4f} +/- {r['mean_Q_plus_se']:<6.4f}  "
            f"{r['mean_Q_minus_mean']:8.4f} +/- {r['mean_Q_minus_se']:<6.4f}  "
            f"{r['q_gap_mean']:8.4f} +/- {r['q_gap_se']:<6.4f}"
        )
    print("-" * 98)


def main() -> None:
    labels = [
        "SBM + echo chamber",
        "SBM + random Q",
        "ER + echo chamber Q",
        "ER + random Q",
    ]
    grouped = {label: [] for label in labels}
    paired_diffs = {
        "echo_gap": [],
        "echo_var": [],
        "random_gap": [],
        "random_var": [],
    }

    print(f"Running 4 community cases x {N_SEEDS} seeds ...")
    for rep in range(N_SEEDS):
        seed_shift = SEED_STEP * rep
        print(f"\nSeed batch {rep + 1}/{N_SEEDS} (seed shift={seed_shift})")

        A_sbm, block_id = build_sbm(seed=SEED + seed_shift)
        A_er = build_er(seed=SEED + seed_shift)

        Q_comm = assign_Q_community(block_id, seed=SEED + 2 + seed_shift)
        Q_rand = assign_Q_random(seed=SEED + 3 + seed_shift)

        rows = [
            run_case(labels[0], A_sbm, Q_comm, seed_dyn=200 + seed_shift),
            run_case(labels[1], A_sbm, Q_rand, seed_dyn=201 + seed_shift),
            run_case(labels[2], A_er, Q_comm, seed_dyn=202 + seed_shift),
            run_case(labels[3], A_er, Q_rand, seed_dyn=203 + seed_shift),
        ]

        for row in rows:
            grouped[row["label"]].append(row)
            print(
                f"  {row['label']:<22} "
                f"Var={row['var']:.4f}, E+= {row['mean_Q_plus']:+.4f}, "
                f"E-= {row['mean_Q_minus']:+.4f}, gap={row['q_gap']:+.4f}"
            )

        sbm_echo, sbm_random, er_echo, er_random = rows
        paired_diffs["echo_gap"].append(sbm_echo["q_gap"] - er_echo["q_gap"])
        paired_diffs["echo_var"].append(sbm_echo["var"] - er_echo["var"])
        paired_diffs["random_gap"].append(sbm_random["q_gap"] - er_random["q_gap"])
        paired_diffs["random_var"].append(sbm_random["var"] - er_random["var"])

    results = [aggregate(grouped[label]) for label in labels]
    print_summary(results)

    print("\nPaired SBM - ER differences:")
    for key, label in [
        ("echo_gap", "Echo chamber Q gap"),
        ("echo_var", "Echo chamber variance"),
        ("random_gap", "Random-Q gap"),
        ("random_var", "Random-Q variance"),
    ]:
        mean, se = mean_se(paired_diffs[key])
        print(f"  {label:<24} = {mean:+.4f} +/- {se:.4f}")

    x = np.arange(len(results))
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#e15759"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_var, ax_gap, ax_means, ax_diff = axes.flat

    def setup_cases(ax) -> None:
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in results], rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    ax_var.bar(
        x,
        [r["var_mean"] for r in results],
        yerr=[r["var_se"] for r in results],
        color=colors,
        alpha=0.85,
        capsize=4,
    )
    ax_var.axhline(PAPER["var"], color="black", ls="--", lw=1.1, label="Source Fig. 4")
    ax_var.set_ylabel(r"$\mathrm{Var}(R^*)$")
    ax_var.set_title("Stationary variance, 5 seeds")
    setup_cases(ax_var)
    ax_var.legend(fontsize=8)

    ax_gap.bar(
        x,
        [r["q_gap_mean"] for r in results],
        yerr=[r["q_gap_se"] for r in results],
        color=colors,
        alpha=0.85,
        capsize=4,
    )
    paper_gap = PAPER["mean_plus"] - PAPER["mean_minus"]
    ax_gap.axhline(paper_gap, color="black", ls="--", lw=1.1, label="Source Fig. 4")
    ax_gap.set_ylabel(r"$E[R|Q=+1]-E[R|Q=-1]$")
    ax_gap.set_title("Conditional mean gap, 5 seeds")
    setup_cases(ax_gap)
    ax_gap.legend(fontsize=8)

    width = 0.36
    ax_means.bar(
        x - width / 2,
        [r["mean_Q_plus_mean"] for r in results],
        yerr=[r["mean_Q_plus_se"] for r in results],
        width=width,
        color="#c0392b",
        alpha=0.8,
        capsize=4,
        label=r"$Q=+1$",
    )
    ax_means.bar(
        x + width / 2,
        [r["mean_Q_minus_mean"] for r in results],
        yerr=[r["mean_Q_minus_se"] for r in results],
        width=width,
        color="#2471a3",
        alpha=0.8,
        capsize=4,
        label=r"$Q=-1$",
    )
    ax_means.axhline(0, color="black", lw=0.7)
    ax_means.set_ylabel("Conditional mean")
    ax_means.set_title("Group centers, 5 seeds")
    setup_cases(ax_means)
    ax_means.legend(fontsize=8)

    diff_labels = ["Echo gap", "Echo var", "Random gap", "Random var"]
    diff_keys = ["echo_gap", "echo_var", "random_gap", "random_var"]
    dx = np.arange(len(diff_keys))
    diff_means = [mean_se(paired_diffs[key])[0] for key in diff_keys]
    diff_ses = [mean_se(paired_diffs[key])[1] for key in diff_keys]
    ax_diff.bar(dx, diff_means, yerr=diff_ses, color=["#9467bd", "#8c564b", "#17becf", "#7f7f7f"],
                alpha=0.85, capsize=4)
    ax_diff.axhline(0, color="black", lw=0.8)
    ax_diff.set_xticks(dx)
    ax_diff.set_xticklabels(diff_labels, rotation=15, ha="right", fontsize=8)
    ax_diff.set_ylabel("SBM - ER paired difference")
    ax_diff.set_title("Community amplification, paired by seed")
    ax_diff.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Variation 4 - community structure and selective exposure\n"
        rf"$c={C},\;d={D}$, $n={N}$, mean +/- SE over {N_SEEDS} seeds",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
