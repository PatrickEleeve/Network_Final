"""Variation 1: Topology robustness - ER vs power-law directed graphs.

Critique target:
    The source paper's finite simulations are concentrated on ER-based graphs.
    The theorem covers broader locally convergent graph families, so this
    variation checks whether the Fig. 4 selective-exposure result is sensitive
    to topology, degree scale, and low-degree mass in finite networks.

We compare the Fig. 4 polarization scenario across ER and power-law
configuration-model graphs. The script reports five-seed averages for degree
diagnostics and opinion moments because topology, mean degree, low-degree mass,
and zero in-degree vertices can all affect Var(R*).

Selective exposure: Q in {-1,+1}, Z ~ Beta(8,1) / Beta(1,8)
c=0.50, d=0.45 (memory case), same as Fig. 4 left panel.

Run:
    python3 scripts/variation1_topology.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attributes import sample_attributes
from src.dynamics import run_to_stationarity
from src.graph_construction import directed_cm_powerlaw, directed_er
from src.validation import empirical_moments

# Parameters
N = 1000
N_ITER = 200
SEED = 42
N_SEEDS = 5
SEED_STEP = 1000
SCENARIO = "fig4"

C = 0.50
D = 0.45

P_ER = 0.03            # mean in-degree about 30
P_ER_SPARSE = 0.002   # mean in-degree about 2

TOPOLOGIES = [
    {"name": "ER dense (p=0.03)", "kind": "er", "params": {"p": P_ER}},
    {"name": "PL a=2.5 dmin=12", "kind": "pl", "params": {"alpha": 2.5, "d_min": 12}},
    {"name": "ER sparse (p=0.002)", "kind": "er", "params": {"p": P_ER_SPARSE}},
    {"name": "PL a=2.5 dmin=1", "kind": "pl", "params": {"alpha": 2.5, "d_min": 1}},
    {"name": "PL a=2.0 dmin=5", "kind": "pl", "params": {"alpha": 2.0, "d_min": 5}},
    {"name": "PL a=3.5 dmin=20", "kind": "pl", "params": {"alpha": 3.5, "d_min": 20}},
]

FIG_PATH = Path(__file__).resolve().parents[1] / "figures" / "variation1_topology.png"

# Source-paper Fig. 4 references.
PAPER = dict(var=0.1484, mean_plus=+0.3684, mean_minus=-0.3684, var_cond=0.0095)


def build_graph(topology: dict, seed: int):
    if topology["kind"] == "er":
        return directed_er(N, topology["params"]["p"], C, seed=seed)
    kw = topology["params"]
    return directed_cm_powerlaw(N, alpha=kw["alpha"], d_min=kw["d_min"], c=C, seed=seed)


def degree_summary(A) -> dict:
    deg = np.diff(A.indptr)
    return {
        "mean_deg": float(np.mean(deg)),
        "median_deg": float(np.median(deg)),
        "max_deg": float(np.max(deg)),
        "zero_in": float(np.sum(deg == 0)),
    }


def mean_se(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size))


def run_topology(
    topology: dict,
    Q: np.ndarray,
    S: np.ndarray,
    graph_seed: int,
    seed_dyn: int,
) -> dict:
    A = build_graph(topology, graph_seed)
    out = run_to_stationarity(
        A, Q, S, c=C, d=D, scenario=SCENARIO, n_iter=N_ITER, seed=seed_dyn
    )
    stats = empirical_moments(out["R"], Q)
    stats.update(degree_summary(A))
    stats["rho2"] = float(A.power(2).sum() / N)
    stats["name"] = topology["name"]
    stats["q_gap"] = stats["mean_Q_plus"] - stats["mean_Q_minus"]
    return stats


def aggregate(rows: list[dict]) -> dict:
    out = {"name": rows[0]["name"], "rows": rows}
    for key in [
        "var",
        "mean_Q_plus",
        "mean_Q_minus",
        "q_gap",
        "mean_deg",
        "median_deg",
        "max_deg",
        "zero_in",
        "rho2",
    ]:
        mean, se = mean_se([r[key] for r in rows])
        out[f"{key}_mean"] = mean
        out[f"{key}_se"] = se
    return out


def main() -> None:
    grouped = {topo["name"]: [] for topo in TOPOLOGIES}

    print(f"Running {len(TOPOLOGIES)} topologies x {N_SEEDS} seeds ...")
    for rep in range(N_SEEDS):
        seed_shift = SEED_STEP * rep
        attrs = sample_attributes(N, SCENARIO, seed=SEED + 1 + seed_shift)
        Q = attrs["Q"]
        S = attrs["S"]
        print(f"\nSeed batch {rep + 1}/{N_SEEDS} (seed shift={seed_shift})")

        for i, topo in enumerate(TOPOLOGIES):
            stats = run_topology(
                topo,
                Q,
                S,
                graph_seed=SEED + seed_shift,
                seed_dyn=101 + i + seed_shift,
            )
            grouped[topo["name"]].append(stats)
            print(
                f"  {topo['name']:<22} "
                f"Var={stats['var']:.4f}, gap={stats['q_gap']:+.4f}, "
                f"mean d-={stats['mean_deg']:.2f}, zero-in={int(stats['zero_in'])}"
            )

    results = [aggregate(grouped[topo["name"]]) for topo in TOPOLOGIES]

    print("\n" + "-" * 132)
    print(
        f"{'Topology':>22}  {'Var(R*)':>15}  {'Q gap':>15}  "
        f"{'E[R|+]':>15}  {'E[R|-]':>15}  {'mean d-':>13}  "
        f"{'zero-in':>13}  {'rho2':>13}"
    )
    print("-" * 132)
    for r in results:
        print(
            f"{r['name']:>22}  "
            f"{r['var_mean']:8.4f} +/- {r['var_se']:<6.4f}  "
            f"{r['q_gap_mean']:8.4f} +/- {r['q_gap_se']:<6.4f}  "
            f"{r['mean_Q_plus_mean']:8.4f} +/- {r['mean_Q_plus_se']:<6.4f}  "
            f"{r['mean_Q_minus_mean']:8.4f} +/- {r['mean_Q_minus_se']:<6.4f}  "
            f"{r['mean_deg_mean']:7.2f} +/- {r['mean_deg_se']:<5.2f}  "
            f"{r['zero_in_mean']:7.1f} +/- {r['zero_in_se']:<5.1f}  "
            f"{r['rho2_mean']:7.5f} +/- {r['rho2_se']:<7.5f}"
        )
    print("-" * 132)

    er_result = results[0]
    print("\nPolarization check against dense ER baseline:")
    for r in results[1:]:
        delta_var = r["var_mean"] - er_result["var_mean"]
        delta_gap = r["q_gap_mean"] - er_result["q_gap_mean"]
        print(f"  {r['name']} vs ER:")
        print(
            f"    Delta Var(R*) = {delta_var:+.4f} "
            f"(SE approx {np.sqrt(r['var_se']**2 + er_result['var_se']**2):.4f})"
        )
        print(
            f"    Delta Q gap   = {delta_gap:+.4f} "
            f"(SE approx {np.sqrt(r['q_gap_se']**2 + er_result['q_gap_se']**2):.4f})"
        )

    labels = [r["name"] for r in results]
    x = np.arange(len(results))
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#e15759", "#b07aa1", "#76b7b2"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_var, ax_gap, ax_deg, ax_rho = axes.flat

    def setup_xaxis(ax) -> None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
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
    setup_xaxis(ax_var)
    ax_var.legend(fontsize=8)

    paper_gap = PAPER["mean_plus"] - PAPER["mean_minus"]
    ax_gap.bar(
        x,
        [r["q_gap_mean"] for r in results],
        yerr=[r["q_gap_se"] for r in results],
        color=colors,
        alpha=0.85,
        capsize=4,
    )
    ax_gap.axhline(paper_gap, color="black", ls="--", lw=1.1, label="Source Fig. 4")
    ax_gap.set_ylabel(r"$E[R|Q=+1]-E[R|Q=-1]$")
    ax_gap.set_title("Conditional mean gap, 5 seeds")
    setup_xaxis(ax_gap)
    ax_gap.legend(fontsize=8)

    ax_deg.bar(
        x,
        [r["mean_deg_mean"] for r in results],
        yerr=[r["mean_deg_se"] for r in results],
        color=colors,
        alpha=0.85,
        capsize=4,
        label="mean in-degree",
    )
    ax_deg.set_ylabel("Mean in-degree")
    ax_deg.set_title("Degree scale")
    setup_xaxis(ax_deg)
    ax_zero = ax_deg.twinx()
    ax_zero.plot(
        x,
        [r["zero_in_mean"] for r in results],
        color="#333333",
        marker="o",
        lw=1.5,
        label="zero in-degree",
    )
    ax_zero.set_ylabel("Zero in-degree count")
    lines, line_labels = ax_deg.get_legend_handles_labels()
    lines2, labels2 = ax_zero.get_legend_handles_labels()
    ax_deg.legend(lines + lines2, line_labels + labels2, fontsize=8, loc="upper right")

    ax_rho.bar(
        x,
        [r["rho2_mean"] for r in results],
        yerr=[r["rho2_se"] for r in results],
        color=colors,
        alpha=0.85,
        capsize=4,
    )
    ax_rho.set_ylabel(r"$\rho_2=n^{-1}\sum_i\sum_j A_{ij}^2$")
    ax_rho.set_title("Weight concentration")
    setup_xaxis(ax_rho)

    fig.suptitle(
        "Variation 1 - topology robustness of selective-exposure polarization\n"
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
