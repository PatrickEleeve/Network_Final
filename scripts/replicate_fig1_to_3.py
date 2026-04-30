"""Replicate Figures 1-3 from Fraiman, Lin & Olvera-Cravioto (2024).

Common setup from Section 5.1:
    Graph: directed ER(n=1000, p=0.03)
    Q_i ~ Uniform(-1, 1)
    R_i^(0) ~ Uniform({-1, +1})
    Equal inbound-neighbor weights summing to c.

The no-memory case rescales c and d proportionally so that c + d = 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attributes import sample_attributes
from src.dynamics import run_to_stationarity
from src.graph_construction import directed_er
from src.validation import empirical_moments


N = 1000
P = 0.03
SEED = 42
CONVERGENCE_TOL = 1e-6
ROOT = Path(__file__).resolve().parents[1]


CASES = {
    "fig1": {
        "title": "Figure 1 replication - small-variance media",
        "c": 0.001,
        "d": 0.3,
        "n_iter": 500,
        "ylim": (0, 55),
    },
    "fig2": {
        "title": "Figure 2 replication - polarized media independent of Q",
        "c": 0.5,
        "d": 0.001,
        "n_iter": 16000,
        "ylim": (0, 55),
    },
    "fig3": {
        "title": "Figure 3 replication - skewed media independent of Q",
        "c": 0.5,
        "d": 0.001,
        "n_iter": 16000,
        "ylim": (0, 55),
    },
}


def no_memory_params(c: float, d: float) -> tuple[float, float]:
    total = c + d
    return c / total, d / total


def media_overlay(ax, scenario: str, z: np.ndarray) -> None:
    if scenario == "fig1":
        pdf = np.where((-0.03 <= z) & (z <= 0.03), 1.0 / 0.06, np.nan)
        ax.plot(z, pdf, "--", color="gray", lw=1.5, label="media")
    elif scenario == "fig2":
        ax.axvline(-1, color="gray", ls="--", lw=1.5, label="media support")
        ax.axvline(+1, color="gray", ls="--", lw=1.5)
    elif scenario == "fig3":
        u = (z + 1.0) / 2.0
        ax.plot(z, beta_dist.pdf(u, 1, 8) / 2.0, "--", color="gray", lw=1.5, label="media")


def run_panel(scenario: str, c: float, d: float, Q: np.ndarray, S: np.ndarray,
              seed_dyn: int) -> dict:
    A = directed_er(N, P, c, seed=SEED)
    out = run_to_stationarity(
        A, Q, S, c=c, d=d, scenario=scenario,
        n_iter=CASES[scenario]["n_iter"], seed=seed_dyn,
        coupled_tol=CONVERGENCE_TOL,
    )
    stats = empirical_moments(out["R"], Q=np.sign(Q))
    stats["R"] = out["R"]
    stats["c"] = c
    stats["d"] = d
    stats["n_iter_run"] = out["n_iter_run"]
    stats["converged"] = out["converged"]
    stats["coupled_diff"] = out["coupled_diff"]
    return stats


def make_figure(scenario: str) -> None:
    spec = CASES[scenario]
    c_mem, d_mem = spec["c"], spec["d"]
    c_nom, d_nom = no_memory_params(c_mem, d_mem)

    attrs = sample_attributes(N, scenario, seed=SEED + 1)
    Q, S = attrs["Q"], attrs["S"]
    q_plus = Q > 0
    q_minus = Q < 0

    print(f"Running {scenario}: n_iter={spec['n_iter']} ...")
    mem = run_panel(scenario, c_mem, d_mem, Q, S, seed_dyn=100 + int(scenario[-1]))
    nom = run_panel(scenario, c_nom, d_nom, Q, S, seed_dyn=200 + int(scenario[-1]))

    for label, res in [("Memory", mem), ("No-memory", nom)]:
        print(f"  {label}: c={res['c']:.6g}, d={res['d']:.6g}, "
              f"mean={res['mean']:+.4f}, var={res['var']:.6f}, "
              f"iters={res['n_iter_run']}, "
              f"coupled_diff={res['coupled_diff']:.2e}, "
              f"converged={res['converged']}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bins = np.linspace(-1.0, 1.0, 51)
    z = np.linspace(-1.0, 1.0, 500)
    colors = {"plus": "#c0392b", "minus": "#2471a3"}
    panels = [
        (axes[0], mem, "Memory"),
        (axes[1], nom, "No-memory"),
    ]

    for ax, res, label in panels:
        ax.hist(res["R"][q_minus], bins=bins, density=True, alpha=0.55,
                color=colors["minus"], label=r"$Q<0$")
        ax.hist(res["R"][q_plus], bins=bins, density=True, alpha=0.55,
                color=colors["plus"], label=r"$Q>0$")
        media_overlay(ax, scenario, z)
        ax.set_xlim(-1, 1)
        ax.set_ylim(*spec["ylim"])
        ax.set_xlabel(r"Opinion $R_i^*$", fontsize=11)
        ax.set_title(
            f"{label}\n"
            rf"$c={res['c']:.6g},\ d={res['d']:.6g}$",
            fontsize=10,
        )
        ax.text(
            0.02, 0.96,
            f"Mean={res['mean']:+.4f}\nVar={res['var']:.6f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Probability density", fontsize=11)
    fig.suptitle(
        rf"{spec['title']} - ER($n={N}, p={P}$), $Q\sim U(-1,1)$",
        fontsize=11,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = ROOT / "figures" / f"{scenario}_replication.png"
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved -> {out_path}")
    plt.close(fig)


def main() -> None:
    for scenario in ["fig1", "fig2", "fig3"]:
        make_figure(scenario)


if __name__ == "__main__":
    main()
