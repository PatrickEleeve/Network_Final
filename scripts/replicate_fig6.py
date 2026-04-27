"""Replicate Figure 6 from Fraiman, Lin & Olvera-Cravioto (2024).

Memory reduces variance:
    Var(M*) ≤ Var(R*)
where M* is the stationary distribution with memory (c+d < 1)
and R* is the proportionally-equivalent no-memory case (c+d = 1).

Parameters:
    memory:    c=0.3, d=0.2  (c+d=0.5, self-memory=0.5)
    no-memory: c=0.6, d=0.4  (c+d=1,    same c/d ratio)

Graph: ER(n=1000, p=0.03)
Signal: Z ~ Uniform(-1, 1) i.i.d., independent of Q
Q ~ Uniform({-1, +1})  (discrete, but media is independent → no selective exposure)

Run:
    python3 scripts/replicate_fig6.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attributes import sample_attributes
from src.dynamics import run_to_stationarity
from src.graph_construction import directed_er
from src.validation import empirical_moments

# ── parameters ───────────────────────────────────────────────────────────────
N        = 1000
P        = 0.03
N_ITER   = 300   # d=0.2 → k_ε ≈ 69 for 1e-6; 300 gives safety margin
SEED     = 42
SCENARIO = "fig6"

C_MEM = 0.3
D_MEM = 0.2
_ratio = C_MEM + D_MEM   # 0.5
C_NOM = C_MEM / _ratio   # 0.6
D_NOM = 1.0 - C_NOM      # 0.4

FIG_PATH = Path(__file__).resolve().parents[1] / "figures" / "fig6_replication.png"


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Building ER(n={N}, p={P}), seed={SEED} ...")
    A_mem = directed_er(N, P, C_MEM, seed=SEED)
    A_nom = directed_er(N, P, C_NOM, seed=SEED)

    attrs = sample_attributes(N, SCENARIO, seed=SEED + 1)
    Q = attrs["Q"]
    S = attrs["S"]
    print(f"Q split: {(Q > 0).sum()} = +1, {(Q < 0).sum()} = -1")

    # ── run dynamics ──────────────────────────────────────────────────────
    print(f"\nRunning {N_ITER} iterations ...")
    print(f"  Memory case:    c={C_MEM}, d={D_MEM}, c+d={C_MEM+D_MEM}")
    out_mem = run_to_stationarity(
        A_mem, Q, S, c=C_MEM, d=D_MEM,
        scenario=SCENARIO, n_iter=N_ITER, seed=101,
    )
    print(f"  No-memory case: c={C_NOM:.4f}, d={D_NOM:.4f}, c+d=1")
    out_nom = run_to_stationarity(
        A_nom, Q, S, c=C_NOM, d=D_NOM,
        scenario=SCENARIO, n_iter=N_ITER, seed=102,
    )

    stats_mem = empirical_moments(out_mem["R"], Q)
    stats_nom = empirical_moments(out_nom["R"], Q)

    # ── print ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"{'Quantity':>22}  {'Memory':>12}  {'No-memory':>12}  {'Ratio':>10}")
    print("─" * 70)
    for key, label in [("var", "Var(R*)"),
                        ("mean", "Mean(R*)"),
                        ("mean_Q_plus", "E[R*|Q=+1]"),
                        ("mean_Q_minus", "E[R*|Q=-1]")]:
        v_mem = stats_mem[key]
        v_nom = stats_nom[key]
        ratio = v_mem / v_nom if abs(v_nom) > 1e-9 else float("nan")
        print(f"{label:>22}  {v_mem:12.6f}  {v_nom:12.6f}  {ratio:10.4f}")

    # Proposition 6 check
    var_mem = stats_mem["var"]
    var_nom = stats_nom["var"]
    print("─" * 70)
    print(f"\nProposition 6 check: Var(M*) ≤ Var(R*)?")
    print(f"  Var(M*) = {var_mem:.6f}")
    print(f"  Var(R*) = {var_nom:.6f}")
    if var_mem <= var_nom:
        print(f"  ✓ PASS — memory reduces variance by {(1 - var_mem/var_nom)*100:.1f}%")
    else:
        print(f"  ✗ FAIL — memory DID NOT reduce variance")
    print(f"  (c+d) × Var ratio: {(C_MEM+D_MEM)*var_mem/var_nom:.4f}")

    # ── plot ──────────────────────────────────────────────────────────────
    bins = np.linspace(-1.0, 1.0, 51)
    pos = Q > 0
    neg = Q < 0
    COL_PLUS  = "#c0392b"
    COL_MINUS = "#2471a3"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    cases_plot = [
        (axes[0], out_mem["R"], stats_mem, C_MEM, D_MEM, "Memory"),
        (axes[1], out_nom["R"], stats_nom, C_NOM, D_NOM, "No-memory"),
    ]

    for ax, R, st, c, d, label in cases_plot:
        ax.hist(R[pos], bins=bins, density=True, alpha=0.55,
                color=COL_PLUS,  label=r"$Q=+1$")
        ax.hist(R[neg], bins=bins, density=True, alpha=0.55,
                color=COL_MINUS, label=r"$Q=-1$")

        txt = (f"Var(R*) = {st['var']:.4f}\n"
               f"Mean(R*) = {st['mean']:+.4f}\n"
               f"c={c}, d={d}, c+d={c+d:.1f}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        ax.set_xlim(-1, 1)
        ax.set_xlabel(r"Opinion $R_i^*$", fontsize=11)
        ax.set_title(f"{label}\n$c={c},\\;d={d},\\;c+d={c+d}$", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Probability density", fontsize=11)
    fig.suptitle(
        rf"Figure 6 replication — ER($n={N},\;p={P}$), "
        r"$Z\sim\mathrm{Uniform}(-1,1)$ independent of $Q$",
        fontsize=11,
    )
    fig.tight_layout()
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nFigure saved → {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
