"""Variation 1: Topology robustness — ER vs power-law directed graphs.

Critique from CLAUDE.md (critique_1, STRONGEST):
    "All simulations (Fig 1-7) use ER only. Theorem 2 claims universality across
    'broad class of random graphs', but this is never empirically tested
    on non-ER topologies."

We compare the Fig 4 polarization scenario across ER and power-law
configuration-model graphs.  The script reports degree diagnostics next to
opinion moments because topology, mean degree, low-degree mass, and zero
in-degree vertices can all affect Var(R*).

Selective exposure: Q ∈ {−1,+1}, Z ~ Beta(8,1) / Beta(1,8)
c=0.50, d=0.45 (memory case) — same as Fig 4 left panel.

Key metric: Does Var(R*) and polarization signature change with topology?

Run:
    python3 scripts/variation1_topology.py
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
from src.graph_construction import directed_cm_powerlaw, directed_er
from src.validation import empirical_moments

# ── parameters ───────────────────────────────────────────────────────────────
N        = 1000
N_ITER   = 200
SEED     = 42
SCENARIO = "fig4"        # selective exposure

C = 0.50
D = 0.45

P_ER = 0.03            # ER edge probability → mean degree ≈ 30
P_ER_SPARSE = 0.002   # ER edge probability → mean degree ≈ 2

# Comparison set: use degree diagnostics below before attributing differences
# purely to heavy tails.
TOPOLOGIES = [
    # Dense ER baseline and a broadly comparable power-law graph.
    {"name": "ER dense (p=0.03)",       "kind": "er", "params": {"p": P_ER}},
    {"name": "PL α=2.5 dₘᵢₙ=12",       "kind": "pl", "params": {"alpha": 2.5, "d_min": 12}},
    # Sparse pair: useful as degree/low-in-degree sensitivity, not a pure
    # heavy-tail comparison.
    {"name": "ER sparse (p=0.002)",      "kind": "er", "params": {"p": P_ER_SPARSE}},
    {"name": "PL α=2.5 dₘᵢₙ=1",        "kind": "pl", "params": {"alpha": 2.5, "d_min": 1}},
    # Additional power-law tails.
    {"name": "PL α=2.0 dₘᵢₙ=5",        "kind": "pl", "params": {"alpha": 2.0, "d_min": 5}},
    {"name": "PL α=3.5 dₘᵢₙ=20",       "kind": "pl", "params": {"alpha": 3.5, "d_min": 20}},
]

FIG_PATH = Path(__file__).resolve().parents[1] / "figures" / "variation1_topology.png"

# Paper reference values (Fig 4 baseline)
PAPER = dict(var=0.1484, mean_plus=+0.3684, mean_minus=-0.3684, var_cond=0.0095)


# ── helpers ──────────────────────────────────────────────────────────────────
def build_graph(topology: dict, seed: int):
    if topology["kind"] == "er":
        return directed_er(N, topology["params"]["p"], C, seed=seed)
    else:
        kw = topology["params"]
        return directed_cm_powerlaw(N, alpha=kw["alpha"], d_min=kw["d_min"],
                                     c=C, seed=seed)


def degree_summary(A) -> dict:
    """Summarize realized in-degree distribution for interpretation."""
    deg = np.diff(A.indptr)
    return {
        "mean_deg": float(np.mean(deg)),
        "median_deg": float(np.median(deg)),
        "max_deg": int(np.max(deg)),
        "zero_in": int(np.sum(deg == 0)),
    }


def run_topology(topology: dict, Q: np.ndarray, S: np.ndarray, seed_dyn: int) -> dict:
    A = build_graph(topology, SEED)
    out = run_to_stationarity(A, Q, S, c=C, d=D, scenario=SCENARIO,
                               n_iter=N_ITER, seed=seed_dyn)
    stats = empirical_moments(out["R"], Q)
    stats.update(degree_summary(A))
    # also compute ρ₂ = mean Σ_j A[i,j]²
    rho2 = float(A.power(2).sum() / N)
    stats["rho2"] = rho2
    stats["name"] = topology["name"]
    stats["R"] = out["R"]
    return stats


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    attrs = sample_attributes(N, SCENARIO, seed=SEED + 1)
    Q = attrs["Q"]
    S = attrs["S"]

    # ── run all topologies ───────────────────────────────────────────────
    results = []
    for i, topo in enumerate(TOPOLOGIES):
        print(f"Running: {topo['name']} ...")
        stats = run_topology(topo, Q, S, seed_dyn=101 + i)
        results.append(stats)
        print(f"  Var(R*) = {stats['var']:.4f}  (paper: {PAPER['var']})")
        print(f"  E[R*|Q=+1] = {stats['mean_Q_plus']:+.4f}  (paper: {PAPER['mean_plus']:+.4f})")
        print(f"  E[R*|Q=-1] = {stats['mean_Q_minus']:+.4f}  (paper: {PAPER['mean_minus']:+.4f})")
        print(f"  degree: mean={stats['mean_deg']:.2f}, median={stats['median_deg']:.0f}, "
              f"max={stats['max_deg']}, zero-in={stats['zero_in']}")
        print(f"  ρ₂ = {stats['rho2']:.5f}")
        print()

    # ── summary table ────────────────────────────────────────────────────
    print("─" * 118)
    header = (f"{'Topology':>18}  {'Var(R*)':>10}  {'E[R|+]':>10}  {'E[R|-]':>10}  "
              f"{'mean d-':>8}  {'med':>5}  {'max':>5}  {'zero':>5}  {'ρ₂':>8}")
    print(header)
    print("─" * 118)
    for r in results:
        print(f"{r['name']:>18}  {r['var']:10.4f}  {r['mean_Q_plus']:10.4f}  "
              f"{r['mean_Q_minus']:10.4f}  {r['mean_deg']:8.2f}  "
              f"{r['median_deg']:5.0f}  {r['max_deg']:5d}  {r['zero_in']:5d}  "
              f"{r['rho2']:8.5f}")
    print("─" * 118)
    print("\nInterpretation guard:")
    print("  Treat large changes as topology/degree-distribution sensitivity unless")
    print("  mean degree, low-degree mass, and zero-in-degree counts are comparable.")

    # ── check polarization signature ─────────────────────────────────────
    er_result = results[0]
    print(f"\nPolarization physics check:")
    for r in results[1:]:
        delta_var = r["var"] - er_result["var"]
        delta_gap = abs(r["mean_Q_plus"] - r["mean_Q_minus"]) - \
                    abs(er_result["mean_Q_plus"] - er_result["mean_Q_minus"])
        print(f"  {r['name']} vs ER:")
        print(f"    ΔVar(R*) = {delta_var:+.4f}  ({delta_var/PAPER['var']*100:+.1f}% of paper value)")
        print(f"    ΔQgap     = {delta_gap:+.4f}  (Q+ − Q− gap change)")

    # ── plot: 3×2 subplots ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    axes_flat = axes.flatten()

    bins = np.linspace(-1.0, 1.0, 51)
    pos = Q > 0
    neg = Q < 0
    COL_PLUS  = "#c0392b"
    COL_MINUS = "#2471a3"

    # media signal overlay
    z = np.linspace(-1.0, 1.0, 400)
    u = (z + 1.0) / 2.0
    pdf_plus  = beta_dist.pdf(u, 8, 1) / 2.0
    pdf_minus = beta_dist.pdf(u, 1, 8) / 2.0

    for ax, r in zip(axes_flat, results):
        R = r["R"]
        ax.hist(R[pos], bins=bins, density=True, alpha=0.55,
                color=COL_PLUS,  label=r"$Q=+1$")
        ax.hist(R[neg], bins=bins, density=True, alpha=0.55,
                color=COL_MINUS, label=r"$Q=-1$")
        ax.plot(z, pdf_plus,  "--", color=COL_PLUS,  lw=1.2, alpha=0.7)
        ax.plot(z, pdf_minus, "--", color=COL_MINUS, lw=1.2, alpha=0.7)

        txt = (f"Var(R*) = {r['var']:.4f}\n"
               f"E[R|+] = {r['mean_Q_plus']:+.4f}\n"
               f"E[R|−] = {r['mean_Q_minus']:+.4f}\n"
               f"ρ₂ = {r['rho2']:.4f}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        ax.set_xlim(-1, 1)
        ax.set_xlabel("Opinion $R^*$", fontsize=10)
        ax.set_title(r["name"], fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.3)

    for ax in axes[:, 0]:
        ax.set_ylabel("Density", fontsize=10)

    fig.suptitle(
        "Variation 1 — Topology robustness of polarization\n"
        rf"$c={C},\;d={D}$, selective exposure, $n={N}$",
        fontsize=12, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
