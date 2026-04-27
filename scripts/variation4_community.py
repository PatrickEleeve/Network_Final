"""Variation 4: Community structure — SBM echo chambers vs ER random mixing.

Critique from CLAUDE.md (critique_4):
    "Real social networks are scale-free with heavy-tailed in-degree distributions;
    also have clustering, which violates tree-like-limit assumption."
    This variation tests whether community structure amplifies polarization
    beyond what selective exposure alone produces.

Design:
    1. SBM (2 blocks, 500 each): p_in=0.05, p_out=0.01 → strong modularity
       - Community-aligned Q: Q=+1 in block 0, Q=−1 in block 1  (echo chamber)
       - Community-uncorrelated Q: random Q across blocks
    2. ER (n=1000, p=0.03): comparable mean degree ~30
       - Same two Q assignments

All use selective exposure: Z ~ Beta(8,1)/Beta(1,8) keyed on Q.
c=0.50, d=0.45 (memory case, same as Fig 4).

Key question: does community structure amplify the Q+ vs Q− gap beyond
what selective exposure alone achieves on a random-mixing graph?

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

# ── parameters ───────────────────────────────────────────────────────────────
N         = 1000
N_BLOCK   = 500        # 2 blocks of 500 each
N_ITER    = 200
SEED      = 42
SCENARIO  = "fig4"     # selective exposure

C = 0.50
D = 0.45

# ER: mean degree ≈ 999 * 0.03 = 30
P_ER = 0.03

# SBM: mean degree = 499 * p_in + 500 * p_out ≈ 30
P_IN  = 0.05           # within-block
P_OUT = 0.01           # cross-block
# modularity ≈ (500*0.05 - 500*0.01) / (500*(0.05+0.01)) ≈ 0.67

FIG_PATH = Path(__file__).resolve().parents[1] / "figures" / "variation4_community.png"

# Paper reference (Fig 4 baseline)
PAPER = dict(var=0.1484, mean_plus=+0.3684, mean_minus=-0.3684, var_cond=0.0095)


# ── helpers ──────────────────────────────────────────────────────────────────
def build_sbm(seed: int):
    """2-block SBM with modularity."""
    p_matrix = np.array([[P_IN, P_OUT],
                          [P_OUT, P_IN]], dtype=np.float64)
    A = directed_sbm([N_BLOCK, N_BLOCK], p_matrix, C, seed=seed)
    # block_id[i] = 0 or 1
    block_id = np.zeros(N, dtype=np.int64)
    block_id[N_BLOCK:] = 1
    return A, block_id


def build_er(seed: int):
    return directed_er(N, P_ER, C, seed=seed)


def assign_Q_community(block_id: np.ndarray, seed: int) -> np.ndarray:
    """Q = +1 for block 0, −1 for block 1 (echo chamber)."""
    rng = np.random.default_rng(seed)
    Q = np.where(block_id == 0, +1.0, -1.0)
    # Add small noise to avoid deterministic symmetry
    flip = rng.random(N) < 0.02  # 2% random flips
    Q[flip] *= -1.0
    return Q


def assign_Q_random(seed: int) -> np.ndarray:
    """Q uniform on {−1, +1} (no correlation with community)."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1.0, 1.0]), size=N)


def run_case(label: str, A, Q: np.ndarray, seed_dyn: int) -> dict:
    S = np.zeros(N, dtype=np.int8)
    out = run_to_stationarity(A, Q, S, c=C, d=D, scenario=SCENARIO,
                               n_iter=N_ITER, seed=seed_dyn)
    stats = empirical_moments(out["R"], Q)
    stats["label"] = label
    stats["R"] = out["R"]
    stats["Q"] = Q
    return stats


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── build graphs ──────────────────────────────────────────────────────
    print("Building graphs ...")
    A_sbm, block_id = build_sbm(seed=SEED)
    A_er = build_er(seed=SEED)

    # ── prepare Q assignments ─────────────────────────────────────────────
    Q_comm  = assign_Q_community(block_id, seed=SEED + 2)    # echo chamber
    Q_rand  = assign_Q_random(seed=SEED + 3)                  # random mixing

    # Also make an ER-specific random Q for fair paired comparison
    Q_rand2 = assign_Q_random(seed=SEED + 4)

    # ── 4 cases ───────────────────────────────────────────────────────────
    cases = [
        ("SBM + echo chamber",   A_sbm, Q_comm,  200),
        ("SBM + random Q",       A_sbm, Q_rand,  201),
        ("ER + echo chamber Q",  A_er,  Q_comm,  202),
        ("ER + random Q",        A_er,  Q_rand2, 203),
    ]

    results = []
    for label, A, Q, seed_dyn in cases:
        print(f"  Running: {label} ...")
        stats = run_case(label, A, Q, seed_dyn)
        results.append(stats)
        print(f"    Var(R*) = {stats['var']:.4f}")
        print(f"    E[R|+]  = {stats['mean_Q_plus']:+.4f}")
        print(f"    E[R|−]  = {stats['mean_Q_minus']:+.4f}")
        print(f"    ΔQ gap  = {stats['mean_Q_plus'] - stats['mean_Q_minus']:+.4f}")

    # ── summary table ────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print(f"{'Case':>22}  {'Var(R*)':>10}  {'E[R|+]':>10}  {'E[R|−]':>10}  {'ΔQ gap':>10}")
    print("─" * 80)
    for r in results:
        gap = r["mean_Q_plus"] - r["mean_Q_minus"]
        print(f"{r['label']:>22}  {r['var']:10.4f}  {r['mean_Q_plus']:10.4f}  "
              f"{r['mean_Q_minus']:10.4f}  {gap:10.4f}")
    print("─" * 80)

    # ── key comparisons ──────────────────────────────────────────────────
    r_sbm_echo = results[0]
    r_er_echo  = results[2]
    r_sbm_rand = results[1]
    r_er_rand  = results[3]

    echo_gap_diff = (r_sbm_echo["mean_Q_plus"] - r_sbm_echo["mean_Q_minus"]) - \
                    (r_er_echo["mean_Q_plus"] - r_er_echo["mean_Q_minus"])
    echo_var_diff = r_sbm_echo["var"] - r_er_echo["var"]

    print(f"\nCommunity amplification (echo chamber):")
    print(f"  SBM ΔQ − ER ΔQ = {echo_gap_diff:+.4f}")
    print(f"  SBM Var − ER Var = {echo_var_diff:+.4f}")
    print(f"  → Community structure {'amplifies' if echo_gap_diff > 0.01 else 'does NOT amplify'} "
          f"polarization beyond selective exposure alone.")

    rand_gap_diff = (r_sbm_rand["mean_Q_plus"] - r_sbm_rand["mean_Q_minus"]) - \
                    (r_er_rand["mean_Q_plus"] - r_er_rand["mean_Q_minus"])
    print(f"\nCommunity without echo (random Q):")
    print(f"  SBM ΔQ − ER ΔQ = {rand_gap_diff:+.4f}")
    print(f"  → {'Community alone adds some polarization' if abs(rand_gap_diff) > 0.01 else 'No significant effect without echo chamber alignment'}")

    # ── plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    bins = np.linspace(-1.0, 1.0, 51)
    COL_PLUS  = "#c0392b"
    COL_MINUS = "#2471a3"

    # Order: SBM-echo, ER-echo, SBM-random, ER-random
    for ax, r in zip(axes.flat, results):
        R = r["R"]
        Q = r["Q"]
        pos = Q > 0
        neg = Q < 0

        ax.hist(R[pos], bins=bins, density=True, alpha=0.55,
                color=COL_PLUS,  label=r"$Q=+1$")
        ax.hist(R[neg], bins=bins, density=True, alpha=0.55,
                color=COL_MINUS, label=r"$Q=-1$")

        gap = r["mean_Q_plus"] - r["mean_Q_minus"]
        txt = (f"Var(R*) = {r['var']:.4f}\n"
               f"E[R|+] = {r['mean_Q_plus']:+.4f}\n"
               f"E[R|−] = {r['mean_Q_minus']:+.4f}\n"
               f"ΔQ gap = {gap:+.4f}")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        # Mark conditional means
        ax.axvline(x=r["mean_Q_plus"],  color=COL_PLUS,  ls="--", lw=1.2, alpha=0.6)
        ax.axvline(x=r["mean_Q_minus"], color=COL_MINUS, ls="--", lw=1.2, alpha=0.6)

        ax.set_xlim(-1, 1)
        ax.set_xlabel("Opinion $R^*$", fontsize=10)
        ax.set_title(r["label"], fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.3)

    for ax in axes[:, 0]:
        ax.set_ylabel("Density", fontsize=10)

    fig.suptitle(
        "Variation 4 — Community structure (SBM) vs random mixing (ER)\n"
        rf"$c={C},\;d={D}$, selective exposure, $n={N}$",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
