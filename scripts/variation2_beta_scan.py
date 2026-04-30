"""Variation 2: Polarization phase transition — scan Beta asymmetry.

Critique from CLAUDE.md (critique_2):
    "Single parameter point Beta(8,1)/Beta(1,8). Our test: scan Beta asymmetry
    from (8,1) to (1.5, 1). Identify polarization phase transition threshold."

We scan the Beta shape parameter b from 1 (uniform, no selective exposure) to 8
(paper's value, strong selective exposure):
    Q=+1: Z ~ −1 + 2·Beta(b, 1)   (skewed toward +1)
    Q=-1: Z ~ −1 + 2·Beta(1, b)   (skewed toward −1)

Key metrics across b:
    - Var(R*): overall opinion dispersion
    - ΔQ = E[R*|Q=+1] − E[R*|Q=−1]: polarization gap
    - ρ₂: structural heterogeneity

Run:
    python3 scripts/variation2_beta_scan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dynamics import run_to_stationarity
from src.graph_construction import directed_er
from src.validation import empirical_moments

# ── parameters ───────────────────────────────────────────────────────────────
N        = 1000
P        = 0.03
C        = 0.50
D        = 0.45
N_ITER   = 200
SEED     = 42
SCENARIO = "fig4"      # we'll override the signal logic below

# Scan range for Beta shape parameter b
B_VALUES = np.unique(np.round(np.concatenate([
    np.linspace(1.0, 2.0, 11),    # fine near symmetric
    np.linspace(2.5, 8.0, 12),    # coarser further out
]), 2))

FIG_PATH = Path(__file__).resolve().parents[1] / "figures" / "variation2_beta_scan.png"

PAPER = {"var": 0.1484, "mean_plus": +0.3684, "mean_minus": -0.3684, "var_cond": 0.0095}


# ── custom signal sampler ────────────────────────────────────────────────────
def sample_media_parametric(Q: np.ndarray, b: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Z ~ -1 + 2*Beta(b,1) for Q=+1, -1 + 2*Beta(1,b) for Q=-1."""
    n = Q.shape[0]
    Z = np.empty(n, dtype=np.float64)
    plus  = Q > 0
    minus = ~plus
    Z[plus]  = -1.0 + 2.0 * rng.beta(b, 1.0, size=plus.sum())
    Z[minus] = -1.0 + 2.0 * rng.beta(1.0, b,   size=minus.sum())
    return Z


def run_simulation(b: float, A, Q: np.ndarray, S: np.ndarray,
                   seed_dyn: int) -> dict:
    """Run opinion dynamics with parametrized Beta(b,1)/(1,b) media."""
    def media_sampler(Q_: np.ndarray, _S: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
        return sample_media_parametric(Q_, b, rng)

    out = run_to_stationarity(
        A, Q, S, c=C, d=D, scenario=SCENARIO,
        n_iter=N_ITER, seed=seed_dyn, media_sampler=media_sampler,
    )
    return empirical_moments(out["R"], Q)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # build graph & attributes once
    print(f"Building ER(n={N}, p={P}), seed={SEED} ...")
    A = directed_er(N, P, C, seed=SEED)

    rng_attr = np.random.default_rng(SEED + 1)
    Q = rng_attr.choice(np.array([-1.0, 1.0]), size=N)
    S = np.zeros(N, dtype=np.int8)
    print(f"Q: {(Q>0).sum()} = +1, {(Q<0).sum()} = -1")

    # ── scan b values ────────────────────────────────────────────────────
    vars_r   = []
    gaps_q   = []
    means_p  = []
    means_m  = []
    vars_cond_p = []
    vars_cond_m = []

    print(f"\nScanning {len(B_VALUES)} Beta asymmetry values ...")
    for i, b in enumerate(B_VALUES):
        stats = run_simulation(b, A, Q, S, seed_dyn=101 + i)
        vars_r.append(stats["var"])
        gaps_q.append(stats["mean_Q_plus"] - stats["mean_Q_minus"])
        means_p.append(stats["mean_Q_plus"])
        means_m.append(stats["mean_Q_minus"])
        vars_cond_p.append(stats["var_Q_plus"])
        vars_cond_m.append(stats["var_Q_minus"])

        if b in [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            print(f"  b={b:3.1f}: Var={stats['var']:.4f}, "
                  f"gap={stats['mean_Q_plus']-stats['mean_Q_minus']:.4f}, "
                  f"E[+|Q=+1]={stats['mean_Q_plus']:+.4f}, "
                  f"E[+|Q=-1]={stats['mean_Q_minus']:+.4f}")

    vars_r = np.array(vars_r)
    gaps_q = np.array(gaps_q)
    means_p = np.array(means_p)
    means_m = np.array(means_m)
    vars_cond_p = np.array(vars_cond_p)
    vars_cond_m = np.array(vars_cond_m)

    # ── analytical: E[Z|Q=+1] for Beta(b,1) ──────────────────────────────
    analytical_mean_Z_plus = -1.0 + 2.0 * B_VALUES / (B_VALUES + 1.0)

    # ── identify transition ──────────────────────────────────────────────
    # Polarization onset: where gap exceeds 10% of paper's gap at b=8
    paper_gap = PAPER["mean_plus"] - PAPER["mean_minus"]  # ≈ 0.7368
    threshold = 0.10 * paper_gap
    transitional = gaps_q > threshold
    if transitional.any():
        b_crit = B_VALUES[transitional][0]
        print(f"\nPolarization onset (>10% paper gap): b ≈ {b_crit:.1f}")
    else:
        print("\nNo clear polarization threshold detected in scanned range.")

    # ── plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: Var(R*) vs b
    ax = axes[0, 0]
    ax.plot(B_VALUES, vars_r, "o-", color="#2471a3", lw=1.8, ms=5)
    ax.axhline(y=PAPER["var"], color="#c0392b", ls="--", lw=1.2,
               label=f"Paper (b=8): {PAPER['var']}")
    ax.set_xlabel(r"Beta shape parameter $b$", fontsize=11)
    ax.set_ylabel(r"$\mathrm{Var}(R^*)$", fontsize=12)
    ax.set_title("Opinion variance vs selective exposure strength", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Polarization gap vs b
    ax = axes[0, 1]
    ax.plot(B_VALUES, gaps_q, "o-", color="#c0392b", lw=1.8, ms=5)
    ax.axhline(y=paper_gap, color="gray", ls="--", lw=1.2,
               label=f"Paper (b=8): {paper_gap:.4f}")
    ax.axhline(y=threshold, color="orange", ls=":", lw=1.0,
               label=f"10% threshold: {threshold:.4f}")
    ax.set_xlabel(r"Beta shape parameter $b$", fontsize=11)
    ax.set_ylabel(r"$E[R^*|Q=+1] - E[R^*|Q=-1]$", fontsize=12)
    ax.set_title("Polarization gap", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: conditional means
    ax = axes[1, 0]
    ax.plot(B_VALUES, means_p, "o-", color="#c0392b", lw=1.5, ms=4,
            label=r"$E[R^*|Q=+1]$")
    ax.plot(B_VALUES, means_m, "o-", color="#2471a3", lw=1.5, ms=4,
            label=r"$E[R^*|Q=-1]$")
    ax.plot(B_VALUES, analytical_mean_Z_plus, "--", color="gray", lw=1.2, alpha=0.7,
            label=r"$E[Z|Q=+1]$ (analytical)")
    ax.plot(B_VALUES, -analytical_mean_Z_plus, "--", color="gray", lw=1.2, alpha=0.7)
    ax.axhline(y=0, color="black", lw=0.5)
    ax.set_xlabel(r"Beta shape parameter $b$", fontsize=11)
    ax.set_ylabel("Opinion / signal mean", fontsize=11)
    ax.set_title("Conditional opinion means", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Panel 4: conditional variances
    ax = axes[1, 1]
    ax.plot(B_VALUES, vars_cond_p, "o-", color="#c0392b", lw=1.5, ms=4,
            label=r"$\mathrm{Var}(R^*|Q=+1)$")
    ax.plot(B_VALUES, vars_cond_m, "o-", color="#2471a3", lw=1.5, ms=4,
            label=r"$\mathrm{Var}(R^*|Q=-1)$")
    ax.axhline(y=PAPER["var_cond"], color="gray", ls="--", lw=1.0,
               label=f"Paper: {PAPER['var_cond']}")
    ax.set_xlabel(r"Beta shape parameter $b$", fontsize=11)
    ax.set_ylabel("Conditional variance", fontsize=11)
    ax.set_title("Within-group opinion dispersion", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Variation 2 — Polarization phase transition with selective exposure strength\n"
        rf"ER($n={N},\;p={P}$), $c={C},\;d={D}$, "
        r"$Z|Q{=}+1\sim\mathrm{Beta}(b,1)$, "
        r"$Z|Q{=}-1\sim\mathrm{Beta}(1,b)$",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
