"""Replicate Figure 5 from Fraiman, Lin & Olvera-Cravioto (2024).

Bot scenario: 800 regular vertices + 200 stubborn bots.
Bots broadcast Q=+1, Z=+1 always; they have zero in-degree (never listen).
Regular vertices undergo selective exposure (fig4/f5 scenario).

Two panels:
  Left  — memory case    (c=0.50, d=0.45)
  Right — no-memory case (c≈0.5263, d≈0.4737)

Key observation from paper: bots produce a SHIFTED unimodal distribution
(not true bimodal polarization).  Paper section 5.3 acknowledges this
distinction — bots cannot target based on internal opinion.

Run:
    python3 scripts/replicate_fig5.py
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
from src.graph_construction import directed_er_with_bots
from src.validation import empirical_moments

# ── parameters ───────────────────────────────────────────────────────────────
N_REGULAR = 800
N_BOTS    = 200
N         = N_REGULAR + N_BOTS  # 1000 total
P         = 0.03
N_ITER    = 200
SEED      = 42
SCENARIO  = "fig5"

C_MEM = 0.50
D_MEM = 0.45
_ratio = C_MEM + D_MEM
C_NOM = C_MEM / _ratio
D_NOM = 1.0 - C_NOM

FIG_PATH = Path(__file__).resolve().parents[1] / "figures" / "fig5_replication.png"


# ── helpers ──────────────────────────────────────────────────────────────────
def run_case_bot(label: str, c: float, d: float,
                 A: object, Q: np.ndarray, S: np.ndarray, seed_dyn: int) -> dict:
    out = run_to_stationarity(
        A, Q, S, c=c, d=d, scenario=SCENARIO,
        n_iter=N_ITER, seed=seed_dyn,
    )
    R = out["R"]
    m = empirical_moments(R, Q)
    m.update(label=label, R=R, c=c, d=d)
    return m


def print_stats(regular: np.ndarray, bots: np.ndarray, label: str,
                R: np.ndarray, Q: np.ndarray) -> None:
    """Print descriptive stats for regular v bot subpopulations."""
    print(f"\n  {label}:")
    print(f"    Regular (n={regular.sum()}):  "
          f"mean(R)={R[regular].mean():+.4f},  var(R)={R[regular].var():.4f}")
    print(f"    Bots    (n={bots.sum()}):  "
          f"mean(R)={R[bots].mean():+.4f},  var(R)={R[bots].var():.4f}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Building bot graph: {N_REGULAR} regular + {N_BOTS} bots, p={P}, seed={SEED} ...")
    A_mem, is_bot = directed_er_with_bots(N_REGULAR, N_BOTS, P, C_MEM, seed=SEED)
    A_nom, _      = directed_er_with_bots(N_REGULAR, N_BOTS, P, C_NOM, seed=SEED)

    attrs = sample_attributes(N, SCENARIO, seed=SEED + 1, is_bot=is_bot)
    Q = attrs["Q"]
    S = attrs["S"]

    regular = ~is_bot
    bots    = is_bot
    print(f"Regular vertices: {regular.sum()}, Bots: {bots.sum()}")
    print(f"Bots Q: all {Q[bots].mean():+.1f}, Bots S: all {S[bots].sum()}")

    # ── run dynamics ──────────────────────────────────────────────────────
    print(f"\nRunning {N_ITER} iterations per case ...")
    mem = run_case_bot("Memory", C_MEM, D_MEM, A_mem, Q, S, seed_dyn=101)
    nom = run_case_bot("No-memory", C_NOM, D_NOM, A_nom, Q, S, seed_dyn=102)

    for case in [mem, nom]:
        print_stats(regular, bots, f"{case['label']} (c={case['c']:.4f}, d={case['d']:.4f})",
                    case["R"], Q)

    # ── print comparison: regular only vs all ─────────────────────────────
    print("\n" + "─" * 80)
    print("Regular-only opinion moments (exclude bots from stats):")
    for case in [mem, nom]:
        Rreg = case["R"][regular]
        Qreg = Q[regular]
        pos = Qreg > 0
        neg = Qreg < 0
        print(f"  {case['label']}:")
        print(f"    Mean(R)       = {Rreg.mean():+.4f}")
        print(f"    Var(R)        = {Rreg.var():.4f}")
        print(f"    E[R|Q=+1]     = {Rreg[pos].mean():+.4f}")
        print(f"    E[R|Q=-1]     = {Rreg[neg].mean():+.4f}")
        print(f"    Var(R|Q=+1)   = {Rreg[pos].var():.4f}")
        print(f"    Var(R|Q=-1)   = {Rreg[neg].var():.4f}")

    # ── plot ──────────────────────────────────────────────────────────────
    bins = np.linspace(-1.0, 1.0, 51)
    COL_REG_PLUS  = "#c0392b"   # red for regular Q=+1
    COL_REG_MINUS = "#2471a3"   # blue for regular Q=-1
    COL_BOT       = "#7d3c98"   # purple for bots

    # media signal overlay densities
    z = np.linspace(-1.0, 1.0, 400)
    u = (z + 1.0) / 2.0
    pdf_plus  = beta_dist.pdf(u, 8, 1) / 2.0
    pdf_minus = beta_dist.pdf(u, 1, 8) / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    panel_titles = [
        rf"Memory ($c={C_MEM},\;d={D_MEM},\;c+d=0.95$)",
        rf"No-memory ($c={C_NOM:.4f},\;d={D_NOM:.4f},\;c+d=1$)",
    ]

    ymax = 0.0
    for ax, case, title in zip(axes, [mem, nom], panel_titles):
        R = case["R"]

        # Regular vertices split by Q
        reg_pos = regular & (Q > 0)
        reg_neg = regular & (Q < 0)

        n_pos, _, _ = ax.hist(R[reg_pos], bins=bins, density=True, alpha=0.50,
                              color=COL_REG_PLUS,  label=r"Regular $Q=+1$")
        n_neg, _, _ = ax.hist(R[reg_neg], bins=bins, density=True, alpha=0.50,
                              color=COL_REG_MINUS, label=r"Regular $Q=-1$")
        ymax = max(ymax, float(n_pos.max()), float(n_neg.max()))

        # media signal overlays
        ax.plot(z, pdf_plus,  "--", color=COL_REG_PLUS,  lw=1.4, alpha=0.7,
                label=r"signal $Q{=}+1$")
        ax.plot(z, pdf_minus, "--", color=COL_REG_MINUS, lw=1.4, alpha=0.7,
                label=r"signal $Q{=}-1$")
        ymax = max(ymax, float(pdf_plus.max()), float(pdf_minus.max()))

        # Bots are a point mass at +1. Plotting them as a density histogram
        # creates a height 1 / bin_width = 25 spike and compresses regular
        # opinions, so mark the point mass separately.
        ax.axvline(0.995, color=COL_BOT, lw=2.0, alpha=0.9,
                   label="Bots: point mass at +1")

        # stats annotation
        Rreg = R[regular]
        txt = (f"Mean(R) all     = {R.mean():+.4f}\n"
               f"Var(R)  all     = {R.var():.4f}\n"
               f"Mean(R) regular = {Rreg.mean():+.4f}\n"
               f"Var(R)  regular = {Rreg.var():.4f}\n"
               f"Bots: 200 at R=+1")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        ax.set_xlim(-1, 1)
        ax.set_xlabel(r"Opinion $R_i^*$", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.3)

    for ax in axes:
        ax.set_ylim(0, ymax * 1.18)

    axes[0].set_ylabel("Probability density", fontsize=11)
    fig.suptitle(
        rf"Figure 5 replication — {N_REGULAR} regular + {N_BOTS} bots, "
        rf"ER($p={P}$), selective exposure",
        fontsize=11,
    )
    fig.tight_layout()
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nFigure saved → {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
