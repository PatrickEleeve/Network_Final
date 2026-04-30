"""Variation 3: Bot influence deep dive — do bots polarize or merely shift?

Critique from CLAUDE.md (critique_3):
    The paper claims "stubborn agents (bots) can polarize opinions" but Fig 5
    shows only a SHIFTED unimodal distribution.  Paper section 5.3 acknowledges
    "bots cannot target individuals based on their internal opinions, which
    reduces their effectiveness for polarizing."  Our test distinguishes
    distribution shift from true bimodal polarization.

Two sub-experiments scanning bot proportion (0% → 50%):
    A. Asymmetric bots (paper's setup): all bots push Q=+1, Z=+1
    B. Balanced bots: 50% fixed at Q=+1, 50% fixed at Q=-1

Metrics across bot proportion:
    - Var(R*): overall opinion dispersion
    - ΔQ = E[R*|Q=+1] − E[R*|Q=−1]: polarization gap (regular vertices only)
    - Mean(R*): regular-vertex distribution shift
    - Var(R* | Q=+1), Var(R* | Q=−1): within-group dispersion

Key question: does bot proportion correlate with bimodality (enlarged Q-gap) or
merely with mean shift (regular opinions drift toward +1)?

Run:
    python3 scripts/variation3_bot_scan.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attributes import sample_attributes
from src.graph_construction import directed_er, directed_er_with_bots, in_degree
from src.signals import external_signal, sample_media
from src.validation import empirical_moments

# ── parameters ───────────────────────────────────────────────────────────────
N_TOTAL   = 1000
P         = 0.03
C         = 0.50
D         = 0.45
N_ITER    = 200
SEED      = 42
SCENARIO  = "fig4"       # selective exposure for regular vertices

BOT_PROPORTIONS = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])

FIG_PATH = Path(__file__).resolve().parents[1] / "figures" / "variation3_bot_scan.png"

# Paper reference (Fig 4, no bots)
PAPER = dict(var=0.1484, mean_plus=+0.3684, mean_minus=-0.3684, var_cond=0.0095)


# ── simulation ───────────────────────────────────────────────────────────────
def run_bot_dynamics(
    A,
    Q: np.ndarray,
    S: np.ndarray,
    is_bot: np.ndarray,
    balanced: bool,
    seed_dyn: int,
) -> np.ndarray:
    """Run dynamics while keeping bots fixed at their intended opinions.

    This is stricter than only correcting bots at the end: regular vertices
    listen to bot opinions during every update, so bots must be pinned from
    initialization onward for the balanced-bot experiment to mean what it says.
    """
    n = len(Q)
    rng = np.random.default_rng(seed_dyn)
    deg = in_degree(A)
    memory = 1.0 - C - D

    R = rng.choice(np.array([-1.0, 1.0]), size=n).astype(np.float64)
    if is_bot.any():
        R[is_bot] = Q[is_bot]

    scenario = "fig5" if not balanced else "fig4"
    for _ in range(N_ITER):
        Z = sample_media(Q, S, scenario, rng)
        if is_bot.any():
            Z[is_bot] = Q[is_bot]
        W = external_signal(Q, Z, deg, C, D)
        R = A @ R + memory * R + W
        if is_bot.any():
            R[is_bot] = Q[is_bot]

    return R


def run_bot_case(prop_bots: float, balanced: bool, seed_base: int) -> dict:
    """Run one bot-proportion trial.

    If balanced=True, half the bots push +1 and half push −1.
    Otherwise (paper's), all bots push +1.
    """
    n_bots = int(N_TOTAL * prop_bots)
    n_regular = N_TOTAL - n_bots

    if n_bots == 0:
        # Fall back to plain ER (no bots)
        A = directed_er(N_TOTAL, P, C, seed=SEED)
        attrs = sample_attributes(N_TOTAL, SCENARIO, seed=SEED + 1)
        Q = attrs["Q"]
        S = attrs["S"]
        is_bot = np.zeros(N_TOTAL, dtype=bool)
    else:
        A, is_bot = directed_er_with_bots(n_regular, n_bots, P, C, seed=SEED)
        attrs = sample_attributes(N_TOTAL, SCENARIO, seed=SEED + 1, is_bot=is_bot)
        Q = attrs["Q"]
        S = attrs["S"]
        # Override: in paper's case, all bots Q=+1; in balanced case, split
        bot_indices = np.where(is_bot)[0]
        if balanced:
            n_half = n_bots // 2
            Q[bot_indices[:n_half]] = +1.0
            Q[bot_indices[n_half:]] = -1.0
        else:
            Q[bot_indices] = +1.0
        S[bot_indices] = 1

    R = run_bot_dynamics(A, Q, S, is_bot, balanced=balanced, seed_dyn=seed_base)

    # Compute stats on regular vertices only (bots excluded from opinion analysis)
    regular = ~is_bot
    stats = empirical_moments(R[regular], Q[regular])
    stats["mean_regular"] = float(np.mean(R[regular]))
    stats["mean_all"] = float(np.mean(R))
    stats["prop_bots"] = prop_bots
    stats["balanced"] = balanced
    stats["R_regular"] = R[regular]
    stats["Q_regular"] = Q[regular]
    stats["R_all"] = R
    stats["Q_all"] = Q
    stats["is_bot"] = is_bot
    return stats


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    results_asym = []
    results_bal  = []

    for prop in BOT_PROPORTIONS:
        seed_dyn = 200 + int(prop * 100)
        print(f"\nBot proportion {prop:.0%} — asymmetric ...")
        r = run_bot_case(prop, balanced=False, seed_base=seed_dyn)
        results_asym.append(r)
        print(f"  Var(R*)={r['var']:.4f},  ΔQ={r['mean_Q_plus']-r['mean_Q_minus']:+.4f},  "
              f"Regular mean={r['mean_regular']:+.4f}")

        print(f"Bot proportion {prop:.0%} — balanced ...")
        r = run_bot_case(prop, balanced=True, seed_base=seed_dyn)
        results_bal.append(r)
        print(f"  Var(R*)={r['var']:.4f},  ΔQ={r['mean_Q_plus']-r['mean_Q_minus']:+.4f},  "
              f"Regular mean={r['mean_regular']:+.4f}")

    # ── extract vectors ──────────────────────────────────────────────────
    def extract(results, key):
        return np.array([r[key] for r in results])

    asym_var   = extract(results_asym, "var")
    asym_gap   = extract(results_asym, "mean_Q_plus") - extract(results_asym, "mean_Q_minus")
    asym_mean  = extract(results_asym, "mean_regular")
    asym_mean_p = extract(results_asym, "mean_Q_plus")
    asym_mean_m = extract(results_asym, "mean_Q_minus")

    bal_var    = extract(results_bal, "var")
    bal_gap    = extract(results_bal, "mean_Q_plus") - extract(results_bal, "mean_Q_minus")
    bal_mean   = extract(results_bal, "mean_regular")
    bal_mean_p = extract(results_bal, "mean_Q_plus")
    bal_mean_m = extract(results_bal, "mean_Q_minus")

    # ── summary table ────────────────────────────────────────────────────
    print("\n" + "─" * 95)
    print(f"{'Prop':>6}  {'Asym Var':>10}  {'Asym ΔQ':>10}  {'Asym RegMean':>12}  "
          f"{'Bal Var':>10}  {'Bal ΔQ':>10}  {'Bal RegMean':>12}")
    print("─" * 95)
    for i, prop in enumerate(BOT_PROPORTIONS):
        print(f"{prop:6.0%}  {asym_var[i]:10.4f}  {asym_gap[i]:10.4f}  {asym_mean[i]:12.4f}  "
              f"{bal_var[i]:10.4f}  {bal_gap[i]:10.4f}  {bal_mean[i]:12.4f}")

    paper_gap = PAPER["mean_plus"] - PAPER["mean_minus"]

    # Identify: does asymmetric bot presence increase or decrease ΔQ?
    print(f"\nPaper Q-gap (no bots): {paper_gap:.4f}")
    print(f"Asymmetric: Q-gap at 0% bots = {asym_gap[0]:.4f}")
    print(f"  Q-gap at max bots = {asym_gap[-1]:.4f} "
          f"({(asym_gap[-1]/asym_gap[0]-1)*100:+.1f}% change)")
    print(f"Balanced:   Q-gap at 0% bots = {bal_gap[0]:.4f}")
    print(f"  Q-gap at max bots = {bal_gap[-1]:.4f} "
          f"({(bal_gap[-1]/bal_gap[0]-1)*100:+.1f}% change)")

    # ── plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    prop_pct = BOT_PROPORTIONS * 100

    # Panel 1: Var(R*) vs bot proportion
    ax = axes[0, 0]
    ax.plot(prop_pct, asym_var, "o-", color="#c0392b", lw=1.8, ms=5, label="Asymmetric bots")
    ax.plot(prop_pct, bal_var,  "s--", color="#2471a3", lw=1.8, ms=5, label="Balanced bots")
    ax.axhline(y=PAPER["var"], color="gray", ls=":", lw=1.2,
               label=f"Paper (no bots): {PAPER['var']}")
    ax.set_xlabel("Bot proportion (%)", fontsize=10)
    ax.set_ylabel(r"$\mathrm{Var}(R^*)$", fontsize=11)
    ax.set_title("Opinion variance", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Panel 2: Polarization gap ΔQ vs bot proportion
    ax = axes[0, 1]
    ax.plot(prop_pct, asym_gap, "o-", color="#c0392b", lw=1.8, ms=5)
    ax.plot(prop_pct, bal_gap,  "s--", color="#2471a3", lw=1.8, ms=5)
    ax.axhline(y=paper_gap, color="gray", ls=":", lw=1.2,
               label=f"Paper baseline: {paper_gap:.4f}")
    ax.set_xlabel("Bot proportion (%)", fontsize=10)
    ax.set_ylabel(r"$E[R^*|Q{=}+1] - E[R^*|Q{=}-1]$", fontsize=10)
    ax.set_title("Polarization gap (regular vertices)", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Panel 3: regular-vertex mean shift
    ax = axes[0, 2]
    ax.plot(prop_pct, asym_mean, "o-", color="#c0392b", lw=1.8, ms=5, label="Asymmetric")
    ax.plot(prop_pct, bal_mean,  "s--", color="#2471a3", lw=1.8, ms=5, label="Balanced")
    ax.axhline(y=0, color="black", lw=0.6)
    ax.set_xlabel("Bot proportion (%)", fontsize=10)
    ax.set_ylabel(r"Regular mean of $R^*$", fontsize=11)
    ax.set_title("Regular opinion shift", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # Panel 4: Conditional means (asymmetric)
    ax = axes[1, 0]
    ax.plot(prop_pct, asym_mean_p, "o-", color="#c0392b", lw=1.5, ms=4,
            label=r"$E[R^*|Q{=}+1]$")
    ax.plot(prop_pct, asym_mean_m, "o-", color="#2471a3", lw=1.5, ms=4,
            label=r"$E[R^*|Q{=}-1]$")
    ax.axhline(y=0, color="black", lw=0.5)
    ax.set_xlabel("Bot proportion (%)", fontsize=10)
    ax.set_ylabel("Conditional mean", fontsize=10)
    ax.set_title("Asymmetric bots — conditional means", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 5: Conditional means (balanced)
    ax = axes[1, 1]
    ax.plot(prop_pct, bal_mean_p, "o-", color="#c0392b", lw=1.5, ms=4,
            label=r"$E[R^*|Q{=}+1]$")
    ax.plot(prop_pct, bal_mean_m, "o-", color="#2471a3", lw=1.5, ms=4,
            label=r"$E[R^*|Q{=}-1]$")
    ax.axhline(y=0, color="black", lw=0.5)
    ax.set_xlabel("Bot proportion (%)", fontsize=10)
    ax.set_ylabel("Conditional mean", fontsize=10)
    ax.set_title("Balanced bots — conditional means", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 6: Histograms at selected proportions (asymmetric)
    ax = axes[1, 2]
    bins = np.linspace(-1.0, 1.0, 41)
    COL_PLUS  = "#c0392b"
    COL_MINUS = "#2471a3"

    # Show histograms for 3 key proportions: 0%, 20%, 50%
    for idx, (prop_show, alpha) in enumerate([(0, 0.9), (4, 0.7), (8, 0.5)]):
        r = results_asym[prop_show]
        Rreg = r["R_regular"]
        Qreg = r["Q_regular"]
        pos = Qreg > 0
        neg = Qreg < 0
        prop_val = BOT_PROPORTIONS[prop_show]
        ls = "-" if idx == 0 else "--" if idx == 1 else ":"
        ax.hist(Rreg[pos], bins=bins, density=True, alpha=alpha * 0.5,
                color=COL_PLUS, histtype="stepfilled")
        ax.hist(Rreg[neg], bins=bins, density=True, alpha=alpha * 0.5,
                color=COL_MINUS, histtype="stepfilled")

        # Add a line annotation
        ax.axvline(x=r["mean_Q_plus"], color=COL_PLUS, ls=ls, lw=1.2, alpha=alpha,
                   label=f"{prop_val:.0%} bots: E[+|+]={r['mean_Q_plus']:+.3f}")
        ax.axvline(x=r["mean_Q_minus"], color=COL_MINUS, ls=ls, lw=1.2, alpha=alpha,
                   label=f"{prop_val:.0%} bots: E[+|−]={r['mean_Q_minus']:+.3f}")

    ax.set_xlim(-1, 1)
    ax.set_xlabel("Opinion $R^*$", fontsize=10)
    ax.set_title("Asymmetric bots — opinion histograms", fontsize=10)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Variation 3 — Bot influence: polarization vs regular-opinion shift\n"
        rf"ER($n={N_TOTAL},\;p={P}$), $c={C},\;d={D}$, selective exposure for regular vertices",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
