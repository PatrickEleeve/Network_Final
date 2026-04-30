"""Replicate Figure 4 from Fraiman, Lin & Olvera-Cravioto (2024).

Two panels side-by-side:
  Left  — memory case    (c=0.50, d=0.45, 1-c-d=0.05)
  Right — no-memory case (c=0.5263, d=0.4737, 1-c-d=0)

Graph: directed ER(n=1000, p=0.03), selective-exposure signals (fig4 scenario).
Expected: Var(R*)≈0.1484, E[R*|Q=+1]≈+0.3684, E[R*|Q=-1]≈-0.3684.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.attributes import sample_attributes
from src.dynamics import run_to_stationarity
from src.graph_construction import directed_er
from src.validation import empirical_moments, fig4_proposition4_prediction

# ── parameters ────────────────────────────────────────────────────────────────
N = 1000
P = 0.03
SEED_GRAPH = 42
SEED_ATTR  = 43
N_ITER     = 200   # d≈0.45 → k_ε≈31 for 1e-6 precision; 200 gives large safety margin
N_REPLICATES = 10

_C_MEM = 0.50
_D_MEM = 0.45
_RATIO = _C_MEM + _D_MEM
CASES = [
    {"label": "With Memory",    "c": _C_MEM,          "d": _D_MEM,          "seed_dyn": 101},
    {"label": "Without Memory", "c": _C_MEM / _RATIO, "d": _D_MEM / _RATIO, "seed_dyn": 102},
]

# Source-paper Figure 4 benchmark values.  Proposition 4/5 is evaluated below
# only for the no-memory case where c + d = 1.
TARGET = {
    "var":         0.1484,
    "mean_Q_plus":  0.3684,
    "mean_Q_minus": -0.3684,
    "var_Q_cond":   0.0095,
}

# ── build graph & attributes once ─────────────────────────────────────────────
print("Building graph ...")
attrs = sample_attributes(N, scenario="fig4", seed=SEED_ATTR)
Q = attrs["Q"]
S = attrs["S"]

# ── run dynamics for each case ─────────────────────────────────────────────────
results = []
for case in CASES:
    c, d = case["c"], case["d"]
    # rebuild A with correct c (weights scale with c)
    A_case = directed_er(N, P, c=c, seed=SEED_GRAPH)
    print(f"Running {case['label']}  (c={c}, d={d}, n_iter={N_ITER}) ...")
    out = run_to_stationarity(
        A_case, Q, S, c=c, d=d,
        scenario="fig4", n_iter=N_ITER, seed=case["seed_dyn"],
    )
    R = out["R"]
    stats = empirical_moments(R, Q)
    theory = None
    if np.isclose(c + d, 1.0):
        theory = fig4_proposition4_prediction(A_case, Q, c, d)
    results.append({"case": case, "R": R, "stats": stats, "theory": theory})
    print(f"  Var(R*)        = {stats['var']:.4f}  (source benchmark ≈ {TARGET['var']})")
    print(f"  E[R*|Q=+1]     = {stats['mean_Q_plus']:.4f}  (source benchmark ≈ {TARGET['mean_Q_plus']})")
    print(f"  E[R*|Q=-1]     = {stats['mean_Q_minus']:.4f}  (source benchmark ≈ {TARGET['mean_Q_minus']})")
    print(f"  Var(R*|Q=+1)   = {stats['var_Q_plus']:.4f}  (source benchmark ≈ {TARGET['var_Q_cond']})")
    print(f"  Var(R*|Q=-1)   = {stats['var_Q_minus']:.4f}  (source benchmark ≈ {TARGET['var_Q_cond']})")
    if theory is not None:
        print(f"  Prop. 4 plug-in Var = {theory['var']:.4f},  ρ₂ = {theory['rho2']:.5f}")
    print()


def _mean_se(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr)))


print(f"Seed-averaged summary over {N_REPLICATES} independent graph/attribute/dynamics seeds ...")
for case in CASES:
    c, d = case["c"], case["d"]
    rows = []
    theory_rows = []
    for rep in range(N_REPLICATES):
        seed_shift = 1000 * rep
        A_rep = directed_er(N, P, c=c, seed=SEED_GRAPH + seed_shift)
        attrs_rep = sample_attributes(N, scenario="fig4", seed=SEED_ATTR + seed_shift)
        Q_rep = attrs_rep["Q"]
        S_rep = attrs_rep["S"]
        out = run_to_stationarity(
            A_rep, Q_rep, S_rep, c=c, d=d,
            scenario="fig4", n_iter=N_ITER,
            seed=case["seed_dyn"] + seed_shift,
        )
        rows.append(empirical_moments(out["R"], Q_rep))
        if np.isclose(c + d, 1.0):
            theory_rows.append(fig4_proposition4_prediction(A_rep, Q_rep, c, d))

    print(f"  {case['label']}:")
    for key, label in [
        ("var", "Var(R*)"),
        ("mean_Q_plus", "E[R*|Q=+1]"),
        ("mean_Q_minus", "E[R*|Q=-1]"),
        ("var_Q_plus", "Var(R*|Q=+1)"),
        ("var_Q_minus", "Var(R*|Q=-1)"),
    ]:
        mean, se = _mean_se([row[key] for row in rows])
        print(f"    empirical {label:<16} = {mean:+.4f} ± {se:.4f}")
    if theory_rows:
        mean, se = _mean_se([row["var"] for row in theory_rows])
        rho_mean, rho_se = _mean_se([row["rho2"] for row in theory_rows])
        print(f"    Prop. 4 plug-in Var    = {mean:+.4f} ± {se:.4f}")
        print(f"    ρ₂                     = {rho_mean:.5f} ± {rho_se:.5f}")
print()

# ── media signal overlay densities ────────────────────────────────────────────
x = np.linspace(-1, 1, 400)
from scipy.stats import beta as beta_dist

# Beta(8,1) shifted to [-1,1]: pdf_Z(z) = beta(8,1).pdf((z+1)/2) / 2
def shifted_beta_pdf(x_vals, a, b):
    u = (x_vals + 1) / 2
    return beta_dist.pdf(u, a, b) / 2.0

pdf_plus  = shifted_beta_pdf(x, 8, 1)   # Q=+1 media
pdf_minus = shifted_beta_pdf(x, 1, 8)   # Q=-1 media

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

BINS = 50
COLOR_PLUS  = "#C0392B"   # red for Q=+1
COLOR_MINUS = "#1A5276"   # navy for Q=-1
ALPHA_HIST  = 0.55

for ax, res in zip(axes, results):
    R   = res["R"]
    stats = res["stats"]
    case  = res["case"]

    pos = Q > 0
    neg = Q < 0

    ax.hist(R[pos], bins=BINS, range=(-1, 1), density=True,
            color=COLOR_PLUS,  alpha=ALPHA_HIST, label=r"$Q=+1$")
    ax.hist(R[neg], bins=BINS, range=(-1, 1), density=True,
            color=COLOR_MINUS, alpha=ALPHA_HIST, label=r"$Q=-1$")

    # dashed overlay: underlying media distributions
    ax.plot(x, pdf_plus,  color=COLOR_PLUS,  lw=1.4, ls="--", alpha=0.8)
    ax.plot(x, pdf_minus, color=COLOR_MINUS, lw=1.4, ls="--", alpha=0.8)

    ax.set_xlim(-1, 1)
    ax.set_xlabel("Opinion $R^*$", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        f"{case['label']}\n"
        rf"$c={case['c']},\ d={case['d']}$",
        fontsize=12,
    )

    stat_text = (
        rf"$\mathrm{{Var}}(R^*)={stats['var']:.4f}$" + "\n"
        rf"$\mathbb{{E}}[R^*|Q=+1]={stats['mean_Q_plus']:.4f}$" + "\n"
        rf"$\mathbb{{E}}[R^*|Q=-1]={stats['mean_Q_minus']:.4f}$"
    )
    ax.text(0.03, 0.97, stat_text, transform=ax.transAxes,
            va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.legend(fontsize=10, loc="upper center")

fig.suptitle(
    "Figure 4 — Selective Exposure (ER, $n=1000$, $p=0.03$)",
    fontsize=13, y=1.01,
)
plt.tight_layout()

out_path = Path(__file__).parent.parent / "figures" / "fig4_replication.png"
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.close(fig)
