"""Numerical verification of Theorem 1 (geometric contraction at rate <= 1 - d).

Two coupled chains with different R^(0) see the same W^(k) realization, so their
difference evolves deterministically as (R1 - R2)^(k+1) = Delta (R1 - R2)^(k),
giving
    ||R1 - R2||_inf  <=  (1 - d)^k ||R1 - R2||^(0).

We scan several d values and plot measured decay against the theoretical bound
on a log scale.

Run from repo root:
    python -m scripts.validate_theorem1
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attributes import sample_attributes
from src.graph_construction import directed_er
from src.validation import validate_theorem1_convergence


FIG_PATH = Path(__file__).resolve().parents[1] / "figures" / "theorem1_convergence.png"


def run_case(n: int, p: float, c: float, d: float, scenario: str,
             n_iter: int, seed: int) -> dict:
    A = directed_er(n, p, c, seed=seed)
    attrs = sample_attributes(n, scenario, seed=seed + 1)
    result = validate_theorem1_convergence(
        A, attrs["Q"], attrs["S"], c=c, d=d,
        scenario=scenario, n_iter=n_iter, seed=seed + 2,
    )

    diffs = result["diffs"]
    k = np.arange(len(diffs))
    bound = diffs[0] * (1 - d) ** k

    valid = bound > 1e-12
    max_ratio = float(np.max(diffs[valid] / bound[valid]))

    mask = (diffs > 1e-12) & (diffs < diffs[0])
    if mask.sum() >= 2:
        slope = np.polyfit(k[mask], np.log(diffs[mask]), 1)[0]
        rate_trim = float(np.exp(slope))
    else:
        rate_trim = float("nan")

    passed = (max_ratio <= 1.0 + 1e-8) and (rate_trim <= 1 - d + 1e-6)
    return {
        "c": c, "d": d, "n_iter": n_iter,
        "diffs": diffs, "bound": bound, "k": k,
        "max_ratio": max_ratio, "empirical_rate": rate_trim,
        "passed": passed,
    }


def print_case(r: dict) -> None:
    print(f"  c={r['c']:.3f}  d={r['d']:.3f}  bound rate={1 - r['d']:.4f}  "
          f"empirical={r['empirical_rate']:.4f}  max ratio={r['max_ratio']:.4f}  "
          f"{'PASS' if r['passed'] else 'FAIL'}")


def main() -> None:
    n, p = 1000, 0.03
    scenario = "fig4"
    seed = 0

    # Scan d values; c stays at 0.5 (fig 4/5 regime) except when d > 0.5 would
    # violate c + d <= 1, in which case lower c so d can grow.
    cases_spec = [
        # (c, d, n_iter)
        (0.5, 0.05,   200),
        (0.5, 0.20,   100),
        (0.5, 0.45,    80),
        (0.1, 0.80,    40),
    ]

    print(f"ER({n}, {p})  scenario={scenario}")
    print("Theorem 1: ||R1-R2||_inf should decay at rate <= 1 - d")
    cases = []
    for c, d, n_iter in cases_spec:
        r = run_case(n, p, c, d, scenario, n_iter, seed)
        print_case(r)
        cases.append(r)

    overall_pass = all(r["passed"] for r in cases)
    print(f"\nAll cases: {'PASS' if overall_pass else 'FAIL'}")

    # ----- Plot -----
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(cases)))

    for r, color in zip(cases, colors):
        # Mask fp floor on measured curve so the plot doesn't show random noise
        # once we hit machine epsilon; the theoretical bound is plotted full.
        d = r["d"]
        diffs = r["diffs"].copy()
        diffs_plot = np.where(diffs > 5e-16, diffs, np.nan)
        ax.semilogy(r["k"], diffs_plot, marker="o", ms=3, lw=1.2, color=color,
                    label=f"measured, d={d:.2f}")
        ax.semilogy(r["k"], r["bound"], linestyle="--", lw=1.0, color=color,
                    alpha=0.7, label=f"bound (1-d)$^k$, d={d:.2f}")

    ax.axhline(5e-16, color="gray", lw=0.6, linestyle=":")
    ax.text(ax.get_xlim()[1], 5e-16, "  fp floor", color="gray",
            va="center", fontsize=8)

    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$\|R_1^{(k)} - R_2^{(k)}\|_\infty$")
    ax.set_title("Theorem 1: geometric contraction of the opinion recursion\n"
                 f"ER({n}, {p}), coupled chains with common media noise")
    ax.set_ylim(1e-17, 5)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nFigure saved to: {FIG_PATH}")


if __name__ == "__main__":
    main()
