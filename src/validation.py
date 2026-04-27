"""Numerical checks for the main theorems and propositions.

validate_theorem1_convergence:
    Run two coupled chains from different initial conditions under identical
    media signal realizations. Under Theorem 1, their difference evolves
    deterministically as (R1 - R2)^(k+1) = Delta (R1 - R2)^(k), so
        ||R1 - R2||_inf  <=  (1 - d)^k  ||R1 - R2||^(0).
    We record ||R1 - R2||_inf over time and confirm the geometric rate.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .graph_construction import in_degree
from .signals import external_signal, sample_media


def validate_theorem1_convergence(
    A: sparse.csr_matrix,
    Q: np.ndarray,
    S: np.ndarray,
    c: float,
    d: float,
    scenario: str,
    n_iter: int,
    seed: int,
) -> dict:
    """Return an array of ||R1_k - R2_k||_inf for k = 0, ..., n_iter.

    Uses *common random noise*: both chains see the same W^(k), so the
    difference obeys the deterministic contraction
        (R1 - R2)^(k+1) = Delta (R1 - R2)^(k).
    """
    n = A.shape[0]
    in_deg = in_degree(A)
    memory = 1.0 - c - d

    rng_init = np.random.default_rng(seed)
    R1 = rng_init.choice(np.array([-1.0, 1.0]), size=n).astype(np.float64)
    R2 = rng_init.choice(np.array([-1.0, 1.0]), size=n).astype(np.float64)

    # One shared RNG for signals -> both chains see the same W^(k)
    rng_signals = np.random.default_rng(seed + 10_000)

    diffs = [np.max(np.abs(R1 - R2))]
    for _ in range(n_iter):
        Z = sample_media(Q, S, scenario, rng_signals)
        W = external_signal(Q, Z, in_deg, c, d)
        R1 = A @ R1 + memory * R1 + W
        R2 = A @ R2 + memory * R2 + W
        diffs.append(np.max(np.abs(R1 - R2)))

    diffs = np.array(diffs)
    # empirical rate from log-linear fit, skipping k=0
    k = np.arange(len(diffs))
    mask = diffs > 0
    if mask.sum() >= 2:
        slope = np.polyfit(k[mask], np.log(diffs[mask]), 1)[0]
        empirical_rate = float(np.exp(slope))
    else:
        empirical_rate = 0.0

    return {
        "diffs": diffs,
        "empirical_rate": empirical_rate,
        "theoretical_bound": 1.0 - d,
    }


def empirical_moments(R: np.ndarray, Q: np.ndarray | None = None) -> dict:
    """Basic sample statistics; optional conditioning on Q in {-1, +1}."""
    out = {
        "mean": float(np.mean(R)),
        "var": float(np.var(R)),
    }
    if Q is not None:
        pos = Q > 0
        neg = Q < 0
        out.update(
            {
                "mean_Q_plus":  float(np.mean(R[pos]))  if pos.any() else np.nan,
                "mean_Q_minus": float(np.mean(R[neg]))  if neg.any() else np.nan,
                "var_Q_plus":   float(np.var(R[pos]))   if pos.any() else np.nan,
                "var_Q_minus":  float(np.var(R[neg]))   if neg.any() else np.nan,
            }
        )
    return out
