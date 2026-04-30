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
    # empirical rate from log-linear fit, skipping the initial plateau and
    # machine-precision tail where log(diffs) is numerically unstable.
    k = np.arange(len(diffs))
    floor = max(np.finfo(float).eps * max(float(diffs[0]), 1.0), 1e-14)
    mask = (k > 0) & (diffs > floor) & (diffs < diffs[0])
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


def fig4_proposition4_prediction(
    A: sparse.csr_matrix,
    Q: np.ndarray,
    c: float,
    d: float,
) -> dict:
    """Finite-graph Proposition 4/5 prediction for the Fig. 4 no-memory case.

    Proposition 4/5 assumes c + d = 1.  This helper evaluates the closed-form
    prediction using the realized graph through rho_2 = mean_i sum_j A[i,j]^2
    and the realized Q mix through the media-signal moments.
    """
    if not np.isclose(c + d, 1.0):
        raise ValueError("Proposition 4/5 prediction requires c + d = 1")

    row_sq = np.asarray(A.power(2).sum(axis=1)).ravel()
    rho2 = float(np.mean(row_sq))

    # If U ~ Beta(8,1), then Z = -1 + 2U has mean 7/9 and variance 32/810.
    z_cond_mean = np.where(Q > 0, 7.0 / 9.0, -7.0 / 9.0)
    z_cond_var = 32.0 / 810.0
    z_mean = float(np.mean(z_cond_mean))
    z_var = float(np.mean(z_cond_var + z_cond_mean**2) - z_mean**2)

    cond_var_by_row = d**2 * z_cond_var + (d**2 / (1.0 - rho2)) * row_sq * z_var
    pos = Q > 0
    neg = Q < 0

    return {
        "rho2": rho2,
        "mean_Z": z_mean,
        "var_Z": z_var,
        "var": float(d**2 * z_var + (rho2 * d**2 / (1.0 - rho2)) * z_var),
        "mean_Q_plus": float(d * (7.0 / 9.0) + c * z_mean),
        "mean_Q_minus": float(d * (-7.0 / 9.0) + c * z_mean),
        "var_Q_plus": float(np.mean(cond_var_by_row[pos])) if pos.any() else np.nan,
        "var_Q_minus": float(np.mean(cond_var_by_row[neg])) if neg.any() else np.nan,
    }
