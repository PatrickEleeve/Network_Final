"""Media signals Z_i^(k) and external term W_i^(k).

Per paper equation (2):
    W_i^(k) = q_i * (c - sum_r c(i, r)) + d * Z_i^(k)
With equal weights: sum_r c(i, r) = c if d_i^- > 0, else 0. So:
    W_i^(k) = d * Z_i^(k)                  if d_i^- > 0
    W_i^(k) = c * q_i + d * Z_i^(k)        if d_i^- = 0

Figures differ only in how Z is sampled.
"""
from __future__ import annotations

import numpy as np


def sample_media(
    Q: np.ndarray,
    S: np.ndarray,
    scenario: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one time step of media signals Z^(k), one per vertex."""
    n = Q.shape[0]
    Z = np.empty(n, dtype=np.float64)

    if scenario == "fig1":
        Z[:] = rng.uniform(-0.03, 0.03, size=n)
    elif scenario == "fig2":
        Z[:] = rng.choice(np.array([-1.0, 1.0]), size=n)
    elif scenario == "fig3":
        # Z ~ -1 + 2*Beta(1, 8)  (shifted to [-1, 1])
        Z[:] = -1.0 + 2.0 * rng.beta(1.0, 8.0, size=n)
    elif scenario in ("fig4", "fig5"):
        # Selective exposure keyed on Q:
        #   Q = +1 -> Beta(8, 1) (skewed toward +1)
        #   Q = -1 -> Beta(1, 8) (skewed toward -1)
        plus = Q > 0
        minus = ~plus
        Z[plus] = -1.0 + 2.0 * rng.beta(8.0, 1.0, size=plus.sum())
        Z[minus] = -1.0 + 2.0 * rng.beta(1.0, 8.0, size=minus.sum())
        if scenario == "fig5":
            bot_mask = S == 1
            Z[bot_mask] = 1.0  # bots broadcast constant +1
    elif scenario in ("fig6", "fig7"):
        Z[:] = rng.uniform(-1.0, 1.0, size=n)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return Z


def external_signal(
    Q: np.ndarray,
    Z: np.ndarray,
    in_deg: np.ndarray,
    c: float,
    d: float,
) -> np.ndarray:
    """W_i = q_i * (c - sum_r c(i,r)) + d * Z_i.

    With equal weights, sum_r c(i,r) = c when in_deg > 0, else 0.
    """
    has_inbound = in_deg > 0
    W = d * Z
    # vertices with no inbound neighbors absorb the full c*q_i
    W[~has_inbound] += c * Q[~has_inbound]
    return W
