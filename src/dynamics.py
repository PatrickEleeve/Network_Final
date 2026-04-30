"""Opinion dynamics recursion (paper Equation 1).

R_i^(k+1) = (A @ R^(k))[i] + W_i^(k) + (1 - c - d) * R_i^(k)
         = (Delta R^(k))[i] + W_i^(k)
where Delta = A + (1 - c - d) I.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .graph_construction import in_degree
from .signals import external_signal, sample_media


def one_step_update(
    R: np.ndarray,
    A: sparse.csr_matrix,
    W: np.ndarray,
    c: float,
    d: float,
) -> np.ndarray:
    """One time step: R_{k+1} = A R_k + (1-c-d) R_k + W_k."""
    memory = 1.0 - c - d
    return A @ R + memory * R + W


def run_to_stationarity(
    A: sparse.csr_matrix,
    Q: np.ndarray,
    S: np.ndarray,
    c: float,
    d: float,
    scenario: str,
    n_iter: int,
    seed: int,
    R0: np.ndarray | None = None,
    record_trajectory: bool = False,
    record_every: int = 1,
) -> dict:
    """Iterate the recursion for n_iter steps and return the final state.

    If record_trajectory, store R at every `record_every`-th step (plus step 0).
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    in_deg = in_degree(A)

    if R0 is None:
        # Paper p.12: "Everyone's initial opinion R_i^(0) is chosen to be
        # uniform on {-1, 1}." Stationary is independent of R0 by Theorem 1.
        R = rng.choice(np.array([-1.0, 1.0]), size=n).astype(np.float64)
    else:
        R = R0.astype(np.float64).copy()

    fixed = S == 1
    if fixed.any():
        # In the paper's bot experiment, bots have zero in-degree, Q=Z=+1,
        # and therefore never change opinion.  Initialize and keep them fixed
        # so the finite-time simulation matches that construction exactly.
        R[fixed] = Q[fixed]

    traj = None
    if record_trajectory:
        traj = [R.copy()]

    for k in range(n_iter):
        Z = sample_media(Q, S, scenario, rng)
        W = external_signal(Q, Z, in_deg, c, d)
        R = one_step_update(R, A, W, c, d)
        if fixed.any():
            R[fixed] = Q[fixed]
        if record_trajectory and ((k + 1) % record_every == 0):
            traj.append(R.copy())

    out: dict = {"R": R}
    if record_trajectory:
        out["trajectory"] = np.array(traj)
    return out
