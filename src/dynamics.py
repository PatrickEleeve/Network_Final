"""Opinion dynamics recursion (paper Equation 1).

R_i^(k+1) = (A @ R^(k))[i] + W_i^(k) + (1 - c - d) * R_i^(k)
         = (Delta R^(k))[i] + W_i^(k)
where Delta = A + (1 - c - d) I.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import sparse

from .graph_construction import in_degree
from .signals import external_signal, sample_media

MediaSampler = Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray]


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
    media_sampler: MediaSampler | None = None,
    coupled_tol: float | None = None,
) -> dict:
    """Iterate the recursion for n_iter steps and return the final state.

    If record_trajectory, store R at every `record_every`-th step (plus step 0).
    If media_sampler is provided, it overrides scenario-based Z sampling.

    coupled_tol uses a second chain with the same media shocks and a different
    initial condition.  Stopping when the two chains are close checks decay of
    initial-condition dependence, which is the contraction in Theorem 1.
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

    R_coupled = None
    coupled_diff = None
    if coupled_tol is not None:
        if coupled_tol <= 0:
            raise ValueError("coupled_tol must be positive")
        R_coupled = -R
        if fixed.any():
            R_coupled[fixed] = Q[fixed]
        coupled_diff = float(np.max(np.abs(R - R_coupled)))

    sampler = media_sampler
    n_iter_run = 0
    converged = False
    for k in range(n_iter):
        if sampler is not None:
            Z = sampler(Q, S, rng)
        else:
            Z = sample_media(Q, S, scenario, rng)
        W = external_signal(Q, Z, in_deg, c, d)
        R = one_step_update(R, A, W, c, d)
        if R_coupled is not None:
            R_coupled = one_step_update(R_coupled, A, W, c, d)
        if fixed.any():
            R[fixed] = Q[fixed]
            if R_coupled is not None:
                R_coupled[fixed] = Q[fixed]
        if record_trajectory and ((k + 1) % record_every == 0):
            traj.append(R.copy())
        n_iter_run = k + 1
        if R_coupled is not None:
            coupled_diff = float(np.max(np.abs(R - R_coupled)))
            if coupled_diff <= coupled_tol:
                converged = True
                break

    out: dict = {"R": R, "n_iter_run": n_iter_run}
    if record_trajectory:
        out["trajectory"] = np.array(traj)
    if coupled_tol is not None:
        out["converged"] = converged
        out["coupled_diff"] = coupled_diff
    return out
