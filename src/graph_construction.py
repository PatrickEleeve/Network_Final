"""Directed random graph construction.

Convention: edge (i, j) means j listens to i (j's opinion depends on i's).
We return a sparse weight matrix A of shape (n, n) with
    A[i, j] = weight vertex i places on inbound neighbor j
so that the neighbor term of the recursion is (A @ R)[i].

For equal weights with total neighbor budget c:
    A[i, j] = c / d_i^-     if (j, i) is an edge,     0 otherwise.
Row i sums to c when d_i^- > 0, and to 0 when d_i^- = 0.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def directed_er(n: int, p: float, c: float, seed: int) -> sparse.csr_matrix:
    """Directed Erdős-Rényi: for each ordered pair (i, j) with i != j,
    include edge (i, j) with probability p independently.

    Returns weight matrix A (CSR), A[i, j] = c / d_i^- if j -> i is an edge.
    """
    rng = np.random.default_rng(seed)
    # mask[j, i] = True iff edge j -> i exists (j is inbound neighbor of i)
    mask = rng.random((n, n)) < p
    np.fill_diagonal(mask, False)
    # A[i, j] = 1 iff j -> i exists  ==>  A = mask.T
    inbound = mask.T.astype(np.float64)          # inbound[i, j] = 1 iff j -> i
    in_deg = inbound.sum(axis=1)                 # d_i^-
    with np.errstate(divide="ignore", invalid="ignore"):
        row_scale = np.where(in_deg > 0, c / np.maximum(in_deg, 1), 0.0)
    A = inbound * row_scale[:, None]
    return sparse.csr_matrix(A)


def directed_er_with_bots(
    n_regular: int, n_bots: int, p: float, c: float, seed: int
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Regular vertices follow ER among themselves; bots broadcast to regulars
    with probability p each, and receive from nobody (zero in-degree).

    Returns (A, is_bot) where is_bot is a boolean array of length n_regular + n_bots.
    Vertices 0..n_regular-1 are regular; n_regular..n-1 are bots.
    """
    n = n_regular + n_bots
    rng = np.random.default_rng(seed)
    # regular-regular edges
    rr = rng.random((n_regular, n_regular)) < p
    np.fill_diagonal(rr, False)
    # bot -> regular edges (bots influence regulars)
    br = rng.random((n_bots, n_regular)) < p
    # Build full inbound[i, j] = 1 iff j -> i
    inbound = np.zeros((n, n), dtype=np.float64)
    # inbound for regulars: from other regulars AND from bots
    inbound[:n_regular, :n_regular] = rr.T.astype(np.float64)          # regular -> regular
    inbound[:n_regular, n_regular:] = br.T.astype(np.float64)          # bot -> regular
    # bots have zero in-degree by construction
    in_deg = inbound.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_scale = np.where(in_deg > 0, c / np.maximum(in_deg, 1), 0.0)
    A = inbound * row_scale[:, None]
    is_bot = np.zeros(n, dtype=bool)
    is_bot[n_regular:] = True
    return sparse.csr_matrix(A), is_bot


def _trim_stubs(degrees: np.ndarray, excess: int, d_min: int,
                rng: np.random.Generator) -> np.ndarray:
    """Randomly reduce degrees (above d_min) to remove `excess` total stubs."""
    d = degrees.copy()
    removed = 0
    while removed < excess:
        candidates = np.where(d > d_min)[0]
        if len(candidates) == 0:
            break
        idx = rng.choice(candidates, size=min(len(candidates), excess - removed),
                         replace=False)
        d[idx] -= 1
        removed += len(idx)
    return d


def directed_cm_powerlaw(
    n: int, alpha: float, d_min: int, c: float, seed: int
) -> sparse.csr_matrix:
    """Directed configuration model with power-law degree distributions.

    In-degree and out-degree are drawn i.i.d. from P(D=k) ∝ k^{-α} for k ≥ d_min.
    Multi-edges and self-loops are removed (simple graph constraint).

    Returns weight matrix A (CSR) with equal weights summing to c per row.
    """
    rng = np.random.default_rng(seed)

    # ── sample degree sequences from discrete power-law ──────────────────
    # P(D=k) ∝ k^{-α} for k ∈ [d_min, k_max]
    k_max = n - 1  # cannot exceed n-1 neighbors
    support = np.arange(d_min, k_max + 1, dtype=np.float64)
    probs = support ** (-alpha)
    probs /= probs.sum()

    # sample in-degree and out-degree independently
    d_in  = rng.choice(support, size=n, p=probs).astype(np.int64)
    d_out = rng.choice(support, size=n, p=probs).astype(np.int64)

    # ensure in/out stub counts are equal by trimming the larger side
    total_in  = d_in.sum()
    total_out = d_out.sum()
    m_target = min(total_in, total_out)
    # randomly remove excess stubs from the larger side
    if total_in > m_target:
        excess = total_in - m_target
        d_in = _trim_stubs(d_in, excess, d_min, rng)
    if total_out > m_target:
        excess = total_out - m_target
        d_out = _trim_stubs(d_out, excess, d_min, rng)

    # ── build stub lists ─────────────────────────────────────────────────
    in_stubs  = np.repeat(np.arange(n), d_in)
    out_stubs = np.repeat(np.arange(n), d_out)
    m = len(in_stubs)
    assert m == len(out_stubs), f"stub mismatch: {len(in_stubs)} vs {len(out_stubs)}"

    # shuffle and pair
    perm = rng.permutation(m)
    src = out_stubs          # source = out-stub owner
    dst = in_stubs[perm]     # destination = in-stub owner

    # ── remove self-loops & multi-edges ───────────────────────────────────
    mask = src != dst
    edges = set()
    for s, d in zip(src[mask], dst[mask]):
        if (s, d) not in edges:
            edges.add((s, d))
    edges = list(edges)

    if not edges:
        # degenerate: fall back to a sparse ER
        return directed_er(n, 0.03, c, seed + 100_000)

    s_arr = np.array([e[0] for e in edges], dtype=np.int64)
    d_arr = np.array([e[1] for e in edges], dtype=np.int64)

    # ── build A: A[i, j] = weight i places on inbound neighbor j ─────────
    # edge (s → d) means d listens to s → A[d, s] nonzero
    inbound = sparse.coo_matrix(
        (np.ones(len(s_arr), dtype=np.float64), (d_arr, s_arr)),
        shape=(n, n),
    ).tocsr()
    in_deg = np.diff(inbound.indptr)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_scale = np.where(in_deg > 0, c / np.maximum(in_deg, 1), 0.0)
    # Scale each row i by row_scale[i] via left-multiplication by diag matrix
    A = sparse.diags(row_scale, format="csr") @ inbound
    return A


def directed_sbm(
    block_sizes: list[int],
    p_matrix: np.ndarray,
    c: float,
    seed: int,
) -> sparse.csr_matrix:
    """Directed stochastic block model (SBM).

    Vertices partitioned into K blocks. Edge (j → i) exists with probability
    p_matrix[block(i), block(j)] independently for each ordered pair i ≠ j.

    Parameters
    ----------
    block_sizes: Size of each block, sum = n.
    p_matrix:    K×K matrix; p_matrix[b_i, b_j] = prob of edge from block j to block i.
    c:           Total neighbor trust budget per vertex.
    seed:        RNG seed.

    Returns weight matrix A (CSR), A[i, j] = c / d_i^- if j → i is an edge.
    """
    rng = np.random.default_rng(seed)
    n = sum(block_sizes)
    K = len(block_sizes)

    # block membership
    block_id = np.empty(n, dtype=np.int64)
    start = 0
    for b, size in enumerate(block_sizes):
        block_id[start : start + size] = b
        start += size

    # sample edges
    inbound = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        bi = block_id[i]
        for j in range(n):
            if i == j:
                continue
            bj = block_id[j]
            if rng.random() < p_matrix[bi, bj]:
                inbound[i, j] = 1.0  # j → i

    in_deg = inbound.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_scale = np.where(in_deg > 0, c / np.maximum(in_deg, 1), 0.0)
    A = inbound * row_scale[:, None]
    return sparse.csr_matrix(A)


def in_degree(A: sparse.csr_matrix) -> np.ndarray:
    """Number of inbound neighbors per vertex (nonzero count of row i)."""
    return np.diff(A.indptr)
