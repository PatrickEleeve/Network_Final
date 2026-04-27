"""Replicate Figure 7 from Fraiman, Lin & Olvera-Cravioto (2024).

Individual opinion trajectories over time, comparing memory vs no-memory.
Memory smooths trajectories (less jagged), consistent with Proposition 6.

Same parameters as Fig 6:
    memory:    c=0.3, d=0.2  (c+d=0.5)
    no-memory: c=0.6, d=0.4  (c+d=1)

Matches the paper's Figure 7 format: two distinct vertices over 50 iterations,
one initialized at -1 and one initialized at +1.

Run:
    python3 scripts/replicate_fig7.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.attributes import sample_attributes
from src.dynamics import run_to_stationarity
from src.graph_construction import directed_er

# ── parameters ───────────────────────────────────────────────────────────────
N          = 1000
P          = 0.03
N_ITER     = 50
RECORD_EVERY = 1
SEED        = 42
SCENARIO    = "fig7"

C_MEM = 0.3
D_MEM = 0.2
_ratio = C_MEM + D_MEM
C_NOM = C_MEM / _ratio
D_NOM = 1.0 - C_NOM

FIG_PATH = Path(__file__).resolve().parents[1] / "figures" / "fig7_replication.png"


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Building ER(n={N}, p={P}), seed={SEED} ...")
    A_mem = directed_er(N, P, C_MEM, seed=SEED)
    A_nom = directed_er(N, P, C_NOM, seed=SEED)

    attrs = sample_attributes(N, SCENARIO, seed=SEED + 1)
    Q = attrs["Q"]
    S = attrs["S"]

    # Paper Figure 7 traces two vertices, initialized at -1 and +1.
    trace_ids = np.array([0, 1])
    R0 = np.random.default_rng(SEED + 2).choice(np.array([-1.0, 1.0]), size=N).astype(np.float64)
    R0[trace_ids[0]] = -1.0
    R0[trace_ids[1]] = +1.0
    print(f"Tracing vertices: {trace_ids.tolist()}")
    print(f"  R0 values: {R0[trace_ids]}")
    print(f"  Q values: {Q[trace_ids]}")

    # ── run dynamics with trajectory recording ────────────────────────────
    print(f"\nRunning {N_ITER} iterations with trajectory recording ...")
    out_mem = run_to_stationarity(
        A_mem, Q, S, c=C_MEM, d=D_MEM, scenario=SCENARIO,
        n_iter=N_ITER, seed=101, R0=R0,
        record_trajectory=True, record_every=RECORD_EVERY,
    )
    out_nom = run_to_stationarity(
        A_nom, Q, S, c=C_NOM, d=D_NOM, scenario=SCENARIO,
        n_iter=N_ITER, seed=102, R0=R0,
        record_trajectory=True, record_every=RECORD_EVERY,
    )

    traj_mem = out_mem["trajectory"]   # shape: (n_snapshots, n)
    traj_nom = out_nom["trajectory"]

    n_snapshots = traj_mem.shape[0]
    time_axis = np.arange(n_snapshots) * RECORD_EVERY
    print(f"Recorded {n_snapshots} snapshots (every {RECORD_EVERY} steps)")

    # ── compute roughness metric for each trace ───────────────────────────
    def roughness(traj: np.ndarray) -> float:
        """Average absolute step-to-step change, normalized by range."""
        step_diffs = np.abs(np.diff(traj))
        return float(np.mean(step_diffs))

    rough_mem = [roughness(traj_mem[:, i]) for i in trace_ids]
    rough_nom = [roughness(traj_nom[:, i]) for i in trace_ids]
    print(f"\nMemory roughness:    {[f'{r:.4f}' for r in rough_mem]}")
    print(f"No-memory roughness: {[f'{r:.4f}' for r in rough_nom]}")
    avg_r_mem = np.mean(rough_mem)
    avg_r_nom = np.mean(rough_nom)
    print(f"Average roughness: memory={avg_r_mem:.4f}, no-memory={avg_r_nom:.4f}, "
          f"ratio={avg_r_mem/avg_r_nom:.3f}")

    # ── plot ──────────────────────────────────────────────────────────────
    colors = ["#2471a3", "#c0392b"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, traj, c, d, label in [
        (axes[0], traj_mem, C_MEM, D_MEM, "Memory"),
        (axes[1], traj_nom, C_NOM, D_NOM, "No-memory"),
    ]:
        for idx, vertex_id in enumerate(trace_ids):
            ax.plot(time_axis, traj[:, vertex_id], color=colors[idx], lw=1.4,
                    label=rf"$R_{{{vertex_id}}}^{{(0)}}={R0[vertex_id]:+.0f}$")

        ax.axhline(y=0, color="gray", lw=0.6, linestyle="--", alpha=0.5)
        ax.set_xlim(0, N_ITER)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Iteration $k$", fontsize=11)
        ax.set_title(
            f"{label}\n$c={c},\\;d={d},\\;c+d={c+d}$",
            fontsize=10,
        )
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(r"Opinion $R_i^{(k)}$", fontsize=11)
    fig.suptitle(
        rf"Figure 7 replication — ER($n={N},\;p={P}$), "
        r"$Z\sim\mathrm{Uniform}(-1,1)$ independent of $Q$",
        fontsize=11,
    )
    fig.tight_layout()
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"\nFigure saved → {FIG_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
