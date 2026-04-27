"""Vertex attribute sampling per figure scenario.

Attributes (per vertex):
    Q: internal opinion in [-1, +1]  (discrete {-1,+1} for most figures)
    S: stubborn flag in {0, 1}       (1 for bots)
"""
from __future__ import annotations

import numpy as np


def sample_attributes(n: int, scenario: str, seed: int, is_bot: np.ndarray | None = None) -> dict:
    """Sample Q and S vectors for a given figure scenario.

    Per paper (Sections 5.1-5.4):
        fig1, fig2, fig3 : Q ~ Uniform(-1, 1) continuous (Z independent of Q)
        fig4, fig5, fig6, fig7 : Q ~ Uniform({-1, +1}) discrete
        bots (Fig 5)     : S = 1, Q = +1, Z = +1 always
    """
    rng = np.random.default_rng(seed)
    if scenario in ("fig1", "fig2", "fig3"):
        Q = rng.uniform(-1.0, 1.0, size=n)
    else:
        Q = rng.choice(np.array([-1.0, 1.0]), size=n)
    S = np.zeros(n, dtype=np.int8)
    if is_bot is not None:
        S[is_bot] = 1
        # Bots in Fig 5 hold Q = +1 (they push one side).
        Q[is_bot] = 1.0
    return {"Q": Q, "S": S}
