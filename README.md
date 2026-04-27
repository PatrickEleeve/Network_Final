# Opinion Dynamics on Directed Complex Networks

Replication and variation study for Fraiman, Lin, and Olvera-Cravioto (2024),
arXiv:2209.00969v2.

The project implements the paper's linear stochastic opinion recursion on
directed graphs, reproduces Figures 1-7, and adds topology, selective-exposure,
bot, and community-structure variations.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Core Replications

```bash
python3 scripts/replicate_fig1_to_3.py
python3 scripts/replicate_fig4.py
python3 scripts/replicate_fig5.py
python3 scripts/replicate_fig6.py
python3 scripts/replicate_fig7.py
python3 scripts/validate_theorem1.py
```

## Variation Studies

```bash
python3 scripts/variation1_topology.py
python3 scripts/variation2_beta_scan.py
python3 scripts/variation3_bot_scan.py
python3 scripts/variation4_community.py
```

Generated figures are saved in `figures/`.

## Notes

- The original paper does not report random seeds, iteration counts, or ensemble
  replicates, so these scripts use fixed seeds and documented iteration counts.
- Figures are single-run replications unless otherwise noted.
- Variation 1 should be interpreted as topology/degree-distribution sensitivity;
  mean degree and low-degree mass are reported because they can confound a pure
  heavy-tail comparison.
