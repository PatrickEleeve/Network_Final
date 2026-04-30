# MATLAB Figure Reproductions

This directory contains one MATLAB entry script per reproduced paper figure.
Each script builds its own graph, samples attributes, runs the stochastic opinion
dynamics, and exports both high-resolution PNG and vector PDF files.

## Run One Figure

From MATLAB:

```matlab
run("matlab/replicate_fig4_matlab.m")
```

Available scripts:

```text
replicate_fig1_matlab.m
replicate_fig2_matlab.m
replicate_fig3_matlab.m
replicate_fig4_matlab.m
replicate_fig5_matlab.m
replicate_fig6_matlab.m
replicate_fig7_matlab.m
```

## Run All

```matlab
run("matlab/run_all_matlab_reproductions.m")
```

Outputs are saved to `figures_matlab/` as:

```text
fig1_replication_matlab.png
fig1_replication_matlab.pdf
...
fig7_replication_matlab.png
fig7_replication_matlab.pdf
```

The PNG files are exported at 450 DPI for quick comparison. The PDF files are
vector exports intended for manuscript layout.

The scripts avoid toolbox-only beta random sampling by using the inverse CDF
for the specific Beta(8,1) and Beta(1,8) laws used in the paper.
