% Replicate Figure 1 with a standalone MATLAB entry point.
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
addpath(fullfile(scriptDir, "lib"));
flo_set_light_theme();

N = 1000;
P = 0.03;
SEED_GRAPH = 42;
SEED_ATTR = 43;
N_ITER = 500;
SCENARIO = "fig1";

C_MEM = 0.001;
D_MEM = 0.300;
total = C_MEM + D_MEM;
C_NOM = C_MEM / total;
D_NOM = D_MEM / total;

fprintf("Figure 1 MATLAB replication\n");
[Q, S] = flo_sample_attributes(N, SCENARIO, SEED_ATTR);
pos = Q > 0;
neg = Q < 0;

cases = {
    "Memory", C_MEM, D_MEM, 101;
    "No-memory", C_NOM, D_NOM, 102
};

results = cell(2, 1);
for j = 1:2
    label = cases{j, 1};
    c = cases{j, 2};
    d = cases{j, 3};
    seedDyn = cases{j, 4};
    fprintf("  Running %s: c=%.6g, d=%.6g, iter=%d\n", label, c, d, N_ITER);
    A = flo_directed_er(N, P, c, SEED_GRAPH);
    [R, ~] = flo_run_to_stationarity(A, Q, S, c, d, SCENARIO, N_ITER, seedDyn);
    stats = flo_empirical_moments(R, sign(Q));
    results{j} = struct("label", label, "c", c, "d", d, "R", R, "stats", stats);
    fprintf("    mean=%+.4f, var=%.6f\n", stats.mean, stats.var);
end

colors = flo_colors();
bins = linspace(-1, 1, 51);
z = linspace(-1, 1, 600);
mediaPdf = nan(size(z));
mediaPdf(abs(z) <= 0.03) = 1 / 0.06;

fig = figure("Units", "inches", "Position", [1, 1, 10.8, 4.2], "Color", "w");
tl = tiledlayout(fig, 1, 2, "TileSpacing", "compact", "Padding", "compact");
title(tl, "Figure 1 replication: small-variance media", "FontName", "Helvetica", "FontSize", 12, "Color", colors.dark);

for j = 1:2
    ax = nexttile(tl);
    hold(ax, "on");
    R = results{j}.R;
    flo_hist_pdf(ax, R(neg), bins, colors.blue, 0.55, "Q < 0");
    flo_hist_pdf(ax, R(pos), bins, colors.red, 0.55, "Q > 0");
    plot(ax, z, mediaPdf, "--", "Color", colors.gray, "LineWidth", 1.4, "DisplayName", "media");
    xlim(ax, [-1, 1]);
    ylim(ax, [0, 55]);
    xlabel(ax, "Opinion R_i^*", "Interpreter", "tex");
    ylabel(ax, "Density");
    title(ax, {results{j}.label, sprintf("c=%.6g, d=%.6g", results{j}.c, results{j}.d)}, "FontSize", 10);
    flo_add_stats_box(ax, sprintf("Mean = %+.4f\nVar  = %.6f", results{j}.stats.mean, results{j}.stats.var));
    legend(ax, "Location", "northeast", "Box", "off", "FontSize", 8);
    flo_apply_paper_style(ax);
end

outBase = fullfile(repoRoot, "figures_matlab", "fig1_replication_matlab");
flo_save_figure(fig, outBase);
