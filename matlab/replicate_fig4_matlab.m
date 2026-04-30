% Replicate Figure 4 with a standalone MATLAB entry point.
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
addpath(fullfile(scriptDir, "lib"));
flo_set_light_theme();

N = 1000;
P = 0.03;
SEED_GRAPH = 42;
SEED_ATTR = 43;
N_ITER = 200;
SCENARIO = "fig4";

cases = {
    "Memory", 0.5000, 0.4500, 401;
    "No-memory", 0.5263, 0.4737, 402
};

fprintf("Figure 4 MATLAB replication\n");
[Q, S] = flo_sample_attributes(N, SCENARIO, SEED_ATTR);
pos = Q > 0;
neg = Q < 0;

results = cell(2, 1);
for j = 1:2
    label = cases{j, 1};
    c = cases{j, 2};
    d = cases{j, 3};
    seedDyn = cases{j, 4};
    fprintf("  Running %s: c=%.4f, d=%.4f, iter=%d\n", label, c, d, N_ITER);
    A = flo_directed_er(N, P, c, SEED_GRAPH);
    [R, ~] = flo_run_to_stationarity(A, Q, S, c, d, SCENARIO, N_ITER, seedDyn);
    stats = flo_empirical_moments(R, Q);
    results{j} = struct("label", label, "c", c, "d", d, "R", R, "stats", stats);
    fprintf("    var=%.4f, E[R|Q=+1]=%+.4f, E[R|Q=-1]=%+.4f\n", ...
        stats.var, stats.mean_Q_plus, stats.mean_Q_minus);
end

colors = flo_colors();
bins = linspace(-1, 1, 51);
z = linspace(-1, 1, 600);
pdfPlus = flo_shifted_beta_pdf(z, 8, 1);
pdfMinus = flo_shifted_beta_pdf(z, 1, 8);

fig = figure("Units", "inches", "Position", [1, 1, 11.2, 4.3], "Color", "w");
tl = tiledlayout(fig, 1, 2, "TileSpacing", "compact", "Padding", "compact");
title(tl, "Figure 4 replication: selective exposure", "FontName", "Helvetica", "FontSize", 12, "Color", colors.dark);

for j = 1:2
    ax = nexttile(tl);
    hold(ax, "on");
    R = results{j}.R;
    flo_hist_pdf(ax, R(neg), bins, colors.blue, 0.52, "Q = -1");
    flo_hist_pdf(ax, R(pos), bins, colors.red, 0.52, "Q = +1");
    plot(ax, z, pdfMinus, "--", "Color", colors.blue, "LineWidth", 1.4, "HandleVisibility", "off");
    plot(ax, z, pdfPlus, "--", "Color", colors.red, "LineWidth", 1.4, "HandleVisibility", "off");
    xlim(ax, [-1, 1]);
    xlabel(ax, "Opinion R^*", "Interpreter", "tex");
    ylabel(ax, "Density");
    title(ax, {results{j}.label, sprintf("c=%.4f, d=%.4f", results{j}.c, results{j}.d)}, "FontSize", 10);
    stats = results{j}.stats;
    txt = sprintf("Var(R*) = %.4f\nE[R*|Q=+1] = %+.4f\nE[R*|Q=-1] = %+.4f", ...
        stats.var, stats.mean_Q_plus, stats.mean_Q_minus);
    flo_add_stats_box(ax, txt);
    legend(ax, "Location", "north", "Orientation", "horizontal", "Box", "off", "FontSize", 8);
    flo_apply_paper_style(ax);
end

outBase = fullfile(repoRoot, "figures_matlab", "fig4_replication_matlab");
flo_save_figure(fig, outBase);
