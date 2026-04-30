% Replicate Figure 5 with a standalone MATLAB entry point.
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
addpath(fullfile(scriptDir, "lib"));
flo_set_light_theme();

N_REGULAR = 800;
N_BOTS = 200;
N = N_REGULAR + N_BOTS;
P = 0.03;
SEED_GRAPH = 42;
SEED_ATTR = 43;
N_ITER = 200;
SCENARIO = "fig5";

C_MEM = 0.5000;
D_MEM = 0.4500;
total = C_MEM + D_MEM;
C_NOM = C_MEM / total;
D_NOM = D_MEM / total;

fprintf("Figure 5 MATLAB replication\n");
[~, isBot] = flo_directed_er_with_bots(N_REGULAR, N_BOTS, P, C_MEM, SEED_GRAPH);
[Q, S] = flo_sample_attributes(N, SCENARIO, SEED_ATTR, isBot);
regular = ~isBot;
regPos = regular & (Q > 0);
regNeg = regular & (Q < 0);

cases = {
    "Memory", C_MEM, D_MEM, 501;
    "No-memory", C_NOM, D_NOM, 502
};

results = cell(2, 1);
for j = 1:2
    label = cases{j, 1};
    c = cases{j, 2};
    d = cases{j, 3};
    seedDyn = cases{j, 4};
    fprintf("  Running %s: c=%.4f, d=%.4f, iter=%d\n", label, c, d, N_ITER);
    [A, isBotCase] = flo_directed_er_with_bots(N_REGULAR, N_BOTS, P, c, SEED_GRAPH);
    [R, ~] = flo_run_to_stationarity(A, Q, S, c, d, SCENARIO, N_ITER, seedDyn);
    statsAll = flo_empirical_moments(R, Q);
    statsReg = flo_empirical_moments(R(regular), Q(regular));
    results{j} = struct("label", label, "c", c, "d", d, "R", R, ...
        "statsAll", statsAll, "statsReg", statsReg, "isBot", isBotCase);
    fprintf("    regular mean=%+.4f, regular var=%.4f, bot mean=%+.4f\n", ...
        mean(R(regular)), var(R(regular), 1), mean(R(isBotCase)));
end

colors = flo_colors();
bins = linspace(-1, 1, 51);
z = linspace(-1, 1, 600);
pdfPlus = flo_shifted_beta_pdf(z, 8, 1);
pdfMinus = flo_shifted_beta_pdf(z, 1, 8);

fig = figure("Units", "inches", "Position", [1, 1, 11.8, 4.5], "Color", "w");
tl = tiledlayout(fig, 1, 2, "TileSpacing", "compact", "Padding", "compact");
title(tl, "Figure 5 replication: stubborn +1 bots shift the distribution", ...
    "FontName", "Helvetica", "FontSize", 12, "Color", colors.dark);

ymax = 0;
axesList = gobjects(2, 1);
for j = 1:2
    ax = nexttile(tl);
    axesList(j) = ax;
    hold(ax, "on");
    R = results{j}.R;
    c1 = flo_hist_pdf(ax, R(regNeg), bins, colors.blue, 0.50, "regular Q = -1");
    c2 = flo_hist_pdf(ax, R(regPos), bins, colors.red, 0.50, "regular Q = +1");
    plot(ax, z, pdfMinus, "--", "Color", colors.blue, "LineWidth", 1.2, "HandleVisibility", "off");
    plot(ax, z, pdfPlus, "--", "Color", colors.red, "LineWidth", 1.2, "HandleVisibility", "off");
    xline(ax, 0.995, "-", "Color", colors.purple, "LineWidth", 2.0, "DisplayName", "bots at +1");
    ymax = max([ymax, c1, c2, pdfMinus, pdfPlus]);

    xlim(ax, [-1, 1]);
    xlabel(ax, "Opinion R_i^*", "Interpreter", "tex");
    ylabel(ax, "Density");
    title(ax, {results{j}.label, sprintf("c=%.4f, d=%.4f", results{j}.c, results{j}.d)}, "FontSize", 10);
    statsReg = results{j}.statsReg;
    statsAll = results{j}.statsAll;
    txt = sprintf("Mean all = %+.4f\nVar all  = %.4f\nMean regular = %+.4f\nVar regular  = %.4f", ...
        statsAll.mean, statsAll.var, statsReg.mean, statsReg.var);
    flo_add_stats_box(ax, txt);
    legend(ax, "Location", "northeast", "Box", "off", "FontSize", 7.5);
    flo_apply_paper_style(ax);
end

for j = 1:2
    ylim(axesList(j), [0, ymax * 1.18]);
end

outBase = fullfile(repoRoot, "figures_matlab", "fig5_replication_matlab");
flo_save_figure(fig, outBase);
