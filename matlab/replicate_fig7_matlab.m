% Replicate Figure 7 with a standalone MATLAB entry point.
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
addpath(fullfile(scriptDir, "lib"));
flo_set_light_theme();

N = 1000;
P = 0.03;
SEED_GRAPH = 42;
SEED_ATTR = 43;
SEED_R0 = 44;
N_ITER = 50;
RECORD_EVERY = 1;
SCENARIO = "fig7";

C_MEM = 0.3000;
D_MEM = 0.2000;
total = C_MEM + D_MEM;
C_NOM = C_MEM / total;
D_NOM = D_MEM / total;

fprintf("Figure 7 MATLAB replication\n");
[Q, S] = flo_sample_attributes(N, SCENARIO, SEED_ATTR);
rng(SEED_R0, "twister");
R0 = 2 .* (rand(N, 1) >= 0.5) - 1;
traceIds = [1, 2];
R0(traceIds(1)) = -1;
R0(traceIds(2)) = 1;

cases = {
    "Memory", C_MEM, D_MEM, 701;
    "No-memory", C_NOM, D_NOM, 702
};

results = cell(2, 1);
for j = 1:2
    label = cases{j, 1};
    c = cases{j, 2};
    d = cases{j, 3};
    seedDyn = cases{j, 4};
    fprintf("  Running %s: c=%.4f, d=%.4f, iter=%d\n", label, c, d, N_ITER);
    A = flo_directed_er(N, P, c, SEED_GRAPH);
    [R, trajectory] = flo_run_to_stationarity(A, Q, S, c, d, SCENARIO, N_ITER, seedDyn, R0, true, RECORD_EVERY);
    rough = mean(abs(diff(trajectory(:, traceIds))), 1);
    results{j} = struct("label", label, "c", c, "d", d, "R", R, "trajectory", trajectory, "rough", rough);
    fprintf("    roughness trace 1=%.4f, trace 2=%.4f\n", rough(1), rough(2));
end

colors = flo_colors();
lineColors = [colors.blue; colors.red];
timeAxis = 0:RECORD_EVERY:N_ITER;

fig = figure("Units", "inches", "Position", [1, 1, 11.5, 4.3], "Color", "w");
tl = tiledlayout(fig, 1, 2, "TileSpacing", "compact", "Padding", "compact");
title(tl, "Figure 7 replication: memory smooths individual trajectories", ...
    "FontName", "Helvetica", "FontSize", 12, "Color", colors.dark);

for j = 1:2
    ax = nexttile(tl);
    hold(ax, "on");
    trajectory = results{j}.trajectory;
    for t = 1:numel(traceIds)
        id = traceIds(t);
        plot(ax, timeAxis, trajectory(:, id), ...
            "Color", lineColors(t, :), ...
            "LineWidth", 1.45, ...
            "DisplayName", sprintf("vertex %d, R0=%+.0f", id, R0(id)));
    end
    yline(ax, 0, "--", "Color", colors.gray, "LineWidth", 0.8, "HandleVisibility", "off");
    xlim(ax, [0, N_ITER]);
    ylim(ax, [-1, 1]);
    xlabel(ax, "Iteration k");
    ylabel(ax, "Opinion R_i^{(k)}", "Interpreter", "tex");
    title(ax, {results{j}.label, sprintf("c=%.4f, d=%.4f, c+d=%.1f", ...
        results{j}.c, results{j}.d, results{j}.c + results{j}.d)}, "FontSize", 10);
    flo_add_stats_box(ax, sprintf("Mean abs step\ntrace 1 = %.4f\ntrace 2 = %.4f", ...
        results{j}.rough(1), results{j}.rough(2)), 0.04, 0.94);
    legend(ax, "Location", "northeast", "Box", "off", "FontSize", 8);
    flo_apply_paper_style(ax);
end

outBase = fullfile(repoRoot, "figures_matlab", "fig7_replication_matlab");
flo_save_figure(fig, outBase);
