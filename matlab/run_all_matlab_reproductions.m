% Run all MATLAB figure reproductions, one script per paper figure.
close all; clc;

scriptDir = fileparts(mfilename('fullpath'));
scripts = [
    "replicate_fig1_matlab.m"
    "replicate_fig2_matlab.m"
    "replicate_fig3_matlab.m"
    "replicate_fig4_matlab.m"
    "replicate_fig5_matlab.m"
    "replicate_fig6_matlab.m"
    "replicate_fig7_matlab.m"
];

for k = 1:numel(scripts)
    fprintf("\n============================================================\n");
    fprintf("Running %s\n", scripts(k));
    fprintf("============================================================\n");
    run(fullfile(scriptDir, scripts(k)));
end
