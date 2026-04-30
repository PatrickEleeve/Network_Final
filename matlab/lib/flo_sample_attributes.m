function [Q, S] = flo_sample_attributes(n, scenario, seed, isBot)
%FLO_SAMPLE_ATTRIBUTES Vertex attributes used by Figures 1-7.
rng(seed, "twister");

if any(strcmp(scenario, ["fig1", "fig2", "fig3"]))
    Q = -1 + 2 .* rand(n, 1);
else
    Q = 2 .* (rand(n, 1) >= 0.5) - 1;
end

S = zeros(n, 1);
if nargin >= 4 && ~isempty(isBot)
    isBot = logical(isBot(:));
    S(isBot) = 1;
    Q(isBot) = 1;
end
end
