function Z = flo_sample_media(Q, S, scenario)
%FLO_SAMPLE_MEDIA Draw one time step of media signals Z_i^(k).
% Beta(8,1) and Beta(1,8) are sampled by inverse CDF to avoid toolboxes.
n = numel(Q);
Z = zeros(n, 1);

switch scenario
    case "fig1"
        Z = -0.03 + 0.06 .* rand(n, 1);
    case "fig2"
        Z = 2 .* (rand(n, 1) >= 0.5) - 1;
    case "fig3"
        Z = -1 + 2 .* beta_1_8(n);
    case {"fig4", "fig5"}
        plus = Q > 0;
        minus = ~plus;
        Z(plus) = -1 + 2 .* beta_8_1(nnz(plus));
        Z(minus) = -1 + 2 .* beta_1_8(nnz(minus));
        if scenario == "fig5"
            Z(S == 1) = 1;
        end
    case {"fig6", "fig7"}
        Z = -1 + 2 .* rand(n, 1);
    otherwise
        error("Unknown scenario: %s", scenario);
end
end

function x = beta_8_1(n)
u = rand(n, 1);
x = u .^ (1 / 8);
end

function x = beta_1_8(n)
u = rand(n, 1);
x = 1 - u .^ (1 / 8);
end
