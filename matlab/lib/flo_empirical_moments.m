function stats = flo_empirical_moments(R, Q)
%FLO_EMPIRICAL_MOMENTS Population-style sample moments, matching numpy var.
R = R(:);
stats.mean = mean(R);
stats.var = var(R, 1);

if nargin >= 2 && ~isempty(Q)
    Q = Q(:);
    pos = Q > 0;
    neg = Q < 0;
    stats.mean_Q_plus = mean_or_nan(R(pos));
    stats.mean_Q_minus = mean_or_nan(R(neg));
    stats.var_Q_plus = var_or_nan(R(pos));
    stats.var_Q_minus = var_or_nan(R(neg));
end
end

function y = mean_or_nan(x)
if isempty(x)
    y = NaN;
else
    y = mean(x);
end
end

function y = var_or_nan(x)
if isempty(x)
    y = NaN;
else
    y = var(x, 1);
end
end
