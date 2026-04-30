function [R, trajectory] = flo_run_to_stationarity(A, Q, S, c, d, scenario, nIter, seed, R0, recordTrajectory, recordEvery)
%FLO_RUN_TO_STATIONARITY Iterate the paper's linear stochastic recursion.
if nargin < 9 || isempty(R0)
    R0 = [];
end
if nargin < 10 || isempty(recordTrajectory)
    recordTrajectory = false;
end
if nargin < 11 || isempty(recordEvery)
    recordEvery = 1;
end

rng(seed, "twister");
n = size(A, 1);
inDeg = full(sum(A ~= 0, 2));
memory = 1 - c - d;

if isempty(R0)
    R = 2 .* (rand(n, 1) >= 0.5) - 1;
else
    R = R0(:);
end

fixed = S == 1;
if any(fixed)
    % In the paper's bot experiment, bots have zero in-degree and Q=Z, so
    % their opinions remain fixed.  Enforce this exactly in finite runs.
    R(fixed) = Q(fixed);
end

if recordTrajectory
    nSnapshots = floor(nIter / recordEvery) + 1;
    trajectory = zeros(nSnapshots, n);
    trajectory(1, :) = R.';
    snap = 2;
else
    trajectory = [];
end

for k = 1:nIter
    Z = flo_sample_media(Q, S, scenario);
    W = d .* Z;
    noInbound = inDeg == 0;
    W(noInbound) = W(noInbound) + c .* Q(noInbound);
    R = A * R + memory .* R + W;
    if any(fixed)
        R(fixed) = Q(fixed);
    end

    if recordTrajectory && mod(k, recordEvery) == 0
        trajectory(snap, :) = R.';
        snap = snap + 1;
    end
end
end
