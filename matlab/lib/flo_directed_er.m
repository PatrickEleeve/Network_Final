function A = flo_directed_er(n, p, c, seed)
%FLO_DIRECTED_ER Directed ER graph with row weights summing to c.
% Edge j -> i means vertex i listens to j, so A(i,j) is nonzero.
rng(seed, "twister");
mask = rand(n, n) < p;
mask(1:n+1:end) = false;

inbound = sparse(mask.');
inDeg = full(sum(inbound ~= 0, 2));
rowScale = zeros(n, 1);
hasInbound = inDeg > 0;
rowScale(hasInbound) = c ./ inDeg(hasInbound);

A = spdiags(rowScale, 0, n, n) * inbound;
end
