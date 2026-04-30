function [A, isBot] = flo_directed_er_with_bots(nRegular, nBots, p, c, seed)
%FLO_DIRECTED_ER_WITH_BOTS ER regular graph plus stubborn broadcasting bots.
% Regular vertices are 1:nRegular; bots are nRegular+1:n.
rng(seed, "twister");
n = nRegular + nBots;

rr = rand(nRegular, nRegular) < p;
rr(1:nRegular+1:end) = false;
br = rand(nBots, nRegular) < p;  % bot -> regular

inbound = sparse(n, n);
inbound(1:nRegular, 1:nRegular) = sparse(rr.');
inbound(1:nRegular, nRegular+1:n) = sparse(br.');

inDeg = full(sum(inbound ~= 0, 2));
rowScale = zeros(n, 1);
hasInbound = inDeg > 0;
rowScale(hasInbound) = c ./ inDeg(hasInbound);

A = spdiags(rowScale, 0, n, n) * inbound;
isBot = false(n, 1);
isBot(nRegular+1:n) = true;
end
