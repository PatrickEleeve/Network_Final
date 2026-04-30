function y = flo_shifted_beta_pdf(z, a, b)
%FLO_SHIFTED_BETA_PDF PDF of -1 + 2 * Beta(a,b), no toolbox required.
u = (z + 1) ./ 2;
y = zeros(size(z));
inside = (u >= 0) & (u <= 1);
coef = gamma(a + b) ./ (gamma(a) .* gamma(b));
y(inside) = 0.5 .* coef .* u(inside).^(a - 1) .* (1 - u(inside)).^(b - 1);
end
