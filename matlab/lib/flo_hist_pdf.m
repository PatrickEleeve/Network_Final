function counts = flo_hist_pdf(ax, values, bins, color, alphaValue, labelText)
%FLO_HIST_PDF Draw a compact density histogram and return bin heights.
values = values(:);
if isempty(values)
    counts = zeros(1, numel(bins) - 1);
else
    counts = histcounts(values, bins, "Normalization", "pdf");
end
centers = 0.5 .* (bins(1:end-1) + bins(2:end));
bar(ax, centers, counts, 1.0, ...
    "FaceColor", color, ...
    "FaceAlpha", alphaValue, ...
    "EdgeColor", "none", ...
    "DisplayName", labelText);
end
