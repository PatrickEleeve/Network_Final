function flo_save_figure(fig, outBase)
%FLO_SAVE_FIGURE Export high-resolution PNG and vector PDF.
outBase = string(outBase);
[outDir, ~, ~] = fileparts(outBase);
if ~exist(outDir, "dir")
    mkdir(outDir);
end

pngPath = outBase + ".png";
pdfPath = outBase + ".pdf";

set(fig, "Color", "w");
set(findall(fig, "Type", "axes"), "Color", "w");
set(findall(fig, "Type", "text"), "Color", [35, 39, 47] ./ 255);
set(findall(fig, "Type", "legend"), "Color", "w", "TextColor", [35, 39, 47] ./ 255);
if exist("exportgraphics", "file") == 2
    exportgraphics(fig, pngPath, "Resolution", 450, "BackgroundColor", "white");
    exportgraphics(fig, pdfPath, "ContentType", "vector", "BackgroundColor", "white");
else
    print(fig, pngPath, "-dpng", "-r450");
    print(fig, pdfPath, "-dpdf", "-painters");
end

fprintf("Saved PNG -> %s\n", pngPath);
fprintf("Saved PDF -> %s\n", pdfPath);
end
