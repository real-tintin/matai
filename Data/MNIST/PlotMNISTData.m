function PlotMNISTData(XRaw, YRaw, C)

Pixels = length(XRaw);
PixelM = sqrt(Pixels);

X = reshape(XRaw, PixelM, PixelM);
X = 255 * X;

Label = (0:9) * YRaw;
Clas = (0:9) * C;

GrayCMap = linspace(1, 0, 256)' * [1 1 1];

figure;
image(X);
colormap(GrayCMap);
daspect([1 1 1]);

TitleString = sprintf('MNIST %i (%ix%i) pixels, label = %i, clas = %i', ...
    Pixels, PixelM, PixelM, Label, Clas);
title(TitleString, 'FontSize', 12);

set(gca, 'Xtick', ((0:PixelM) + 0.5));
set(gca, 'Ytick', ((0:PixelM) + 0.5));
set(gca, 'XTickLabel', []);
set(gca, 'YTickLabel', []);
grid on;
