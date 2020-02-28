% Machine Learning ANN: PlotMNISTData
function PlotHandle = PlotMNISTData(XRaw, YRaw)

% Pixels
Pixels = length(XRaw);
PixelM = sqrt(Pixels);

% Reshape and rescale pixels
X = reshape(XRaw, PixelM, PixelM);
X = 255*X;

% Get label
Y = (0:9)*YRaw;

% Colormap in grayescale
map = linspace(1, 0, 256)' * [1 1 1];

% Plot
PlotHandle = figure;
image(X);
colormap(map);
daspect([1 1 1]);
TitleString = sprintf('Plot MNIST image, %i (%ix%i) pixels, label = %i', ...
    Pixels, PixelM, PixelM, Y);
title(TitleString, 'FontSize', 12);
set(gca, 'Xtick', ((0:PixelM) + 0.5));
set(gca, 'Ytick', ((0:PixelM) + 0.5));
set(gca, 'XTickLabel', []);
set(gca, 'YTickLabel', []);
grid on;