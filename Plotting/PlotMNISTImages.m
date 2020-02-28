function Handle = PlotMNISTImages(ANN, Data, Settings)

% Settings
NumImage  = Settings.Plot.MNISTImages{1};
TypeImage = Settings.Plot.MNISTImages{2};

% Get test data
X = Data.TestX;
Y = Data.TestY;

% Get classification error
[~, Error] = GetClassification(ANN, X, Y);

% Select images
switch TypeImage
    case 'Incorrect'
        Index = find(Error > 0);
    case 'Correct'
        Index = find(Error == 0);
end
Index = Index(1:min(NumImage, length(Index)));

% Plot images
for iIndex = 1:length(Index)
    XTmp = X(:, Index(iIndex));
    YTmp = Y(:, Index(iIndex));
    PlotMNISTData(XTmp, YTmp);
end
