function PlotMNISTImages(ANN, Data, Settings)

NumImage  = Settings.Plot.MNISTImages{1};
TypeImage = Settings.Plot.MNISTImages{2};

[CAll, Error] = GetClassification(ANN, Data.TestX, Data.TestY);

switch TypeImage
    case 'Incorrect'
        Index = find(Error > 0);
    case 'Correct'
        Index = find(Error == 0);
end
Index = Index(1:min(NumImage, length(Index)));

for iIndex = 1:length(Index)
    X = Data.TestX(:, Index(iIndex));
    Y = Data.TestY(:, Index(iIndex));
    C = CAll(:, Index(iIndex));
    PlotMNISTData(X, Y, C);
end
