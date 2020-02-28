% Machine Learning ANN: SplitData
function [TrainX, TrainY, TestX, TestY] = SplitData(X, Y, TrainRatio)

% Data size
DataSize = size(X, 2);

% Random permute index
rand('twister');
Index = randperm(DataSize);

% Train and test index
SplitIndex = round(TrainRatio*DataSize);
TrainIndex = Index(1:SplitIndex);
TestIndex  = Index((SplitIndex + 1):DataSize);

% Get data
TrainX = X(:, TrainIndex);
TrainY = Y(:, TrainIndex);
TestX  = X(:, TestIndex);
TestY  = Y(:, TestIndex);