function [TrainX, TrainY, TestX, TestY] = SplitData(X, Y, TrainRatio)
DataSize = size(X, 2);

rng('shuffle');
Index = randperm(DataSize);

SplitIndex = round(TrainRatio * DataSize);
TrainIndex = Index(1:SplitIndex);
TestIndex  = Index((SplitIndex + 1):DataSize);

TrainX = X(:, TrainIndex);
TrainY = Y(:, TrainIndex);
TestX  = X(:, TestIndex);
TestY  = Y(:, TestIndex);
