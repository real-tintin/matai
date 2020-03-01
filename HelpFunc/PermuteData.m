function [X, Y] = PermuteData(X, Y)
DataSize = size(X, 2);

rng('shuffle');
Index = randperm(DataSize);

X = X(:, Index);
Y = Y(:, Index);
