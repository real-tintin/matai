% Machine Learning ANN: PermuteData
function [X, Y] = PermuteData(X, Y)

% Data size
DataSize = size(X, 2);

% Random permute index
rand('twister');
Index = randperm(DataSize);

% Permute data
X = X(:, Index);
Y = Y(:, Index);