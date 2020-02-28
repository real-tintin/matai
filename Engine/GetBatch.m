% Machine Learning ANN: GetBatch
function [XBatch YBatch BatchSize] = GetBatch(X, Y, BatchIndex)

% Get data index
DataIndex = BatchIndex(1):BatchIndex(2);

% Get data
XBatch = X(:, DataIndex);
YBatch = Y(:, DataIndex);

% Batch size
BatchSize = BatchIndex(2) - BatchIndex(1) + 1;