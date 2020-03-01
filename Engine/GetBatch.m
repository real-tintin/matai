function [XBatch, YBatch, BatchSize] = GetBatch(X, Y, BatchIndex)
DataIndex = BatchIndex(1):BatchIndex(2);

XBatch = X(:, DataIndex);
YBatch = Y(:, DataIndex);

BatchSize = BatchIndex(2) - BatchIndex(1) + 1;
