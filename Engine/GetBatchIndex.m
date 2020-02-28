% Machine Learning ANN: GetBatchIndex
function BatchIndex = GetBatchIndex(X, Y, BatchSize)

% Get data size
DataSize = size(X, 2);

% Check if BatchSize feasible
BatchSize = min(BatchSize, DataSize);

% Batch index
if isinf(BatchSize)
    BatchIndex = [1 DataSize];
else
    BatchIndex = [(1:BatchSize:(DataSize - BatchSize + 1))' ...
        ((1 + BatchSize):BatchSize:(DataSize + 1))' - 1];
    if BatchIndex(end, 2) < DataSize
        BatchIndex(end + 1, :) = [(BatchIndex(end, 2) + 1) DataSize];
    end
end