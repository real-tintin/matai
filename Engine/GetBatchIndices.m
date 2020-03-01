function BatchIndices = GetBatchIndices(DataSize, BatchSize)

BatchSize = min(BatchSize, DataSize);

if isinf(BatchSize)
    BatchIndices = [1 DataSize];
else
    BatchIndices = [(1:BatchSize:(DataSize - BatchSize + 1))' ...
        ((1 + BatchSize):BatchSize:(DataSize + 1))' - 1];
    if BatchIndices(end, 2) < DataSize
        BatchIndices(end + 1, :) = [(BatchIndices(end, 2) + 1) DataSize];
    end
end
