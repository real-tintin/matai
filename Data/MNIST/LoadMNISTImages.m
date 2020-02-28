% Machine Learning ANN: LoadMNISTImages
function Images = LoadMNISTImages(FilePath)

FileH = fopen(FilePath, 'rb');
assert(FileH ~= -1, ['Could not open ', FilePath, '']);

Magic = fread(FileH, 1, 'int32', 0, 'ieee-be');
assert(Magic == 2051, ['Bad magic number in ', FilePath, '']);

numImages = fread(FileH, 1, 'int32', 0, 'ieee-be');
numRows = fread(FileH, 1, 'int32', 0, 'ieee-be');
numCols = fread(FileH, 1, 'int32', 0, 'ieee-be');

Images = fread(FileH, inf, 'uchar');
Images = reshape(Images, numCols, numRows, numImages);
Images = permute(Images,[2 1 3]);

fclose(FileH);

% Reshape to and convert to double and rescale to [0,1]
Images = reshape(Images, size(Images, 1) * size(Images, 2), size(Images, 3));
Images = double(Images) / 255;