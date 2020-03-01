function Labels = LoadMNISTLabels(FilePath)

FileH = fopen(FilePath, 'rb');
assert(FileH ~= -1, ['Could not open ', FilePath, '']);

magic = fread(FileH, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', FilePath, '']);

nLabels = fread(FileH, 1, 'int32', 0, 'ieee-be');
Labels = fread(FileH, inf, 'uchar');

assert(size(Labels, 1) == nLabels, 'Mismatch in label count');
fclose(FileH);
