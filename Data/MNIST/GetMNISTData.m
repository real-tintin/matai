% Machine Learning ANN: GetMNISTData
function [X, Y] = GetMNISTData(DataType)

% Get file name part based on DataType
switch DataType
    case 'Train'
        FileNamePart = 'train';
    case 'Test'
        FileNamePart = 't10k';
end

% Get images (input)
FileName = [FileNamePart '-images-idx3-ubyte'];
FilePath = GetMNISTFilePath(FileName);
Images = LoadMNISTImages(FilePath);

% Get labels (output)
FileName = [FileNamePart '-labels-idx1-ubyte'];
FilePath = GetMNISTFilePath(FileName);
Labels = LoadMNISTLabels(FilePath);

% Set/format X and Y
X          = Images;
DataLength = length(Labels);
Y = zeros(10, DataLength);
for iData = 1:DataLength
    Y(:, iData) = (Labels(iData) == 0:9)';
end

function FilePath = GetMNISTFilePath(FileName)
DataRoot = fileparts(mfilename('fullpath'));
FilePath = fullfile(DataRoot, FileName);

if ~isfile(FilePath)
    disp([FileName ' not found locally, downloading instead...'])
    DownloadMNISTFile(FileName, FilePath);
end

function DownloadMNISTFile(FileName, FilePath)
BaseUrl = 'http://yann.lecun.com/exdb/mnist/';

Url = [BaseUrl FileName '.gz'];
websave(FilePath, Url);

gunzip([FilePath '.gz']);
delete([FilePath '.gz']);
