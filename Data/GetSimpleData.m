% Machine Learning ANN: GetSimpleData
function [X, Y] = GetSimpleData(DataSize, Case)

% Random shuffle of rand
rand('twister');

% Generate data
switch Case
    case 1
        X = rand(3, DataSize) > 0.5;
        Y = X(1, :) & X(2, :);
    case 2
        X = rand(9, DataSize) > 0.5;
        Y = (X(1, :) & X(5, :) & X(9, :)) | ...
            (X(3, :) & X(5, :) & X(7, :));
    case 3
        Xdec = round(rand(1, DataSize)*7);
        Ydec = Xdec.^2;
        Xbin = dec2bin(Xdec, 4);
        Ybin = dec2bin(Ydec, 6);
        X    = zeros(4, DataSize);
        Y    = zeros(6, DataSize);
        for iData = 1:DataSize
            X(:, iData) = (Xbin(iData, :) - '0')';
            Y(:, iData) = (Ybin(iData, :) - '0')';
        end
end