% Machine Learning ANN: GetClassification
function [C, ErrorBin, ErrorCon] = GetClassification(ANN, X, Y)

% Data size
DataSize = size(X, 2);

% Forward propagation
ANN = ForwardPropagation(ANN, X, DataSize);

% Continues output error
switch ANN.CostFunc
    case 'CrossEntropy'
        VCon     = -(Y.*log(ANN.A{end}) + (1 - Y).*log(1 - ANN.A{end}));
        ErrorCon = sum(VCon, 1);
    case 'Quadratic'
        VCon     = Y - ANN.A{end};
        ErrorCon = 0.5*(sum(VCon.^2, 1));
end

% Classification and binary output error
Outputs = ANN.Outputs;
C       = zeros(Outputs, DataSize);
switch ANN.ClassMethod
    case 'max1'
        [Val Idx] = max(ANN.A{end}, [], 1);
        for iC = 1:DataSize
            C(Idx(iC), iC) = 1;
        end
        VBin = Y - C;
    case 'above50%'
        C    = (ANN.A{end} > 0.5);
        VBin = Y - C;
    otherwise
        C    = (ANN.A{end} > 0.5);
        VBin = Y - C;
end
ErrorBin = 0.5*(sum(VBin.^2, 1));