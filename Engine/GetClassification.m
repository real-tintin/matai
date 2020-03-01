function [C, ErrorBin, ErrorCon] = GetClassification(ANN, X, Y)

ANN = ForwardPropagation(ANN, X, size(X, 2));

ErrorCon = GetErrorCon(ANN, Y);
[ErrorBin, C] = GetErrorBin(ANN, Y);

function ErrorCon = GetErrorCon(ANN, Y)
switch ANN.CostFunc
    case 'CrossEntropy'
        VCon     = -(Y.*log(ANN.A{end}) + (1 - Y).*log(1 - ANN.A{end}));
        ErrorCon = sum(VCon, 1);
    case 'Quadratic'
        VCon     = Y - ANN.A{end};
        ErrorCon = 0.5*(sum(VCon.^2, 1));
end

function [ErrorBin, C] = GetErrorBin(ANN, Y)
Outputs = ANN.Outputs;
C       = zeros(Outputs, size(Y, 2));
switch ANN.ClassMethod
    case 'max1'
        [~, Idx] = max(ANN.A{end}, [], 1);
        for iC = 1 : size(C, 2)
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
