% Machine Learning ANN: sigmoid
function Y = Sigmoid(X, ifDerivative)
if ifDerivative
    Y = X.*(1-X);
else
    Y = 1./(1 + exp(-X));
end