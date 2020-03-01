function ANN = ForwardPropagation(ANN, X, DataSize)
for Layer = 1:ANN.Layers
    if Layer == 1
        ANN.A{Layer} = X;
    else
        ANN.Z{Layer} = ANN.W{Layer} * ANN.A{Layer-1} + ...
            ANN.b{Layer} * ones(1, DataSize);
        ANN.A{Layer} = Sigmoid(ANN.Z{Layer}, false);
    end
end
