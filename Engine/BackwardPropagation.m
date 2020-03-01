function ANN = BackwardPropagation(ANN, Y)
for Layer = ANN.Layers:-1:2
    if Layer == ANN.Layers
        switch ANN.CostFunc
            case 'CrossEntropy'
                ANN.D{Layer} = (ANN.A{Layer} - Y);
            case 'Quadratic'
                ANN.D{Layer} = (ANN.A{Layer} - Y) .* ...
                    Sigmoid(ANN.A{Layer}, true);
        end
    else
        ANN.D{Layer} = (ANN.W{Layer+1}' * ANN.D{Layer+1}) ...
            .* Sigmoid(ANN.A{Layer}, true);
    end
end
