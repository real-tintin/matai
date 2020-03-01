function ANN = UpdateWeightBias(ANN, Alfa, Lambda, Eta, DataSize)
for Layer = ANN.Layers:-1:2
    dWold = ANN.dW{Layer};
    dWnew = ANN.D{Layer} * ANN.A{Layer-1}';
    db    = ANN.D{Layer};
    
    WUpdate = - Alfa * dWnew + ...      % gradient decent
        + Eta * dWold + ...             % momentum
        - Alfa * Lambda * ANN.W{Layer}; % weight deacy
    
    bUpdate = - Alfa * db;              % gradient decent
    
    ANN.W{Layer}  = ANN.W{Layer} + WUpdate;
    ANN.b{Layer}  = ANN.b{Layer} + bUpdate*ones(DataSize, 1);
    ANN.dW{Layer} = dWnew; % Save dW (for momentum term)
end
