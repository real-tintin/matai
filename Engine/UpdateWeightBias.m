% Machine Learning ANN: UpdateWeightBias
function ANN = UpdateWeightBias(ANN, Alfa, Lambda, Eta, DataSize)
for Layer = ANN.Layers:-1:2

    % Derivatives
    dWold = ANN.dW{Layer};
    dWnew = ANN.D{Layer}*ANN.A{Layer-1}';
    db    = ANN.D{Layer};

    % Updates W
    WUpdate = - Alfa*dWnew + ...     % gradient decent
        + Eta*dWold + ...            % momentum
        - Alfa*Lambda*ANN.W{Layer};  % weight deacy

    % Update b
    bUpdate = - Alfa*db; % gradient decent

    % Update W and b
    ANN.W{Layer}  = ANN.W{Layer} + WUpdate;
    ANN.b{Layer}  = ANN.b{Layer} + bUpdate*ones(DataSize, 1);

    % Save dW (for momentum term)
    ANN.dW{Layer} = dWnew;

end
end