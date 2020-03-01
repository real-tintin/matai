function ANN = InizializeANN(Settings)

Neurons     = Settings.ANN.Neurons;
CostFunc    = Settings.ANN.CostFunc;
ClassMethod = Settings.ANN.ClassMethod;

ANN.Neurons     = Neurons;
ANN.Layers      = length(Neurons);
ANN.HidenLayers = length(Neurons)-2;
ANN.Inputs      = Neurons(1);
ANN.Outputs     = Neurons(end);

ANN.CostFunc    = CostFunc;
ANN.ClassMethod = ClassMethod;

ANN.W  = cell(1, ANN.Layers);
ANN.dW = cell(1, ANN.Layers);
ANN.b  = cell(1, ANN.Layers);
ANN.Z  = cell(1, ANN.Layers);
ANN.A  = cell(1, ANN.Layers);
ANN.D  = cell(1, ANN.Layers);

for Layer = 2:ANN.Layers
    
    % Layer weight (W, dW) and bias (b)
    if Layer == 2
        ANN.W{Layer}  = randn(Neurons(Layer), ANN.Inputs);
        ANN.dW{Layer} = zeros(Neurons(Layer), ANN.Inputs);
    else
        ANN.W{Layer}  = randn(Neurons(Layer), Neurons(Layer-1));
        ANN.dW{Layer} = zeros(Neurons(Layer), Neurons(Layer-1));
    end
    ANN.b{Layer} = randn(Neurons(Layer), 1);
    
    % Layer input (Z), output/activation (A) and error (D)
    ANN.Z{Layer} = zeros(1, Neurons(Layer));
    ANN.A{Layer} = zeros(1, Neurons(Layer));
    ANN.D{Layer} = zeros(1, Neurons(Layer));
end
