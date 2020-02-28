function PlotSelected(ANN, Data, State, Settings)
if Settings.Plot.ErrorConvergence
    PlotErrorConvergence(State, Settings);
end

if Settings.Plot.WBConvergence
    PlotWBConvergence(State, Settings);
end

if Settings.Plot.NeuralNetwork
    PlotNeuralNetwork(ANN, Settings);
end

if ~isempty(Settings.Plot.MNISTImages)
    PlotMNISTImages(ANN, Data, Settings);
end
