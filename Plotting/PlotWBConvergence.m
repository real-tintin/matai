function PlotWBConvergence(State, Settings)
Epochs = Settings.Train.Epochs;
Epoch  = 1:Epochs;

Layers = State.ANN{1}.Layers;
for Layer = 2:Layers
    figure;
    annotation('textbox', [0 0.9 1 0.1], 'String', ['Weight and bias '...
        'convergence for layer l_{' num2str(Layer) '}'], 'EdgeColor', ...
        'none', 'HorizontalAlignment', 'center', 'FontSize', 11);
    
    Nodes    = size(State.ANN{1}.W{Layer}, 1);
    Weights  = size(State.ANN{1}.W{Layer}, 2);
    SubPlotM = ceil(sqrt(Nodes));
    
    for Node = 1:Nodes
        W = inf(Epochs, Weights);
        b = inf(1, Epochs);
        for iEpoch = Epoch
            ANN          = State.ANN{iEpoch};
            W(iEpoch, :) = ANN.W{Layer}(Node, :);
            b(iEpoch)    = ANN.b{Layer}(Node);
        end
        
        subplot(SubPlotM, SubPlotM, Node);
        line(Epoch, W, 'LineStyle', '-', 'LineWidth', 2);
        line(Epoch, b, 'LineStyle', '--', 'LineWidth', 2);
        title(['Node n_{' num2str(Node) '}'], 'FontSize', 9);
        xlabel('Epoch', 'FontSize', 8);
        ylabel('W (-) b (--)', 'FontSize', 8);
        grid on;
    end
end
