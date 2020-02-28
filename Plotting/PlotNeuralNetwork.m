% Machine Learning ANN: PlotNeuralNetwork
function Handle = PlotNeuralNetwork(ANN, Settings)

% Settings
LayerMargin  = 1;
NeuronMargin = 1;

% Get some data
Layers  = ANN.Layers;
Neurons = ANN.Neurons;

% Compute position of each neuron
NeuronPos = cell(1, Layers);
for Layer = 1:Layers
    NeuronPos{Layer} = zeros(Neurons(Layer), 2);
    for Neuron = 1:Neurons(Layer)
        NeuronXPos = Layer;
        NeuronYPos = (Neurons(Layer) - Neuron) - (Neurons(Layer) - 1)/2;
        NeuronPos{Layer}(Neuron, :) = [LayerMargin*NeuronXPos ...
            NeuronMargin*NeuronYPos];
    end
end

% Plot axis dimensions
XAxisDim = [1.5 LayerMargin*Layers + 0.5];
YAxisDim = [-0.5 0.5]*NeuronMargin*max(Neurons);

% Create and setup figure
Handle     = figure;
AxisHandle = gca;
xlim(XAxisDim);
ylim(YAxisDim);
daspect([1 1 1]);
grid on;

% Create template neuron
Rad          = 0.15;
Phi          = 0:0.01:2*pi;
NeuronX      = Rad*cos(Phi);
NeuronY      = Rad*sin(Phi);
FaceColor    = 0.75*[1 1 1];
NeuronHandle = patch(NeuronX, NeuronY, FaceColor);
EdgeColor    = [0 0 0];
LineWidth    = 2;
set(NeuronHandle, ...
    'EdgeColor', EdgeColor, ...
    'LineWidth', LineWidth);

% Plot neurons
for Layer = 1:Layers
    for Neuron = 1:Neurons(Layer)

        % Get position
        NeuronXPos = NeuronPos{Layer}(Neuron, 1);
        NeuronYPos = NeuronPos{Layer}(Neuron, 2);

        % Copy graphical neruon (if not first)
        if Layer == 1 && Neuron == 1
            NewNeuronHandle = NeuronHandle;
        else
            NewNeuronHandle = copyobj(NeuronHandle, AxisHandle);
        end

        % Set positon
        set(NewNeuronHandle, ...
            'XData', NeuronX + NeuronXPos, ...
            'YData', NeuronY + NeuronYPos);

    end
end

% Create edge template
EdgeX      = [0 1];
EdgeY      = [0 0];
EdgeHandle = line(EdgeX, EdgeY);
EdgeColor  = [0 0 0];
LineWidth  = 2;
set(EdgeHandle, ...
    'Color', EdgeColor, ...
    'LineWidth', LineWidth);

% Plot edges
for Layer = 2:Layers
    for Neuron2 = 1:Neurons(Layer)
        for Neuron1 = 1:Neurons(Layer - 1)

            % Get neuron positions
            NeuronXPos1  = NeuronPos{Layer - 1}(Neuron1, 1);
            NeuronXPos2  = NeuronPos{Layer}(Neuron2, 1);
            NeuronYPos1  = NeuronPos{Layer - 1}(Neuron1, 2);
            NeuronYPos2  = NeuronPos{Layer}(Neuron2, 2);

            % Get angle
            X = (NeuronXPos2 - NeuronXPos1);
            Y = (NeuronYPos2 - NeuronYPos1);
            Theta = atan(Y/X);

            % Compute edge position
            EdgeXPos1 = NeuronXPos1 + Rad*cos(Theta);
            EdgeXPos2 = NeuronXPos2 - Rad*cos(Theta);
            EdgeYPos1 = NeuronYPos1 + Rad*sin(Theta);
            EdgeYPos2 = NeuronYPos2 - Rad*sin(Theta);

            % Copy graphical edge (if not first)
            if Layer == 2 && Neuron1 == 1 && Neuron2 == 1
                NewEdgeHandle = EdgeHandle;
            else
                NewEdgeHandle = copyobj(EdgeHandle, AxisHandle);
            end

            % Determine edge color using weights and bias
            W          = ANN.W{Layer}(Neuron2, :);
            b          = ANN.b{Layer}(Neuron2, :);
            Z          = W + b; % assume normalized positive input
            Z(Z < 0)   = 0;
            normZ      = max(max(Z), eps);
            ColorScale = Z(Neuron1) / normZ;
            EdgeColor  = (1 - ColorScale)*[1 1 1];

            % Set positon
            set(NewEdgeHandle, ...
                'XData', [EdgeXPos1 EdgeXPos2], ...
                'YData', [EdgeYPos1 EdgeYPos2], ...
                'Color', EdgeColor);

        end
    end
end

title('Neurons & Edges (strength: white -> black)');
xlabel('Layer [-]');
set(gca, 'XLim', [0, ANN.Layers + 1]);
set(gca, 'XTick', 0:(ANN.Layers + 1))
set(gca, 'YTick', []);
