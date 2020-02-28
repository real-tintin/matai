% Machine Learning ANN: TrainANN
function [ANN, State] = TrainANN(ANN, Data, Settings)

% Get train settings
Alfa        = Settings.Train.Alfa;
Lambda      = Settings.Train.Lambda;
Eta         = Settings.Train.Eta;
Breakpoint  = Settings.Train.BreakPoint;
BatchSize   = Settings.Train.BatchSize;
Epochs      = Settings.Train.Epochs;

% Get train data
X = Data.TrainX;
Y = Data.TrainY;

% Print start and start timer
tic;
PrintStart(ANN, Epochs, BatchSize)

% Get segment batch index
BatchIndex = GetBatchIndex(X, Y, BatchSize);
Batches    = size(BatchIndex, 1);

% Setup State
State.ErrorBin = inf(1, Epochs);
State.ErrorCon = inf(1, Epochs);
if Settings.Plot.WBConvergence
    State.ANN = cell(1, Epochs);
end

% Initialize train parameter state
TrainParamState = 1;

% Train ANN
HandleWaitbar  = waitbar(0, 'Training progress 0 %');
for Epoch = 1:Epochs

    % Get current train parameters
    TmpAlfa   = Alfa(TrainParamState);
    TmpLambda = Lambda(TrainParamState);
    TmpEta    = Eta(TrainParamState);

    % For data for each batch
    for iBatch = 1:Batches

        % Get current batch data and size
        [XBatch YBatch BatchSize] = GetBatch(X, Y, BatchIndex(iBatch, :));

        % Forward propagation
        ANN = ForwardPropagation(ANN, XBatch, BatchSize);

        % Backwards propagation
        ANN = BackwardPropagation(ANN, YBatch);

        % Update weight and bias
        ANN = UpdateWeightBias(ANN, TmpAlfa, TmpLambda, TmpEta, BatchSize);

    end

    % Update waitbar
    Completed     = Epoch/Epochs;
    HandleWaitbar = waitbar(Completed, HandleWaitbar, ...
        ['Training progress ' num2str(Completed*100, '%.0f') ' %']);

    % Get and save output error
    [C, ErrorBin, ErrorCon] = GetClassification(ANN, X, Y);
    State.ErrorBin(Epoch)   = sum(ErrorBin) / length(ErrorBin);
    State.ErrorCon(Epoch)   = sum(ErrorCon) / length(ErrorCon);

    % Print information (per iteration)
    fprintf('       + Epoch %i\n', Epoch);
    fprintf('           Train error: %.2f %%\n', State.ErrorBin(Epoch)*100);

    % Save ANN to State
    if Settings.Plot.WBConvergence
        State.ANN{Epoch} = ANN;
    end

    % Get train parameter state depening on error
    TrainParamState = GetTrainParamState(TrainParamState, ...
        Breakpoint, State.ErrorBin(Epoch));

end

% Delete waitbar
delete(HandleWaitbar)

% Print end
TimeEnd = toc;
PrintEnd(TimeEnd);

% Get train parameter state depending on error
function TrainParamState = GetTrainParamState(TrainParamState, ...
    Breakpoint, Error)
if TrainParamState < (length(Breakpoint) + 1)
    TmpBreakpoint = Breakpoint(TrainParamState);
    if Error < TmpBreakpoint
        TrainParamState = TrainParamState + 1;
        fprintf('    Info: Breakpoint reached, parameter state updated.\n');
    end
end

% Print start
function PrintStart(ANN, Epochs, BatchSize)

% Local variables
Layers  = ANN.Layers;
Neurons = ANN.Neurons;

% Build neuron string
NeruonString = [num2str(sum(Neurons)) ' (' ...
    sprintf([repmat('%ix', 1, length(Neurons) - 1) '%i'], Neurons) ')'];

% Print
fprintf(['*** Training started ' ...
    datestr(now, 'yyyy-mm-dd HH:MM:SS') ' ***\n']);
fprintf(['    Layers        :   ' num2str(Layers) ' \n']);
fprintf(['    Neurons       :   ' NeruonString ' \n']);
fprintf(['    Batch size    :   ' num2str(BatchSize) ' \n']);
fprintf(['    Epochs        :   ' num2str(Epochs) ' \n']);

% Print end
function PrintEnd(TimeEnd)
fprintf(['    Training time  :   ' num2str(TimeEnd) ' s\n']);
fprintf(['*** Training completed ' ...
    datestr(now, 'yyyy-mm-dd HH:MM:SS') ' ***\n']);
