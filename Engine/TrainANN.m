function [ANN, State] = TrainANN(ANN, Data, Settings)
Alfas       = Settings.Train.Alfas;
Lambdas     = Settings.Train.Lambdas;
Etas        = Settings.Train.Etas;
Breakpoints = Settings.Train.BreakPoints;
BatchSize   = Settings.Train.BatchSize;
Epochs      = Settings.Train.Epochs;

X = Data.TrainX;
Y = Data.TrainY;

VerifyIOLayerSize(ANN, X, Y);

BatchIndices = GetBatchIndices(size(X, 2), BatchSize);
Batches = size(BatchIndices, 1);

TrainParamState = 1;
State = InitState(Epochs);
HandleWaitbar = waitbar(0, 'Training progress 0 %');

tic; PrintStart(ANN, Epochs, BatchSize);

for iEpoch = 1:Epochs
    Alfa   = Alfas(TrainParamState);
    Lambda = Lambdas(TrainParamState);
    Eta    = Etas(TrainParamState);
    
    for iBatch = 1:Batches
        [XBatch, YBatch, BatchSize] = GetBatch(X, Y, BatchIndices(iBatch, :));
        
        ANN = ForwardPropagation(ANN, XBatch, BatchSize);
        ANN = BackwardPropagation(ANN, YBatch);
        ANN = UpdateWeightBias(ANN, Alfa, Lambda, Eta, BatchSize);
    end
    
    [~, ErrorBin, ErrorCon] = GetClassification(ANN, X, Y);
    State.ErrorBin(iEpoch)  = sum(ErrorBin) / length(ErrorBin);
    State.ErrorCon(iEpoch)  = sum(ErrorCon) / length(ErrorCon);
    State.ANN{iEpoch}       = ANN;
    
    PrintEpoch(iEpoch, State.ErrorBin(iEpoch) * 100);
    UpdateWaitBar(HandleWaitbar, iEpoch, Epochs);
    
    TrainParamState = GetTrainParamState(TrainParamState, ...
        Breakpoints, State.ErrorBin(iEpoch));
end

delete(HandleWaitbar);
PrintEnd(toc);

function State = InitState(Epochs)
State.ErrorBin = inf(1, Epochs);
State.ErrorCon = inf(1, Epochs);
State.ANN = cell(1, Epochs);

function VerifyIOLayerSize(ANN, X, Y)
assert(ANN.Neurons(1) == size(X, 1), ...
    ['First layer should contain ' num2str(ANN.Neurons(1)) ' neurons'])

assert(ANN.Neurons(end) == size(Y, 1), ...
    ['Last layer should contain ' num2str(ANN.Neurons(end)) ' neurons'])

function TrainParamState = GetTrainParamState(TrainParamState, ...
    Breakpoints, Error)
if TrainParamState < (length(Breakpoints) + 1)
    Breakpoint = Breakpoints(TrainParamState);
    if Error < Breakpoint
        TrainParamState = TrainParamState + 1;
        fprintf('    Info: Breakpoint reached, parameter state updated.\n');
    end
end

function PrintStart(ANN, Epochs, BatchSize)
NeruonString = [num2str(sum(ANN.Neurons)) ' (' ...
    sprintf([repmat('%ix', 1, length(ANN.Neurons) - 1) '%i'], ...
    ANN.Neurons) ')'];

fprintf(['*** Training started ' ...
    datestr(now, 'yyyy-mm-dd HH:MM:SS') ' ***\n']);
fprintf(['    Layers        :   ' num2str(ANN.Layers) ' \n']);
fprintf(['    Neurons       :   ' NeruonString ' \n']);
fprintf(['    Batch size    :   ' num2str(BatchSize) ' \n']);
fprintf(['    Epochs        :   ' num2str(Epochs) ' \n']);

function UpdateWaitBar(HandleWaitbar, iEpoch, Epochs)
Completed = iEpoch / Epochs;
waitbar(Completed, HandleWaitbar, ...
    ['Training progress ' num2str(Completed * 100, '%.0f') ' %']);

function PrintEpoch(iEpoch, ErrorBin)
fprintf('       + Epoch %i\n', iEpoch);
fprintf('           Train error: %.2f %%\n', ErrorBin);

function PrintEnd(TimeEnd)
fprintf(['    Training time  :   ' num2str(TimeEnd) ' s\n']);
fprintf(['*** Training completed ' ...
    datestr(now, 'yyyy-mm-dd HH:MM:SS') ' ***\n']);
