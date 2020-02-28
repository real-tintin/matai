% Machine Learning ANN: MainANN_SimpleData
function ANN = MainANN_SimpleData

%% *** Setup MATLAB ***
SetupMATLAB();

%% *** Settings for ANN ***
Settings.ANN.Neurons                = [4 4 6];        % With inputs/outputs
Settings.ANN.CostFunc               = 'CrossEntropy'; % CrossEntropy or Quadratic
Settings.ANN.ClassMethod            = 'above50%';     % max1 or above50%

%% *** Settings for traning ***
Settings.Train.Alfa                 = 0.3;   % Learning rate [0,1]
Settings.Train.Lambda               = 0.001; % Weight decay [0,1]
Settings.Train.Eta                  = 0.01;  % Momentum [0,1]
Settings.Train.BreakPoint           = 0.00;  % Breakpoint(s) to change parameters
Settings.Train.BatchSize            = 1;     % Batch size
Settings.Train.Epochs               = 30;    % Epochs (forward/backward passes)

%% *** Settings for plotting ***
Settings.Plot.ErrorConvergence      = true; % Error convergence
Settings.Plot.WBConvergence         = true; % Weight & bias convergence
Settings.Plot.NeuralNetwork         = true; % Plot neural network
Settings.Plot.MNISTImages           = {};   % Plot MNIST images

%% *** Select train and test data ***
Case                            = 3;
DataSize                        = 1e3;
TrainRatio                      = 0.5;
[X, Y]                          = GetSimpleData(DataSize, Case);
[Data.TrainX, Data.TrainY, ...
    Data.TestX, Data.TestY]     = SplitData(X, Y, TrainRatio);

%% *** Inizialize/load ANN ***
% load(fullfile(pwd, 'Networks', 'ANN_SimpleData-784-24-10.mat'), 'ANN');
ANN = InizializeANN(Settings);

%% *** Train ANN ***
[ANN, State] = TrainANN(ANN, Data, Settings);

%% *** Print train error ***
PrintClassError(ANN, Data, 'Train');

%% *** Print test error ***
PrintClassError(ANN, Data, 'Test');

%% *** Plot selected ***
PlotSelected(ANN, Data, State, Settings);

%% *** Save ANN ***
ANNSaveName = ['ANN_SimpleData' datestr(now, 'yyyymmdd_HHMMSS')];
save(fullfile(pwd, 'Networks', ANNSaveName), 'ANN');
