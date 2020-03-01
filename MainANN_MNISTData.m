function ANN = MainANN_MNISTData()

% TODO: ACTIVATION FUNCTIONS

%% *** Setup MATLAB ***
SetupMATLAB();

%% *** Settings for ANN ***
Settings.ANN.Neurons                = [784 192 96 48 24 10]; % With inputs/outputs
Settings.ANN.CostFunc               = 'CrossEntropy';        % CrossEntropy or Quadratic
Settings.ANN.ClassMethod            = 'max1';                % max1 or above50%

%% *** Settings for traning ***
Settings.Train.Alfas                = [2e-2 1e-2 1e-2];  % Learning rate [0,1]
Settings.Train.Lambdas              = [0.02 1e-6 1e-9];  % Weight decay [0,1]
Settings.Train.Etas                 = [1e-3 1e-9 0];     % Momentum [0,1]
Settings.Train.BreakPoints          = [0.03 0.01];       % Breakpoint(s) to change parameters
Settings.Train.BatchSize            = 100;               % Batch size
Settings.Train.Epochs               = 5;                 % Epochs (forward/backward passes)

%% *** Settings for plotting ***
Settings.Plot.ErrorConvergence      = true;  % Error convergence
Settings.Plot.WBConvergence         = false; % Weight & bias convergence
Settings.Plot.NeuralNetwork         = false; % Plot neural network
Settings.Plot.MNISTImages           = {10, 'Incorrect'}; % Plot MNIST images

%% *** Select & preprocess data ***
DistRatio                           = 0.001;
[Data.TrainX, Data.TrainY]          = GetMNISTData('Train');
[Data.DistX, Data.DistY]            = SplitData(Data.TrainX, Data.TrainY, DistRatio);
[Data.DistX, Data.DistY]            = DistortImage(Data.DistX, Data.DistY);
Data.TrainX                         = [Data.TrainX Data.DistX];
Data.TrainY                         = [Data.TrainY Data.DistY];
[Data.TrainX, Data.TrainY]          = PermuteData(Data.TrainX, Data.TrainY);
[Data.TestX, Data.TestY]            = GetMNISTData('Test');

%% *** Inizialize/load ANN ***
% load(fullfile(pwd, 'ANN_MNIST_20171203_145443.mat'), 'ANN');
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
ANNSaveName = ['ANN_MNIST_' datestr(now, 'yyyymmdd_HHMMSS')];
save(fullfile(pwd, 'Networks', ANNSaveName), 'ANN');
