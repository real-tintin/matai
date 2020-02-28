% Machine Learning ANN: PrintClassError
function ANN = PrintClassError(ANN, Data, Type)

% Get data
X = Data.([Type 'X']);
Y = Data.([Type 'Y']);

% Get classification error
[C, Error] = GetClassification(ANN, X, Y);

% Print classification error
ErrorAvg     = sum(Error) / length(Error);
ErrorPercent = ErrorAvg * 100;
fprintf('%s error: %.2f %%\n', Type, ErrorPercent);