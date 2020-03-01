function ANN = PrintClassError(ANN, Data, Type)
X = Data.([Type 'X']);
Y = Data.([Type 'Y']);

[~, ErrorBin] = GetClassification(ANN, X, Y);

ErrorAvg     = sum(ErrorBin) / length(ErrorBin);
ErrorPercent = ErrorAvg * 100;
fprintf('%s error: %.2f %%\n', Type, ErrorPercent);
