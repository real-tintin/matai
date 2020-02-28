% Machine Learning ANN: PlotErrorConvergence
function PlotHandle = PlotErrorConvergence(State, Settings)

% Data
Epochs   = Settings.Train.Epochs;
Epoch    = 1:Epochs;
ErrorCon = State.ErrorCon*100;
ErrorBin = State.ErrorBin*100;

% Plot error convergence
PlotHandle = figure;
plot(Epoch, ErrorCon, Epoch, ErrorBin);
title('Output error e_{bin} and e_{con} convergence');
xlabel('Epoch');
ylabel('\Sigma |e|');
legend('e_{con}', 'e_{bin}');
grid on;