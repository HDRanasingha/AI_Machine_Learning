% Number of users
numberOfUsers = 10;

% Store combined data
Acc_TimeD_FDay = [];
Acc_TimeD_MDay = [];

% File to store accuracies
accuracyLogFile = 'optimizetimeaccuracy_log.mat';

% Check if the file exists
if isfile(accuracyLogFile)
    load(accuracyLogFile, 'accuracyLog');
else
    accuracyLog = [];
end

% Load data for all users
for userIdx = 1:numberOfUsers
    userPrefix = sprintf('U%02d', userIdx);
    
    % Load the first day time dataset
    dataFDay = load(sprintf('%s_Acc_TimeD_FDay.mat', userPrefix));
    firstDayData = dataFDay.Acc_TD_Feat_Vec;  
    Acc_TimeD_FDay = [Acc_TimeD_FDay; firstDayData];
    
    % Load the multiple day time dataset
    dataMDay = load(sprintf('%s_Acc_TimeD_MDay.mat', userPrefix));
    multiDayData = dataMDay.Acc_TD_Feat_Vec; 
    Acc_TimeD_MDay = [Acc_TimeD_MDay; multiDayData];
end

% Combine the datasets into a single dataset
combinedData = [Acc_TimeD_FDay; Acc_TimeD_MDay];
labelsFDay = ones(size(Acc_TimeD_FDay, 1), 1);  
labelsMDay = zeros(size(Acc_TimeD_MDay, 1), 1); 
combinedLabels = [labelsFDay; labelsMDay];  

% Normalize the combined data
normalizedData = zscore(combinedData);

% PCA for feature selection
[coeff, score, latent] = pca(normalizedData);
explainedVariance = cumsum(latent) / sum(latent);
numComponents = find(explainedVariance >= 0.95, 1); 
reducedData = score(:, 1:numComponents);

% Split the dataset into training, validation, and test sets
[trainData, valData, testData, trainLabels, valLabels, testLabels] = splitData(reducedData, combinedLabels, 0.8, 0.1);

% Create and configure the neural network with increased hidden layers and dropout
net = feedforwardnet([50, 30, 15]);  % Increased number of neurons
net.layers{1}.transferFcn = 'relu';   % Changed to ReLU activation function
net.layers{2}.transferFcn = 'relu';   % ReLU for hidden layers
net.layers{3}.transferFcn = 'tansig'; % Tansig for output layer
net.trainParam.lr = 0.001;            % Keep the learning rate
net.trainFcn = 'trainscg';            % Scaled conjugate gradient
net.trainParam.epochs = 150;          % Increased epochs
net.performFcn = 'mse';               % Mean squared error for performance

% Add Dropout layer
dropoutLayer = dropoutLayer(0.3);     % 30% dropout rate

% Combine the layers (you can experiment with where to add dropout)
net = addlayers(net, dropoutLayer, 2); % Add dropout after the second layer

% Set division of data for training, validation, and testing
net.divideFcn = 'divideind'; 
net.divideParam.trainInd = 1:size(trainData, 1);
net.divideParam.valInd = size(trainData, 1) + (1:size(valData, 1));
net.divideParam.testInd = size(trainData, 1) + size(valData, 1) + (1:size(testData, 1));

% Train the network with early stopping
[net, tr] = train(net, trainData', trainLabels');

% Test the network on multi-day data
testOutputs = net(testData');

% Accuracy calculation (Binary classification)
testPredictions = testOutputs > 0.5;  % Binary classification (0 or 1)
accuracy = sum(testPredictions' == testLabels) / length(testLabels);
accuracyPercentage = accuracy * 100;

% Display performance
disp(['Test Accuracy: ', num2str(accuracyPercentage), '%']);

% Log the accuracy
accuracyLog = [accuracyLog; accuracyPercentage];

% Save the updated accuracy log
save(accuracyLogFile, 'accuracyLog');

% Accuracy Trend Visualization
figure;
plot(accuracyLog, '-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Run Number');
ylabel('Accuracy (%)');
title('Time optimize Step Accuracy');
ylim([0 100]);
grid on;

% Plot training performance
figure;
plot(tr.perf, 'LineWidth', 2);
hold on;
plot(tr.vperf, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Performance (MSE)');
legend('Training', 'Validation');
title('Training and Validation Performance');
grid on;

% Function to split data into train, validation, and test sets
function [trainData, valData, testData, trainLabels, valLabels, testLabels] = splitData(data, labels, trainRatio, valRatio)
    numSamples = size(data, 1);
    trainSize = round(trainRatio * numSamples);
    valSize = round(valRatio * numSamples);
    indices = randperm(numSamples);
    trainData = data(indices(1:trainSize), :);
    valData = data(indices(trainSize+1:trainSize+valSize), :);
    testData = data(indices(trainSize+valSize+1:end), :);
    
    trainLabels = labels(indices(1:trainSize));
    valLabels = labels(indices(trainSize+1:trainSize+valSize));
    testLabels = labels(indices(trainSize+valSize+1:end));
end
