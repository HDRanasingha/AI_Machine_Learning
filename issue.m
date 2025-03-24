% Number of users
numberOfUsers = 10;

% Store combined data
Acc_FreqD_FDay = [];
Acc_FreqD_MDay = [];

% File to store accuracies
accuracyLogFile = 'accuracy_log.mat';

% Check if the file exists
if isfile(accuracyLogFile)
    % Load previous accuracy data
    load(accuracyLogFile, 'accuracyLog');
else
    % Initialize an empty array for storing accuracies
    accuracyLog = [];
end

% Load data for all users
for userIdx = 1:numberOfUsers
    userPrefix = sprintf('U%02d', userIdx);
    
    % Load the first-day time dataset
    dataFuDay = load(sprintf('%s_Acc_FreqD_FDay.mat', userPrefix));
    disp('Variables in the first-day time .mat file:');
    disp(fieldnames(dataFuDay));
    
    % Load the first-day dataset
    oneDayData = dataFuDay.Acc_FD_Feat_Vec;  % Replace with the correct variable name
    Acc_FreqD_FDay = [Acc_FreqD_FDay; oneDayData];
    
    % Load the multi-day time dataset
    dataMuDay = load(sprintf('%s_Acc_FreqD_MDay.mat', userPrefix));
    disp('Variables in the multi-day time .mat file:');
    disp(fieldnames(dataMuDay));
    
    % Load the multi-day dataset
    secondDayData = dataMuDay.Acc_FD_Feat_Vec;  % Replace with the correct variable name
    Acc_FreqD_MDay = [Acc_FreqD_MDay; secondDayData];
end

% Display summary of collected data
disp('Summary of Loaded Data:');
disp(['Acc_FreqD_FDay: ', num2str(size(Acc_FreqD_FDay))]);
disp(['Acc_FreqD_MDay: ', num2str(size(Acc_FreqD_MDay))]);

% Split the dataset (train on FDay data, test on MDay data)
[trainData, valData, testData] = splitData(Acc_FreqD_FDay, 0.8, 0.1);

% Define labels (use appropriate labels from your dataset)
% Assuming the labels are also available from the data
trainLabels = trainData;  % Replace with actual labels if available
valLabels = valData;      % Replace with actual labels if available
testLabels = testData;    % Replace with actual labels if available

% Create and configure the neural network
net = feedforwardnet([10, 5]);
net.layers{1}.transferFcn = 'tansig'; 
net.trainParam.lr = 0.01;
net.trainFcn = 'trainscg'; 
net.trainParam.epochs = 100; 
net.performFcn = 'mse';  % Mean squared error for regression or cross-entropy for classification

% Set division of data for training, validation, and testing
net.divideFcn = 'divideind';  % Specify manual division
net.divideParam.trainInd = 1:size(trainData, 1);
net.divideParam.valInd = size(trainData, 1) + (1:size(valData, 1));
net.divideParam.testInd = size(trainData, 1) + size(valData, 1) + (1:size(testData, 1));

% Train the network
[net, tr] = train(net, trainData', trainLabels');

% Test the network on multi-day data (MDay)
testOutputs = net(testData');

% Accuracy calculation
[~, testPredictions] = max(testOutputs, [], 1);  % Class predictions
[~, testTargets] = max(testLabels, [], 2);       % Actual class labels
accuracy = sum(testPredictions' == testTargets) / length(testTargets);
accuracyPercentage = accuracy * 100;

% Display performance results
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
title('Accuracy Trend Over Multiple Runs');
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
function [trainData, valData, testData] = splitData(data, trainRatio, valRatio)
    numSamples = size(data, 1);
    trainSize = round(trainRatio * numSamples);
    valSize = round(valRatio * numSamples);
    indices = randperm(numSamples);
    trainData = data(indices(1:trainSize), :);
    valData = data(indices(trainSize+1:trainSize+valSize), :);
    testData = data(indices(trainSize+valSize+1:end), :);
end


