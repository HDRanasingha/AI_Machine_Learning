% Number of users
numberOfUsers = 10;

% Initialize combined dataset for time domain features (First Day and Multi-Day)
Acc_TimeD_FDay = [];
Acc_TimeD_MDay = [];
Acc_FreqD_Fday = [];
Acc_FreqD_Mday = [];
Acc_TimeD_FreqD_Mday = [];
Acc_TimeD_FreqD_Fday = [];

% Load data for all users
for userIdx = 1:numberOfUsers
    userPrefix = sprintf('U%02d', userIdx);
    
    % Load the Time Domain First Day dataset
    dataTimeFDay = load(sprintf('%s_Acc_TimeD_FDay.mat', userPrefix));
    if isfield(dataTimeFDay, 'Acc_TD_Feat_Vec')
        Acc_TimeD_FDay = [Acc_TimeD_FDay; dataTimeFDay.Acc_TD_Feat_Vec];  % Combine TimeFDay data
    end
    
    % Load the Time Domain Multi-Day dataset
    dataTimeMDay = load(sprintf('%s_Acc_TimeD_MDay.mat', userPrefix));
    if isfield(dataTimeMDay, 'Acc_TD_Feat_Vec')
        Acc_TimeD_MDay = [Acc_TimeD_MDay; dataTimeMDay.Acc_TD_Feat_Vec];  % Combine TimeMDay data
    end
end

% Display summary of combined data
disp('Summary of Loaded Data:');
disp(['Acc_TimeD_FDay size: ', num2str(size(Acc_TimeD_FDay))]);
disp(['Acc_TimeD_MDay size: ', num2str(size(Acc_TimeD_MDay))]);

% Combine both TimeFDay and TimeMDay datasets
combinedData = [Acc_TimeD_FDay; Acc_TimeD_MDay];

% Split the dataset into training, validation, and test sets
[trainData, valData, testData] = splitData(combinedData, 0.8, 0.1);  % 80% train, 10% validation, 10% test

% Create and configure the Feedforward MLP
net = feedforwardnet([10, 5]); % Two hidden layers with 10 and 5 neurons
net.layers{1}.transferFcn = 'tansig'; % Activation function for the first layer
net.trainParam.lr = 0.01;  % Learning rate
net.trainFcn = 'trainscg'; % Optimizer (scaled conjugate gradient)
net.trainParam.epochs = 500;  % Max epochs

% Train the network
[net, tr] = train(net, trainData', trainData');  % Training with self-prediction (auto-associative)

% Test the network
testOutputs = net(testData');
testPerformance = perform(net, testData', testOutputs);
disp(['Test Performance (MSE): ', num2str(testPerformance)]);

% Calculate accuracy
accuracy = calculateAccuracy(testOutputs', testData);
accuracyPercentage = accuracy * 100;
disp(['Test Accuracy: ', num2str(accuracyPercentage), '%']);

% Function to split data into training, validation, and testing sets
function [trainData, valData, testData] = splitData(data, trainRatio, valRatio)
    numSamples = size(data, 1);
    trainSize = round(trainRatio * numSamples);
    valSize = round(valRatio * numSamples);
    indices = randperm(numSamples);
    trainData = data(indices(1:trainSize), :);
    valData = data(indices(trainSize+1:trainSize+valSize), :);
    testData = data(indices(trainSize+valSize+1:end), :);
end

% Function to calculate accuracy
function accuracy = calculateAccuracy(predictions, targets)
    [~, predictedLabels] = max(predictions, [], 1);
    [~, trueLabels] = max(targets, [], 1);
    accuracy = sum(predictedLabels == trueLabels) / length(trueLabels);
end





