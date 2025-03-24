% Number of users
numberOfUsers = 10;

% Store combined data
Acc_TimeD_FreqD_FDay = [];
Acc_TimeD_FreqD_MDay = [];

% File to store accuracies
accuracyLogFile = 'FTAccuracy_log.mat';

% Check if the file exists
if isfile(accuracyLogFile)
    % Load previous accuracy data
    load(accuracyLogFile, 'accuracyLog');
else
    accuracyLog = [];
end

% Load data for all users
for userIdx = 1:numberOfUsers
    userPrefix = sprintf('U%02d', userIdx);
    
    % Load the first-day time-frequency combine dataset
    datacombineFDay = load(sprintf('%s_Acc_TimeD_FreqD_FDay.mat', userPrefix));
    FIRSTDayData = datacombineFDay.Acc_TDFD_Feat_Vec;
    Acc_TimeD_FreqD_FDay = [Acc_TimeD_FreqD_FDay; FIRSTDayData];
    
    % Load the multi-day time-frequency combine dataset
    datacombineMDay = load(sprintf('%s_Acc_TimeD_FreqD_MDay.mat', userPrefix));
    multipleDayData = datacombineMDay.Acc_TDFD_Feat_Vec;
    Acc_TimeD_FreqD_MDay = [Acc_TimeD_FreqD_MDay; multipleDayData];
end

% Combine the datasets
combinedData = [Acc_TimeD_FreqD_FDay; Acc_TimeD_FreqD_MDay];
labelsFDay = ones(size(Acc_TimeD_FreqD_FDay, 1), 1); 
labelsMDay = zeros(size(Acc_TimeD_FreqD_MDay, 1), 1);  
combinedLabels = [labelsFDay; labelsMDay];

% Split the dataset into training, validation, and test sets
trainSize = round(0.8 * size(combinedData, 1));
valSize = round(0.1 * size(combinedData, 1));
indices = randperm(size(combinedData, 1));
trainData = combinedData(indices(1:trainSize), :);
valData = combinedData(indices(trainSize+1:trainSize+valSize), :);
testData = combinedData(indices(trainSize+valSize+1:end), :);

trainLabels = combinedLabels(indices(1:trainSize));
valLabels = combinedLabels(indices(trainSize+1:trainSize+valSize));
testLabels = combinedLabels(indices(trainSize+valSize+1:end));

% Create and configure the neural network
net = feedforwardnet([20,15]);
net.layers{1}.transferFcn = 'tansig'; 
net.trainParam.lr = 0.0001;
net.trainFcn = 'trainscg'; 
net.trainParam.epochs = 200; 
net.performFcn = 'mse'; 

% Set division of data for training, validation, and testing
net.divideFcn = 'divideind'; 
net.divideParam.trainInd = 1:size(trainData, 1);
net.divideParam.valInd = size(trainData, 1) + (1:size(valData, 1));
net.divideParam.testInd = size(trainData, 1) + size(valData, 1) + (1:size(testData, 1));

% Train the network
[net, tr] = train(net, trainData', trainLabels');

% Test the network on multi-day data
testOutputs = net(testData');

% Accuracy calculation (Binary classification)
testPredictions = testOutputs > 0.5; 
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
title('combine Evaluate Step Accuracy');
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



