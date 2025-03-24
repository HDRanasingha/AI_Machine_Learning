% Number of users
numberOfUsers = 10;

% Store combined data
Acc_TimeD_FreqD_FDay = [];
Acc_TimeD_FreqD_MDay = [];

% File to store accuracies
accuracyLogFile = 'FrequencyTime_accuracy_log.mat';

% Check if the file exists
if isfile(accuracyLogFile)
    load(accuracyLogFile, 'accuracyLog');
else
    accuracyLog = [];
end

% Load data for all users
for userIdx = 1:numberOfUsers
    userPrefix = sprintf('U%02d', userIdx);
    
    % Load the first.day time dataset
    datacombineFDay = load(sprintf('%s_Acc_TimeD_FreqD_FDay.mat', userPrefix));
    disp('Variables in the f.day frequency .mat file:');
    disp(fieldnames(datacombineFDay));
    
    % Load the first.day dataset
    FIRSTDayData = datacombineFDay.Acc_TDFD_Feat_Vec; 
    Acc_TimeD_FreqD_FDay = [Acc_TimeD_FreqD_FDay; FIRSTDayData];
    
    % Load the m.day time dataset
    datacombineMDay = load(sprintf('%s_Acc_TimeD_FreqD_MDay.mat', userPrefix));
    disp('Variables in the m.day frequency .mat file:');
    disp(fieldnames(datacombineMDay));
    
    % Load the multi-day dataset
    multipleDayData = datacombineMDay.Acc_TDFD_Feat_Vec; 
    Acc_TimeD_FreqD_MDay = [Acc_TimeD_FreqD_MDay; multipleDayData];  
end

% Display summary of collected data
disp('Summary of Loaded Data:');
disp(['Acc_TimeD_FreqD_FDay: ', num2str(size(Acc_TimeD_FreqD_FDay))]);
disp(['Acc_TimeD_FreqD_MDay: ', num2str(size(Acc_TimeD_FreqD_MDay))]);

% Step 1: Split the dataset
[trainData, valData, testData, trainLabels, valLabels, testLabels] = splitData(Acc_TimeD_FreqD_FDay, Acc_TimeD_FreqD_MDay, 0.8, 0.1);

% Step 2: Perform PCA for dimensionality reduction
[coeff, score, latent] = pca(trainData);
explainedVariance = cumsum(latent) / sum(latent);
numComponents = find(explainedVariance >= 0.95, 1);  % Retain enough components to explain 95% variance
trainDataPCA = score(:, 1:numComponents);
testDataPCA = (testData - mean(trainData)) * coeff(:, 1:numComponents);  % Apply PCA to test data

% PCA Explained Variance Plot
figure;
plot(explainedVariance, 'o-', 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance');
title('Explained Variance by Principal Components');
grid on;

% Step 3: Create and configure the neural network
net = feedforwardnet([20, 15]);  % Two hidden layers, 20 and 15 neurons
net.layers{1}.transferFcn = 'tansig';  % Hyperbolic tangent sigmoid transfer function
net.trainParam.lr = 0.01;  % Learning rate
net.trainFcn = 'trainscg';  % Scaled conjugate gradient training function
net.trainParam.epochs = 100;  % Max epochs
net.performFcn = 'mse';  % Mean squared error for performance
net.divideFcn = 'divideind';  % Manual data division for training, validation, and testing
net.divideParam.trainInd = 1:size(trainDataPCA, 1);
net.divideParam.valInd = size(trainDataPCA, 1) + (1:size(valData, 1));
net.divideParam.testInd = size(trainDataPCA, 1) + size(valData, 1) + (1:size(testData, 1));

% Train the network
[net, tr] = train(net, trainDataPCA', trainLabels');

% Step 4: Test the network on m.day data
testOutputs = net(testDataPCA');
[~, testPredictions] = max(testOutputs, [], 1);  % Class predictions
[~, testTargets] = max(testLabels, [], 2);       % Actual class labels
accuracy = sum(testPredictions' == testTargets) / length(testTargets);
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
title('Evaluate Combined Accuracy');
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
function [trainData, valData, testData, trainLabels, valLabels, testLabels] = splitData(dataFDay, dataMDay, trainRatio, valRatio)
    numSamples = size(dataFDay, 1);
    trainSize = round(trainRatio * numSamples);
    valSize = round(valRatio * numSamples);
    indices = randperm(numSamples);
    
    % Split data
    trainData = dataFDay(indices(1:trainSize), :);
    valData = dataFDay(indices(trainSize+1:trainSize+valSize), :);
    testData = dataMDay(indices(trainSize+valSize+1:end), :);
    
    % Split labels (assuming labels are available)
    trainLabels = dataFDay(indices(1:trainSize), :);  % Modify this as per the actual label structure
    valLabels = dataFDay(indices(trainSize+1:trainSize+valSize), :);
    testLabels = dataMDay(indices(trainSize+valSize+1:end), :);
end

