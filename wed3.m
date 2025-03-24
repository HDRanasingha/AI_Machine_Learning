% Number of users and files per user
numberofUsers = 10; 
numberofFilesPerUser = 6; 

% Store combined data
Acc_FreqD_FDay = [];
Acc_TimeD_FDay = [];
Acc_TimeD_FreqD_FDay = [];
Acc_FreqD_MDay = [];
Acc_TimeD_MDay = [];
Acc_TimeD_FreqD_MDay = [];

% Load data from files
for userIdx = 1:numberofUsers
    userPrefix = sprintf('U%02d', userIdx);
    
    % File names corresponding
    fileNames = {
        sprintf('%s_Acc_FreqD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FreqD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_FreqD_MDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_MDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FreqD_MDay.mat', userPrefix)
    };

    % Load and store data based on variable name
    for fileIdx = 1:numberofFilesPerUser
        data = load(fileNames{fileIdx});
        if isfield(data, 'Acc_FD_Feat_Vec')
            Acc_FreqD_FDay = [Acc_FreqD_FDay; data.Acc_FD_Feat_Vec];
        elseif isfield(data, 'Acc_TD_Feat_Vec')
            Acc_TimeD_FDay = [Acc_TimeD_FDay; data.Acc_TD_Feat_Vec];
        elseif isfield(data, 'Acc_TDFD_Feat_Vec')
            Acc_TimeD_FreqD_FDay = [Acc_TimeD_FreqD_FDay; data.Acc_TDFD_Feat_Vec];
        elseif isfield(data, 'FreqD_Feat_Vec')
            Acc_FreqD_MDay = [Acc_FreqD_MDay; data.FreqD_Feat_Vec];
        elseif isfield(data, 'TimeD_Feat_Vec')
            Acc_TimeD_MDay = [Acc_TimeD_MDay; data.TimeD_Feat_Vec];
        elseif isfield(data, 'TDFD_Feat_Vec')
            Acc_TimeD_FreqD_MDay = [Acc_TimeD_FreqD_MDay; data.TDFD_Feat_Vec];
        else
            warning('Unrecognized variable in file %s', fileNames{fileIdx});
        end
    end
end

% Display summary of collected data
disp('Summary of Loaded Data:');
disp(['Acc_FreqD_FDay: ', num2str(size(Acc_FreqD_FDay))]);
disp(['Acc_TimeD_FDay: ', num2str(size(Acc_TimeD_FDay))]);
disp(['Acc_TimeD_FreqD_FDay: ', num2str(size(Acc_TimeD_FreqD_FDay))]);
disp(['Acc_FreqD_MDay: ', num2str(size(Acc_FreqD_MDay))]);
disp(['Acc_TimeD_MDay: ', num2str(size(Acc_TimeD_MDay))]);
disp(['Acc_TimeD_FreqD_MDay: ', num2str(size(Acc_TimeD_FreqD_MDay))]);

% Step 2: Prepare Training and Testing Data
trainDataFDay = Acc_TimeD_FreqD_FDay; % Training/validation data
testDataMDay = Acc_TimeD_FreqD_MDay;  % Testing data

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

% File to store accuracies
TimeFreqFile = 'TimeFreq.mat';

% Check if the file exists
if isfile(TimeFreqFile)
    % Load previous accuracy data
    load(TimeFreqFile, 'TimeFreq');
else
    % Initialize an empty array for storing accuracies
    TimeFreq = [];
end

% Example: Train, validate, and test the MLP for Acc_FreqD_FDay
if ~isempty(Acc_TimeD_FreqD_FDay)
    % Split the dataset
    [trainData, valData, testData] = splitData(Acc_TimeD_FreqD_FDay, 0.8, 0.1);
    
    % Perform PCA
    [coeff, score, latent] = pca(trainData);
    explainedVariance = cumsum(latent) / sum(latent);
    numComponents = find(explainedVariance >= 0.95, 1);
    trainDataPCA = score(:, 1:numComponents);
    testDataPCA = (testData - mean(trainData)) * coeff(:, 1:numComponents);
    
    % Create and configure the neural network
    net = feedforwardnet([20, 15]);
    net.layers{1}.transferFcn = 'tansig'; 
    net.trainParam.lr = 0.02;
    net.trainFcn = 'trainscg'; 
    net.trainParam.epochs = 100; 
    
    % Train the network
    [net, tr] = train(net, trainDataPCA', trainDataPCA');
    testOutputs = net(testDataPCA');
    testPerformance = perform(net, testDataPCA', testOutputs);
    disp(['Test Performance: ', num2str(testPerformance)]);
    
    % Accuracy calculation
    [~, testPredictions] = max(testOutputs); 
    [~, testTargets] = max(testDataPCA, [], 2); 
    accuracy = sum(testPredictions' == testTargets) / length(testTargets);
    accuracyPercentage = accuracy * 100;
    disp(['Test Accuracy: ', num2str(accuracyPercentage), '%']);
    TimeFreq = [TimeFreq; accuracyPercentage];
    save(TimeFreqFile, 'TimeFreq');

    % PCA Explained Variance
    figure;
    plot(cumsum(latent) / sum(latent), 'o-', 'LineWidth', 2);
    xlabel('Number of Principal Components');
    ylabel('Cumulative Explained Variance');
    title('Explained Variance by Principal Components');
    grid on;

    % Training Performance
    figure;
    plot(tr.perf, 'LineWidth', 2);
    hold on;
    plot(tr.vperf, 'LineWidth', 2);
    xlabel('Epoch');
    ylabel('Performance (MSE)');
    legend('Training', 'Validation');
    title('Training and Validation Performance');
    grid on;

    % Confusion Matrix
    figure;
    confusionchart(testTargets, testPredictions);
    title('Confusion Matrix for Test Data');

    % Accuracy Trend Visualization
    figure;
    plot(TimeFreq, '-o', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('Run Number');
    ylabel('Accuracy (%)');
    title('TimeFrequncy accuracy');
    ylim([0 100]);
    grid on;
end
