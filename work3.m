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

%Prepare Training and Testing Data
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
historyTimeFreqModalFile = 'historyTimeFreqModal.mat';

% Check if the file exists
if isfile(historyTimeFreqModalFile)
    % Load previous accuracy data
    load(historyTimeFreqModalFile, 'historyTimeFreqModal');
else
    % Initialize an empty array for storing accuracies
    historyTimeFreqModal = [];
end

% Example: Train, validate, and test the MLP for Acc_FreqD_FDay
if ~isempty(Acc_TimeD_FreqD_FDay)
    % Split the dataset
    [trainData, valData, testData] = splitData(Acc_TimeD_FreqD_FDay, 0.8, 0.1);
    
    % Create and configure the neural network
    net = feedforwardnet([10, 5]);
    net.layers{1}.transferFcn = 'tansig'; 
    net.trainParam.lr = 0.01;
    net.trainFcn = 'trainscg'; 
    net.trainParam.epochs = 500; 
    
    % Train the network
    [net, tr] = train(net, trainData', trainData');
    testOutputs = net(testData');
    testPerformance = perform(net, testData', testOutputs);
    disp(['Test Performance: ', num2str(testPerformance)]);
    
    % Accuracy calculation
    [~, testPredictions] = max(testOutputs); 
    [~, testTargets] = max(testData, [], 2); 
    accuracy = sum(testPredictions' == testTargets) / length(testTargets);
    accuracyPercentage = accuracy * 100;
    disp(['Test Accuracy: ', num2str(accuracyPercentage), '%']);
    historyTimeFreqModal = [historyTimeFreqModal; accuracyPercentage];
    save(historyTimeFreqModalFile, 'historyTimeFreqModal');

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

    % Accuracy Trend Visualization
    figure;
    plot(historyTimeFreqModal, '-o', 'LineWidth', 2, 'MarkerSize', 6);
    xlabel('Run Number');
    ylabel('Accuracy (%)');
    title('Accuracy Trend Over Multiple Runs');
    ylim([0 100]);
    grid on;
end
