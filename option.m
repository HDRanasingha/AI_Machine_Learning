% Define the number of users and files per user
numUsers = 10; % Total number of users
numFilesPerUser = 6; % Files per user

% Initialize arrays to store combined data for each feature type
Acc_FreqD_FDay = [];
Acc_TimeD_FDay = [];
Acc_TimeD_FreqD_FDay = [];
Acc_FreqD_MDay = [];
Acc_TimeD_MDay = [];
Acc_TimeD_FreqD_MDay = [];

% Load data
for userIdx = 1:numUsers
    userPrefix = sprintf('U%02d', userIdx);
    fileNames = {
        sprintf('%s_Acc_FreqD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FreqD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_FreqD_MDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_MDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FreqD_MDay.mat', userPrefix)
    };

    for fileIdx = 1:numFilesPerUser
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

% Split data function
function [trainData, valData, testData] = splitData(data, trainRatio, valRatio)
    numSamples = size(data, 1);
    trainSize = round(trainRatio * numSamples);
    valSize = round(valRatio * numSamples);
    testSize = numSamples - trainSize - valSize;

    indices = randperm(numSamples);
    trainData = data(indices(1:trainSize), :);
    valData = data(indices(trainSize+1:trainSize+valSize), :);
    testData = data(indices(trainSize+valSize+1:end), :);
end

% Example: Train, validate, and evaluate the MLP for Acc_FreqD_FDay
if ~isempty(Acc_FreqD_FDay)
    % Split the dataset (80% train, 10% validation)
    [trainData, valData, ~] = splitData(Acc_FreqD_FDay, 0.8, 0.1);
    
    % Define the number of features and classes
    numFeatures = size(trainData, 2);
    numClasses = size(trainData, 2); % Assuming one-hot encoded targets

    % Create and configure the neural network
    net = feedforwardnet(10); % 10 hidden neurons
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 10/100;
    net.divideParam.testRatio = 0; % Not using a separate test ratio within the network

    % Train the network
    [net, tr] = train(net, trainData', trainData'); % Adjust target accordingly

    % Evaluate on validation set
    valOutputs = net(valData');
    valPerformance = perform(net, valData', valOutputs);
    disp(['Validation Performance for Acc_FreqD_FDay: ', num2str(valPerformance)]);
    
    % Calculate training accuracy
    trainOutputs = net(trainData'); % Get outputs for training data
    [~, trainPredictions] = max(trainOutputs); % Get predicted classes
    [~, trainTargets] = max(trainData, [], 2); % True classes for one-hot encoding
    trainAccuracy = sum(trainPredictions' == trainTargets) / length(trainTargets);
    disp(['Training Accuracy for Acc_FreqD_FDay: ', num2str(trainAccuracy * 100), '%']);
    
    % Calculate validation accuracy
    [~, valPredictions] = max(valOutputs); % Get predicted classes
    [~, valTargets] = max(valData, [], 2); % True classes for one-hot encoding
    valAccuracy = sum(valPredictions' == valTargets) / length(valTargets);
    disp(['Validation Accuracy for Acc_FreqD_FDay: ', num2str(valAccuracy * 100), '%']);
end
