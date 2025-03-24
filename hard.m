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

% Iterate through all users and their corresponding files
for userIdx = 1:numUsers
    % Construct user ID prefix (e.g., U01, U02, ...)
    userPrefix = sprintf('U%02d', userIdx);
    
    % File names corresponding to the variables
    fileNames = {
        sprintf('%s_Acc_FreqD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FreqD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_FreqD_MDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_MDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FreqD_MDay.mat', userPrefix)
    };

    % Iterate through each file and load data
    for fileIdx = 1:numFilesPerUser
        % Load the .mat file
        data = load(fileNames{fileIdx});
        
        % Append data to corresponding dataset
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

% Split data into training, validation, and testing
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

% Example: Train, validate, and test the MLP for Acc_FreqD_FDay
if ~isempty(Acc_FreqD_FDay)
    % Split the dataset (80% train, 10% validation, 10% test)
    [trainData, valData, testData] = splitData(Acc_FreqD_FDay, 0.8, 0.1);
    
    % Create and configure the neural network
    net = feedforwardnet(10); % 10 hidden neurons
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.1;
    net.divideParam.testRatio = 0.1;
    
    % Train the network
    [net, tr] = train(net, trainData', trainData'); % Adjust target accordingly
    
    % Evaluate the performance on the test data
    testOutputs = net(testData');
    
    % Calculate performance metrics
    testPerformance = perform(net, testData', testOutputs);
    disp(['Test Performance for Acc_FreqD_FDay: ', num2str(testPerformance)]);
    
    % Calculate accuracy
    [~, testPredictions] = max(testOutputs); % Predicted classes
    [~, testTargets] = max(testData, [], 2); % True classes for one-hot encoding
    accuracy = sum(testPredictions' == testTargets) / length(testTargets);
    
    % Display accuracy as a percentage
    accuracyPercentage = accuracy * 100;
    disp(['Test Accuracy for Acc_FreqD_FDay: ', num2str(accuracyPercentage), '%']);
end