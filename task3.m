% Define the number of users and files per user
numofUsers = 10; 
numofFilesPerUser = 6; 

%  store combined data 
Acc_FreqD_FDay = [];
Acc_TimeD_FDay = [];
Acc_TimeD_FreqD_FDay = [];
Acc_FreqD_MDay = [];
Acc_TimeD_MDay = [];
Acc_TimeD_FreqD_MDay = [];


for userIdx = 1:numofUsers
   
    userPrefix = sprintf('U%02d', userIdx);
    

    fileNames = {
        sprintf('%s_Acc_FreqD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FreqD_FDay.mat', userPrefix), ...
        sprintf('%s_Acc_FreqD_MDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_MDay.mat', userPrefix), ...
        sprintf('%s_Acc_TimeD_FreqD_MDay.mat', userPrefix)
    };

   
    for fileIdx = 1:numofFilesPerUser
        % Load the .mat file
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

% Function to split data into training, validation, and testing
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
if ~isempty(Acc_TimeD_FreqD_FDay)
    % Split the dataset (80% train, 10% validation, 10% test)
    [trainData, valData, testData] = splitData(Acc_TimeD_FreqD_FDay, 0.8, 0.1);
    
    % Perform PCA
    [coeff, score, latent] = pca(trainData);
    explainedVariance = cumsum(latent) / sum(latent);
    
    % Choose the number of components that explain 95% variance
    numComponents = find(explainedVariance >= 0.95, 1);
    trainDataPCA = score(:, 1:numComponents);
    testDataPCA = (testData - mean(trainData)) * coeff(:, 1:numComponents);
    
    % Create and configure the neural network with optimized architecture
    net = feedforwardnet([15, 10]); % Example with 2 hidden layers
    
    % Set activation function
    net.layers{1}.transferFcn = 'tansig'; % Adjust as needed
    
    % Set learning rate and training function
    net.trainParam.lr = 0.01;
    net.trainFcn = 'trainscg'; % Scaled conjugate gradient
    
    % Set the number of epochs
    net.trainParam.epochs = 100; 
    
    % Train the network
    [net, tr] = train(net, trainDataPCA', trainDataPCA'); % Adjust target accordingly
    
    % Evaluate the performance on the test data
    testOutputs = net(testDataPCA');
    
    % Calculate performance metrics
    testPerformance = perform(net, testDataPCA', testOutputs);
    disp(['Test Performance for Acc_TimeD_FreqD_FDay: ', num2str(testPerformance)]);
    
    % Calculate accuracy
    [~, testPredictions] = max(testOutputs); 
    [~, testTargets] = max(testDataPCA, [], 2); 
    accuracy = sum(testPredictions' == testTargets) / length(testTargets);
    
    % Display accuracy as a percentage
    accuracyPercentage = accuracy * 100;
    disp(['Test Accuracy for Acc_TimeD_FreqD_FDay: ', num2str(accuracyPercentage), '%']);

 % Visualization
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
    end

