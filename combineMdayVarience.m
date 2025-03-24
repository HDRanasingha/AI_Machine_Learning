% amount of users
numUsers = 10;

allData = []; % combine mean data
userVariances = []; % Store intra-variance for each user

for userIdx = 1:numUsers
    
    fileName = sprintf('U%02d_Acc_TimeD_FreqD_MDay.mat', userIdx);
    datacombineMDay = load(fileName);

    % Display the variables 
    fprintf('Variables in %s:\n', fileName);
    disp(fieldnames(datacombineMDay));

    
    if isfield(datacombineMDay, 'Acc_TDFD_Feat_Vec')
        userData = datacombineMDay.Acc_TDFD_Feat_Vec; 
    else
        error('Variable "Acc_TDFD_Feat_Vec" not found in %s', fileName);
    end

    %calculate intraVariance
    intraVariance = var(userData, 0, 1); 

    %  mean data for the current user (for inter-variance)
    userMeanData = mean(userData, 1);

    %  user's mean data for inter-variance calculation
    allData = [allData; userMeanData];

    % Store intra-variance for the current user
    userVariances = [userVariances; intraVariance];

    % Display the intra-variance for the current user
    fprintf('User %d Intra-Variance per Variable:\n', userIdx);
    disp(intraVariance);
end

% calculate inter-variance 
interVariance = var(allData, 0, 1);

% Display inter-variance for all users combined
fprintf('Inter-Variance Across Users (Based on Mean Data):\n');
disp(interVariance);

% Display overall statistics
fprintf('Summary of Variances Across All Users:\n');
fprintf('Average Intra-Variance Across Users:\n');
disp(mean(userVariances, 1));
fprintf('Inter-Variance Across All Users:\n');
disp(interVariance);



