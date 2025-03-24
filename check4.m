% Load the first-day frequency dataset
dataFDay = load("U02_Acc_FreqD_FDay.mat");

% Display the variables in the .mat file
disp('Variables in the .mat file:');
disp(fieldnames(dataFDay));

% Load the first-day dataset
crossDayDataFDay = dataFDay.Acc_FD_Feat_Vec;  % Replace with the correct variable name

% Convert to table if needed (optional)
tFDay = array2table(crossDayDataFDay);

% Calculate descriptive statistics
meanValueFDay = mean(crossDayDataFDay, 1); % Mean of each column
stdValueFDay = std(crossDayDataFDay, 0, 1); % Standard deviation of each column
varianceValueFDay = var(crossDayDataFDay, 0, 1); % Variance of each column

% Display the results
disp('Descriptive Statistics for First Day Frequency:');
disp(['Mean: ', num2str(meanValueFDay)]);
disp(['Standard Deviation: ', num2str(stdValueFDay)]);
disp(['Variance: ', num2str(varianceValueFDay)]);