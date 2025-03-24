% Load the multi-day time dataset
dataMDay = load("U02_Acc_TimeD_MDay.mat");

% Display the variables in the .mat file
disp('Variables in the .mat file:');
disp(fieldnames(dataMDay));

% Load the multi-day dataset
crossDayDataMDay = dataMDay.Acc_TD_Feat_Vec;  % Replace with the correct variable name

% Convert to table if needed (optional)
tMDay = array2table(crossDayDataMDay);

% Calculate descriptive statistics
meanValueMDay = mean(crossDayDataMDay, 1); % Mean of each column
stdValueMDay = std(crossDayDataMDay, 0, 1); % Standard deviation of each column
varianceValueMDay = var(crossDayDataMDay, 0, 1); % Variance of each column

% Display the results
disp('Descriptive Statistics for Multi-Day Time Domain:');
disp(['Mean: ', num2str(meanValueMDay)]);
disp(['Standard Deviation: ', num2str(stdValueMDay)]);
disp(['Variance: ', num2str(varianceValueMDay)]);