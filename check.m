%data= load ("U01_Acc_FreqD_MDay.mat");
%t=array2table(data);
% Load the .mat file
%data = load("U01_Acc_FreqD_MDay.mat");
data = load("U01_Acc_FreqD_MDay.mat");
disp('Variables in the .mat file:');
disp(fieldnames(data));
data = load("U01_Acc_FreqD_MDay.mat");
disp('Variables in the .mat file:');
disp(fieldnames(data));




% Load the cross day dataset
dataFreqMDay = load("U01_Acc_FreqD_MDay.mat");
crossDayData = dataFreqMDay.Acc_FD_Feat_Vec;
% Display the variables in the loaded data
%disp('Variables in the .mat file:');
%disp(fieldnames(data));

% Extract the desired variable (replace 'your_variable_name' with the actual name)
%yourVariable = data.Acc_FD_Feat_Vec; % Replace with the correct variable name from the file

 %Convert to table if needed (optional)
t = array2table(crossDayData);

% Calculate descriptive statistics
meanValue = mean(crossDayData, 1); % Mean of each column
stdValue = std(crossDayData, 0, 1); % Standard deviation of each column
varianceValue = var(crossDayData, 0, 1); % Variance of each column

% Display the results
disp('Descriptive Statistics:');
disp(['Mean: ', num2str(meanValue)]);
disp(['Standard Deviation: ', num2str(stdValue)]);
disp(['Variance: ', num2str(varianceValue)]);

