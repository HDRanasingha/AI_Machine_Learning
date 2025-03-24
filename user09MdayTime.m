% Load user 09 multiple time domain
dataMDay = load("U09_Acc_TimeD_MDay.mat");

% Display the variables
disp('Variables in the .mat file:');
disp(fieldnames(dataMDay));

multiDayData = dataMDay.Acc_TD_Feat_Vec;  

% Convert to table 
%tMDay = array2table(multiDayData);

% Calculate descriptive statistics
meanValueMDay = mean(multiDayData, 1); 
stdValueMDay = std(multiDayData, 0, 1); 
varianceValueMDay = var(multiDayData, 0, 1); 

% Display mean & stdv&varience
disp(['Mean: ', num2str(meanValueMDay)]);
disp(['Standard Deviation: ', num2str(stdValueMDay)]);
disp(['Variance: ', num2str(varianceValueMDay)]);