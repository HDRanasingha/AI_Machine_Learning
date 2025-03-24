% load user 04 First day time domain
dataFDay = load("U04_Acc_TimeD_FDay.mat");

% variables display
disp(fieldnames(dataFDay));

firstDayData = dataFDay.Acc_TD_Feat_Vec;  

% convert to table
%tMDay = array2table(firstDayData);

% mean,std,verience calculate
meanValueMDay = mean(firstDayData, 1);
stdValueMDay = std(firstDayData, 0, 1); 
varianceValueMDay = var(firstDayData, 0, 1); 

% Display the results
disp(['Mean: ', num2str(meanValueMDay)]);
disp(['Standard Deviation: ', num2str(stdValueMDay)]);
disp(['Variance: ', num2str(varianceValueMDay)]);