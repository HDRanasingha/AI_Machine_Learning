
% load matfile
dataFirstDay = load("U02_Acc_FreqD_FDay.mat");

% variable display
disp(fieldnames(dataFirstDay));

oneDayData = dataFirstDay.Acc_FD_Feat_Vec;

% Calculate mean,stdvalue,varience
meanValue = mean(oneDayData, 1); 
stdValue = std(oneDayData, 0, 1); 
varianceValue = var(oneDayData, 0, 1); 

% Display the results

disp(['Mean: ', num2str(meanValue)]);
disp(['Standard Deviation: ', num2str(stdValue)]);
disp(['Variance: ', num2str(varianceValue)]);