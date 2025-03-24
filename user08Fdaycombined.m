% Load user8 combined Fday
datacombineFDay = load("U08_Acc_TimeD_FreqD_FDay.mat");

% Display the variables
disp(fieldnames(datacombineFDay));

FIRSTDayData = datacombineFDay.Acc_TDFD_Feat_Vec;  

% Convert to table 
%tMDay = array2table(firstDayData);

% calculate mean,std,verience
meanValueMDay = mean(FIRSTDayData, 1); 
stdValueMDay = std(FIRSTDayData, 0, 1); 
varianceValueMDay = var(FIRSTDayData, 0, 1); 

% Display 

disp(['Mean: ', num2str(meanValueMDay)]);
disp(['Standard Deviation: ', num2str(stdValueMDay)]);
disp(['Variance: ', num2str(varianceValueMDay)]);