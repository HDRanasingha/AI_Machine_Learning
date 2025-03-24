% Load Mday combined data
datacombineMDay = load("U04_Acc_TimeD_FreqD_MDay.mat");

% Display the variables
disp(fieldnames(datacombineMDay));

multipleDayData = datacombineMDay.Acc_TDFD_Feat_Vec;  

% Convert to table 
%tMDay = array2table(multiDayData);

% mean,stdv,veriance calculate
meanValueMDay = mean(multipleDayData, 1); 
stdValueMDay = std(multipleDayData, 0, 1); 
varianceValueMDay = var(multipleDayData, 0, 1); 

% display mean,stdv and varience
disp(['Mean: ', num2str(meanValueMDay)]);
disp(['Standard Deviation: ', num2str(stdValueMDay)]);
disp(['Variance: ', num2str(varianceValueMDay)]);