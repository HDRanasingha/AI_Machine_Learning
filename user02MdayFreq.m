% load user02 Mday frequncy domain.
dataMuDay = load("U02_Acc_FreqD_MDay.mat");

% Display the variables in the loaded .mat file
disp(fieldnames(dataMuDay));

secondDayData = dataMuDay.Acc_FD_Feat_Vec;

% calculate mean,std,varience
meanValue = mean(secondDayData, 1); 
stdValue = std(secondDayData, 0, 1); 
varianceValue = var(secondDayData, 0, 1);

% Display the stdv,varience,mean

disp(['Mean: ', num2str(meanValue)]);
disp(['Standard Deviation: ', num2str(stdValue)]);
disp(['Variance: ', num2str(varianceValue)]);
