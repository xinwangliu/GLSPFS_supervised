function [labels, accuracy,choose_c, model] = SVM_Test_LOOV(train_x,train_y,test_x,test_y,kernel,CR)
% tic
trainsize = size(train_x,1);
testsize  = size(test_x,1);
switch kernel
    case 'linear'
        K_Train = full(train_x*train_x');
        K_Test  = full(test_x*train_x');
    case 'chi2'
        KM_Train_r = chi2rbfkernel((train_x)',(train_x)',1);
        KM_Test_r  = chi2rbfkernel((test_x)',(train_x)',1);
        alpha = 2*mean(KM_Train_r(:));
        K_Train = exp(-KM_Train_r/alpha);
        K_Test = exp(-KM_Test_r/alpha);
    case 'user'
        K_Train = train_x;
        K_Test = test_x;
end
   
KTrain = [[1:trainsize]',K_Train];
KTest = [[1:testsize]',K_Test];

rmax = 0;
N = 10;
for c = CR
    option = ['-q -v ',int2str(N),' -t 4 -c ',num2str(2^c)];
    r = svmtrain(train_y, KTrain,option);
    if r > rmax
        rmax = r;
        choose_c = c;
    end
end
option = ['-q -t 4 -c ',num2str(2^choose_c)];
model = svmtrain(train_y, KTrain,option);

[labels, accuracy] = svmpredict(test_y,KTest,model);
