function [model,para] = SVM_Train_kernel(train_x,train_y,kernel,CR)
% tic
trainsize = size(train_x,1);
para = [];
if strcmp(kernel,'linear')
    K_Train = train_x*train_x';
end
if strcmp(kernel,'chi2')
    KM_Train_r = chi2rbfkernel((train_x)',(train_x)',1);
    alpha = 2*mean(KM_Train_r(:));
    K_Train = exp(-KM_Train_r/alpha);
    para.alpha = alpha;
end
   
KTrain = [[1:trainsize]',K_Train];


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
para.choose_c = choose_c;
% [labels, accuracy] = svmpredict(test_y,KTest,model);
