function [labels, accuracy,choose_c] = FastBestSVM_Test_kernel_largescale(train_x,train_y,test_x,test_y,kernel,CR)

% tic
trainsize = train_x.S;
testsize  = test_x.S;
if strcmp(kernel,'linear')
    tic
    K_Train = sl_segment_multiply_sstm(train_x,[]);
    toc
    tic
	K_Test  = sl_segment_multiply_sstm(test_x,train_x);
    toc
end

   
KTrain = [[1:trainsize]',K_Train];
KTest = [[1:testsize]',K_Test];

rmax = 0;
N = 10;
for c = CR
    option = ['-q -v ',int2str(N),' -t 4 -c ',num2str(2^c)];
    if size(train_y,1) == 1
        train_y = train_y';
    end
    r = svmtrain(train_y, KTrain,option);
    if r > rmax
        rmax = r;
        choose_c = c;
    end
end
option = ['-q -t 4 -c ',num2str(2^choose_c)];
model = svmtrain(train_y, KTrain,option);
clear KTrain
clear train_x
clear test_x
clear K_Test
clear K_Train
if size(test_y,1) == 1
	test_y = test_y';
end
[labels, accuracy] = svmpredict(test_y,KTest,model);
