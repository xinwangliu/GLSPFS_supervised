function accuracy = SVM_OneVsAll_Test(Train_data, Test_data, Train_label, Test_label,SVM_C)
[N,D] = size(Train_data);
C     = max(Train_label);
% SVM_C = [-9:-4];

% Training
for c = 1:C
    pos_set = c;
    neg_set = [1:C];
    neg_set(c) = [];
    indexp = find(ismember(Train_label,pos_set));
    indexn = find(ismember(Train_label,neg_set));
    train_data = [Train_data(indexp,:);Train_data(indexn,:)];
    train_label = -ones(length(indexp)+length(indexn),1);
    train_label(1:length(indexp)) = 1; 
    model = BinarySVM(train_data,train_label,'linear',SVM_C);
    [w{c},b{c}] = GetWeightBinary(model,train_data);
end

% Test
result = zeros(C,size(Test_data,1));
for i = 1:C
    result(i,:) = (Test_data*w{i}' - b{i})';
end
[v,label] = max(result);
accuracy = sum(label' == Test_label)/length(Test_label);





function model = BinarySVM(train_x,train_y,kernel,CR)
% tic
trainsize = size(train_x,1);

if strcmp(kernel,'linear')
    K_Train = train_x*train_x';
end
if strcmp(kernel,'chi2')
    KM_Train_r = chi2rbfkernel((train_x)',(train_x)',1);
    alpha = 2*mean(KM_Train_r(:));
    K_Train = exp(-KM_Train_r/alpha);
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

