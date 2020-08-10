clear;
clc;
path = '/home/xliu/FeatureSelectionToolBox/';
addpath(genpath(path));

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rset1 = 10.^[-3:1:2];
rset2 = 10.^[-3:1:2];
cset1 = 10.^[-3:1:3];
CR = [-5:2:15];
kfold = 5;
kernel = 'gaussian';
kerneloptions = 0;
dataName = 'ALLAML'; %%%  CLL-SUB-111; TOX-171; warpAR10P; pixraw10P; warpPIE10P; orlraws10P; SMK-CAN-187
%%% ALLAML LUNG; GLIOMA; Carcinom; Prostate-GE
numFea0 = 200;
nn = 20;

ksize = [1:2:21];

load([path,'supervisedFSalgorithms/20groupdata/',dataName,'_data.mat'],'X','Y');
load([path,'supervisedFSalgorithms/20groupdata/',dataName,'_partition.mat'],'Partition');

for idata = 1 : 1 %%%
    trnX = X(Partition(:,idata)==1,:);
    tstX = X(Partition(:,idata)==-1,:);
    trnY = Y(Partition(:,idata)==1);
    tstY = Y(Partition(:,idata)==-1);
    
    nclass = length(unique(tstY));
    rangY = 1:nclass;
    setEm = [];
    for di = 1:size(trnX,2)
        tmp = trnX(:,di);
        stdV = std(tmp);
        if stdV<1e-8;
            setEm = [setEm,di];
        end
    end
    valIndx = setdiff(1:size(trnX,2),setEm);
    trnX = trnX(:,valIndx);
    tstX = tstX(:,valIndx);
    
    for di = 1:size(trnX,2)
        tmp = trnX(:,di);
        meanV = mean(tmp);
        stdV = std(tmp);
        trnX(:,di) = (tmp -meanV)/stdV;
        tstX(:,di) = (tstX(:,di) -meanV)/stdV;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Kmatrix = mysvmkernel(trnX,trnX,kernel,kerneloptions);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    knnACC1 = zeros(nn,1);
    fea1 = cell(length(ksize),1);
    k = 7; % need to be carefully tuned.
    [optimalR11,optimalR12,acc] = cvLPSPFSSVM(trnX,trnY,Kmatrix,rset1,rset2,kfold,k,numFea0,CR);
    [feaIndx1,W1,obj1] = LPSPFS(trnX,optimalR11(1),optimalR12(1),trnY,Kmatrix,ksize(ki),numFea0);
     for feaI = 1:nn
            numFea = 1:feaI*10;
            %%%%%%%% Cross Validation
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            trnX1 = trnX(:,feaIndx1(numFea));
            tstX1 = tstX(:,feaIndx1(numFea));
            [labels1, accuracy1,choose_c1, model1] = SVM_Test_LOOV(trnX1,trnY,tstX1,tstY,'linear',CR);
            knnACC1(feaI) = accuracy1(1);
        end
end
