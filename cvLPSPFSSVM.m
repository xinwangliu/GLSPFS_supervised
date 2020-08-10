function [optimalR1,optimalR2,acc] = cvLPSPFSSVM(X,Y,Ker,rset1,rset2,kfold,options,numFea,CR)
n = size(X,1);
numTest = round(n/kfold);
indx = randperm(n);

acc = zeros(length(rset1),length(rset2));
for rindx1 = 1:length(rset1)
    r1 = rset1(rindx1);
    for rindx2 = 1:length(rset2)
        r2 = rset2(rindx2);
        knnACC1 = zeros(kfold,1);
        for p =1:kfold
            tstIndx = indx((p-1)*numTest+1:min(p*numTest,n));
            trnIndx = setdiff(indx,tstIndx);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            tstX = X(tstIndx,:);
            tstY = Y(tstIndx,:);
            trnX = X(trnIndx,:);
            trnY = Y(trnIndx,:);
            Ktrntrn = Ker(trnIndx,trnIndx);
            [feaIndx1] = LPSPFS(trnX,r1,r2,trnY,Ktrntrn,options,numFea);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            trnX1 = trnX(:,feaIndx1);
            tstX1 = tstX(:,feaIndx1);
            kernel = 'linear';
            [labels] = SVM_Test_LOOV(trnX1,trnY,tstX1,tstY,kernel,CR);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            knnACC1(p) = mean(labels==tstY);
        end
        acc(rindx1,rindx2) = mean(knnACC1);
    end
end
[optimalRindx1, optimalRindx2]= findIndex(acc);
optimalR1 = rset1(optimalRindx1(end));
optimalR2 = rset2(optimalRindx2(end));