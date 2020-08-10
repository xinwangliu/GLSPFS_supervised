function [feaIndx,W,obj] = semiRFSLTSA(XL,Y,XU,Kmatrix,mu,lambda,options,numFea)

maxIter = 100;
flag =1;
iter = 0;
XL = XL';
XU = XU';
[dim,numLabeled] = size(XL);
[numUnlabeled] = size(XU,2);
X = [XL XU]; %% dim*(numLabeled+numUnlabeled)
ntrn = size(X,2);
%%%%%%%
L = ComputeMLTSA(X',Kmatrix,options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% XL = X*AL;
AL = [eye(numLabeled); zeros(numUnlabeled,numLabeled)];
U = AL*AL' + mu* L;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
klength = length(unique(Y));
YL = zeros(numLabeled,klength);
for i = 1:klength
    YL(Y==i,i) = 1;
end
YL = YL';

dvec = ones(dim,1);
objold = inf;
% dvec1 = ones(num,1);
while flag
    if (dim>ntrn)
        iter = iter +1;
        D = spdiags(dvec,0,dim,dim);%%d*d
        %% W = (XS'*XS + r*D)\(XS'*Y');%%d*n;
        DX = D*X;
        W = DX*((U*X'*DX + lambda*eye(ntrn))\(AL*YL'));
        
        Xi = sqrt(sum(W.*W,2));
        Xi1 = (W'*XL-YL).*(W'*XL-YL);
        Xi2 = W'*X*L*X'*W;
        obj(iter) = sum(sum(Xi1)) + mu*trace(Xi2) + lambda*sum(Xi);
        
        dvec = 2*Xi;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (iter > maxIter) || abs((objold-obj(iter))/obj(iter)) <1e-4
            flag = 0;
        end
        objold = obj(iter);
    else
        iter = iter +1;
        D = spdiags(dvec,0,dim,dim);%%d*d
        %% W = (XS'*XS + r*D)\(XS'*Y');%%d*n;
        DX = D*X;
        W = ((DX*T*X' + lambda*eye(dim))\DX)*AL*YL';
        
        Xi = sqrt(sum(W.*W,2));
        Xi1 = (W'*XL-YL).*(W'*XL-YL);
        Xi2 = W'*X*L*X'*W;
        obj(iter) = sum(sum(Xi1)) + mu*trace(Xi2) + lambda*sum(Xi);
        
        dvec = 2*Xi;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (iter > maxIter) || abs((objold-obj(iter))/obj(iter)) <1e-4
            flag = 0;
        end
        objold = obj(iter);
    end     
end
Xi0 = sqrt(sum(W.*W,2));
[val0, indx0] = sort(Xi0,'descend');
feaIndx = indx0(1:numFea);