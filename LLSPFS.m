function [feaIndx,W,obj] = LLSPFS(X,r1,r2,Y,Kmatrix,options,numFea) %% 
[num, dim] = size(X);
d = ones(dim,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L = computeM(X,Kmatrix,options);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nclass = length(unique(Y));
YL = zeros(num,nclass);
for ii = 1:num
    for c = 1:nclass
        if(Y(ii)==c)
            YL(ii,c) = 1;
        end
    end
end

NIter = 100;
flag =1;
objold = inf;
iter = 0;
if num<dim    
    while flag
        iter = iter +1;
        D = spdiags(d,0,dim,dim);
        DX = D*X';
        %%% Notive trick!!!
        W = DX*(((eye(num)+ r2*num*L)*X*DX + r1*num*eye(num))\YL);
        Xi = sqrt(sum(W.*W,2));
        d = 2*Xi;     
        XW = X*W -YL;
        obj(iter) = trace(XW*XW')/num + r2*trace(L*((X*W)*(X*W)')) + r1*sum(Xi);        
        if abs((objold-obj(iter))/obj(iter)) <1e-4 || iter>NIter
            flag = 0;
        end
        objold = obj(iter);
    end
else
    while flag
        iter = iter +1;
        D = spdiags(d,0,dim,dim);
        DX = D*X';
        %%% Notive trick!!!
        W = (DX*(eye(num)+ num*r2*L)*X + r1*num*eye(dim) )\(DX*YL);     
        Xi = sqrt(sum(W.*W,2));
        d = 2*Xi;    
        XW = X*W -YL;
        obj(iter) = trace(XW*XW')/num + r2*trace(L*((X*W)*(X*W)')) + r1*sum(Xi);        
        if abs((objold-obj(iter))/obj(iter)) <1e-4 || iter>NIter
            flag = 0;
        end
        objold = obj(iter);
    end  
end
Xi = sqrt(sum(W.*W,2));
[val0,indx0] = sort(Xi,'descend');
feaIndx = indx0(1:numFea);