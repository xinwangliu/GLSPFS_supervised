function [x1,x2,maxValue]= findIndex(A)

[n,m]=size(A);
if n==1
    x1=1;
    [maxValue,x2]=max(A);
end
if m==1
    x2=1;
    [maxValue,x1]=max(A);
end
if n>1 && m>1
    B=reshape(A,m*n,1);
    [maxValue,maxIndex]=max(B);
    t=mod(maxIndex,n);
    if t==0
        x1=n;
    else
        x1=t;
    end
    x2=(maxIndex-x1)/n +1;
end