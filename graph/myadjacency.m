function A = myadjacency(X,NN,Kmatrix)
% {adjacency} computes the graph adjacency matrix.
n=size(X,1);
p=2:(NN+1);


[Z,I]=sort(Kmatrix,2,'descend');

Z=Z(:,p)'; % it picks the neighbors from 2nd to NN+1th
I=I(:,p)'; % it picks the indices of neighbors from 2nd to NN+1th
idy = I(:);
DI  = Z(:);

I=repmat((1:n),[NN 1]);
I=I(:);

A=sparse(I,idy,DI,n,n);

A=A+((A~=A').*A'); % symmetrize