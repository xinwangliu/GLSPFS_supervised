function [w,b] = GetWeightBinary(model,train_x)
    SVs     = full(model.SVs);
    SVCoef  = model.sv_coef;
    [num,dim] = size(SVs);
    w = zeros(1,dim);
    for i = 1:size(SVs,1)
        if size(SVs(i,:),2) == 1
            h = train_x(SVs(i),:);
        else
            h = SVs(i,:);
        end
        w = w + SVCoef(i,:)'*double(h);
    end
    w = model.Label(1)*w;
    b = model.Label(1)*model.rho;