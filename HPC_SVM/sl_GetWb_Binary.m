function [w,b] = sl_GetWb_Binary(model,prefix,train_index)
SV_index = model.SVs;
for i = 1:length(SV_index)
    clear data
    index = train_index(SV_index(i));
    load([prefix,int2str(index),'.mat'],'data')
    if i == 1
        w = data*model.sv_coef(i);
    else
        w = w + data*model.sv_coef(i);
    end
end
b = -model.rho;
if model.Label(1) == -1
  w = -w;
  b = -b;
end

