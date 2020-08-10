% This make.m is used under Windows

% add -largeArrayDims on 64-bit machines

mex -O -largeArrayDims -c svm.cpp 
mex -O -largeArrayDims -c svm_model_matlab.c
mex -O svmtrain.c svm.o svm_model_matlab.o
mex -O svmpredict.c svm.o svm_model_matlab.o

