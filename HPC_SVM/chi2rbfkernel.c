#include "math.h"
#include "stdio.h"
#include "mex.h"  
#include "memory.h"
#define SMALLNUM 0.00001
double kernel_linear(double *f1,double *f2, int dim)
{

	int i;
	double s = 0;
	for(i = 0; i < dim; i++)
	{
		s += f1[i]*f2[i];
	}
	return s;
}

double kernel_chi2rbf(double *f1,double *f2, int dim, double alpha)
{

	int i;
	double s = 0;

	for(i = 0; i < dim; i++)
	{
		s += (f1[i]-f2[i])*(f1[i]-f2[i])/(f1[i]+f2[i]+SMALLNUM);
	}
	s = exp(-s/alpha);
	return s;
}

double kernel_chi2(double *f1,double *f2, int dim, double alpha)
{

	int i;
	double s = 0;

	for(i = 0; i < dim; i++)
	{
		s += (f1[i]-f2[i])*(f1[i]-f2[i])/(f1[i]+f2[i]+SMALLNUM);
	}

	return s;
}

double kernel_rbf(double *f1,double *f2, int dim, double alpha)
{

	int i;
	double s = 0;

	for(i = 0; i < dim; i++)
	{
		s += (f1[i]-f2[i])*(f1[i]-f2[i]);
	}

	return s;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/*Input Variables*/
	mxArray *data1_temp;
	int  samples1;
	int  dim1;
	double *data1;

	
	mxArray *data2_temp;
	int  samples2;
	int  dim2;
	double *data2;

	mxArray *alpha_temp;
	double alpha;

	/* Process Variable */
	int i,j,k;
	double *f1,*f2;
	/* Output Variable */
	double *KM;

	/* debug */
	double vdebug;

	/* Get Input */
	data1_temp = prhs[0];
	samples1   = mxGetN(data1_temp);
	dim1	   = mxGetM(data1_temp);
	data1	   = mxGetPr(data1_temp);

	data2_temp = prhs[1];
	samples2   = mxGetN(data2_temp);
	dim2	   = mxGetM(data2_temp);
	data2	   = mxGetPr(data2_temp);

	alpha_temp	= prhs[2];
	alpha		= mxGetScalar(alpha_temp);

	printf("%d\n",dim1);
	printf("%d\n",dim2);
	if(dim1 != dim2)
	{
		printf("the dimension of features in two datasets is not same");
		return;
	}
	/* get output */
	plhs[0] = mxCreateDoubleMatrix(samples1,samples2,mxREAL);
	KM		= mxGetPr(plhs[0]);
	



	printf("%d\n",samples1);
	printf("%d\n",samples2);

	/* ------------------------------ core algorithm -----------------------*/
	f1 = (double *)malloc(dim1*sizeof(double));
	f2 = (double *)malloc(dim1*sizeof(double));
	for(i = 0; i < samples1; i++)
	{

		for(j = 0; j < samples2; j++)
		{
			/*if(j == 118)
			{
				vdebug = 0;
			}*/
			for(k = 0; k < dim1; k++) f1[k] = data1[k + i*dim1];

			for(k = 0; k < dim1; k++) f2[k] = data2[k + j*dim2];
	
			KM[j*samples1+i] = kernel_chi2(f1,f2,dim1,alpha);
		}
	}
	free(f1);
	free(f2);
}
	
