/*

version of psf2otf for m = n

*/
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "helper_cuda.h"
#include <cufft.h>

template <typename T> __global__ void copyRealFromComplex(cufftDoubleComplex* C, T* R, int n);
void check_cufft(cufftResult status);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){

    mxGPUArray *A, *B;
    double *d_A, *d_B;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

    if ((nrhs < 2) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    A = mxGPUCopyFromMxArray (prhs[0]); // mxGPUCreateFromMxArray(prhs[0]);
	int N = mxGetScalar(prhs[1]);

    if ((mxGPUGetClassID(A) != mxDOUBLE_CLASS)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    d_A = (double *)(mxGPUGetData(A));

    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_INITIALIZE_VALUES );
    d_B = (double *)(mxGPUGetData(B));
	// ----------------- END OF MEX FILE STARTING CONFIGURATION ----------------------------

	cufftHandle plan, plan_inverse;
	cufftDoubleComplex *output;
	int n = (N/2)+1;
	checkCudaErrors(cudaMalloc((void**)&output, sizeof(cufftDoubleComplex)*n));


	// cufft plan initialization
	if (cufftPlan1d(&plan, N, CUFFT_D2Z, 1) != CUFFT_SUCCESS){	// N - number of input samples (length of input data)
		cudaFree(output);
		mexErrMsgIdAndTxt(errId, "plan initialization failed, cufft error code\n");
	}

	if (cufftPlan1d(&plan_inverse, N, CUFFT_Z2D, 1) != CUFFT_SUCCESS){
		mexErrMsgIdAndTxt(errId, "inverse plan initialization failed, cufft error code\n");
	}

	// exec
	if (cufftExecD2Z(plan, d_A, output) != CUFFT_SUCCESS){
		cudaFree(output);
		cufftDestroy(plan);		
		mexErrMsgIdAndTxt(errId, "cufft exec failed, cufft error code\n");
	}

	
	if (cufftExecZ2D(plan_inverse, output, d_B) != CUFFT_SUCCESS){
		mexErrMsgIdAndTxt(errId, "cufft exec failed, cufft error code\n");
	}


	int threads = N;
	int blocks = N;
	//copyRealFromComplex<<<threads, blocks>>>(output, d_B, n);
	//cudaDeviceSynchronize();
	//checkCudaErrors(cudaGetLastError());		

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);

    mxGPUDestroyGPUArray(A);
	mxGPUDestroyGPUArray(B);

	cufftDestroy(plan);
	cudaFree(output);
}

template <typename T> __global__ void copyRealFromComplex(cufftDoubleComplex* C, T* R, int n){

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index < n){
		R[index] = C[index].x;
	}
}

void check_cufft(cufftResult status){
	
	char const * const errId = "parallel:gpu:mexGPUExample:CufftError";
	if (status != CUFFT_SUCCESS){
		mexErrMsgIdAndTxt(errId, "cufft error code %d\n", status);
	}
}