/*

version of psf2otf for m = n

*/
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "helper_cuda.h"

__global__ void pow_vs_xx(double * A, double *B, double *C, int N);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){

	 /* Declare all variables.*/
    mxGPUArray *A, *B, *C;
    double *d_A, *d_B, *d_C;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs < 2) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    A = mxGPUCopyFromMxArray (prhs[0]); // mxGPUCreateFromMxArray(prhs[0]);
	int N = mxGetScalar(prhs[1]);

    /*
     * Verify that A really is a double array before extracting the pointer.
     */
    if ((mxGPUGetClassID(A) != mxDOUBLE_CLASS)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_A = (double *)(mxGPUGetData(A));

    /* Create a GPUArray to hold the result and get its underlying pointer. */

    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_INITIALIZE_VALUES );
    d_B = (double *)(mxGPUGetData(B));

	C = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_INITIALIZE_VALUES );
    d_C = (double *)(mxGPUGetData(C));
	// ----------------- END OF MEX FILE STARTING CONFIGURATION ----------------------------

	int threads = N;
	int blocks = 1;
	pow_vs_xx<<<blocks, threads>>>(d_A, d_B, d_C, N);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());		

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);
	plhs[1] = mxGPUCreateMxArrayOnGPU(C);

    mxGPUDestroyGPUArray(A);
	mxGPUDestroyGPUArray(B);
	mxGPUDestroyGPUArray(C);

}

__global__ void pow_vs_xx(double * A, double *B, double *C, int N){


	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x){
		
		double a = A[i];
		double p = a;
		double xx = a;

		for (int j=0; j < 5;  j++){
			p = pow(p, 2.0);
			xx = xx*xx;
		}
		B[i] = p;
		C[i] = xx;
	}
}
