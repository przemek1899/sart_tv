
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))


void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const *A;
    mxGPUArray *B;
    double const *d_A;
    double *d_B;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";
	char const * const cublasErrId = "Cublas error";

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs < 3) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    A = mxGPUCreateFromMxArray(prhs[0]);
	int m = mxGetScalar(prhs[1]);
	int n = mxGetScalar(prhs[2]);

    /*
     * Verify that A really is a double array before extracting the pointer.
     */
    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_A = (double const *)(mxGPUGetDataReadOnly(A));



	// ------------------- CUBLAS STARTS --------------------------------
	// ------------------- CUBLAS STARTS --------------------------------

	cublasStatus_t cublas_status;
	cublasHandle_t cublas_handle;
	cublas_status = cublasCreate(&cublas_handle);
	if (cublas_status != CUBLAS_STATUS_SUCCESS){
		mexErrMsgIdAndTxt(cublasErrId, "error while creating handle, error code %d\n", cublas_status);
	}

	double * sum_result = (double*) malloc(sizeof(double));
	cublas_status = cublasDasum(cublas_handle, m*n, d_A, 1, sum_result);
	
	cudaDeviceSynchronize();
	mexPrintf("total sum is %f\n", *sum_result);
	cublas_status = cublasDestroy(cublas_handle);
	
	free(sum_result);

	// ------------------- CUBLAS ENDS --------------------------------
	// ------------------- CUBLAS ENDS --------------------------------

	/* Create a GPUArray to hold the result and get its underlying pointer. */
	mwSize B_num_dim = 1;
	mwSize B_dims[1];
	B_dims[0] = 1;
    B = mxGPUCreateGPUArray(B_num_dim, B_dims, mxGPUGetClassID(A), mxGPUGetComplexity(A), MX_GPU_INITIALIZE_VALUES);
    d_B = (double *)(mxGPUGetData(B));
	// typedef enum mxComplexity {mxREAL=0, mxCOMPLEX};
	
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);

}
