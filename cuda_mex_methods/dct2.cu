#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "cublas_v2.h"
#include "helper_cuda.h"

#define BLOCK_THREAD_DIM 32
__global__ void generate_dct_matrix_coefficients(double *A, double *AT, double N);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){

	 /* Declare all variables.*/
    mxGPUArray *imageArray, *result;
    double *d_imageArray, *d_result;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    mxInitGPU();

    if ((nrhs < 2) || !(mxIsGPUArray(prhs[0])) ) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    imageArray = mxGPUCopyFromMxArray(prhs[0]);
	int N = mxGetScalar(prhs[1]);

/*
    if ((mxGPUGetClassID(imageArray) != mxDOUBLE_CLASS) ) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }*/

    d_imageArray = (double *)(mxGPUGetData(imageArray));

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    result = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(imageArray),
                            mxGPUGetDimensions(imageArray),
                            mxGPUGetClassID(imageArray),
                            mxGPUGetComplexity(imageArray),
                            MX_GPU_INITIALIZE_VALUES );
    d_result = (double *)(mxGPUGetData(result));

	double * d_A, *d_AT, *d_C;
	cudaMalloc((void**)&d_A, N*N*sizeof(double));
	cudaMalloc((void**)&d_AT, N*N*sizeof(double));
	cudaMalloc((void**)&d_C, N*N*sizeof(double));

	const double C_beta = 0.0;
	const double alpha = 1.0;
	
	// YOUR CODE HERE
	cudaError_t status;
	cublasStatus_t cublas_status;

	// ---------------------------------- CUBLAS initialization ---------------------------------------
	cublasHandle_t cublas_handle;
	cublasCreate(&cublas_handle);

	dim3 blocks(BLOCK_THREAD_DIM, BLOCK_THREAD_DIM);
	dim3 grid(N / BLOCK_THREAD_DIM, N / BLOCK_THREAD_DIM);

	generate_dct_matrix_coefficients<<<grid, blocks>>>(d_A, d_AT, N);

	cudaDeviceSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){
		mexErrMsgIdAndTxt(errId, "cuda error code %d\n", status);
	}

	// dct2: AT*X*A
	cublas_status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_AT, N, d_imageArray, N, &C_beta, d_C, N);
	if(cublas_status != CUBLAS_STATUS_SUCCESS){
		mexErrMsgIdAndTxt(errId, "cublas error code %d\n", cublas_status);
	}
	cublas_status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_C, N, d_A, N, &C_beta, d_result, N);
	if(cublas_status != CUBLAS_STATUS_SUCCESS){
		mexErrMsgIdAndTxt(errId, "cublas error code %d\n", cublas_status);
	}


	// idct2: A*X*AT
	cublas_status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_result, N, &C_beta, d_C, N);
	if(cublas_status != CUBLAS_STATUS_SUCCESS){
		mexErrMsgIdAndTxt(errId, "cublas error code %d\n", cublas_status);
	}
	cublas_status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_C, N, d_AT, N, &C_beta, d_result, N);
	if(cublas_status != CUBLAS_STATUS_SUCCESS){
		mexErrMsgIdAndTxt(errId, "cublas error code %d\n", cublas_status);
	}

	cudaFree(d_A);
	cudaFree(d_AT);
	cudaFree(d_C);

	cublasDestroy(cublas_handle);

    plhs[0] = mxGPUCreateMxArrayOnGPU(result);
    mxGPUDestroyGPUArray(imageArray);
    mxGPUDestroyGPUArray(result);
}


__global__ void generate_dct_matrix_coefficients(double *A, double *AT, double N){

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	
	double lx = 1.0 + (1.0)*(x>0);
	double ly = 1.0 + (1.0)*(y>0);
	// row major order
	// A[x + y*N] = cospi((2*x+1)*y/(2*N));
	int n = N;

	// column major order
	AT[x + y*n] = sqrt(lx/N) * cospi((2.0*y+1.0)*x/(2.0*N));
	A[x + y*n] = sqrt(ly/N) * cospi((2.0*x+1.0)*y/(2.0*N));

}