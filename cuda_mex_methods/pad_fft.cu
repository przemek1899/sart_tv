
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "helper_cuda.h"
#include <cufft.h>

__global__ void fill_after_fft(cufftDoubleComplex* v, int N, int n, double * real_fft);
__global__ void psf_from_fft(cufftDoubleComplex* vn, cufftDoubleComplex* vm, int N, int n, int M, int m, double * real_fft);

void check_cufft(cufftResult status);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){

	 /* Declare all variables.*/
    mxGPUArray *A;
    mxGPUArray *B;
    double *d_A;
    double *d_B;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs < 3) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    A = mxGPUCopyFromMxArray (prhs[0]); // mxGPUCreateFromMxArray(prhs[0]);
	int N = mxGetScalar(prhs[1]);
	int M = mxGetScalar(prhs[2]);

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
    d_A = (double *)(mxGPUGetData(A));

    /* Create a GPUArray to hold the result and get its underlying pointer. */
	mwSize B_dims[2];
	B_dims[0] = M;
	B_dims[1] = N;
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            B_dims,//mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_B = (double *)(mxGPUGetData(B));
	// ----------------- END OF MEX FILE STARTING CONFIGURATION ----------------------------

	//  --------------- REC BREGMAN METHODS STARTS HERE -------------------------------------------
	cufftHandle plan, plan1d_m;
	cufftDoubleComplex *output, *fft1d_m;
	double prd[2] = {0.57, -0.57};
	double * input_n, *input_m;

	int n = (N/2)+1;
	//int m_fft = (M/2)+1;

	//---------------------- MEMORY ALLOCATION
	checkCudaErrors(cudaMalloc((void**)&output, sizeof(cufftDoubleComplex)*n));
	checkCudaErrors(cudaMalloc((void**)&fft1d_m, sizeof(cufftDoubleComplex)*m_fft);

	cudaMalloc((void**)&input_n, sizeof(double)*N);
	cudaMemset(input_n, 0, sizeof(double)*N);

	cudaMalloc((void**)&input_m, sizeof(double)*M);
	cudaMemset(input_m, 0, sizeof(double)*M);

	mexPrintf("po memset\n");
	// SET prd
	cudaMemcpy(input_n, &prd[0], sizeof(double)*2, cudaMemcpyHostToDevice);
	cudaMemcpy(input_m, &prd[0], sizeof(double)*2, cudaMemcpyHostToDevice);
	mexPrintf("po 0 i 1\n");

	// cufft plan initialization
	if (cufftPlan1d(&plan, N, CUFFT_D2Z, 1) != CUFFT_SUCCESS){	// N - number of input samples (length of input data)
		cudaFree(output);
		mexErrMsgIdAndTxt(errId, "plan initialization failed, cufft error code\n");
	}
	if (cufftPlan1d(&plan1d_m, M, CUFFT_D2Z, 1) != CUFFT_SUCCESS){	// N - number of input samples (length of input data)
		cudaFree(output);
		mexErrMsgIdAndTxt(errId, "plan initialization failed, cufft error code\n");
	}

	// exec
	if (cufftExecD2Z(plan, input_n, output) != CUFFT_SUCCESS){
		cudaFree(output);
		cufftDestroy(plan);		
		mexErrMsgIdAndTxt(errId, "cufft exec failed, cufft error code\n");
	}
	if (cufftExecD2Z(plan1d_m, input_m, fft1d_m) != CUFFT_SUCCESS){
		cudaFree(fft1d_m);
		cufftDestroy(plan1d_m);		
		mexErrMsgIdAndTxt(errId, "cufft exec failed, cufft error code\n");
	}


	int threads = 256;
	int blocks = 32;
	fill_after_fft<<<threads, blocks>>>(output, N, n, d_B);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());		
	// END OF MY CODE

	/* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);

	cufftDestroy(plan);
	cufftDestroy(plan1d_m);

	cudaFree(output);
	cudaFree(input_n);
	cudaFree(fft1d_m);
}

__global__ void fill_after_fft(cufftDoubleComplex* v, int N, int n, double * real_fft){

	int index = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (index >= n && index < N){
		v[index] = v[N-index];
	}

	if (index < N){
		cufftDoubleComplex a = v[index];
		//real_fft[index] = pow(a.x, 2.0) + pow(a.y, 2.0);
		v[index] = pow(a.x, 2.0) + pow(a.y, 2.0);
	}
}

__global__ void psf_from_fft(cufftDoubleComplex* vn, cufftDoubleComplex* vm, int N, int n, int M, int m, double * real_fft){

	int index = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (index >= n && index < N){
		v[index] = v[N-index];
	}

	if (index < N){
		cufftDoubleComplex a = v[index];
		//real_fft[index] = pow(a.x, 2.0) + pow(a.y, 2.0);
		v[index] = pow(a.x, 2.0) + pow(a.y, 2.0);
	}

	// now comes part of merging to vectors into matrix
	// in this point we assume that vectors vm and vn are already normalized to pow
	__shared__ double[M] vm_shared;
	__shared__ double[N] vn_shared;

	vm_shared[some_m_index] = vm[some_m_index];
	vn_shared[some_n_index] = vn[some_n_index];
}


void check_cufft(cufftResult status){
	
	char const * const errId = "parallel:gpu:mexGPUExample:CufftError";
	if (status != CUFFT_SUCCESS){
		mexErrMsgIdAndTxt(errId, "cufft error code %d\n", status);
	}
}