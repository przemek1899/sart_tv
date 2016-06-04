/*

version of psf2otf for m = n

*/
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "helper_cuda.h"
#include <cufft.h>

template<typename T> __global__ void psf_from_fft(cufftDoubleComplex* v, int N, int n, T* real_fft, T* result);
template<typename T> __global__ void add_and_divide_cut_complex(T * Numer1, cufftDoubleCompex* fft2_rhs, T* Denom, int fft2_rows, int fft2_cols);

void check_cufft(cufftResult status);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){

	 /* Declare all variables.*/
    mxGPUArray *Numer1, *rhs, *Denom, *U;
    double *d_Numer1, *d_rhs, *d_Denom, *d_U;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs < 4) || !(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1])) || !(mxIsGPUArray(prhs[2]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    Numer1 = mxGPUCopyFromMxArray (prhs[0]); // mxGPUCreateFromMxArray(prhs[0]);
	rhs = mxGPUCopyFromMxArray (prhs[1]);
	Denom = mxGPUCopyFromMxArray (prhs[2]);
	int N = mxGetScalar(prhs[3]);

    /*
     * Verify that A really is a double array before extracting the pointer.
     */
    if ((mxGPUGetClassID(Numer1) != mxDOUBLE_CLASS) ||(mxGPUGetClassID(rhs) != mxDOUBLE_CLASS) || (mxGPUGetClassID(Denom) != mxDOUBLE_CLASS)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_Numer1 = (double *)(mxGPUGetData(Numer1));
	d_rhs = (double *)(mxGPUGetData(rhs));
	d_Denom = (double *)(mxGPUGetData(Denom));

    /* Create a GPUArray to hold the result and get its underlying pointer. */

    U = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(Numer1),
                            mxGPUGetDimensions(Numer1),
                            mxGPUGetClassID(Numer1),
                            mxGPUGetComplexity(Numer1),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_U = (double *)(mxGPUGetData(U));
	// ----------------- END OF MEX FILE STARTING CONFIGURATION ----------------------------

	//  --------------- REC BREGMAN METHODS STARTS HERE -------------------------------------------

	
	cufftHandle plan;


	cufftDoubleComplex *output;
	double prd[2] = {0.57, -0.57};
	double * input, *temp_worksapce;

	int n = (N/2)+1;

	//---------------------- MEMORY ALLOCATION
	checkCudaErrors(cudaMalloc((void**)&output, sizeof(cufftDoubleComplex)*n));
	cudaMalloc((void**)&input, sizeof(double)*N);
	cudaMemset(input, 0, sizeof(double)*N);
	cudaMalloc((void**)&temp_worksapce, sizeof(double)*N);

	cudaMemcpy(input, &prd[0], sizeof(double)*2, cudaMemcpyHostToDevice);


	// cufft plan initialization
	if (cufftPlan1d(&plan, N, CUFFT_D2Z, 1) != CUFFT_SUCCESS){	// N - number of input samples (length of input data)
		cudaFree(output);
		mexErrMsgIdAndTxt(errId, "plan initialization failed, cufft error code\n");
	}

	// exec
	if (cufftExecD2Z(plan, input, output) != CUFFT_SUCCESS){
		cudaFree(output);
		cufftDestroy(plan);		
		mexErrMsgIdAndTxt(errId, "cufft exec failed, cufft error code\n");
	}


	int threads = N;
	int blocks = N;
	int shared_mem_size = N*sizeof(double);
	psf_from_fft<<<threads, blocks, shared_mem_size>>>(output, N, n, temp_worksapce, d_U);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());		

	// -------------------------------------- U = IFFT2((Numer1 + FFT2(rhs))./Denom) --------------------------------------------

	// --------------------------------------- CUFFT PLAN INITIALIZATION -------------------------------

	cufftHandle plan2d, plan_ifft;
	int fft2_output_size = N*(N/2+1);
	cufftDoubleComplex* fft2_rhs;
	cudaMalloc((void**)&fft2_rhs, sizeof(cufftDoubleComplex)*fft2_output_size);

	if(cufftPlan2D(&plan2d, N, N, CUFFT_D2Z) != CUFFT_SUCCESS){
		mexErrMsgIdAndTxt(errId, "plan2d initialization failed, cufft error code\n");
	}
	if(cufftPlan2D(&plan_ifft, N, N, CUFFT_Z2D) != CUFFT_SUCCESS){
		mexErrMsgIdAndTxt(errId, "ifft plan initialization failed, cufft error code\n");
	}

	// --------------------------------------- FFT2 EXECUTION -------------------------------
	// ftt(rhs)
	if(cufftExecD2Z(plan2d, d_rhs, fft2_rhs) != CUFFT_SUCCESS){
		mexErrMsgIdAndTxt(errId, "cufft exec failed, cufft error code\n");
	}

	// ------------------------------------- (Numer1 + fft2(rhs))./Denom
	threads = N;
	blocks = N/2+1;
	add_and_divide_cut_complex<<<threads, block>>>(d_Numer1, fft2_rhs, d_Denom, N, N/2+1);

	// inverse fft ifft2((Numer1 + fft2(rhs))./Denom)
	if(cufftExecZ2D(plan_ifft, fft2_rhs, d_U) != CUFFT_SUCCESS){
		mexErrMsgIdAndTxt(errId, "cufft exec failed, cufft error code\n");
	}

	// END OF MY CODE

	/* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(U);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(Numer1);
	mxGPUDestroyGPUArray(rhs);
	mxGPUDestroyGPUArray(Denom);
    mxGPUDestroyGPUArray(U);

	cufftDestroy(plan);

	cudaFree(output);
	cudaFree(input);
	cudaFree(temp_worksapce);
	cudaFree(fft2_output_data);
}

template<typename T>
__global__ void psf_from_fft(cufftDoubleComplex* v, int N, int n, T* real_fft, T* result){

	// blocks are rows which cover the whole length, usually the matrix is of size 512x512, the NVIDIA GPU's allows maximum number
	// of threads per block of 1024
	
	if (threadIdx.x >= n && threadIdx.x < N){
		v[threadIdx.x] = v[N-threadIdx.x];
	}

	if (threadIdx.x < N){
		cufftDoubleComplex a = v[threadIdx.x];
		real_fft[threadIdx.x] = pow(a.x, 2.0) + pow(a.y, 2.0);
	}

	if (blockIdx.x < N && threadIdx.x < N){
		extern __shared__ double v_shared[];
		v_shared[threadIdx.x] = real_fft[threadIdx.x];

		result[threadIdx.x+blockIdx.x*N] = v_shared[threadIdx.x] + v_shared[blockIdx.x];
	}
}

template<typename T>
__global__ void add_and_divide_cut_complex(T * Numer1, cufftDoubleCompex* fft2_rhs, T* Denom, int fft2_rows, int fft2_cols){

	// IMPORTANT: Numer1 and Denom are matrices of size rows x cols, but fft2_rhs was computed by cufft in R2C plan so its total length is N*(N/2+1);
	 // operation: (Numer1 + fft2(rhs))./Denom
/*
Similar to the one-dimensional case, the frequency domain representation of real-valued
input data satisfies Hermitian symmetry, defined as: X(n1, n2, ..., nd) = X*(N1-n1,N2-n2,...,Nd-nd)
for two dimensional fft, i.e. fft2 on NxM matrix indexing is the following: X(n,m) = X*(N-n, M-m);
the length of fft2 done by cufft from NxM is: N*(M/2+1);

kernel run configuration should be fitted to this size N*(M/2+1)

*/

	if (threadIdx.x < fft2_rows && blockIdx.x < fft2_cols){

		int index = threadIdx.x + blockIdx.x*fft2_cols;
		cufftDoubleCompex t = fft_rhs[index];
		t.x += Numer1[index];
		
		// UWAGA CO JESLI JEST DZIELENIE PRZEZ ZERO??
		double divider = Denom[index];
		if (divider != 0.0){
			t.x /= divider;
			t.y /= divider;
			fft_rhs[index] = t;
		}
	}
}

void check_cufft(cufftResult status){
	
	char const * const errId = "parallel:gpu:mexGPUExample:CufftError";
	if (status != CUFFT_SUCCESS){
		mexErrMsgIdAndTxt(errId, "cufft error code %d\n", status);
	}
}