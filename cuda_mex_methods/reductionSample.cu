

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"

/*
 * Device code
 */
void __global__ TimesTwo(double const * const A,
                         double * const B,
                         int const N)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        B[i] = 2.0 * A[i];
    }
}

void __global__ reduction1(double const *A, int m, int n, double *result, int rm, int rn){
	
	extern __shared__ double sh_data[];

	//each thread loads one element from global memory
	unsigned int tid = threadIdx.x;
	sh_data[tid] = 0;
	unsigned int global_row_index = blockIdx.x * blockDim.x + threadIdx.x;
	// zakladamy, ze blockDim.y = 1
	unsigned int data_index = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y*m;
	if (global_row_index < m && blockIdx.y < n){
		sh_data[tid] = A[data_index];
	}
	__syncthreads();

	//do reduction in shared memory
	for(unsigned int s=1; s<blockDim.x; s*=2){
		if (tid % (s*2) == 0){
			sh_data[tid] += sh_data[tid+s];
		}
		__syncthreads();
	}

	//write results for this block to global memory
	if (blockIdx.x < rm && blockIdx.y < rn){
		if (tid==0){
			result[blockIdx.x + blockIdx.y*rm] = sh_data[0];
		}
	}
}

void __global__ reduction2(double const *A, int m, int n, double *result, int rm, int rn){

	extern __shared__ double sh_data[];

	//each thread loads one element from global memory
	unsigned int tid = threadIdx.x;
	sh_data[tid] = 0;
	unsigned int global_row_index = blockIdx.x * blockDim.x + threadIdx.x;
	// zakladamy, ze blockDim.y = 1
	unsigned int data_index = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y*m;
	if (global_row_index < m && blockIdx.y < n){
		sh_data[tid] = A[data_index];
	}
	__syncthreads();

	//do reduction in shared memory
	for(unsigned int s=1; s<blockDim.x; s*=2){
		int index = 2*s*tid;
		if (index < blockDim.x){
			sh_data[index] = sh_data[index+s];
		}
		__syncthreads();
	}

	//write results for this block to global memory
	if (blockIdx.x < rm && blockIdx.y < rn){
		if (tid==0){
			result[blockIdx.x + blockIdx.y*rm] = sh_data[0];
		}
	}
}

void __global__ reduction3(double const *A, int m, int n, double *result, int rm, int rn){

	extern __shared__ double sh_data[];

	//each thread loads one element from global memory
	unsigned int tid = threadIdx.x;
	sh_data[tid] = 0;
	unsigned int global_row_index = blockIdx.x * blockDim.x + threadIdx.x;
	// zakladamy, ze blockDim.y = 1
	unsigned int data_index = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y*m;
	if (global_row_index < m && blockIdx.y < n){
		sh_data[tid] = A[data_index];
	}
	__syncthreads();

	// do reduction in shared memory
	// SEQUENTIAL ADDRESSING (CONFLICT FREE)
	for(unsigned int s=blockDim.x/2; s>0; s>>=1){
		if (tid < s){
			sh_data[tid] += sh_data[tid+s];
		}
		__syncthreads();
	}

	//write results for this block to global memory
	if (blockIdx.x < rm && blockIdx.y < rn){
		if (tid==0){
			result[blockIdx.x + blockIdx.y*rm] = sh_data[0];
		}
	}
}

void __global__ reduction4(double const *A, int m, int n, double *result, int rm, int rn){
	/*
		4th version - First Add During Load
		HALVE THE NUMBER OF THE BLOCKS !!!!!
	*/

	extern __shared__ double sh_data[];

	//each thread loads one element from global memory
	unsigned int tid = threadIdx.x;
	sh_data[tid] = 0;
	unsigned int global_row_index = blockIdx.x * blockDim.x + threadIdx.x;
	// zakladamy, ze blockDim.y = 1
	unsigned int data_index = (blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y*m;
	if (global_row_index < m && blockIdx.y < n){
		sh_data[tid] = A[data_index];
	}
	__syncthreads();

	// do reduction in shared memory
	//TODO

	//write results for this block to global memory
	if (blockIdx.x < rm && blockIdx.y < rn){
		if (tid==0){
			result[blockIdx.x + blockIdx.y*rm] = sh_data[0];
		}
	}
}

/*
 * Host code
 */

int getRound(int m, int n){

	if (m % n == 0)
		return m;
	else
		return (m/n) * n + n;
}


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
	
    /* Choose a reasonably sized number of threads for the block. */
    int threadsPerBlock = 256;
	int rm = getRound(m, threadsPerBlock)/threadsPerBlock;
    int blocksPerGrid = getRound(rm, 32);
	dim3 gridDim(32, 32*128);
	int shared_mem_size = threadsPerBlock*sizeof(double);
	
	double *d_AB;
	cudaMalloc(&d_AB, sizeof(double)*rm*n);
	
	mwSize B_dims[2];
	B_dims[0] = 1;
	B_dims[1] = n;

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            B_dims,//mxGPUGetDimensions(A),
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_B = (double *)(mxGPUGetData(B));

    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
	reduction1<<<gridDim, threadsPerBlock, shared_mem_size>>>(d_A, m, n, d_AB, rm, n);
    cudaDeviceSynchronize();

	threadsPerBlock = getRound(rm, 32);
	shared_mem_size = threadsPerBlock*sizeof(double);
	gridDim.x = 1;
	reduction1<<<gridDim, threadsPerBlock, shared_mem_size>>>(d_AB, rm, n, d_B, 1, n);
	cudaDeviceSynchronize();

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);

	cudaFree(d_AB);
    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
}
