
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"

#define TILE_DIM 32
#define BLOCK_ROWS 4
/*
wydaje sie ze klopot sprawia wymiar macierzy
n = 2500;
przy BLOCK_ROWS = 4, wynik jest ok, ale gdy zwiekszamy BLOCK_ROWS pojawiaja sie juz bledy
*/

/*
	device code
*/

void __global__ checkMajorOrder(const double *A, double *AT){
	// one block 256
	//A - matrix 16x16

	int index = threadIdx.x;
	AT[index] = A[index];
}

void __global__ copy(const double *A, int m, int n, double *AT){

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int index;
	int mn = m*n;

	if (x < m && y < n){
		for (int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			index = x + (y+j)*m;
			if(index < mn){
				AT[x + (y+j)*m] = A[x + (y+j)*m];
			}
		}
	}
}

/*
uwaga - dla macierzy trzymanych kolumnowo (matlab) w wywo³aniu poni¿szego kernela trzeba zamieniæ miejscami m z n
*/
void __global__ transposeNaive(const double *A, int m, int n, double *AT){

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int mn = m*n;
	int index;

	if (x < m && y < n){
		for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			index = x + (y+j)*m;
			if(index < mn){
				AT[x*n + y +j] = A[index];
			}
		}
	}
}

void __global__ transposeCoalescedBankConfilict(const double *A, int m, int n, double *AT){

	__shared__ double tile [TILE_DIM][TILE_DIM];

	int x_in = blockIdx.x * TILE_DIM + threadIdx.x;
	int y_in = blockIdx.y * TILE_DIM + threadIdx.y;
	int mn = m*n;
	int index;
	
	//zerowanie shared memory
	
	for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
		tile[threadIdx.y+j][threadIdx.x] = 0.0;
	}
	__syncthreads();

	if (x_in < m && y_in < n){
		// kopiowanie do pamiêci wspó³dzielonej - shared memory
		for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			tile[threadIdx.y+j][threadIdx.x] = 0.0;
			index = x_in + (y_in+j)*m;
			if(index < mn){
				tile[threadIdx.y+j][threadIdx.x] = A[index];	
			}
		}
		__syncthreads();

	}

	int x_out = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
	int y_out = blockIdx.x * TILE_DIM + threadIdx.y;
	
	if((blockIdx.x == (gridDim.x - 1)) && (blockIdx.y == (gridDim.y -1))){
		int y_threshold = TILE_DIM + m -gridDim.x*TILE_DIM;
		int x_threshold = TILE_DIM + n - gridDim.y*TILE_DIM;
		for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			index = x_out + (y_out+j)*n;
			if((index < mn) && (threadIdx.x < x_threshold) && ((threadIdx.y +j) < y_threshold)){
				AT[index] = tile[threadIdx.x][threadIdx.y+j];
			}
		}
	}
	else if (blockIdx.x == (gridDim.x - 1)){
		int threshold = TILE_DIM + m -gridDim.x*TILE_DIM;
		for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			index = x_out + (y_out+j)*n;
			if((index < mn) && ((threadIdx.y + j) < threshold)){
				AT[index] = tile[threadIdx.x][threadIdx.y+j];
			}
		}
	}
	else if(blockIdx.y == (gridDim.y -1)){
		int threshold = TILE_DIM + n - gridDim.y*TILE_DIM;
		for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			int y = y_out+j;
			index = x_out + y*n;
			//tutaj moze indeks wypierdalac kod
			if((index < mn) && (threadIdx.x < threshold)){
				AT[index] = tile[threadIdx.x][threadIdx.y+j];
			}
		}
	}
	else{
		for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			index = x_out + (y_out+j)*n;
			if(index < mn){
				AT[index] = tile[threadIdx.x][threadIdx.y+j];
			}
		}
	}
}

void __global__ transposeCoalesced(const double *A, int m, int n, double *AT){

	__shared__ double tile [TILE_DIM][TILE_DIM + 1];

	int x_in = blockIdx.x * TILE_DIM + threadIdx.x;
	int y_in = blockIdx.y * TILE_DIM + threadIdx.y;
	int mn = m*n;
	int index;
	
	//zerowanie shared memory
	
	for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
		tile[threadIdx.y+j][threadIdx.x] = 0.0;
	}
	__syncthreads();

	if (x_in < m && y_in < n){
		// kopiowanie do pamiêci wspó³dzielonej - shared memory
		for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			tile[threadIdx.y+j][threadIdx.x] = 0.0;
			index = x_in + (y_in+j)*m;
			if(index < mn){
				tile[threadIdx.y+j][threadIdx.x] = A[index];	
			}
		}
		__syncthreads();

	}

	int x_out = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
	int y_out = blockIdx.x * TILE_DIM + threadIdx.y;
	
	if((blockIdx.x == (gridDim.x - 1)) && (blockIdx.y == (gridDim.y -1))){
		int y_threshold = TILE_DIM + m -gridDim.x*TILE_DIM;
		int x_threshold = TILE_DIM + n - gridDim.y*TILE_DIM;
		for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			index = x_out + (y_out+j)*n;
			if((index < mn) && (threadIdx.x < x_threshold) && ((threadIdx.y +j) < y_threshold)){
				AT[index] = tile[threadIdx.x][threadIdx.y+j];
			}
		}
	}
	else if (blockIdx.x == (gridDim.x - 1)){
		int threshold = TILE_DIM + m -gridDim.x*TILE_DIM;
		for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			index = x_out + (y_out+j)*n;
			if((index < mn) && ((threadIdx.y + j) < threshold)){
				AT[index] = tile[threadIdx.x][threadIdx.y+j];
			}
		}
	}
	else if(blockIdx.y == (gridDim.y -1)){
		int threshold = TILE_DIM + n - gridDim.y*TILE_DIM;
		for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			int y = y_out+j;
			index = x_out + y*n;
			//tutaj moze indeks wypierdalac kod
			if((index < mn) && (threadIdx.x < threshold)){
				AT[index] = tile[threadIdx.x][threadIdx.y+j];
			}
		}
	}
	else{
		for(int j=0; j<TILE_DIM; j+=BLOCK_ROWS){
			index = x_out + (y_out+j)*n;
			if(index < mn){
				AT[index] = tile[threadIdx.x][threadIdx.y+j];
			}
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
    dim3 threadsPerBlock (TILE_DIM, BLOCK_ROWS);

	int grid_x = getRound(m, TILE_DIM) / TILE_DIM;
	int grid_y = getRound(n, TILE_DIM) / TILE_DIM;
	dim3 blocks(grid_x, grid_y);
	
	
	mwSize B_dims[2];
	B_dims[0] = n;
	B_dims[1] = m;

	// for copy
	//B_dims[0] = m;
	//B_dims[1] = n;


    /* Create a GPUArray to hold the result and get its underlying pointer. */
    B = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
                            B_dims,//mxGPUGetDimensions(A), //B_dims
                            mxGPUGetClassID(A),
                            mxGPUGetComplexity(A),
                            MX_GPU_INITIALIZE_VALUES);
    d_B = (double *)(mxGPUGetData(B));

    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */

	//copy<<<blocks, threadsPerBlock>>>(d_A, n, m, d_B);
	//transposeNaive<<<blocks, threadsPerBlock>>>(d_A, m, n, d_B);
	//transposeCoalescedBankConfilict<<<blocks, threadsPerBlock>>>(d_A, m, n, d_B);
	transposeCoalesced<<<blocks, threadsPerBlock>>>(d_A, m, n, d_B);

    cudaDeviceSynchronize();

    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
}
