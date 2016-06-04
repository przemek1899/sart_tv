
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"


template<typename T> __global__ void compute_wx_wy(T* Ux, T* Uy, T* bx, T* by, T* Wx, T* Wy, int rows, int cols, double tau);
template<typename T> __global__ void compute_rhs_DxtU_DytU_column_mayor_order(T* bx, T* by, T* Wx, T* Wy, T* RHS, int rows, int cols, double tau);
template<typename T> __global__ void compute_rhs_DxtU_DytU_row_mayor_order(T* bx, T* by, T* Wx, T* Wy, T* RHS, int rows, int cols, double tau);
template<typename T> __global__ void compute_Ux_Uy_column_major_order(T* U, T* Ux, T* Uy, int rows, int cols);
template<typename T> __global__ void compute_Ux_Uy_row_major_order(T* U, T* Ux, T* Uy, int rows, int cols);
template<typename T> __global__ void bregman_update(T* b, T* U, T* W, int rows, int cols, T gamma);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){

	 /* Declare all variables.*/
    mxGPUArray *Ux, *Uy, *bx, *by;
    mxGPUArray *Wx, *Wy;
    double *d_Ux, *d_Uy, *d_bx, *d_by;
    double *d_Wx, *d_Wy;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs < 6) || !(mxIsGPUArray(prhs[0])) || !(mxIsGPUArray(prhs[1])) || !(mxIsGPUArray(prhs[2])) || !(mxIsGPUArray(prhs[3]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    Ux = mxGPUCopyFromMxArray(prhs[0]);
    Uy = mxGPUCopyFromMxArray(prhs[1]);
	bx = mxGPUCopyFromMxArray(prhs[2]);
    by = mxGPUCopyFromMxArray(prhs[3]);
	int N = mxGetScalar(prhs[4]);
	double tau = mxGetScalar(prhs[5]);

    /*
     * Verify that Ux, Uy, bx, by really are double array before extracting the pointer.
     */
    if ((mxGPUGetClassID(Ux) != mxDOUBLE_CLASS) || (mxGPUGetClassID(Uy) != mxDOUBLE_CLASS) || (mxGPUGetClassID(bx) != mxDOUBLE_CLASS) || (mxGPUGetClassID(by) != mxDOUBLE_CLASS)) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_Ux = (double *)(mxGPUGetData(Ux));
	d_Uy = (double *)(mxGPUGetData(Uy));
	d_bx = (double *)(mxGPUGetData(bx));
	d_by = (double *)(mxGPUGetData(by));

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    Wx = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(Ux),
                            mxGPUGetDimensions(Ux),
                            mxGPUGetClassID(Ux),
                            mxGPUGetComplexity(Ux),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_Wx = (double *)(mxGPUGetData(Wx));

	Wy = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(Uy),
                            mxGPUGetDimensions(Uy),
                            mxGPUGetClassID(Uy),
                            mxGPUGetComplexity(Uy),
                            MX_GPU_DO_NOT_INITIALIZE);
    d_Wy = (double *)(mxGPUGetData(Wy));

	// YOUR CODE HERE
	cudaError_t status;
	int threads = N;
	int blocks = N;
	compute_wx_wy<<<threads, blocks>>>(d_Ux, d_Uy, d_bx, d_by, d_Wx, d_Wy, N, N, tau);
	cudaDeviceSynchronize();
	status = cudaGetLastError();
	if(status != cudaSuccess){
		mexErrMsgIdAndTxt(errId, "cuda error code %d\n", status);
	}
	// END YOUR CODE HERE

	/* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(Wx);
	plhs[1] = mxGPUCreateMxArrayOnGPU(Wy);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
	mxGPUDestroyGPUArray(Ux);
    mxGPUDestroyGPUArray(Uy);
	mxGPUDestroyGPUArray(bx);
    mxGPUDestroyGPUArray(by);
    mxGPUDestroyGPUArray(Wx);
    mxGPUDestroyGPUArray(Wy);

}

//version for real numbers
/*
	Compute_Wx_Wy
	
	it's very simple

	xr = Ux[i] + bx[i]
	yr = Uy[i] + by[i]
	Vr = sqrt(xr*xr + yr*yr)
	
	if ...

*/
template<typename T>
__global__ void compute_wx_wy(T* Ux, T* Uy, T* bx, T* by, T* Wx, T* Wy, int rows, int cols, double tau){
	 

	// tutaj trzeba dodac wykrywanie konfiguracji kernela tzn. siatkê bloków i w¹tków, wymiary bloków
	if (blockDim.x == 1){

	}
	else if(blockDim.y == 1){

	}
	else{

	}

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int index = x + blockIdx.x * cols;

	if (x < cols && y < rows){

		T xr = Ux[index] + bx[index];
		T yr = Uy[index] + by[index];
		xr = pow(xr, 2.0);
		yr = pow(yr, 2.0);
		T vr = sqrt(xr+yr);
		if (vr <= tau)
            {
                Wx[index] = 0; Wy[index] = 0;
            }
            else
            {
                vr = (vr - tau) / vr;
                Wx[index] = xr*vr; Wy[index] = yr*vr;
            }
	}

}

//version for real numbers
template<typename T>
__global__ void compute_rhs_DxtU_DytU_column_mayor_order(T* bx, T* by, T* Wx, T* Wy, T* RHS, int rows, int cols, double tau){
	 
	/*
	it is assumed that data in every matrix (bx, by, Wx, Wy, RHS) are in column major order, which is typicall for matlab

	kernel configuration: blocks are spanned to cover columns i.e. every block is one-dimensional and may visualized as a column of matrix
	
	*/

	int index = threadIdx.x+blockIdx.x*rows;

	int colt = rows*(cols-1);
    int rowt = rows-1;

	// predicates depend on block configuration
	// predicate1 - if a thread belogns to the first column (0 indexed)
	// predicate2 - if a thread (cell) is a first thread in a column (0 indexed)
	int index1 = index + (colt * (blockIdx.x == 0)) - (rows*(blockIdx.x != 0));
	int index2 = index + (rowt * (threadIdx.x == 0)) - (1*(threadIdx.x != 0));

	if (blockIdx.x < cols && threadIdx.x < rows){

		RHS[index] = tau*(bx[index] - bx[index1] - Wx[index] + Wx[index1] + by[index] - by[index2] - Wy[index] + Wy[index2]);
	}
}

template<typename T>
__global__ void compute_rhs_DxtU_DytU_row_mayor_order(T* bx, T* by, T* Wx, T* Wy, T* RHS, int rows, int cols, double tau){

	/*
	it is assumed that data in every matrix (bx, by, Wx, Wy, RHS) are in row major order, which is NOT typicall for matlab

	*/
	 
	int index = threadIdx.x+blockIdx.x*cols;

	int colt = rows*(cols-1);
    int rowt = rows-1;

	// predicates depend on block configuration
	int index1 = index + (colt * (threadIdx.x == 0)) - (rows*(threadIdx.x != 0));
	int index2 = index + (rowt * (blockIdx.x == 0)) - (1*(blockIdx.x != 0));

	if (threadIdx.x < cols && blockIdx.x < rows){

		RHS[index] = tau*(bx[index] - bx[index1] - Wx[index] + Wx[index1] + by[index] - by[index2] - Wy[index] + Wy[index2]);
	}
}

template<typename T>
__global__ void compute_Ux_Uy_column_major_order(T* U, T* Ux, T* Uy, int rows, int cols){


	// shuffle instructions ??
	int index = threadIdx.x+blockIdx.x*rows;

	int ux_index = (index + rows) % (rows*cols);
	int uy_index = index + 1 - (rows*(threadIdx.x == (cols-1)));

	if (threadIdx.x < cols && blockIdx.x < rows){

		T u = U[index];
		Ux[index] = U[ux_index] -u;
		Uy[index] = U[uy_index] -u;
	}
}

template<typename T>
__global__ void compute_Ux_Uy_row_major_order(T* U, T* Ux, T* Uy, int rows, int cols){

	// mo¿e t¹ funkcjê rozbiæ na dwa kernele, obliczaj¹ce Ux i Uy i wtedy mo¿na by uzyskaæ memory access coalesced albo shuffle instructions
	//shuffle instructions ??
	int index = threadIdx.x+blockIdx.x*cols;

	// TODO przekopiowane rozwi¹zanie z góry, czy w wersje column i row powinny siê zamieniaæ rows z cols ??? chyba nie!!!
	int ux_index = (index + 1) % cols;
	int uy_index = (index + cols) % (rows*cols);

	if (threadIdx.x < rows && blockIdx.x < cols){

		T u = U[index];
		Ux[index] = U[ux_index] -u;
		Uy[index] = U[uy_index] -u;
	}
}

template<typename T>
__global__ void bregman_update(T* b, T* U, T* W, int rows, int cols, T gamma){

	int index = threadIdx.x + blockIdx.x*blockDim.x + threadIdx.y*cols;

	if (index < (cols*rows)){
		b[index] = gamma*(U[index]-W[index]);
	}
}

/*
	Compute_Ux_Uy

	matrices Ux and Uy are computed independently
	
	IMPORTANT !!: this computational procedures are explained assuming that data are stored in a column major order (specific for Matlab)

	computations for Ux:	Ux[i] = U[i+rows] - U[i], basically U[i+rows] is a neighboor element from next COLUMNS (but the same row) e.g. for Aij it is Ai(j+1),
                                                      except for the last column, in this case we take the first column

	computations for Uy:    Uy[i] = U[i+1] - U[i], so taking into account matlab store order U[i+1] is the next element from the same column,
												   e.g. for Aij it is A(i+1)j, for the last element of the column, the first element is taken

*/