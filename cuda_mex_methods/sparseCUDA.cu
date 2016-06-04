
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "cusparse.h"

__global__ void sparse_reduce_sum(double* v, int n, double* result){
	// v - vector, n - length of the vector,
	// zakladamy ze blok y = 1; rozciaga sie tylko wzdluz x

	int index = threadIdx.x + blockDim.x * blockIdx.x;

	// do reduction sum
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
	char const * const cusparseErrId = "Cusparse error occurred";

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

	/*	--------------------- CUDA SPARSE --------------------------------------
		--------------------- CUDA SPARSE --------------------------------------
		--------------------- CUDA SPARSE --------------------------------------
	*/
	// for csc format matrix is stored in column-major format
	mexPrintf("przed deklaracja zmiennych\n");

	// initialize variables
	cusparseStatus_t cusparse_status;
	cusparseHandle_t cusparse_handle = 0;
	cusparseDirection_t dirA = CUSPARSE_DIRECTION_COLUMN;
	int* nnzTotal;
	// chyba trzeba zainicjalizowac nnzTotal
	//cudaMalloc((void**)&nnzTotal, sizeof(int));
	nnzTotal = (int*)malloc(sizeof(int));

	// input arguments
	cusparseMatDescr_t descrA=0;
	int* nnzPerCol;

	// output arguments
	//double* cscSortedValA;
	int* cscSortedRowIndA;
	int* cscSortedColPtrA;

	mexPrintf("przed inicjalizacja handle\n");
	// initialize cusparse library
	cusparse_status = cusparseCreate(&cusparse_handle);
	if (cusparse_status != CUSPARSE_STATUS_SUCCESS){
		mexErrMsgIdAndTxt(cusparseErrId, "initialization failed, error code %d\n", cusparse_status);
	}

	// JESZCZE NIE WIEMY ILE JEST NNZ, A JUZ TRZEBA ZAALOKOWAC TABLICE O TAKIM ROZMIARZE

	// CREATE AND SETUP MATRIX DESCRIPTOR FOR MATRIX A
	cusparse_status = cusparseCreateMatDescr(&descrA);
	if (cusparse_status != CUSPARSE_STATUS_SUCCESS){
		mexErrMsgIdAndTxt(cusparseErrId, "error while creating MatDescr, error code %d\n", cusparse_status);
	}
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	int lda = m;

	mwSize B_num_dim = 1;
	mwSize B_dims[1];
	B_dims[0] = n;
    B = mxGPUCreateGPUArray(B_num_dim, B_dims, mxINT32_CLASS /* albo mxINT64_CLASS*/, mxGPUGetComplexity(A), MX_GPU_DO_NOT_INITIALIZE);
    nnzPerCol = (int *)(mxGPUGetData(B));

	// PRECOMPUTE THE nnzPerCol
	//cudaMalloc((void**)&nnzPerCol, n*sizeof(int));
	mexPrintf("przed obliczniem nnzPerCol\n");
	cusparse_status = cusparseDnnz(cusparse_handle, dirA, m, n, descrA, d_A, lda, nnzPerCol, nnzTotal);
	if (cusparse_status != CUSPARSE_STATUS_SUCCESS){
		mexErrMsgIdAndTxt(cusparseErrId, "error while computing nnzPerCol, error code %d\n", cusparse_status);
	}

	
    /* Create a GPUArray to hold the result and get its underlying pointer. */
	/*mwSize B_num_dim = 1;
	mwSize B_dims[1];
	B_dims[0] = *nnzTotal;
    B = mxGPUCreateGPUArray(B_num_dim, B_dims, mxGPUGetClassID(A), mxGPUGetComplexity(A), MX_GPU_DO_NOT_INITIALIZE);
    d_B = (double *)(mxGPUGetData(B));*/
	// typedef enum mxComplexity {mxREAL=0, mxCOMPLEX};

	//cudaMalloc((void**)&cscSortedValA, (*nnzTotal)*sizeof(d_A[0]));
	cudaMalloc((void**)&cscSortedRowIndA, (*nnzTotal)*sizeof(int));
	cudaMalloc((void**)&cscSortedColPtrA, (n+1)*sizeof(int));

	mexPrintf("przed konwersja\n");
	// CONVERSION FROM DENSE TO SPARSE MATRIX (CSC FORMAT)
	//cusparse_status = cusparseDdense2csc(cusparse_handle, m, n, descrA, d_A, lda, nnzPerCol, /*cscSortedValA*/d_B, cscSortedRowIndA, cscSortedColPtrA);
	if (cusparse_status != CUSPARSE_STATUS_SUCCESS){
		mexErrMsgIdAndTxt(cusparseErrId, "error while doing conversion from dense to csc, error code %d\n", cusparse_status);
	}

	// CLEAN / DESTROY CUSPRASE STUFF
	cusparse_status = cusparseDestroyMatDescr(descrA);
	if (cusparse_status != CUSPARSE_STATUS_SUCCESS) { 	
		mexErrMsgIdAndTxt(cusparseErrId, "error while destroying MatDescr, error code %d\n", cusparse_status);
	}

	cusparse_status = cusparseDestroy(cusparse_handle);
	if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
		mexErrMsgIdAndTxt(cusparseErrId, "error while destroying handle, error code %d\n", cusparse_status);
	}
	
	// CUDA CLEANUP
	//cudaFree(nnzPerCol);
	free(nnzTotal);
	cudaFree(cscSortedRowIndA);
	cudaFree(cscSortedColPtrA);
	//cudaFree();
	
	// CUDA SPARSE ENDS HERE
	
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(B);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);

}
