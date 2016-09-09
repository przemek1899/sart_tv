#include "errorsUtils.h"
#include "sart_constants.h"

void printUsage(void){
	//X = sart_tv_new(Aval_g, rowInd_g, colInd_g, nnz_el, rows, cols, b_g, K);

	mexPrintf("Method's calling: X = sart_tv_new(Aval_g, rowInd_g, colInd_g, nnz_el, rows, cols, b_g, K);\n");
	mexPrintf("Input arguments:\n");
	mexPrintf("Aval_g - gpu vector of nonzero elements from matrix A\n");
	mexPrintf("rowInd_g - gpu vector of row indices of elements in Aval_g\n");
	mexPrintf("colInd_g - gpu vector of column indices of elements in Aval_g\n");
	mexPrintf("nnz_el - number of nonzero elements in matrix A\n");
	mexPrintf("rows - number of rows in matrix A\n");
	mexPrintf("cols - number of columns in matrix A\n");
	mexPrintf("b_g - gpu vector b - projection vector\n");
	mexPrintf("K - number of iteration\n");
}

void exitProgramWithErrorMessage(const char * error_message){
	mexErrMsgIdAndTxt("%s\n", error_message);
}

void verifyArguments(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){
	/* Throw an error if the input is not a GPU array. */
	// mexFunction arguments (Aval, rowInd, colInd, nnz, rows, cols, b, K)
	if (nrhs < 8){
		cudaDeviceReset();
		printUsage();
		mexErrMsgIdAndTxt(errId, "Invalid Input argument: nrhs < 8");
	}
	else if(!(mxIsGPUArray(prhs[0]))){
		cudaDeviceReset();
		printUsage();
        mexErrMsgIdAndTxt(errId, "Invalid Input argument: prhs[0] is not a GPU array");
	}
	else if(!(mxIsGPUArray(prhs[1]))){
		cudaDeviceReset();
		printUsage();
        mexErrMsgIdAndTxt(errId, "Invalid Input argument: prhs[1] is not a GPU array");
	}
	else if(!(mxIsGPUArray(prhs[2]))){
		cudaDeviceReset();
		printUsage();
        mexErrMsgIdAndTxt(errId, "Invalid Input argument: prhs[2] is not a GPU array");
	}
	else if(!(mxIsGPUArray(prhs[6]))){
		cudaDeviceReset();
		printUsage();
        mexErrMsgIdAndTxt(errId, "Invalid Input argument: prhs[6] is not a GPU array");
	}
}

void verifyRetrievedPointers(mxGPUArray const *Aval, mxGPUArray const *rowInd, mxGPUArray const *colInd, mxGPUArray const *b){
	// Verify that Aval, rowInd, colInd and b really are double array before extracting the pointer.
    if (mxGPUGetClassID(Aval) != mxDOUBLE_CLASS) {
		cudaDeviceReset();
		const char * errMsg = "Invalid Input argument: Aval is not a double array";
        mexErrMsgIdAndTxt(errId, errMsg); // errMsg
    }
	if (mxGPUGetClassID(rowInd) != mxINT32_CLASS) {
		cudaDeviceReset();
        mexErrMsgIdAndTxt(errId, "Invalid Input argument: rowInd is not a mxINT32_CLASS array, it is %d\n", mxGPUGetClassID(rowInd)); // errMsg
    }
	if (mxGPUGetClassID(colInd) != mxINT32_CLASS) {
		cudaDeviceReset();
		const char * errMsg = "Invalid Input argument: colInd is not a mxINT32_CLASS array";
        mexErrMsgIdAndTxt(errId, errMsg);
    }
	if (mxGPUGetClassID(b) != mxDOUBLE_CLASS) {
		cudaDeviceReset();
		const char * errMsg = "Invalid Input argument: b is not a double array";
        mexErrMsgIdAndTxt(errId, errMsg);
    }
}

void checkCublas(cublasStatus_t status){
	if (status != CUBLAS_STATUS_SUCCESS){
		cudaDeviceReset();
		mexErrMsgIdAndTxt("Cublas error ", "code error: %d\n", status);
	}
}

void checkCufft(cufftResult_t status){
	if (status != CUFFT_SUCCESS) {
		cudaDeviceReset();
		mexErrMsgIdAndTxt("cuFFT error ", "code error %d\n", status);
	}
}
