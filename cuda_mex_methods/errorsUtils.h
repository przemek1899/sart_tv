#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include "cusparse_v2.h"
#include "cublas_v2.h"
#include "cufft.h"

void printUsage(void);
void verifyArguments(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]);
void verifyRetrievedPointers(mxGPUArray const *Aval, mxGPUArray const *rowInd, mxGPUArray const *colInd, mxGPUArray const *b);
void exitProgramWithErrorMessage(const char *);
void checkCublas(cublasStatus_t status);
void checkCufft(cufftResult_t status);

static const char *_cusparseGetErrorEnum(cusparseStatus_t status);

static const char *_cusparseGetErrorEnum(cusparseStatus_t status)
{
    switch (status)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "cusparse_success";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "cusparseNotInitialized";

		case CUSPARSE_STATUS_ALLOC_FAILED:
			return "cusparseAllocFailed";

		case CUSPARSE_STATUS_INVALID_VALUE:
			return "cusparseInvalidValue";

		case CUSPARSE_STATUS_ARCH_MISMATCH:
			return "cusparseArchMismatch";

		case CUSPARSE_STATUS_MAPPING_ERROR:
			return "cusparseMappingError";
		
		case CUSPARSE_STATUS_EXECUTION_FAILED:
			return "cusparseExecutionFailed";

		case CUSPARSE_STATUS_INTERNAL_ERROR:
			return "cusparseInternalError";

		case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "cusparseMatrixTypeNotSupported";
	}

    return "<unknown>";
}

static const char *_cublasGetErrorEnum(cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "cublas_success";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "cublasNotInitialized";

		case CUBLAS_STATUS_ALLOC_FAILED:
			return "cublasAllocFailed";

		case CUBLAS_STATUS_INVALID_VALUE:
			return "cublasInvalidValue";

		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "cublasArchMismatch";

		case CUBLAS_STATUS_MAPPING_ERROR:
			return "cublasMappingError";
		
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "cublasExecutionFailed";

		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "cublasInternalError";

		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "cublaseNotSupported";

		case CUBLAS_STATUS_LICENSE_ERROR:
			return "cublasLicenseError";
	}
	   
	return "<unknown>";
}

template< typename T >
void checkCusparse2(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        //fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), _cusparseGetErrorEnum(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
		mexErrMsgIdAndTxt("CUSPARSE ERROR ", "CUSPARSE error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cusparseGetErrorEnum(result), func);
    }
}
