#ifndef SART_CONSTANTS_H
#define SART_CONSTANTS_H

/* Parametry RecPF */
const double aTV = 1.2e-4;
const double aL1 = 0.0;

const double negative = -1.0;
const double positive = 1.0;
const double zero = 0.0;

// Psi - dct2, PsiT - idct2

// opts parameters
const int maxItr = 100;
const double gamma_var = 1.0;
const int beta = 10;
const double relchg_tol = 5e-4;
const int real_sol = 1;
const int normalize = 1;

char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
char const * const errCusparseId = "parallel:gpu:mexGPU:CUSPARSE_ERROR";
char const * const errCublasId = "parallel:gpu:mexGPU:CUBLAS_ERROR";
//char const * const errMsg = "Invalid input to MEX file.";

#endif

/*


#define CLEANUP()									\
do {														\
	if(HOST_ONES)	free(HOST_ONES);						\
	if(nnzTotal)	free(nnzTotal);							\
	if(d_rxk)		cudaFree(d_rxk);						\
	if(d_W)			cudaFree(d_W);							\
	if(d_V)			cudaFree(d_V);							\
	if(d_Wrxk)		cudaFree(d_Wrxk);						\
	if(d_AW)		cudaFree(d_AW);							\
	if(nnzPerRow)	cudaFree(nnzPerRow);					\
	if(csrValA)		cudaFree(csrValA);						\
	if(csrRowPtrA)	cudaFree(csrRowPtrA);					\
	if(csrColIndA)	cudaFree(csrColIndA);					\
	if(fft2_data)	cudaFree(fft2_data);					\
	if(cufft_plan)	cufftDestroy(cufft_plan);				\
	if(cublas_handle)	cublasDestroy(cublas_handle);		\
	if(descrA)	cusparseDestroyMatDescr(descrA);			\
	if(cusparse_handle)	cusparseDestroy(cusparse_handle);	\
	if(A)	mxGPUDestroyGPUArray(A);						\
	if(b)	mxGPUDestroyGPUArray(b);						\
	if(X)	mxGPUDestroyGPUArray(X);						\
	if(d_x0)	cudaFree(d_x0);					\
	cudaDeviceReset();										\
} while (0)

*/

