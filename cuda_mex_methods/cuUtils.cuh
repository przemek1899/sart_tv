#include "cufft.h"

__global__ void reciprocal(double * v, int n);
__global__ void saxdotpy(double a, double * x, double *y, double n, double *z);
__global__ void elemByElem(int n, double *x, double *y, double *z);
__global__ void absComplex(cufftDoubleComplex * idata, double *odata, int n);
__device__ __inline__ cuDoubleComplex sqrtComplex(cuDoubleComplex c);
__global__ void copyRealFromComplexCufft(cuDoubleComplex* complex, double* real, int m, int n);
__global__ void copy_real_from_cufft_1d(cuDoubleComplex* complex, double* real, int n);
__global__ void copy_with_comparison(double * d_U, double * d_xk, double * d_max_X, double * d_min_X, int n);
__global__ void normalize_ifft_result(double* ifft_vector, double denominator, int n);
__global__ void simple_copy_from_complex(cuDoubleComplex* complex, double* real, int n);
__global__ void generate_dct_matrix_coefficients(double *A, double *AT, double N);