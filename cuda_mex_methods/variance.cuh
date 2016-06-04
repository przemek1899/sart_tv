
#ifndef VARIANCE_CUDA_H
#define VARIANCE_CUDA_H

__global__ void variance2(double *g_idata, double *g_odata, double mean);
__global__ void variance3(double *g_idata, double *g_odata, double mean);
__global__ void variance4(double *g_idata, double *g_odata, double mean);

template <unsigned int blockSize>
__global__ void variance5(double *g_idata, double * g_odata, double mean);

template <unsigned int blockSize>
__global__ void variance6(double *g_idata, double* g_odata, unsigned int n, double mean);

#endif