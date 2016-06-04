
#ifndef REDUCTION_CUDA_H
#define REDUCTION_CUDA_H

__global__ void reduce0(double *g_idata, double *g_odata);
__global__ void reduce1(double *g_idata, double *g_odata);
__global__ void reduce2(double *g_idata, double *g_odata);
__global__ void reduce3(double *g_idata, double *g_odata);
__global__ void reduce4(double *g_idata, double *g_odata);

template <unsigned int blockSize>
__global__ void reduce5(double *g_idata, double * g_odata);

template <unsigned int blockSize>
__global__ void reduce6(double *g_idata, double* g_odata, unsigned int n);

#endif