
//#ifndef VARIANCE_CUDA_H
//#define VARIANCE_CUDA_H
__global__ void variance2(double *g_idata, double *g_odata, double* reduce_sum, double sum_divider);
__global__ void norm_fro(double *g_idata, double *g_odata);
__global__ void norm_fro2(double *g_idata1, double *g_idata2, double *g_odata);

__global__ void min_value(double *g_idata, double *g_odata);
__global__ void max_value(double *g_idata, double *g_odata);

template <unsigned int blockSize> __global__ void variance6(double *g_idata, double* g_odata, unsigned int n, double* reduce_sum, double sum_divider);

template <unsigned int blockSize>
__global__ void variance6(double *g_idata, double* g_odata, unsigned int n, double* reduce_sum, double sum_divider){

	double mean = reduce_sum[0] / sum_divider;	
	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;

	while (i < n) { sdata[tid] += pow(g_idata[i] - mean, 2.0) + pow(g_idata[i+blockSize] - mean, 2.0);  i += gridSize; }
	__syncthreads();

	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid <   64) { sdata[tid] += sdata[tid +   64]; } __syncthreads(); }

	if (tid < 32) {
		if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
		if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
		if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
		if (blockSize >=    8) sdata[tid] += sdata[tid +  4];
		if (blockSize >=    4) sdata[tid] += sdata[tid +  2];
		if (blockSize >=    2) sdata[tid] += sdata[tid +  1];
	}

	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

//#endif