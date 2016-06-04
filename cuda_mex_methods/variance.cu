

__global__ void variance2(double *g_idata, double *g_odata, double mean){

	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDimx + threadIdx.x;
	sdata[tid] = pow(g_idata[i] - mean, 2.0);
	__syncthreads();

	for (unsigned ints=blockDim.x/2; s>0; s>>=1){
		if (tid < s){
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}


// HALVE THE NUMBER OF BLOCKS, AND REPLACE SINGLE LOAD
__global__ void variance3(double *g_idata, double *g_odata, double mean){

	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// perform first level of reduction
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = pow(g_idata[i] - mean, 2.0) + pow(g_idata[i+blockDim.x] - mean, 2.0);
	__syncthreads();

	for (unsigned ints=blockDim.x/2; s>0; s>>=1){
		if (tid < s){
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}


// UNROLLING THE LAST WARP
__global__ void variance4(double *g_idata, double *g_odata, double mean){

	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// perform first level of reduction
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = pow(g_idata[i] - mean, 2.0) + pow(g_idata[i+blockDim.x] - mean, 2.0);
	__syncthreads();

	for (unsigned ints=blockDim.x/2; s>32; s>>=1){
		if (tid < s){
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	if (tid < 32){
		sdata[tid] += sdata[tid + 32];
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}

	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void variance5(double *g_idata, double * g_odata, double mean){

	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// perform first level of reduction
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = pow(g_idata[i] - mean, 2.0) + pow(g_idata[i+blockDim.x] - mean, 2.0);
	__syncthreads();

	if (blockSize >= 512) {
		if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); 
	}
	if (blockSize >= 256) {
		if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); 
	}
	if (blockSize >= 128) {
		if (tid <  64)  { sdata[tid] += sdata[tid +   64]; } __syncthreads(); 
	}
	if (tid < 32) {
		if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
		if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
		if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
		if (blockSize >=   8) sdata[tid] += sdata[tid +  4]; 
		if (blockSize >=   4) sdata[tid] += sdata[tid +  2]; 
		if (blockSize >=   2) sdata[tid] += sdata[tid +  1]; 
	}

	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize>
__global__ void variance6(double *g_idata, double* g_odata, unsigned int n, double mean){

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