/*

	reduction for vector only

*/

__global__ void reduce0(double *g_idata, double *g_odata){

	extern __shared__ double sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=1; s<blockDim.x; s*=2){
		if (tid % (2*s) == 0){
			sdata[tid] += sdata[s+tid];
		}
	}
	__syncthreads();

	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce1(double *g_idata, double *g_odata){

	// NEW PROBLEM: Shared Memory Bank Conflicts
	
	extern __shared__ double sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=1; s<blockDim.x; s*=2){
		
		int index = 2*s*tid;

		if(index < blockDim.x){
			sdata[index] += sdata[index+s];
		}
		__syncthreads();
	}

	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce2(double *g_idata, double *g_odata){

	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>0; s>>=1){
		if (tid < s){
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}


// HALVE THE NUMBER OF BLOCKS, AND REPLACE SINGLE LOAD
__global__ void reduce3(double *g_idata, double *g_odata){

	/*
	jeœli zmniejszamy liczbê bloków o po³owê to tablica wynikowa te¿ powinna byæ krótsza po³owê
	*/

	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// perform first level of reduction
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>0; s>>=1){
		if (tid < s){
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	// poczekaj ...
	// konsekwencj¹ tego, ¿e zmniejszamy liczbê bloków o po³owê jest to, ¿e mamy krótsz¹ o po³owê tablicê wynikow¹
	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}


// UNROLLING THE LAST WARP
__global__ void reduce4(double *g_idata, double *g_odata){

	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// perform first level of reduction
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>32; s>>=1){
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
__global__ void reduce5(double *g_idata, double * g_odata){

	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// perform first level of reduction
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
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
__global__ void reduce6(double *g_idata, double* g_odata, unsigned int n){

	extern __shared__ double sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;

	while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize];  i += gridSize; }
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