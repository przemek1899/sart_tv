

__global__ void variance2(double *g_idata, double *g_odata, double* reduce_sum, double sum_divider){

	double mean = reduce_sum[0] / sum_divider;	
	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = pow(g_idata[i] - mean, 2.0);
	//sdata[tid] = g_idata[i] - mean;
	//sdata[tid] *= sdata[tid];
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
__global__ void variance3(double *g_idata, double *g_odata, double* reduce_sum, double sum_divider){

	/*
	jeœli zmniejszamy liczbê bloków o po³owê to tablica wynikowa te¿ powinna byæ krótsza po³owê
	*/
	
	double mean = reduce_sum[0] / sum_divider;	
	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// perform first level of reduction
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = pow(g_idata[i] - mean, 2.0) + pow(g_idata[i+blockDim.x] - mean, 2.0);
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


__global__ void norm_fro(double *g_idata, double *g_odata){

	/*
	function performs operation: sqrt(sum(diag(v'*v)))
	*/

	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	//sdata[tid] = pow(g_idata[i], 2.0);
	sdata[tid] = g_idata[i]*g_idata[i];
	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>0; s>>=1){
		if (tid < s){
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void norm_fro2(double *g_idata1, double *g_idata2, double *g_odata){

	/*
	function performs operation: sqrt(sum(diag(v'*v))) for v = g_idata1 - g_idata2
	*/

	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = pow(g_idata1[i] - g_idata2[i], 2.0);
	//sdata[tid] *= sdata[tid];
	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>0; s>>=1){
		if (tid < s){
			sdata[tid] += sdata[tid+s];
		}
		__syncthreads();
	}

	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void max_value(double *g_idata, double *g_odata){
	
	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = g_idata[i];
	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>0; s>>=1){
		if (tid < s){
			sdata[tid] = sdata[tid]*(sdata[tid]>=sdata[tid+s]) + sdata[tid+s]*(sdata[tid]<sdata[tid+s]);

			// a mo¿e w sdata[tid+s] zapisywaæ min ?? - do przemyœlenia
		}
		__syncthreads();
	}

	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void min_value(double *g_idata, double *g_odata){
	
	// Sequential addressing is conflict free
	extern __shared__ double sdata[];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = g_idata[i];
	__syncthreads();

	for (unsigned int s=blockDim.x/2; s>0; s>>=1){
		if (tid < s){
			sdata[tid] = sdata[tid]*(sdata[tid]<=sdata[tid+s]) + sdata[tid+s]*(sdata[tid]>sdata[tid+s]);

			// a mo¿e w sdata[tid+s] zapisywaæ min ?? - do przemyœlenia
		}
		__syncthreads();
	}

	if (tid==0) g_odata[blockIdx.x] = sdata[0];
}
