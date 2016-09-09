#include <stdio.h>
#include <cuda_runtime.h>
#include "reduction.cu"

int main(){

	double * tab, *odata, *result;
	int N = 512;
	int size = N*N;
	float milliseconds = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocation memory
	cudaMalloc((void**)&tab, size*sizeof(double));
	cudaMalloc((void**)&odata, N*sizeof(double));
	cudaMalloc((void**)&result, sizeof(double));

	int sm_size = sizeof(double)*N;

	// reduction 0
	cudaEventRecord(start);

	reduce0<<<N, N, sm_size>>>(tab, odata);
	reduce0<<<1, N, sm_size>>>(odata, result);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("reduction 0: %f ms\n", milliseconds);

	// reduction 1
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	reduce1<<<N, N, sm_size>>>(tab, odata);
	reduce1<<<1, N, sm_size>>>(odata, result);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("reduction 1: %f ms\n", milliseconds);

	// reduction 2
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	reduce2<<<N, N, sm_size>>>(tab, odata);
	reduce2<<<1, N, sm_size>>>(odata, result);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("reduction 2: %f ms\n", milliseconds);

	// reduction 3
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	reduce3<<<N/2, N, sm_size>>>(tab, odata);
	reduce3<<<1, N/2, sm_size>>>(odata, result);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("reduction 3: %f ms\n", milliseconds);

	// reduction 4
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	reduce4<<<N/2, N, sm_size>>>(tab, odata);
	reduce4<<<1, N/2, sm_size>>>(odata, result);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("reduction 4: %f ms\n", milliseconds);

	// reduction 5
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	reduce5<512><<<N/2, N, sm_size>>>(tab, odata);
	reduce5<256><<<1, N/2, sm_size>>>(odata, result);

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("reduction 5: %f ms\n", milliseconds);



	// free memory
	cudaFree(tab);
	cudaFree(odata);
	cudaFree(result);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}