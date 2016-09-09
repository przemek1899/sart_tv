#include "cuUtils.cuh"



// wczeœniej nazywa³o siê  normalizeVectorSum
__global__ void reciprocal(double * v, int n){

	// inverse values of elements in a vector

	// grid stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){

		if (v[i] != 0.0){
			v[i] = 1.0 / v[i];
		}
	}
}


__global__ void saxdotpy(double a, double * x, double *y, double n, double *z){

	// perform following operation
	// z = z + a*(x.*y);

	// grid stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
		
		z[i] += a*x[i]*y[i]; 
	}
}

__global__ void elemByElem(int n, double *x, double *y, double *z){

	// grid stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){

		z[i] = x[i]*y[i]; 
	}
}

__global__ void absComplex(cufftDoubleComplex * idata, double *odata, int n){
	/*
		Instead of completely eliminating the loop when parallelizing the computation, 
		a grid-stride loop approach is used here
	*/

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
		cufftDoubleComplex c = idata[i];
		double x2 = c.x*c.x; // pow2
		double y2 = c.y*c.y; // pow2
		odata[i] = sqrt(x2+y2);
	}
}

/*compute sqrt root of complex c
	Newtow's method for computing sqrt
*/
__device__ __inline__ cuDoubleComplex sqrtComplex(cuDoubleComplex c){

	//Csub - subtract two double complex number: x - y
	//Cmul - multiplicate two double complex number: x*y

	cuDoubleComplex x = c;
	cuDoubleComplex real2 = make_cuDoubleComplex (2.0, 0.0);
	/*
	for(unsigned iter=0; iter<10; iter++){
		x = cuCsub(x,cuCdivf(cuCsub(cuCmul(x,x), c), cuCmul(real2,x))); //
	}*/

	//we can unroll the loop - czy na pewno??
	/*1*/ x = cuCsub(x,cuCdiv(cuCsub(cuCmul(x,x), c), cuCmul(real2,x)));
	/*2*/ x = cuCsub(x,cuCdiv(cuCsub(cuCmul(x,x), c), cuCmul(real2,x)));
	/*3*/ x = cuCsub(x,cuCdiv(cuCsub(cuCmul(x,x), c), cuCmul(real2,x)));
	/*4*/ x = cuCsub(x,cuCdiv(cuCsub(cuCmul(x,x), c), cuCmul(real2,x)));
	/*5*/ x = cuCsub(x,cuCdiv(cuCsub(cuCmul(x,x), c), cuCmul(real2,x)));
	/*6*/ x = cuCsub(x,cuCdiv(cuCsub(cuCmul(x,x), c), cuCmul(real2,x)));
	/*7*/ x = cuCsub(x,cuCdiv(cuCsub(cuCmul(x,x), c), cuCmul(real2,x)));
	/*8*/ x = cuCsub(x,cuCdiv(cuCsub(cuCmul(x,x), c), cuCmul(real2,x)));
	/*9*/ x = cuCsub(x,cuCdiv(cuCsub(cuCmul(x,x), c), cuCmul(real2,x)));
	/*10*/ x = cuCsub(x,cuCdiv(cuCsub(cuCmul(x,x), c), cuCmul(real2,x)));

/*
	int iter;
	for(iter=0; iter<10; iter++){
		x = cuCsubf(x,cuCdivf(cuCsubf(cuCmulf(x,x), c), cuCmulf(real2,x))); //
	}
*/

	return x;
}


__global__ void copyRealFromComplexCufft(cuDoubleComplex* complex, double* real, int m, int n){
	/*
	int x = threadIdx.x;
	int y = blockIdx.x;

	int real_index = x + y*m;

	int cufft_height = m/2+1;
	int complex_index; 

	if (x < cufft_height){
		//dane trzymane kolumnowo
		complex_index = y*cufft_height + x;
	}
	else{
		// indeks y wychodzi poza macierz skompresowan¹ cufft
		x = m - x;
		y = n - y;
		//dane trzymane kolumnowo
		complex_index = y*cufft_height + x;
	}

	real[real_index] = complex[complex_index].x;
	*/

	int x = threadIdx.x;
	int y = blockIdx.x;

	int x2 = (m - x) % m;		// indeksowanie
	int y2 = (n - y) % n;		// indeksowanie

	int cut_cols = n/2+1;

	int out_index = x + y*m;  // pytanie czy czymœ to siê ró¿ni od threadIdx.x + blockIdx.x*blockDim.x - ró¿ni siê jak widaæ fft2_m a blockDim.x
	int in_index = (x + y*cut_cols)*(x < cut_cols) + (x2 + y2*cut_cols)*(x >= cut_cols); // ale ale kolego!! dla fft2 nie tylko zmieniamy indeks ale bierzemy 
																						 // sprzê¿enie wartoœci zespolonej !
																						// tylko, ¿e tutaj to nie ma znaczenia - obliczamy modu³ liczby zespolonej

	if(in_index < cut_cols*m){

		//real[out_index] = complex[in_index].x;
		real[out_index] = complex[in_index].x;
	}

}

__global__ void copy_real_from_cufft_1d(cuDoubleComplex* complex, double* real, int n){

	int cufft_width = n/2+1;

	int index = threadIdx.x + blockIdx.x*blockDim.x;

	int cufft_index = index *(index < cufft_width) + (n-index)*(index >= cufft_width);

	real[index] = complex[cufft_index].x;
}

__global__ void copy_with_comparison(double * d_U, double * d_xk, double * d_max_X, double * d_min_X, int n){
	
	//rec(rec<minx) = minx; rec(rec>maxx) = maxx;
	//xk = rec;

	// fctr = 1/URange; URange = Umax - Umin

	double max = d_max_X[0];
	double min = d_min_X[0];

	double range = max - min;

	// grid stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){

		double val = d_U[i]*range;
		d_xk[i] = min*(val<min) + max*(val>max) + val*((!(val<min)) && (!(val>max)));
		//d_xk[i] = val;
	}
}

__global__ void normalize_ifft_result(double* ifft_vector, double denominator, int n){

		// grid stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){

		ifft_vector[i] /= denominator;
	}
}

__global__ void simple_copy_from_complex(cuDoubleComplex* complex, double* real, int n){

	// grid stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){

		real[i] = complex[i].y;
	}
}

__global__ void generate_dct_matrix_coefficients(double *A, double *AT, double N){

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	
	double lx = 1.0 + (1.0)*(x>0);
	double ly = 1.0 + (1.0)*(y>0);
	int n = N;

	// row major order
	// A[x + y*N] = cospi((2*x+1)*y/(2*N));

	// column major order
	AT[x + y*n] = sqrt(lx/N) * cospi((2.0*y+1.0)*x/(2.0*N));
	A[x + y*n] = sqrt(ly/N) * cospi((2.0*x+1.0)*y/(2.0*N));

}