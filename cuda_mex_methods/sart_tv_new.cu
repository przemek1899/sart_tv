﻿#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "cusparse_v2.h"
#include "helper_cuda.h"
#include "cufft.h"
#include <math.h>
#include <string.h>
#include "sart_constants.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "reduction.cuh"
#include "rec_pf.cuh"
#include "cuUtils.cuh"
#include "reductionMixed.cuh"
#include "errorsUtils.h"

#include <iostream>
#include <fstream>

using namespace std;

//const cusparseDirection_t dirA_col = CUSPARSE_DIRECTION_COLUMN;
//const cusparseDirection_t dirA_row = CUSPARSE_DIRECTION_ROW;
const cusparseOperation_t NON_TRANS = CUSPARSE_OPERATION_NON_TRANSPOSE;
const cusparseOperation_t TRANS = CUSPARSE_OPERATION_TRANSPOSE;

// JAKAŚ METODA CLEANUP BY SIĘ PRZYDAŁA DO PONIŻSZYCH METOD

#define BLOCK_THREAD_DIM 32
#define checkCusparseErrors(val)           checkCusparse2 ( (val), #val, __FILE__, __LINE__ )
char * N_NOT_SQUARE = "Rozmiar n (liczba kolumn) macierzy A, nie jest kwadratem liczby całkowitej\n";

void writeIntData(const char * filename, int * d, unsigned int count_el);
void writeDoubleData(const char * filename, double * d, unsigned int count_el);

void initOnes(double *p, int n);
int stopping_rule(char * stoprule, int k, int kmax);

double init = 0.0;  //słabo czytelna zmienna
int is_rec_pf = 0;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){

	/*
	[X,info,restart] = sart(A,m,n,b,K)
	[X,info,restart] = sart(A,m,n,b,K,x0)
	mexFunction(Aval, rowInd, colInd, nnzPerRow, nnz, rows, cols, b, K, maxBregIter)

	this function takes following arguments:

	A_val  - vector of non-zero values of matrix A           \
	rowInd - row indices of values in matrix A				  -- this three vectors fully define matrix A
	colInd - column indices of values in matrix A			 /
	nnz  - number of non-zero values (all above vectors have length of nnz)
	rows - number of rows in matrix A
	cols - number of columns in matrix A
	b - vector B
	K - number of iterations
	maxBregIter - max number of RecPF iterations

	optionally:
	X0 - X vector with initial values

	matrix A of size m x n (m = rows, n = cols)
	vector b of length m (rows)
	vector x of length n (cols)
	rxk - size of vector b
	*/

	/* Algorithm's parameters
	char * stoprule = "NO";
    bool nonneg = false;
    bool boxcon = false;
	*/
	int casel = 1;
	double lambda = 1.9;
	int TVtype = 2;
	const double aTV = 1.2e-4;
	const double aL1 = 0.0;

	mxGPUArray const *Aval, *rowInd, *colInd, *b, *X0;
	double const *d_Aval, *d_b;
	int const* d_rowInd, *d_colInd;

	mxGPUArray *X;
	double *d_X, *d_x0;
	double *d_rxk, *d_W, *d_V, *d_Wrxk, *d_AW, *HOST_ONES;
	
	mxInitGPU();
	verifyArguments(nlhs, plhs, nrhs, prhs);

	int args_count = -1;

	/* Retrieve input arguments */
	Aval = mxGPUCreateFromMxArray(prhs[++args_count]); // 0
	rowInd = mxGPUCreateFromMxArray(prhs[++args_count]); // 1
	colInd = mxGPUCreateFromMxArray(prhs[++args_count]); // 2
	int nnz = mxGetScalar(prhs[++args_count]); // 3
	int rows = mxGetScalar(prhs[++args_count]); // 4
	int cols = mxGetScalar(prhs[++args_count]); // 5
	b = mxGPUCreateFromMxArray(prhs[++args_count]); // 6
	int K = mxGetScalar(prhs[++args_count]); // 7
	int maxBregIter = mxGetScalar(prhs[++args_count]); // 8
	
	const int ONES_SIZE = cols*(cols>rows) + rows*(rows>cols);
	int required_args = ++args_count; // 9

	double double_cols = cols;
	int n_sqrt = (int) sqrt(cols);
	if (n_sqrt*n_sqrt != cols)
		exitProgramWithErrorMessage(N_NOT_SQUARE);

	verifyRetrievedPointers(Aval, rowInd, colInd, b); // x0 ??
	// TODO Matlab checking error - check that the sizes of A and b match

	d_Aval = (double const *)(mxGPUGetDataReadOnly(Aval));
	d_rowInd = (int const *)(mxGPUGetDataReadOnly(rowInd));
	d_colInd = (int const *)(mxGPUGetDataReadOnly(colInd));
	d_b = (double const *)(mxGPUGetDataReadOnly(b));

	/* Create a GPUArray to hold the result and get its underlying pointer. */
	mwSize X_num_dim = 1;
	mwSize X_dims[1]; // X_dmis[1] = {m};
	X_dims[0] = cols;
    X = mxGPUCreateGPUArray(X_num_dim, X_dims, mxGPUGetClassID(Aval), mxGPUGetComplexity(Aval), MX_GPU_DO_NOT_INITIALIZE);
    d_X = (double *)(mxGPUGetData(X));

	// temporary write data for profiling needs
	/*
	//A_val
	double * h_Aval;
	h_Aval = (double*) malloc(nnz*sizeof(double));
	checkCudaErrors(cudaMemcpy(h_Aval, d_Aval, nnz*sizeof(double), cudaMemcpyDeviceToHost));

	writeDoubleData("Aval.bin", h_Aval, nnz);
	free(h_Aval);

	// COLS
	int * h_colInd;
	h_colInd = (int*) malloc(nnz*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_colInd, d_colInd, nnz*sizeof(int), cudaMemcpyDeviceToHost));

	writeIntData("colInd.bin", h_colInd, nnz);
	free(h_colInd);

	// ROWS
	int * h_rowInd;
	h_rowInd = (int*) malloc(nnz*sizeof(int));
	checkCudaErrors(cudaMemcpy(h_rowInd, d_rowInd, nnz*sizeof(int), cudaMemcpyDeviceToHost));

	writeIntData("rowInd.bin", h_rowInd, nnz);
	free(h_rowInd);

	// B vector
	int b_size = 5400;
	double * h_b;
	h_b = (double*) malloc(b_size*sizeof(double));
	checkCudaErrors(cudaMemcpy(h_b, d_b, b_size*sizeof(double), cudaMemcpyDeviceToHost));

	writeDoubleData("b.bin", h_b, b_size);
	free(h_b);
	// end of writing data
	*/
	/*
	mxGPUArray *U_matlab; double * d_U_matlab;
	U_matlab = mxGPUCreateGPUArray(X_num_dim, X_dims, mxGPUGetClassID(Aval), mxGPUGetComplexity(Aval), MX_GPU_INITIALIZE_VALUES);
	d_U_matlab = (double *)	(mxGPUGetData(U_matlab));

	mwSize R_dims[1];
	R_dims[0] = 512;
	mxGPUArray *reduction_matlab; double * d_reduction_matlab;
	reduction_matlab = mxGPUCreateGPUArray(X_num_dim, R_dims, mxGPUGetClassID(Aval), mxGPUGetComplexity(Aval), MX_GPU_INITIALIZE_VALUES);
	d_reduction_matlab = (double *)	(mxGPUGetData(reduction_matlab));
	
	mwSize Denom_num_dim = 2;
	mwSize Denom_dims[2];
	Denom_dims[0] = 512;
	Denom_dims[1] = 512;

	mxGPUArray *Denom_matlab; double * d_Denom_matlab;
	Denom_matlab = mxGPUCreateGPUArray(Denom_num_dim, Denom_dims, mxGPUGetClassID(Aval), mxGPUGetComplexity(Aval), MX_GPU_INITIALIZE_VALUES);
	d_Denom_matlab = (double *)	(mxGPUGetData(Denom_matlab));
	
	mxGPUArray *output_512x512; double * d_output_512x512;
	output_512x512 = mxGPUCreateGPUArray(Denom_num_dim, Denom_dims, mxGPUGetClassID(Aval), mxGPUGetComplexity(Aval), MX_GPU_INITIALIZE_VALUES);
	d_output_512x512 = (double *)	(mxGPUGetData(output_512x512));
	*/

	double median, variance_host, thresh, norm_U_Uprev, norm_U; //mean_check var_check median_max, 

	// ------------------------------  EVENTS --------------------------------------
	cudaEvent_t start, event_stop;
	cudaEventCreate(&start);
	cudaEventCreate(&event_stop);


	// ---------------------------------- CUBLAS initialization ---------------------------------------
	cublasHandle_t cublas_handle;
	cublasStatus_t cublas_status;
	checkCublas(cublasCreate(&cublas_handle));

	// ---------------------------------- CUSPARSE initialization -------------------------------------
	cusparseHandle_t cusparse_handle = 0;
	cusparseMatDescr_t descrA=0;
	int *csrRowPtrA;
	//int lda = rows;
	checkCusparseErrors(cusparseCreate(&cusparse_handle));

	// ---------------------------------- CUFFT initialization ----------------------------------------
	cufftHandle plan2d_U, plan_psf;
	checkCufft(cufftPlan2d(&plan2d_U, n_sqrt, n_sqrt, CUFFT_D2Z));
	checkCufft(cufftPlan1d(&plan_psf, n_sqrt, CUFFT_D2Z, 1));

	cufftHandle plan2d_rhs, plan2d_ifft;
	int rhs_fft2_size = n_sqrt*(n_sqrt/2+1);
	cufftDoubleComplex* rhs_fft2;
	cudaMalloc((void**)&rhs_fft2, sizeof(cufftDoubleComplex)*rhs_fft2_size);

	if(cufftPlan2d(&plan2d_rhs, n_sqrt, n_sqrt, CUFFT_D2Z) != CUFFT_SUCCESS){
		mexErrMsgIdAndTxt(errId, "plan2d initialization failed, cufft error code\n");
	}
	if(cufftPlan2d(&plan2d_ifft, n_sqrt, n_sqrt, CUFFT_Z2D) != CUFFT_SUCCESS){
		mexErrMsgIdAndTxt(errId, "ifft plan initialization failed, cufft error code\n");
	}


	// ------------------------------- STREAMS -----------------------------------------	
	cudaStream_t cusparse_stream;
	cudaStream_t cublas_stream;
	cudaStream_t stream_plan2d_U;
	cudaStream_t stream_plan_psf;

	cudaStream_t stream1;
	cudaStream_t stream2;

	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
/*
	result = cudaStreamCreate(&cusparse_stream);
	cudaStreamCreate(&cublas_stream);
	cudaStreamCreate(&stream_plan2d_U);
	cudaStreamCreate(&stream_plan_psf);


//	cufftResult cufftSetStream(plan2d_U, stream_plan2d_U);
//	cufftResult cufftSetStream(plan_psf, stream_plan_psf);

//	cusparseStatus_t cusparseSetStream(cusparse_handle, cusparse_stream);
//	cublas_status = cublasSetStream(cublas_handle, cublas_stream);
*/

	// ---------------------------------- RecPF memory initialization ---------------------------------
	double * Denom, * prd_array, *d_U, *d_Ux, *d_Uy, *d_bx, *d_by, *d_Wx, *d_Wy, *d_RHS, *d_Uprev;
	double relchg = relchg_tol;

	checkCudaErrors(cudaMalloc((void**)&Denom, cols*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&prd_array, n_sqrt*sizeof(double)));
	checkCudaErrors(cudaMemset(prd_array, 0, n_sqrt*sizeof(double)));

	checkCudaErrors(cudaMalloc((void**)&d_U, cols*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_Uprev, cols*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_Ux, cols*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_Uy, cols*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_bx, cols*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_by, cols*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_Wx, cols*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_Wy, cols*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_RHS, cols*sizeof(double)));

	cufftDoubleComplex * psf_fft, * Numer1;
	int psf_fft_size = n_sqrt/2+1;
	checkCudaErrors(cudaMalloc((void**)&psf_fft, psf_fft_size*sizeof(cufftDoubleComplex)));
	checkCudaErrors(cudaMalloc((void**)&Numer1, cols*sizeof(cufftDoubleComplex)));

	// variables dependent of aL1
	double *PsiTU, *Z, *d, *dct_A, *dct_AT, *dct1_result, *dct2_result;
	if (aL1 > 0){
		cudaMalloc((void**)&PsiTU, cols*sizeof(double));
		cudaMalloc((void**)&Z, cols*sizeof(double));
		cudaMalloc((void**)&d, cols*sizeof(double));

		cudaMalloc((void**)&dct_A, cols*sizeof(double));
		cudaMalloc((void**)&dct_AT, cols*sizeof(double));
		
		cudaMalloc((void**)&dct1_result, cols*sizeof(double));
		cudaMalloc((void**)&dct2_result, cols*sizeof(double));
		
		dim3 blocks(BLOCK_THREAD_DIM, BLOCK_THREAD_DIM);
		dim3 grid(cols / BLOCK_THREAD_DIM, cols / BLOCK_THREAD_DIM);

		generate_dct_matrix_coefficients<<<grid, blocks>>>(dct_A, dct_AT, cols);

		// TODO - tutaj aż się prosi o strumienie - kopiowanie z kernelem

		cudaMemset(PsiTU, 0, cols*sizeof(double));
		cudaMemset(Z, 0, cols*sizeof(double));
		cudaMemset(d, 0, cols*sizeof(double));
	}

	// ---------------------------------- rxk and x0 initialization -----------------------------------
	checkCudaErrors(cudaMalloc((void**)&d_rxk, rows*sizeof(double)));
	checkCudaErrors(cudaMemcpy(d_rxk, d_b, rows*sizeof(double), cudaMemcpyDeviceToDevice)); // rxk = b
	if (nrhs < (required_args + 1)){
		checkCudaErrors(cudaMalloc((void**)&d_x0, cols*sizeof(double)));
		checkCudaErrors(cudaMemset(d_x0, 0, cols*sizeof(double)));
	}
	else{
		X0 = mxGPUCreateFromMxArray(prhs[required_args]);
		d_x0 = (double *)(mxGPUGetDataReadOnly(X0)); // czy na pewno read only??
	}

	//checkCudaErrors(cudaMalloc((void**)&nnzPerRow, rows*sizeof(int)));

	// --------------------------------------- CUSPARSE CONVERSE DENSE TO CSR -------------------------------------------
	checkCusparseErrors(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	
	// as we get saprse coo format from matlab we no longer need to compute nnzTotal, and nnzPerRow
	// checkCusparseErrors(cusparseDnnz(cusparse_handle, dirA_row, rows, cols, descrA, d_A, lda, nnzPerRow, nnzTotal));
	
	checkCudaErrors(cudaMalloc((void**)&csrRowPtrA, (rows+1)*sizeof(int)));
	//checkCudaErrors(cudaMalloc((void**)&csrValA, (*nnzTotal)*sizeof(double)));
	//checkCudaErrors(cudaMalloc((void**)&csrColIndA, (*nnzTotal)*sizeof(int)));
	
	// dalej w programie, dla obliczeń A*x0 będziemy potrzebować macierzy A w formacie CSR (compressed sparse row)
	// checkCusparseErrors(cusparseDdense2csr(cusparse_handle, m, n, descrA, d_A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA));

	// --------- convert from coo sparse format (given already from matlab) to csr -------------------
	// a może skoro cols jest znacznie wieksze (262144) to może przechowywać w csc ??
	checkCusparseErrors(cusparseXcoo2csr(cusparse_handle, d_rowInd, nnz, rows, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO));


	// --------------------------------------- rxk CALCULATIONS --------------------------------------------------
	if (nrhs > required_args){ // jesli x0 jest podane jako argument, czyli A*x0 nie jest równe 0
		// rxk = b - A*x0, przy czym rxk juz jest rowne b, wiec robimy tylko, rxk - A*x0
		// Mnożenie A*x0, y = α ∗ op ( A ) ∗ x + β ∗ y
		// stare - checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, m, n, *nnzTotal, &negative, descrA, csrValA, csrRowPtrA, csrColIndA, d_x0, &positive, d_rxk));
		checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, rows, cols, nnz, &negative, descrA, d_Aval, csrRowPtrA, d_colInd, d_x0, &positive, d_rxk));
	}

	// -------------------------------------- V, W VECTORS CALCULATIONS ------------------------------------------
	/*
	1. Algorytm zwykłej redukcji
	2. Pomnożenie macierzy przez wektor jedynek
	3. Pomnożenie macierzy przez wektor jedynek,  - napisanie własnego kernela, gdzie nie ma tablicy wektora, tylko 1.0 z palca jest wpisany
	4. Jakieś próby z macierzą w formacie rzadkim (CSC, CSR)
	*/

	// ----------------- SPOSÓB NR 2 - MNOŻENIE MACIERZY PRZEZ WEKTOR JEDYNEK -------------------------------------
	// ----------------- SPOSÓB NR 2 - MNOŻENIE MACIERZY PRZEZ WEKTOR JEDYNEK -------------------------------------

	checkCudaErrors(cudaMalloc((void**)&d_W, rows*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_V, cols*sizeof(double)));

	HOST_ONES = (double*) malloc(ONES_SIZE*sizeof(double));
	initOnes(HOST_ONES, ONES_SIZE);

	// csrValA - macierz w formacie rzadkim - cusparse
	
	double *d_ones;
	checkCudaErrors(cudaMalloc((void**)&d_ones, ONES_SIZE*sizeof(double)));
	// TODO a może po prostu kernel ustawiający jedynki zamiast kopiowania
	checkCudaErrors(cudaMemcpy(d_ones, HOST_ONES, ONES_SIZE*sizeof(double), cudaMemcpyHostToDevice));

	double *d_reduction; int reduction_len = n_sqrt; // must be <= 1024 (max number of threads per block)
	checkCudaErrors(cudaMalloc((void**)&d_reduction, reduction_len*sizeof(double)));

	double *d_reduction_halved_blocks; int reduction_halved_blocks_len = n_sqrt/2; // must be <= 1024 (max number of threads per block)
	checkCudaErrors(cudaMalloc((void**)&d_reduction_halved_blocks, reduction_halved_blocks_len*sizeof(double)));

	double *d_norm_array;  // for reduction
	checkCudaErrors(cudaMalloc((void**)&d_norm_array, reduction_len*sizeof(double)));
	
	double *d_reduction_result, *d_variance_result, *d_norm_result, *d_max_U, *d_min_U;
	checkCudaErrors(cudaMalloc((void**)&d_reduction_result, sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_variance_result, sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_norm_result, sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_max_U, sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_min_U, sizeof(double)));

	// y = α ∗ op ( A ) ∗ x + β ∗ y - csrmv
	// stare - checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, m, n, *nnzTotal, &positive, descrA, csrValA, csrRowPtrA, csrColIndA, ONES_DEV, &zero, d_W));
	// stare - checkCusparseErrors(cusparseDcsrmv(cusparse_handle, TRANS, m, n, *nnzTotal, &positive, descrA, csrValA, csrRowPtrA, csrColIndA, ONES_DEV, &zero, d_V));
	checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, rows, cols, nnz, &positive, descrA, d_Aval, csrRowPtrA, d_colInd, d_ones, &zero, d_W));
	checkCusparseErrors(cusparseDcsrmv(cusparse_handle, TRANS, rows, cols, nnz, &positive, descrA, d_Aval, csrRowPtrA, d_colInd, d_ones, &zero, d_V));


	// a może strumieniowo ?
	int threads = 256;
	int blocks1D = 32;
	int smemSize;	

	reciprocal<<<blocks1D, threads>>>(d_W, rows);  // reciprocal = inverse, czyli bierzemy odwrotność liczby
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	
	// TODO konfiguracja kernela 
	if (cols < threads)
		threads = cols;
	reciprocal<<<cols/threads, threads>>>(d_V, cols);  // reciprocal = inverse, czyli bierzemy odwrotność liczby
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	// SPRAWDZIĆ JAK ZACHOWUJE SIĘ GPU PRZY DZIELENIU PRZEZ ZERO

	// tutaj możemy sprawdzić, czy wynik jest poprawny

	// Apj, Aip - wektory
    // do algorytmu redukcji możemy dołożyć 1./Aip
	// Apj = full(sum(abs(A),1)); % 1 - sumowanie po kolumnach
    // Aip = full(sum(abs(A),2)); % 2 - sumowanie po wierszach

	// W - rozmiar m, V - rozmiar n
	// W, V - wektory
    //  W = 1./Aip; przykladowe wartosci 0.0123
    // I = (W == Inf); 0 albo 1 - tam gdzie inf
    // W(I) = 0; zamienia tylko inf na zera
    // V = 1./Apj';
    // I = (V == Inf);
    // V(I) = 0;

	// W i V NIE SĄ wektorami rzadkimi, zdecydowana większość to elementy niezerowe
	// zakładamy, że m > n
	checkCudaErrors(cudaMalloc((void**)&d_Wrxk, rows*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_AW, cols*sizeof(double)));
	
	cufftDoubleComplex *U_fft2;
	int U_fft2_size = n_sqrt*(n_sqrt/2+1);
	checkCudaErrors(cudaMalloc((void**)&U_fft2, U_fft2_size*sizeof(cufftDoubleComplex))); // 2*sizeof(double)

	double * abs_fft2, * median_array;
	checkCudaErrors(cudaMalloc((void**)&abs_fft2, cols*sizeof(double))); // tutaj do zastanowenia sie
	checkCudaErrors(cudaMalloc((void**)&median_array, cols*sizeof(double)));

	int stop = 0;
	int iteration = 1;
	while(!stop){
		// rxk jest wektorem dense (gęsty, czyli nie rzadkim)
		// rxk = b - Ax, a wetkor b raczej nie jest rzadki, z 5400 elementow nieco ponad 3000 sa niezerowe
		if (casel == 1){
			// SART using constant value of lambda.
			// xk = xk + lambda*(V.*(A'*(W.*rxk)));
			// xk to x0

			/*
				1. v = W.*rxk
				2. z = A'*v
				3. xk + lambda*(V.*z)
				xk siedzi chyba w d_x0
			*/
			// d_AW =  W.*rxk, dlugosc m - chyba nie
			threads = 256;
			blocks1D = 32;
			elemByElem<<<blocks1D, threads>>>(rows, d_W, d_rxk, d_Wrxk);
			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());

			checkCusparseErrors(cusparseDcsrmv(cusparse_handle, TRANS, rows, cols, nnz, &positive, descrA, d_Aval, csrRowPtrA, d_colInd, d_Wrxk, &zero, d_AW));
			checkCudaErrors(cudaGetLastError());	

			threads = 256;
			blocks1D = 32;
			
			// TODO kernel configuration
			if (cols < threads)
				threads = cols;
			saxdotpy<<<cols/threads, threads>>>(lambda, d_V, d_AW, cols, d_x0);

			cudaDeviceSynchronize();
			checkCudaErrors(cudaGetLastError());
			// jakies sprawdzenie bledów ??
		}
		else if(casel == 2){
			// SART using line search
			// TODO
		}
		else if(casel == 3){
			// SART using psi1 or psi2
			// TODO
		}
		
		// ----------------------- REC_PF  ----------------------------------------------------
		// ----------------------- REC_PF  ----------------------------------------------------
		// ----------------------- REC_PF  ----------------------------------------------------

		// 1. FB = fft2(U)/nn; nn = n_sqrt

		// U = reshape(xx,nn,nn);
		// fft2 dla liczb rzeczywistych
		checkCufft(cufftExecD2Z(plan2d_U, d_x0, U_fft2));

		threads = 256;
		blocks1D = 32;

		// abs(fft2(U)/nn)
		divide_and_abs_fft2<<<n_sqrt, n_sqrt>>>(U_fft2, abs_fft2, median_array, U_fft2_size, n_sqrt, n_sqrt, (double) n_sqrt); cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		//------------------ thresh = var(abs(fb))*median(abs(fb(2:end)))*max(10+k,10+K) ----------------------

		// ŚREDNIA - MEAN   metodą redukcji
		smemSize = reduction_len*sizeof(double);

		//reduce2<<<reduction_len, reduction_len, smemSize>>>(abs_fft2, d_reduction);						cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		//reduce2<<<1, reduction_len, smemSize>>>(d_reduction, d_reduction_result);						cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		reduce3<<<reduction_len/2, reduction_len, smemSize>>>(abs_fft2, d_reduction_halved_blocks);	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		reduce3<<<1, reduction_len/2, smemSize>>>(d_reduction_halved_blocks, d_reduction_result);		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		
		//checkCudaErrors(cudaMemcpy(&variance_host, d_reduction_result, sizeof(double), cudaMemcpyDeviceToHost));

		// WARIANCJA - najpierw trzeba policzyć średnią mean
		variance2<<<reduction_len, reduction_len, smemSize>>>(abs_fft2, d_reduction, d_reduction_result, double_cols);	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		reduce2<<<1, reduction_len, smemSize>>>(d_reduction, d_variance_result);										cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		
		// MEDIANA	- thrust sort,  matlab: median(abs(fb(2:end)))
		thrust::device_ptr<double> median_vector(median_array);
		thrust::sort(median_vector+1, median_vector + cols);
		// now at abs_fft2[n] resides median value
		median = median_vector[cols/2+1];
		//median_max = median*(double)max(10+iteration, 10+K); // duże K - max liczba iteracji, małe k - aktualna iteracja/krok

		// thresh = variance*median*max(10+k, 10+K);
		checkCudaErrors(cudaMemcpy(&variance_host, d_variance_result, sizeof(double), cudaMemcpyDeviceToHost));
		thresh = variance_host/ (double_cols-1.0) * median * (double)max(10+iteration, 10+K);
		// wariancja - należy dzielić przez n - 1 (a nie przez samo n) czyli przez cols-1
	
		// picks = find(abs(FB)>thresh);  - tego nie bedziemy obliczac, w nastpenych wywołaniach będziemy poprawdzu sprawdzac warunek abs(FB)>thresh
		// B = FB(picks);
		// [UU,Out_RecPF] = RecPF(nn,nn,aTV,aL1,picks,B,2,opts,PsiT,Psi,range(U(:)),U);

		checkCudaErrors(cudaMemset(Denom, 0, cols*sizeof(double)));
		checkCudaErrors(cudaMemset(Numer1, 0, cols*sizeof(cufftDoubleComplex)));

		// compute range(U) - czyli max(U) - min(U), gdzie U jest xk;
		min_value<<<reduction_len, reduction_len, smemSize>>>(d_x0, d_reduction);		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		min_value<<<1, reduction_len, smemSize>>>(d_reduction, d_min_U);				cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		max_value<<<reduction_len, reduction_len, smemSize>>>(d_x0, d_reduction);		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		max_value<<<1, reduction_len, smemSize>>>(d_reduction, d_max_U);				cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		// Numer1 = zeros(m,n); Numer1(picks) = sqrt(m*n)*B;
		// Denom1 = zeros(m,n); Denom1(picks) = 1;
		set_Numer1_and_Denom1<<<n_sqrt, n_sqrt>>>(Numer1, Denom, abs_fft2, U_fft2, thresh, d_max_U, d_min_U, n_sqrt, n_sqrt, n_sqrt);
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		// checkCudaErrors(cudaMemcpy(d_Denom_matlab, Denom, cols*sizeof(double), cudaMemcpyDeviceToDevice));

		// Numer1 - jest ok, dopiero na 8 pozycji dwa z 262144 elementów różnią się 
		// simple_copy_from_complex<<<n_sqrt, n_sqrt>>>(Numer1, d_Denom_matlab, cols);  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


		// nnz(Denom)
		//reduce2<<<reduction_len, reduction_len, smemSize>>>(Denom, d_reduction);					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		//reduce2<<<1, reduction_len, smemSize>>>(d_reduction, d_reduction_result);					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		reduce3<<<reduction_len/2, reduction_len, smemSize>>>(Denom, d_reduction_halved_blocks);  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		reduce3<<<1, reduction_len/2, smemSize>>>(d_reduction_halved_blocks, d_reduction_result); cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		

		// d_reduction_result = sum(Denom) (which was set Denom(picks) = 1.0)
		// because we want nnz(picks) we need to subtract from d_reduction_result Denom(0); ?? czy na pewno?

		// spradzamy nnz(Denom1) - uwaga, z reduce3 nie działało to do końca dobrze
		checkCudaErrors(cudaMemcpy(&variance_host, d_reduction_result, sizeof(double), cudaMemcpyDeviceToHost));

		double new_aTV = variance_host / sqrt(double_cols) * aTV;
		double new_aL1 = variance_host / sqrt(double_cols) * aL1;

		// prd_array of length n_sqrt (512)
		calculate_prd<<<1, 32>>>(prd_array, d_reduction_result, Denom, aTV, n_sqrt, n_sqrt, beta);		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		// checkCudaErrors(cudaMemcpy(&variance_host, prd_array, sizeof(double), cudaMemcpyDeviceToHost));

		// Denom2
		//          takie same kolumny				    takie same wiersze (odpowiadające kolumnom z wcześniejszego)
		// Denom2 = abs(psf2otf([prd,-prd],[m,n])).^2 + abs(psf2otf([prd;-prd],[m,n])).^2; 
		// mozemy na potrzeby CUDA zastapic to wywolaniem: fft([prd, -prd], n) - kolumny są takie same

		checkCufft(cufftExecD2Z(plan_psf, prd_array, psf_fft));  // fft - one dimensional / jednowymiarowa tranformata
																 // psf_fft - vector of size n_sqrt/2+1 (for cufft library)
		
		psf_from_fft<<<n_sqrt, n_sqrt, n_sqrt*sizeof(double)>>>(psf_fft, n_sqrt, psf_fft_size, Denom, new_aL1*beta);	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			
		// Denom zgadza się co do 6 pozycji po przecinku z matlabem - checkCudaErrors(cudaMemcpy(d_Denom_matlab, Denom, cols*sizeof(double), cudaMemcpyDeviceToDevice));

		// ------------------------------------ REC_PF LOOP --------------------------------
		// ------------------------------------ REC_PF LOOP --------------------------------
		// ------------------------------------ REC_PF LOOP --------------------------------
		
		int breg_iter = 0;

		// zeros matrix
		checkCudaErrors(cudaMemset(d_U, 0, cols*sizeof(double)));
		checkCudaErrors(cudaMemset(d_Ux, 0, cols*sizeof(double)));
		checkCudaErrors(cudaMemset(d_Uy, 0, cols*sizeof(double)));
		checkCudaErrors(cudaMemset(d_bx, 0, cols*sizeof(double)));
		checkCudaErrors(cudaMemset(d_by, 0, cols*sizeof(double)));

		while (breg_iter < maxBregIter){
			// Uprev = U
			checkCudaErrors(cudaMemcpy(d_Uprev, d_U, cols*sizeof(double), cudaMemcpyDeviceToDevice));	// to można asynchronicznie zrobić

			if (TVtype == 1) { 
				// anisotrpic case 2: 
				Wx_Wy_anisotropic<<<n_sqrt, n_sqrt>>>(d_Wx, d_Wy, d_Ux, d_Uy, d_bx, d_by, cols, beta);
				cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			}
			else{ 
				// isotropic case: [Wx, Wy] = Compute_Wx_Wy(Ux,Uy,bx,by,1/beta);
				compute_wx_wy<<<n_sqrt, n_sqrt>>>(d_Ux, d_Uy, d_bx, d_by, d_Wx, d_Wy, n_sqrt, n_sqrt, (double) 1.0/beta);  
				cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			}
			// checkCudaErrors(cudaMemcpy(d_Denom_matlab, d_Wx, cols*sizeof(double), cudaMemcpyDeviceToDevice));
			// checkCudaErrors(cudaMemcpy(d_output_512x512, d_Wy, cols*sizeof(double), cudaMemcpyDeviceToDevice));

			// ---------------- Z-subprolem ----------------
			if (new_aL1 > 0){
				// PsiTU = PsiTU + d;
				// Z = sign(PsiTU).*max(abs(PsiTU)-1/beta,0) - d;
				z_subproblem<<<n_sqrt, n_sqrt>>>(PsiTU, d, Z, cols, beta);	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			}

			// ------------------------------ U = ifft2((Numer1 + fft2(rhs))./Denom) --------------------------------

			// rhs = Compute_rhs_DxtU_DytU(Wx,Wy,bx,by,(aTV*beta)); 
			compute_rhs_DxtU_DytU_column_mayor_order<<<n_sqrt, n_sqrt>>>(d_bx, d_by, d_Wx, d_Wy, d_RHS, n_sqrt, n_sqrt, new_aTV*beta); cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			//checkCudaErrors(cudaMemcpy(d_Denom_matlab, d_RHS, cols*sizeof(double), cudaMemcpyDeviceToDevice));

			if (new_aL1 > 0){
				// dct2: AT*X*A, X = Z
				cublas_status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, cols, cols, &positive, dct_AT, cols, Z, cols, &zero, dct1_result, cols);
				if(cublas_status != CUBLAS_STATUS_SUCCESS){
					mexErrMsgIdAndTxt(errId, "cublas error code %d\n", cublas_status);
				}
				cublas_status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, cols, cols, &positive, dct1_result, cols, dct_A, cols, &zero, dct2_result, cols);
				if(cublas_status != CUBLAS_STATUS_SUCCESS){
					mexErrMsgIdAndTxt(errId, "cublas error code %d\n", cublas_status);
				}
				rhs_aL1<<<n_sqrt, n_sqrt>>>(d_RHS, dct2_result, cols, new_aL1, beta);		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			}

			// fft2(rhs)
			if(cufftExecD2Z(plan2d_rhs, d_RHS, rhs_fft2) != CUFFT_SUCCESS){		mexErrMsgIdAndTxt(errId, "cufft fft2 exec failed, cufft error code\n");}

			// (Numer1 + fft2(rhs))./Denom
			add_and_divide_cut_complex<<<n_sqrt, n_sqrt/2+1>>>(Numer1, rhs_fft2, Denom, n_sqrt, n_sqrt); cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			// ifft2 				
			if(cufftExecZ2D(plan2d_ifft, rhs_fft2, d_U) != CUFFT_SUCCESS){		mexErrMsgIdAndTxt(errId, "cufft ifft2 exec failed, cufft error code\n"); }

			// wyniki ifft2 należy podzielić (znormalizować) przez wielkość macierzy m*n
			normalize_ifft_result<<<n_sqrt, n_sqrt>>>(d_U, double_cols, cols);	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


			// check stopping criterion relchg = norm(U-Uprev,'fro')/norm(U,'fro');
			// norm(U-Uprev, 'fro');
			norm_fro2<<<reduction_len, reduction_len, smemSize>>>(d_U, d_Uprev, d_norm_array);		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			//reduce2<<<1, reduction_len, smemSize>>>(d_norm_array, d_norm_result);					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			reduce3<<<1, reduction_len/2, smemSize>>>(d_norm_array, d_norm_result);					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			checkCudaErrors(cudaMemcpy(&norm_U_Uprev, d_norm_result, sizeof(double), cudaMemcpyDeviceToHost));

			// norm(U, 'fro')
			norm_fro<<<reduction_len, reduction_len, smemSize>>>(d_U, d_norm_array);				cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			//reduce2<<<1, reduction_len, smemSize>>>(d_norm_array, d_norm_result);					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			reduce3<<<1, reduction_len/2, smemSize>>>(d_norm_array, d_norm_result);					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			checkCudaErrors(cudaMemcpy(&norm_U, d_norm_result, sizeof(double), cudaMemcpyDeviceToHost));

			norm_U_Uprev = sqrt(norm_U_Uprev);
			norm_U = sqrt(norm_U);
			relchg = norm_U_Uprev / norm_U;

			//mexPrintf("norm_U_Uprev %f, norm_U %f\n", norm_U_Uprev, norm_U);
			if (relchg < relchg_tol){
				//mexPrintf("relchg (%f) < relchg_tol (%f), break the loop at iteration: %d\n", relchg, relchg_tol, breg_iter);
				break;
			}

			// if aL1 > 0; PsiTU = PsiT(U); end
			if (aL1 > 0){
				// idct2: A*X*AT, X = U
				cublas_status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, cols, cols, &positive, dct_A, cols, d_U, cols, &zero, dct1_result, cols);
				if(cublas_status != CUBLAS_STATUS_SUCCESS){
					mexErrMsgIdAndTxt(errId, "cublas error code %d\n", cublas_status);
				}
				cublas_status = cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, cols, cols, cols, &positive, dct1_result, cols, dct_AT, cols, &zero, PsiTU, cols);
				if(cublas_status != CUBLAS_STATUS_SUCCESS){
					mexErrMsgIdAndTxt(errId, "cublas error code %d\n", cublas_status);
				}
			}
			
			// [Ux,Uy]=Compute_Ux_Uy(U);
			compute_Ux_Uy_column_major_order<<<n_sqrt, n_sqrt>>>(d_U, d_Ux, d_Uy, n_sqrt, n_sqrt);  cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());

			cudaEventRecord(start);

			// bregman update
			//bregman_update<<<n_sqrt, n_sqrt>>>(d_bx, d_Ux, d_Wx, n_sqrt, n_sqrt, gamma_var);
			//bregman_update<<<n_sqrt, n_sqrt>>>(d_by, d_Uy, d_Wy, n_sqrt, n_sqrt, gamma_var);
			
			bregman_update<<<2*n_sqrt, n_sqrt>>>(d_bx, d_Ux, d_Wx, d_by, d_Uy, d_Wy, cols, gamma_var);
			
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			cudaEventRecord(event_stop);

			//if (aL1 > 0); d = d + gamma*(PsiTU - Z); end
			if (aL1 > 0){
				update_d<<<n_sqrt, n_sqrt>>>(d, PsiTU, Z, cols, gamma_var);			cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
			}

			breg_iter++;
		}

		// normalize U:
		// U = U/fctr;  - robimy to w funkcji copy_with_comparison
		// fctr = 1/URange; URange = Umax - Umin

		// ------------------------------------ REC_PF END --------------------------------
		// ------------------------------------ REC_PF END --------------------------------
		// ------------------------------------ REC_PF END --------------------------------

		/*
		[UU,Out_RecPF, numer1] = RecPF(nn,nn,aTV,aL1,picks,B,2,opts,PsiT,Psi,range(U(:)),U);
		rec= UU(:);
		minx = min(xk); maxx = max(xk);  min i max już liczymy wcześniej i wyniki siedzą w d_min_U i d_max_U
		rec(rec<minx) = minx; rec(rec>maxx) = maxx;
		xk = rec;
		*/

		copy_with_comparison<<<n_sqrt, n_sqrt>>>(d_U, d_x0, d_max_U, d_min_U, cols);		cudaDeviceSynchronize();	checkCudaErrors(cudaGetLastError());
		
		// } // is rec_pf
		// ------------------------------------ NEW RESIDUAL ------------------------------------------------
		// rxk = b - A*xk;
		// rxk - d_rxk, 
		// b - d_b
		// może trzeba zrobić, że wcześniej w rxk siedzi już b

		// TODO poniższa linika kopiowania pamięci do optymalizacji !!!
		checkCudaErrors(cudaMemcpy(d_rxk, d_b, rows*sizeof(double), cudaMemcpyDeviceToDevice));
		checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, rows, cols, nnz, &negative, descrA, d_Aval, csrRowPtrA, d_colInd, d_x0, &positive, d_rxk));

		// stopping rule
		if (iteration >= K)
			stop = 1;
		iteration++;
	} 
	// ------------------------------------THE END OF MAIN SART LOOP --------------------------------
	// ------------------------------------THE END OF MAIN SART LOOP --------------------------------
	// ------------------------------------THE END OF MAIN SART LOOP --------------------------------


	// ----------------------------------------- WRAP RESULTS FOR MATLAB --------------------------------------
	checkCudaErrors(cudaMemcpy(d_X, d_x0, cols*sizeof(double), cudaMemcpyDeviceToDevice));
	//checkCudaErrors(cudaMemcpy(d_Denom_matlab, d_U, cols*sizeof(double), cudaMemcpyDeviceToDevice));
	
	/* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(X);
	
	//plhs[1] = mxGPUCreateMxArrayOnGPU(Denom_matlab);

	mxGPUDestroyGPUArray(b);
    mxGPUDestroyGPUArray(X);

	mxGPUDestroyGPUArray(Aval);
	mxGPUDestroyGPUArray(rowInd);
	mxGPUDestroyGPUArray(colInd);
	//mxGPUDestroyGPUArray(Denom_matlab);

	// ------------------------------ CLEANUP -----------------------------
	cufftDestroy(plan2d_U);
	cufftDestroy(plan_psf);
	cufftDestroy(plan2d_rhs);
	cufftDestroy(plan2d_ifft);
	
	checkCublas(cublasDestroy(cublas_handle));
	checkCusparseErrors(cusparseDestroyMatDescr(descrA));
	checkCusparseErrors(cusparseDestroy(cusparse_handle));
	free(HOST_ONES);
	//free(nnzTotal);
	cudaFree(d_U);
	cudaFree(d_Uprev);
	checkCudaErrors(cudaFree(d_Ux));
	checkCudaErrors(cudaFree(d_Uy));
	checkCudaErrors(cudaFree(d_bx));
	checkCudaErrors(cudaFree(d_by));
	checkCudaErrors(cudaFree(d_Wx));
	checkCudaErrors(cudaFree(d_Wy));
	checkCudaErrors(cudaFree(d_RHS));
	checkCudaErrors(cudaFree(rhs_fft2));
	
	checkCudaErrors(cudaFree(psf_fft));
	checkCudaErrors(cudaFree(Denom));
	checkCudaErrors(cudaFree(Numer1));
	checkCudaErrors(cudaFree(median_array));
	checkCudaErrors(cudaFree(d_reduction));
	checkCudaErrors(cudaFree(d_reduction_halved_blocks));
	checkCudaErrors(cudaFree(d_norm_array));

	checkCudaErrors(cudaFree(d_reduction_result));
	checkCudaErrors(cudaFree(d_variance_result));
	checkCudaErrors(cudaFree(d_norm_result));
	checkCudaErrors(cudaFree(d_max_U));
	checkCudaErrors(cudaFree(d_min_U));
	
	checkCudaErrors(cudaFree(abs_fft2));
	checkCudaErrors(cudaFree(d_ones));
	checkCudaErrors(cudaFree(U_fft2));
	checkCudaErrors(cudaFree(d_rxk));
	checkCudaErrors(cudaFree(d_Wrxk));
	checkCudaErrors(cudaFree(d_AW));
	checkCudaErrors(cudaFree(d_W));
	checkCudaErrors(cudaFree(d_V));
	checkCudaErrors(cudaFree(csrRowPtrA));
	//checkCudaErrors(cudaFree(nnzPerRow));
	//checkCudaErrors(cudaFree(csrValA));
	//checkCudaErrors(cudaFree(csrColIndA));

	if (nrhs < (required_args+1)){
		cudaFree(d_x0);
	}
	else{
		mxGPUDestroyGPUArray(X0);
	}

	if (aL1 > 0){
		cudaFree(PsiTU);
		cudaFree(Z);
		cudaFree(d);
		cudaFree(dct_A);
		cudaFree(dct_AT);
		cudaFree(dct2_result);
		cudaFree(dct1_result);
	}

	// cudaEventRecord(start);
	// cudaEventRecord(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, event_stop);
	mexPrintf("event time: %f\n", milliseconds);

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	cudaEventDestroy(start);
	cudaEventDestroy(event_stop);
	/*
	result = cudaStreamDestroy(cusparse_stream);
	result = cudaStreamDestroy(cublas_stream);
	result = cudaStreamDestroy(stream_plan2d_U);
	result = cudaStreamDestroy(stream_plan_psf);
	*/

	// THE END OF MEX FUNCTION
}

void initOnes(double *p, int n){
	int i;
	for (i=0; i<n; i++){
		p[i] = 1.0;
	}
}

int stopping_rule(char * stoprule, int k, int kmax){
	
	if (strcmp(stoprule, "DP")){
		//TODO
	}
	else if(strcmp(stoprule, "ME")){

	}
	else if(strcmp(stoprule, "NC")){

	}
	else if(strcmp(stoprule, "NO")){
		// NO - no stopping rule
		if(k >= kmax)
			return 1;
		return 0;
	}
	return 0;
}

void writeDoubleData(const char * filename, double * d, unsigned int count_el){
	
	FILE * pf = fopen(filename, "wb");
	if (pf != NULL){
		
		unsigned int readEl = 0;
		
		while (count_el - readEl){
			readEl += fwrite((void*)(d+readEl), sizeof(double), count_el - readEl, pf);
		}
		
		fclose(pf);
	}
}

void writeIntData(const char * filename, int * d, unsigned int count_el){
	
	FILE * pf = fopen(filename, "wb");
	if (pf != NULL){
		
		unsigned int readEl = 0;
		
		while (count_el - readEl){
			readEl += fwrite((void*)(d+readEl), sizeof(int), count_el - readEl, pf);
		}
		
		fclose(pf);
	}
}