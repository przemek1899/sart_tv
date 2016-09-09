//#include "matrix.h"
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
/*
	Profiling
	When using the start and stop functions, you also need to instruct the profiling tool to disable profiling at the start of the application. 
	For nvprof you do this with the --profile-from-start off flag.

	cudaProfilerStart()
	cudaProfilerStop()
	using the CUDA driver API, you get the same functionality with cuProfilerStart() and cuProfilerStop()

*/
#include <cuda_profiler_api.h>
//#include <cudaProfiler.h>
#include <iostream>
#include <fstream>
using namespace std;

static const char *_cusparseGetErrorEnum(cusparseStatus_t status)
{
    switch (status)
    {
        case CUSPARSE_STATUS_SUCCESS:
            return "cusparse_success";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "cusparseNotInitialized";

		case CUSPARSE_STATUS_ALLOC_FAILED:
			return "cusparseAllocFailed";

		case CUSPARSE_STATUS_INVALID_VALUE:
			return "cusparseInvalidValue";

		case CUSPARSE_STATUS_ARCH_MISMATCH:
			return "cusparseArchMismatch";

		case CUSPARSE_STATUS_MAPPING_ERROR:
			return "cusparseMappingError";
		
		case CUSPARSE_STATUS_EXECUTION_FAILED:
			return "cusparseExecutionFailed";

		case CUSPARSE_STATUS_INTERNAL_ERROR:
			return "cusparseInternalError";

		case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "cusparseMatrixTypeNotSupported";
	}

    return "<unknown>";
}

template< typename T >
void checkCusparse2(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        //fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), _cusparseGetErrorEnum(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
		//mexErrMsgIdAndTxt("CUSPARSE ERROR ", "CUSPARSE error at %s:%d code=%d(%s) \"%s\" \n",
          //      file, line, static_cast<unsigned int>(result), _cusparseGetErrorEnum(result), func);
		printf("zle przy checkCusparse2\n");
		exit(1);
    }
}


const cusparseOperation_t NON_TRANS = CUSPARSE_OPERATION_NON_TRANSPOSE;
const cusparseOperation_t TRANS = CUSPARSE_OPERATION_TRANSPOSE;

// JAKAŒ METODA CLEANUP BY SIÊ PRZYDA£A DO PONI¯SZYCH METOD

#define checkCusparseErrors(val)           checkCusparse2 ( (val), #val, __FILE__, __LINE__ )

void readDoubleData(const char * filename, double * data, int size);
void readIntData(const char * filename, int * data, int size);


void checkCublas(cublasStatus_t status);
void checkCufft(cufftResult_t status);
void initOnes(double *p, int n);
int stopping_rule(char * stoprule, int k, int kmax);

double init = 0.0;
int is_rec_pf = 0;

int main(){

//	char * stoprule = "NO";
   // bool nonneg = false;
   // bool boxcon = false;
	int casel = 1;
	double lambda = 1.9;

	double *d_X, *d_x0;
	double *d_rxk, *d_W, *d_V, *d_Wrxk, *d_AW, *HOST_ONES;
	
	int nnz = 2471648;
	int b_size = 5400;

	double * h_Aval = (double*) malloc(nnz*sizeof(double));
	int * h_rowInd = (int*) malloc(nnz*sizeof(int));
	int * h_colInd = (int*) malloc(nnz*sizeof(int));
	double * h_b = (double*) malloc(b_size*sizeof(double));

	readDoubleData("/home/pteodors/sart/mex_try/profileCUDA/Aval.bin", h_Aval, nnz);
	readIntData("/home/pteodors/sart/mex_try/profileCUDA/rowInd.bin", h_rowInd, nnz);
	readIntData("/home/pteodors/sart/mex_try/profileCUDA/colInd.bin", h_colInd, nnz);
	readDoubleData("/home/pteodors/sart/mex_try/profileCUDA/b.bin", h_b, b_size);
	// read data from files
	// A val
/*
	int copy_size = nnz*sizeof(double);
	char * memblock = (char*) malloc(copy_size);

	ifstream file_Aval ("Aval.bin", ios::in|ios::binary);
	if (file_Aval.is_open())
	{
		file_Aval.read (memblock, copy_size);
		file_Aval.close();
		h_Aval = (double *) memblock;
	}
	free(memblock);

	// COLS
	copy_size = nnz*sizeof(int);
	memblock = (char*) malloc(copy_size);

	ifstream file_colInd ("colInd.bin", ios::in|ios::binary);
	if (file_colInd.is_open())
	{
		file_colInd.read (memblock, copy_size);
		file_colInd.close();
		h_colInd = (int *) memblock;
	}

	// ROWS
	ifstream file_rowInd ("colInd.bin", ios::in|ios::binary);
	if (file_rowInd.is_open())
	{
		file_rowInd.read (memblock, copy_size);
		file_rowInd.close();
		h_rowInd = (int *) memblock;
	}
	free(memblock);

	// B vector
	copy_size = b_size*sizeof(double);
	memblock = (char*) malloc(copy_size);

	ifstream file_b ("b.bin", ios::in|ios::binary);
	if (file_b.is_open())
	{
		file_b.read (memblock, copy_size);
		file_b.close();
		h_b = (double *) memblock;
	}
	free(memblock);

	if (h_b == h_Aval){
		return 0;
	}
*/

	// end of reading data

	int rows = 5400;
	int cols = 262144;
	int K = 1; // do profilowania K = 1
	int maxBregIter = 1; // do profilowania maxBregIter = 1;

	cudaSetDevice(0);
	
	double * d_Aval, *d_b;
	int * d_rowInd, * d_colInd;

	checkCudaErrors(cudaMalloc((void**)&d_Aval, nnz*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_b, b_size*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_rowInd, nnz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_colInd, nnz*sizeof(int)));

	checkCudaErrors(cudaMemcpy(d_Aval, h_Aval, nnz*sizeof(double), cudaMemcpyHostToDevice)); 
	checkCudaErrors(cudaMemcpy(d_b, h_b, b_size*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_rowInd, h_rowInd, nnz*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_colInd, h_colInd, nnz*sizeof(int), cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void**)&d_X, cols*sizeof(double)));

	free(h_Aval);
	free(h_rowInd);
	free(h_colInd);
	free(h_b);

	
	const int ONES_SIZE = cols*(cols>rows) + rows*(rows>cols);

	double double_cols = cols;
	int n_sqrt = (int) sqrt(cols);
	if (n_sqrt*n_sqrt != cols){
		printf("wynik sie nie zgadza %d\n", n_sqrt);
		exit(1);
	}
	double median, variance_host, thresh, norm_U_Uprev, norm_U;


	cudaProfilerStart();
	// ---------------------------------- CUBLAS initialization ---------------------------------------
	cublasHandle_t cublas_handle;
	checkCublas(cublasCreate(&cublas_handle));
		
	checkCudaErrors(cudaGetLastError());
	// ---------------------------------- CUSPARSE initialization -------------------------------------
	cusparseHandle_t cusparse_handle = 0;
	cusparseMatDescr_t descrA=0;
	int *csrRowPtrA;//, *csrColPtrA;
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
		printf("zle przy cufft 2d\n");
		exit(1);
	}
	if(cufftPlan2d(&plan2d_ifft, n_sqrt, n_sqrt, CUFFT_Z2D) != CUFFT_SUCCESS){
		printf("zle przy cufft 2d\n");
		exit(1);
	}
	
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

	// ---------------------------------- rxk and x0 initialization -----------------------------------
	checkCudaErrors(cudaMalloc((void**)&d_rxk, rows*sizeof(double)));
	checkCudaErrors(cudaMemcpy(d_rxk, d_b, rows*sizeof(double), cudaMemcpyDeviceToDevice)); // rxk = b

	checkCudaErrors(cudaMalloc((void**)&d_x0, cols*sizeof(double)));
	checkCudaErrors(cudaMemset(d_x0, 0, cols*sizeof(double)));

	//checkCudaErrors(cudaMalloc((void**)&nnzPerRow, rows*sizeof(int)));

	// --------------------------------------- CUSPARSE CONVERSE DENSE TO CSR -------------------------------------------
	checkCusparseErrors(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	
	// as we get saprse coo format from matlab we no longer need to compute nnzTotal, and nnzPerRow
	// checkCusparseErrors(cusparseDnnz(cusparse_handle, dirA_row, rows, cols, descrA, d_A, lda, nnzPerRow, nnzTotal));
	
	checkCudaErrors(cudaMalloc((void**)&csrRowPtrA, (rows+1)*sizeof(int)));
	//checkCudaErrors(cudaMalloc((void**)&csrColPtrA, (cols+1)*sizeof(int)));
	//checkCudaErrors(cudaMalloc((void**)&csrValA, (*nnzTotal)*sizeof(double)));
	//checkCudaErrors(cudaMalloc((void**)&csrColIndA, (*nnzTotal)*sizeof(int)));
	
	/*
	d_Aval - wektor z wartoœciami niezerowymi
	csrRowPtrA - wektor skompresowanych indeksów wierszy
	d_colInd - wektor indeksów kolumn
	*/

	// dalej w programie, dla obliczeñ A*x0 bêdziemy potrzebowaæ macierzy A w formacie CSR (compressed sparse row)
	// checkCusparseErrors(cusparseDdense2csr(cusparse_handle, m, n, descrA, d_A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA));

	// --------- convert from coo sparse format (given already from matlab) to csr -------------------
	// a mo¿e skoro cols jest znacznie wieksze (262144) to mo¿e przechowywaæ w csc ??
	checkCusparseErrors(cusparseXcoo2csr(cusparse_handle, d_rowInd, nnz, rows, csrRowPtrA, CUSPARSE_INDEX_BASE_ZERO));

	// uwaga próbujemy uzyskaæ macierz A' (transponowan¹) poprzez stworzenie nowej skompresowanej macierzy
	// ale zaraz zaraz, mo¿emy przecie¿ uzyskaæ macierz transponowan¹ za pomoc¹ funkcji coo2csr zamieniaj¹c colInd z rowInd
	// checkCusparseErrors(cusparseXcoo2csr(cusparse_handle, d_colInd, nnz, cols, csrColPtrA, CUSPARSE_INDEX_BASE_ZERO));
	
	double *cscVal; checkCudaErrors(cudaMalloc((void**)&cscVal, nnz*sizeof(double)));
	int *cscRowInd; checkCudaErrors(cudaMalloc((void**)&cscRowInd, nnz*sizeof(int)));
	int *cscColPtr; checkCudaErrors(cudaMalloc((void**)&cscColPtr, (cols+1)*sizeof(int)));
	
	checkCusparseErrors(
	cusparseDcsr2csc(cusparse_handle, rows, cols, nnz, d_Aval, csrRowPtrA, d_colInd, cscVal, cscRowInd, cscColPtr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO)
	);


	// --------------------------------------- rxk CALCULATIONS --------------------------------------------------
	/*
	if (nrhs > required_args){ // jesli x0 jest podane jako argument, czyli A*x0 nie jest równe 0
		// rxk = b - A*x0, przy czym rxk juz jest rowne b, wiec robimy tylko, rxk - A*x0
		// Mno¿enie A*x0, y = ? * op ( A ) * x + ß * y
		// stare - checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, m, n, *nnzTotal, &negative, descrA, csrValA, csrRowPtrA, csrColIndA, d_x0, &positive, d_rxk));
		checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, rows, cols, nnz, &negative, descrA, d_Aval, csrRowPtrA, d_colInd, d_x0, &positive, d_rxk));
	}
	*/
	// -------------------------------------- V, W VECTORS CALCULATIONS ------------------------------------------
	/*
	1. Algorytm zwyk³ej redukcji
	2. Pomno¿enie macierzy przez wektor jedynek
	3. Pomno¿enie macierzy przez wektor jedynek,  - napisanie w³asnego kernela, gdzie nie ma tablicy wektora, tylko 1.0 z palca jest wpisany
	4. Jakieœ próby z macierz¹ w formacie rzadkim (CSC, CSR)
	*/

	// ----------------- SPOSÓB NR 2 - MNO¯ENIE MACIERZY PRZEZ WEKTOR JEDYNEK -------------------------------------
	// ----------------- SPOSÓB NR 2 - MNO¯ENIE MACIERZY PRZEZ WEKTOR JEDYNEK -------------------------------------

	checkCudaErrors(cudaMalloc((void**)&d_W, rows*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_V, cols*sizeof(double)));

	HOST_ONES = (double*) malloc(ONES_SIZE*sizeof(double));
	initOnes(HOST_ONES, ONES_SIZE);
	// checkCudaErrors(cudaMemcpyToSymbol(ONES_DEV, HOST_ONES, ONES_SIZE*sizeof(double))); -- constant device memory is not suitable for cusparse operations

	// csrValA - macierz w formacie rzadkim - cusparse
	//checkCublas(cublasDgemv(cublas_handle, cublasOperation_t trans, m, n, const double * alpha, d_A, lda, ONES_DEV, 1, const double * beta, double * y, 1));
	
	double *d_ones;
	checkCudaErrors(cudaMalloc((void**)&d_ones, ONES_SIZE*sizeof(double)));
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

	// y = ? * op ( A ) * x + ß * y - csrmv
	// stare - checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, m, n, *nnzTotal, &positive, descrA, csrValA, csrRowPtrA, csrColIndA, ONES_DEV, &zero, d_W));
	// stare - checkCusparseErrors(cusparseDcsrmv(cusparse_handle, TRANS, m, n, *nnzTotal, &positive, descrA, csrValA, csrRowPtrA, csrColIndA, ONES_DEV, &zero, d_V));
	checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, rows, cols, nnz, &positive, descrA, d_Aval, csrRowPtrA, d_colInd, d_ones, &zero, d_W));
	
	// uwaga - tutaj zamiana
	checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, cols, rows, nnz, &positive, descrA, cscVal, cscColPtr, cscRowInd, d_ones, &zero, d_V));
	//checkCusparseErrors(cusparseDcsrmv(cusparse_handle, TRANS, rows, cols, nnz, &positive, descrA, d_Aval, csrRowPtrA, d_colInd, d_ones, &zero, d_V));
	//checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, cols, rows, nnz, &positive, descrA, d_Aval, csrColPtrA, d_rowInd, d_ones, &zero, d_V));

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	// a mo¿e strumieniowo ?
	int threads = 256;
	int blocks1D = 32;
	int smemSize;	

	checkCudaErrors(cudaGetLastError());
	reciprocal<<<blocks1D, threads>>>(d_W, rows);  // reciprocal = inverse, czyli bierzemy odwrotnoœæ liczby
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	
	// TODO konfiguracja kernela 
	if (cols < threads)
		threads = cols;
	reciprocal<<<cols/threads, threads>>>(d_V, cols);  // reciprocal = inverse, czyli bierzemy odwrotnoœæ liczby
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	// SPRAWDZIÆ JAK ZACHOWUJE SIÊ GPU PRZY DZIELENIU PRZEZ ZERO

	// tutaj mo¿emy sprawdziæ, czy wynik jest poprawny

	// Apj, Aip - wektory
    // do algorytmu redukcji mo¿emy do³o¿yæ 1./Aip
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

	// W i V NIE S¥ wektorami rzadkimi, zdecydowana wiêkszoœæ to elementy niezerowe
	// zak³adamy, ¿e m > n
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
		// rxk jest wektorem dense (gêsty, czyli nie rzadkim)
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
			
			// uwaga tutaj zamiana
			checkCusparseErrors(cusparseDcsrmv(cusparse_handle, TRANS, rows, cols, nnz, &positive, descrA, d_Aval, csrRowPtrA, d_colInd, d_Wrxk, &zero, d_AW));
			// checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, cols, rows, nnz, &positive, descrA, d_Aval, csrColPtrA, d_rowInd, d_Wrxk, &zero, d_AW));
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
		// checkCudaErrors(cudaMemcpy(d_output_512x512, abs_fft2, cols*sizeof(double), cudaMemcpyDeviceToDevice));


		//------------------ thresh = var(abs(fb))*median(abs(fb(2:end)))*max(10+k,10+K) ----------------------

		// ŒREDNIA - MEAN   metod¹ redukcji
		smemSize = reduction_len*sizeof(double);

		//reduce2<<<reduction_len, reduction_len, smemSize>>>(abs_fft2, d_reduction);						cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		//reduce2<<<1, reduction_len, smemSize>>>(d_reduction, d_reduction_result);						cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		reduce3<<<reduction_len/2, reduction_len, smemSize>>>(abs_fft2, d_reduction_halved_blocks);	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		reduce3<<<1, reduction_len/2, smemSize>>>(d_reduction_halved_blocks, d_reduction_result);		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		
		//checkCudaErrors(cudaMemcpy(&variance_host, d_reduction_result, sizeof(double), cudaMemcpyDeviceToHost));

		// WARIANCJA - najpierw trzeba policzyæ œredni¹ mean
		variance2<<<reduction_len, reduction_len, smemSize>>>(abs_fft2, d_reduction, d_reduction_result, double_cols);	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		reduce2<<<1, reduction_len, smemSize>>>(d_reduction, d_variance_result);										cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		
		// MEDIANA	- thrust sort,  matlab: median(abs(fb(2:end)))
		thrust::device_ptr<double> median_vector(median_array);
		thrust::sort(median_vector+1, median_vector + cols);
		// now at abs_fft2[n] resides median value
		median = median_vector[cols/2+1];
	//	median_max = median*(double)max(10+iteration, 10+K); // du¿e K - max liczba iteracji, ma³e k - aktualna iteracja/krok

		// thresh = variance*median*max(10+k, 10+K);
		checkCudaErrors(cudaMemcpy(&variance_host, d_variance_result, sizeof(double), cudaMemcpyDeviceToHost));
		thresh = variance_host/ (double_cols-1.0) * median * (double)max(10+iteration, 10+K);
		// wariancja - nale¿y dzieliæ przez n - 1 (a nie przez samo n) czyli przez cols-1
	
		// picks = find(abs(FB)>thresh);  - tego nie bedziemy obliczac, w nastpenych wywo³aniach bêdziemy poprawdzu sprawdzac warunek abs(FB)>thresh
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

		// Numer1 - jest ok, dopiero na 8 pozycji dwa z 262144 elementów ró¿ni¹ siê 
		// simple_copy_from_complex<<<n_sqrt, n_sqrt>>>(Numer1, d_Denom_matlab, cols);  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


		// nnz(Denom)
		//reduce2<<<reduction_len, reduction_len, smemSize>>>(Denom, d_reduction);					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		//reduce2<<<1, reduction_len, smemSize>>>(d_reduction, d_reduction_result);					cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		reduce3<<<reduction_len/2, reduction_len, smemSize>>>(Denom, d_reduction_halved_blocks);  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		reduce3<<<1, reduction_len/2, smemSize>>>(d_reduction_halved_blocks, d_reduction_result); cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		

		// d_reduction_result = sum(Denom) (which was set Denom(picks) = 1.0)
		// because we want nnz(picks) we need to subtract from d_reduction_result Denom(0); ?? czy na pewno?

		// spradzamy nnz(Denom1) - uwaga, z reduce3 nie dzia³a³o to do koñca dobrze
		checkCudaErrors(cudaMemcpy(&variance_host, d_reduction_result, sizeof(double), cudaMemcpyDeviceToHost));

		double new_aTV = variance_host / sqrt(double_cols) * aTV;

		// prd_array of length n_sqrt (512)
		calculate_prd<<<1, 32>>>(prd_array, d_reduction_result, Denom, aTV, n_sqrt, n_sqrt, beta);		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

		// checkCudaErrors(cudaMemcpy(&variance_host, prd_array, sizeof(double), cudaMemcpyDeviceToHost));

		// Denom2
		//          takie same kolumny				    takie same wiersze (odpowiadaj¹ce kolumnom z wczeœniejszego)
		// Denom2 = abs(psf2otf([prd,-prd],[m,n])).^2 + abs(psf2otf([prd;-prd],[m,n])).^2; 
		// mozemy na potrzeby CUDA zastapic to wywolaniem: fft([prd, -prd], n) - kolumny s¹ takie same

		checkCufft(cufftExecD2Z(plan_psf, prd_array, psf_fft));  // fft - one dimensional / jednowymiarowa tranformata
																 // psf_fft - vector of size n_sqrt/2+1 (for cufft library)
		
		psf_from_fft<<<n_sqrt, n_sqrt, n_sqrt*sizeof(double)>>>(psf_fft, n_sqrt, psf_fft_size, Denom);	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			
		// Denom zgadza siê co do 6 pozycji po przecinku z matlabem - checkCudaErrors(cudaMemcpy(d_Denom_matlab, Denom, cols*sizeof(double), cudaMemcpyDeviceToDevice));

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
			//cudaProfilerStart();
			// Uprev = U
			checkCudaErrors(cudaMemcpy(d_Uprev, d_U, cols*sizeof(double), cudaMemcpyDeviceToDevice));	// to mo¿na asynchronicznie zrobiæ
			
			// anisotrpic case 2: [Wx, Wy] = Compute_Wx_Wy(Ux,Uy,bx,by,1/beta);
			compute_wx_wy<<<n_sqrt, n_sqrt>>>(d_Ux, d_Uy, d_bx, d_by, d_Wx, d_Wy, n_sqrt, n_sqrt, (double) 1.0/beta);  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			// checkCudaErrors(cudaMemcpy(d_Denom_matlab, d_Wx, cols*sizeof(double), cudaMemcpyDeviceToDevice));
			// checkCudaErrors(cudaMemcpy(d_output_512x512, d_Wy, cols*sizeof(double), cudaMemcpyDeviceToDevice));

			// ------------------------------ U = ifft2((Numer1 + fft2(rhs))./Denom) --------------------------------

			// rhs = Compute_rhs_DxtU_DytU(Wx,Wy,bx,by,(aTV*beta)); 
			compute_rhs_DxtU_DytU_column_mayor_order<<<n_sqrt, n_sqrt>>>(d_bx, d_by, d_Wx, d_Wy, d_RHS, n_sqrt, n_sqrt, new_aTV*beta); cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			//checkCudaErrors(cudaMemcpy(d_Denom_matlab, d_RHS, cols*sizeof(double), cudaMemcpyDeviceToDevice));

			// fft2(rhs)
			if(cufftExecD2Z(plan2d_rhs, d_RHS, rhs_fft2) != CUFFT_SUCCESS){		exit(1);}

			// (Numer1 + fft2(rhs))./Denom
			add_and_divide_cut_complex<<<n_sqrt, n_sqrt/2+1>>>(Numer1, rhs_fft2, Denom, n_sqrt, n_sqrt); cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			// ifft2 				
			if(cufftExecZ2D(plan2d_ifft, rhs_fft2, d_U) != CUFFT_SUCCESS){		exit(1); }

			// wyniki ifft2 nale¿y podzieliæ (znormalizowaæ) przez wielkoœæ macierzy m*n
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
			
			// [Ux,Uy]=Compute_Ux_Uy(U);
			compute_Ux_Uy_column_major_order<<<n_sqrt, n_sqrt>>>(d_U, d_Ux, d_Uy, n_sqrt, n_sqrt);  cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());

			// bregman update
			bregman_update<<<n_sqrt, n_sqrt>>>(d_bx, d_Ux, d_Wx, n_sqrt, n_sqrt, gamma_var);
			bregman_update<<<n_sqrt, n_sqrt>>>(d_by, d_Uy, d_Wy, n_sqrt, n_sqrt, gamma_var);
			// moze lepiej zrobic powyzsze dwa kernele za jednym wywolaniem ???
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			//checkCudaErrors(cudaMemcpy(d_Denom_matlab, d_bx, cols*sizeof(double), cudaMemcpyDeviceToDevice));
			//checkCudaErrors(cudaMemcpy(d_output_512x512, d_by, cols*sizeof(double), cudaMemcpyDeviceToDevice));
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
		minx = min(xk); maxx = max(xk);  min i max ju¿ liczymy wczeœniej i wyniki siedz¹ w d_min_U i d_max_U
		rec(rec<minx) = minx; rec(rec>maxx) = maxx;
		xk = rec;
		*/

		copy_with_comparison<<<n_sqrt, n_sqrt>>>(d_U, d_x0, d_max_U, d_min_U, cols);		cudaDeviceSynchronize();	checkCudaErrors(cudaGetLastError());
		
		// } // is rec_pf
		// ------------------------------------ NEW RESIDUAL ------------------------------------------------
		// rxk = b - A*xk;
		// rxk - d_rxk, 
		// b - d_b
		// mo¿e trzeba zrobiæ, ¿e wczeœniej w rxk siedzi ju¿ b

		// TODO poni¿sza linika kopiowania pamiêci do optymalizacji !!!
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

	// ------------------------------ CLEANUP -----------------------------
	cufftDestroy(plan2d_U);
	cufftDestroy(plan_psf);
	cufftDestroy(plan2d_rhs);
	cufftDestroy(plan2d_ifft);

	cudaFree(d_X);
	
	checkCublas(cublasDestroy(cublas_handle));
	checkCusparseErrors(cusparseDestroyMatDescr(descrA));
	checkCusparseErrors(cusparseDestroy(cusparse_handle));
	free(HOST_ONES);
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
	//checkCudaErrors(cudaFree(csrColPtrA));

	cudaFree(d_x0);
	cudaFree(d_Aval);
	cudaFree(d_b);
	cudaFree(d_rowInd);
	cudaFree(d_colInd);

	cudaFree(cscVal);
	cudaFree(cscRowInd);
	cudaFree(cscColPtr);
	// THE END

cudaProfilerStop();
		cudaDeviceSynchronize() ;
	cudaDeviceReset();
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

void checkCublas(cublasStatus_t status){
	if (status != CUBLAS_STATUS_SUCCESS){
		cudaDeviceReset();
		exit(1);
	}
}

void checkCufft(cufftResult_t status){
	if (status != CUFFT_SUCCESS) {
		cudaDeviceReset();
		exit(1);
	}
}

void readDoubleData(const char * filename, double * data, int size){

	int elements = size;
	int readData = 0;

	FILE * pf = fopen(filename, "rb");
	if (pf != NULL){
		while (elements - readData){

			readData += fread((void*)(data+readData), sizeof(double), elements - readData, pf);
		}
		fclose(pf);
	}

}

void readIntData(const char * filename, int * data, int size){

	int elements = size;
	int readData = 0;

	FILE * pf = fopen(filename, "rb");
	if (pf != NULL){

		while (elements - readData){

			readData += fread((void*)(data+readData), sizeof(int), elements - readData, pf);
		}
		fclose(pf);
	}
}

