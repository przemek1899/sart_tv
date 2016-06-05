#include "mex.h"
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

// cublas indexing macro
//#define IDX2C(i,j,ld) (((j)*(ld))+(i))
//#define IN_ARGS_NUM 8

const cusparseDirection_t dirA_col = CUSPARSE_DIRECTION_COLUMN;
const cusparseDirection_t dirA_row = CUSPARSE_DIRECTION_ROW;
const cusparseOperation_t NON_TRANS = CUSPARSE_OPERATION_NON_TRANSPOSE;
const cusparseOperation_t TRANS = CUSPARSE_OPERATION_TRANSPOSE;

// JAKAŚ METODA CLEANUP BY SIĘ PRZYDAŁA DO PONIŻSZYCH METOD
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
		mexErrMsgIdAndTxt("CUSPARSE ERROR ", "CUSPARSE error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cusparseGetErrorEnum(result), func);
    }
}

#define checkCusparseErrors(val)           checkCusparse2 ( (val), #val, __FILE__, __LINE__ )

void checkCuda();
void checkCublas(cublasStatus_t status);
void checkCufft(cufftResult_t status);

void verifyArguments(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]);
void verifyRetrievedPointers(mxGPUArray const *Aval, mxGPUArray const *rowInd, mxGPUArray const *colInd, mxGPUArray const *b);
void exitProgramWithErrorMessage(char *);
void initOnes(double *p, int n);

int stopping_rule(char * stoprule, int k, int kmax);

__global__ void normalizeVectorSum(double * v, int n);
__global__ void saxdotpy(double a, double * x, double *y, double n, double *z);
__global__ void elemByElem(int n, double *x, double *y, double *z);
__global__ void absComplex(cufftDoubleComplex * idata, double *odata, int n);

//thrust::plus<double> binary_op; 
double init = 0.0;  //słabo czytelna zmienna

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){

	// [X,info,restart] = sart(A,m,n,b,K)
	// [X,info,restart] = sart(A,m,n,b,K,x0)
	// mexFunction(Aval, rowInd, colInd, nnzPerRow, nnz, rows, cols, b, K)

	char * stoprule = "NO";
    bool nonneg = false;
    bool boxcon = false;
	int casel = 1;
	double lambda = 1.9;

	/*
	this function takes following arguments:

	A_val  - vector of non-zero values of matrix A           \
	rowInd - row indices of values in matrix A				  -- this three vectors fully define matrix A
	colInd - column indices of values in matrix A			 /
	nnz  - number of non-zero values (all above vectors have length of nnz)
	rows - number of rows in matrix A
	cols - number of columns in matrix A
	b - vector B
	K - number of iterations


	optionally:
	X0 - X vector with initial values

	matrix A of size m x n (m = rows, n = cols)
	vector b of length m (rows)
	vector x of length n (cols)
	rxk - size of vector b
	
	*/

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
	
	const int ONES_SIZE = cols*(cols>rows) + rows*(rows>cols);
	int required_args = ++args_count; // 8

	int n_sqrt = (int) sqrt(cols);
	if (n_sqrt*n_sqrt != cols)
		exitProgramWithErrorMessage("Rozmiar n (liczba kolumn) macierzy A, nie jest kwadratem liczby całkowitej");

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

	// ---------------------------------- CUBLAS initialization ---------------------------------------
	cublasHandle_t cublas_handle;
	checkCublas(cublasCreate(&cublas_handle));

	// ---------------------------------- CUSPARSE initialization -------------------------------------
	cusparseHandle_t cusparse_handle = 0;  // po co przypisywać te zero ??
	cusparseMatDescr_t descrA=0;
	//int *nnzPerRow, *csrRowPtrA, *csrColIndA;
	//double* csrValA;
	int *csrRowPtrA;
	int lda = rows;
	checkCusparseErrors(cusparseCreate(&cusparse_handle));

	// ---------------------------------- CUFFT initialization ----------------------------------------
	cufftHandle cufft_plan;
	checkCufft(cufftPlan2d(&cufft_plan, n_sqrt, n_sqrt, CUFFT_R2C));


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

	// alokacja pamieci dla procedur cusparse
	//checkCudaErrors(cudaMalloc((void**)&nnzPerRow, rows*sizeof(int)));

	// --------------------------------------- CUSPARSE CONVERSE DENSE TO CSR -------------------------------------------
	checkCusparseErrors(cusparseCreateMatDescr(&descrA));
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	
	// as we get saprse coo format from matlab we no longer need compute nnzTotal, and nnzPerRow
	//checkCusparseErrors(cusparseDnnz(cusparse_handle, dirA_row, rows, cols, descrA, d_A, lda, nnzPerRow, nnzTotal));
	
	checkCudaErrors(cudaMalloc((void**)&csrRowPtrA, (rows+1)*sizeof(int)));
	//checkCudaErrors(cudaMalloc((void**)&csrValA, (*nnzTotal)*sizeof(double)));
	//checkCudaErrors(cudaMalloc((void**)&csrColIndA, (*nnzTotal)*sizeof(int)));
	
	// dalej w programie, dla obliczeń A*x0 będziemy potrzebować macierzy A w formacie CSR (compressed sparse row)
	//checkCusparseErrors(cusparseDdense2csr(cusparse_handle, m, n, descrA, d_A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA));

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
	// checkCudaErrors(cudaMemcpyToSymbol(ONES_DEV, HOST_ONES, ONES_SIZE*sizeof(double))); -- constant device memory is not suitable for cusparse operations

	// d_A - macierz zwykla - cublas itp.
	// csrValA - macierz w formacie rzadkim - cusparse
	//checkCublas(cublasDgemv(cublas_handle, cublasOperation_t trans, m, n, const double * alpha, d_A, lda, ONES_DEV, 1, const double * beta, double * y, 1));
	
	double *d_ones;
	checkCudaErrors(cudaMalloc((void**)&d_ones, ONES_SIZE*sizeof(double)));
	checkCudaErrors(cudaMemcpy(d_ones, HOST_ONES, ONES_SIZE*sizeof(double), cudaMemcpyHostToDevice));

	// y = α ∗ op ( A ) ∗ x + β ∗ y - csrmv
	// stare - checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, m, n, *nnzTotal, &positive, descrA, csrValA, csrRowPtrA, csrColIndA, ONES_DEV, &zero, d_W));
	// stare - checkCusparseErrors(cusparseDcsrmv(cusparse_handle, TRANS, m, n, *nnzTotal, &positive, descrA, csrValA, csrRowPtrA, csrColIndA, ONES_DEV, &zero, d_V));
	checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, rows, cols, nnz, &positive, descrA, d_Aval, csrRowPtrA, d_colInd, d_ones, &zero, d_W));
	checkCusparseErrors(cusparseDcsrmv(cusparse_handle, TRANS, rows, cols, nnz, &positive, descrA, d_Aval, csrRowPtrA, d_colInd, d_ones, &zero, d_V));

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	mexPrintf("Po mnozeniu macierzy A przez wektor jedynek\n");
	// a może strumieniowo ?
	int threads = 256;
	int blocks1D = 32;
	normalizeVectorSum<<<blocks1D, threads>>>(d_W, rows);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	
	// TODO konfiguracja kernela 
	normalizeVectorSum<<<cols/threads, threads>>>(d_V, cols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	mexPrintf("po normalizacji normalizeVectorSum\n");
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
	
	cufftDoubleComplex *fft2_data;
	checkCudaErrors(cudaMalloc((void**)&fft2_data, (cols/2+1)*sizeof(cufftDoubleComplex))); // 2*sizeof(double)

	mexPrintf("przed petla while\n");
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
			//mexPrintf("po elemByElem\n");

		
			checkCusparseErrors(cusparseDcsrmv(cusparse_handle, TRANS, rows, cols, nnz, &positive, descrA, d_Aval, csrRowPtrA, d_colInd, d_Wrxk, &zero, d_AW));
			checkCudaErrors(cudaGetLastError());
			//mexPrintf("po cusparse csrmv dot\n");			

			threads = 256;
			blocks1D = 32;
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

		// ----------------------- START REKONSTRUKCJA REC ----------------------------------------------------

		// reshape - bardzo ważne - trzeba przetransformować wektor na macierz
	    // U = reshape(xx,nn,nn); nn = sqrt(n)
	    // założenie jest takie, że nn jest liczbą całkowitą (n jest kwadratem liczby całkowitej)

		// fft2 jest obliczane dla liczb rzeczywistych

		//mexPrintf("przed fft2\n");
		cufftExecD2Z(cufft_plan, d_x0, fft2_data);
		mexPrintf("po fft2\n");

		// FB - tu siedzi FFT2
		// fb = FB(:);
		// thresh = var(abs(fb))*median(abs(fb(2:end)))*max(10+k,10+K);%(K-k+1);
		// picks = find(abs(FB)>thresh);
		// B = FB(picks);
		// [UU,Out_RecPF] = RecPF(nn,nn,aTV,aL1,picks,B,2,opts,PsiT,Psi,range(U(:)),U);

		// ----------------------- KONIEC REKONSTRUKCJI REC ---------------------------------------------------

		// ------------------------------------ NEW RESIDUAL ------------------------------------------------
		// rxk = b - A*xk;
		// rxk - d_rxk, 
		// b - d_b
		// może trzeba zrobić, że wcześniej w rxk siedzi już b

		// TODO poniższa linika kopiowania pamięci do optymalizacji !!!
		checkCudaErrors(cudaMemcpy(d_rxk, d_b, rows*sizeof(double), cudaMemcpyDeviceToDevice));
		checkCusparseErrors(cusparseDcsrmv(cusparse_handle, NON_TRANS, rows, cols, nnz, &negative, descrA, d_Aval, csrRowPtrA, d_colInd, d_x0, &positive, d_rxk));

		// stopping rule - OPAKOWAĆ TO W FUNKCJĘ
		//stop = stopping_rule();
		if (iteration >= K)
			stop = 1;
		iteration++;
	} //koniec pętli while


	// ----------------------------------------- WRAP RESULTS FOR MATLAB --------------------------------------
	checkCudaErrors(cudaMemcpy(d_X, d_x0, cols*sizeof(double), cudaMemcpyDeviceToDevice));
	/* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(X);	

   /* The mxGPUArray pointers are host-side structures that refer to device
    * data. These must be destroyed before leaving the MEX function.  */
    //mxGPUDestroyGPUArray(A);
	mxGPUDestroyGPUArray(b);
    mxGPUDestroyGPUArray(X);

	mxGPUDestroyGPUArray(Aval);
	mxGPUDestroyGPUArray(rowInd);
	mxGPUDestroyGPUArray(colInd);

	// ------------------------------ CLEANUP -----------------------------
	cufftDestroy(cufft_plan);
	checkCublas(cublasDestroy(cublas_handle));
	checkCusparseErrors(cusparseDestroyMatDescr(descrA));
	checkCusparseErrors(cusparseDestroy(cusparse_handle));
	free(HOST_ONES);
	//free(nnzTotal);
	checkCudaErrors(cudaFree(d_ones));
	checkCudaErrors(cudaFree(fft2_data));
	checkCudaErrors(cudaFree(d_rxk));
	//checkCudaErrors(cudaFree(nnzPerRow));
	checkCudaErrors(cudaFree(d_Wrxk));
	checkCudaErrors(cudaFree(d_AW));
	checkCudaErrors(cudaFree(d_W));
	checkCudaErrors(cudaFree(d_V));
	//checkCudaErrors(cudaFree(csrValA));
	checkCudaErrors(cudaFree(csrRowPtrA));
	//checkCudaErrors(cudaFree(csrColIndA));

	if (nrhs < (required_args+1)){
		cudaFree(d_x0);
	}
	else{
		mxGPUDestroyGPUArray(X0);
	}

	// THE END
}

void verifyRetrievedPointers(mxGPUArray const *Aval, mxGPUArray const *rowInd, mxGPUArray const *colInd, mxGPUArray const *b){
	// Verify that Aval, rowInd, colInd and b really are double array before extracting the pointer.
    if (mxGPUGetClassID(Aval) != mxDOUBLE_CLASS) {
		cudaDeviceReset();
		const char * errMsg = "Invalid Input argument: Aval is not a double array";
        mexErrMsgIdAndTxt(errId, errMsg); // errMsg
    }
	if (mxGPUGetClassID(rowInd) != mxINT32_CLASS) {
		cudaDeviceReset();
        mexErrMsgIdAndTxt(errId, "Invalid Input argument: rowInd is not a mxINT32_CLASS array, it is %d\n", mxGPUGetClassID(rowInd)); // errMsg
    }
	if (mxGPUGetClassID(colInd) != mxINT32_CLASS) {
		cudaDeviceReset();
		const char * errMsg = "Invalid Input argument: colInd is not a mxINT32_CLASS array";
        mexErrMsgIdAndTxt(errId, errMsg);
    }
	if (mxGPUGetClassID(b) != mxDOUBLE_CLASS) {
		cudaDeviceReset();
		const char * errMsg = "Invalid Input argument: b is not a double array";
        mexErrMsgIdAndTxt(errId, errMsg);
    }
}

void checkCublas(cublasStatus_t status){
	//mexPrintf("cublas status %d\n", status);
	if (status != CUBLAS_STATUS_SUCCESS){
		cudaDeviceReset();
		mexErrMsgIdAndTxt("Cublas error ", "code error: %d\n", status);
	}
}

void verifyArguments(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[]){
	/* Throw an error if the input is not a GPU array. */
	// mexFunction arguments (Aval, rowInd, colInd, nnz, rows, cols, b, K)
	if (nrhs < 8){
		cudaDeviceReset();
		mexErrMsgIdAndTxt(errId, "Invalid Input argument: nrhs < 8");
	}
	else if(!(mxIsGPUArray(prhs[0]))){
		cudaDeviceReset();
        mexErrMsgIdAndTxt(errId, "Invalid Input argument: prhs[0] is not a GPU array");
	}
	else if(!(mxIsGPUArray(prhs[1]))){
		cudaDeviceReset();
        mexErrMsgIdAndTxt(errId, "Invalid Input argument: prhs[1] is not a GPU array");
	}
	else if(!(mxIsGPUArray(prhs[2]))){
		cudaDeviceReset();
        mexErrMsgIdAndTxt(errId, "Invalid Input argument: prhs[2] is not a GPU array");
	}
	else if(!(mxIsGPUArray(prhs[6]))){
		cudaDeviceReset();
        mexErrMsgIdAndTxt(errId, "Invalid Input argument: prhs[6] is not a GPU array");
	}
}

void checkCufft(cufftResult_t status){
	if (status != CUFFT_SUCCESS) {
		cudaDeviceReset();
		mexErrMsgIdAndTxt("cuFFT error ", "code error %d\n", status);
	}
}


void exitProgramWithErrorMessage(char * error_message){
	mexErrMsgIdAndTxt("%s\n", error_message);
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

__global__ void normalizeVectorSum(double * v, int n){

	int index = threadIdx.x + blockDim.x * blockIdx.x;

	if(index < n){
		if (v[index] != 0.0){
			v[index] = 1.0 / v[index];
		}
	}
}


__global__ void saxdotpy(double a, double * x, double *y, double n, double *z){

	// wykonuje operacje
	// z = z + a*(x.*y);

	// TODO
	// stride version

	int index = threadIdx.x + blockDim.x*blockIdx.x;

	if (index < n){
		z[index] += a*x[index]*y[index]; 
		//x[index] = a*y[index];
	}
}

__global__ void elemByElem(int n, double *x, double *y, double *z){

	int index = threadIdx.x + blockDim.x*blockIdx.x;

	if (index < n){
		z[index] = x[index]*y[index]; 
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

static const char *_cublasGetErrorEnum(cublasStatus_t status)
{
    switch (status)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "cublas_success";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "cublasNotInitialized";

		case CUBLAS_STATUS_ALLOC_FAILED:
			return "cublasAllocFailed";

		case CUBLAS_STATUS_INVALID_VALUE:
			return "cublasInvalidValue";

		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "cublasArchMismatch";

		case CUBLAS_STATUS_MAPPING_ERROR:
			return "cublasMappingError";
		
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "cublasExecutionFailed";

		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "cublasInternalError";

		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "cublaseNotSupported";

		case CUBLAS_STATUS_LICENSE_ERROR:
			return "cublasLicenseError";
	}
	   
	return "<unknown>";
}