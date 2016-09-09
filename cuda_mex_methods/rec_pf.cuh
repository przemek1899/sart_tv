
#include <cuda_runtime.h>
#include "cuUtils.cuh"

template <typename T> __global__ void compute_rhs_DxtU_DytU_column_mayor_order(T* bx, T* by, T* Wx, T* Wy, T* RHS, int rows, int cols, double tau);
template <typename T> __global__ void compute_rhs_DxtU_DytU_row_mayor_order(T* bx, T* by, T* Wx, T* Wy, T* RHS, int rows, int cols, double tau);
template <typename T> __global__ void compute_wx_wy(T* Ux, T* Uy, T* bx, T* by, T* Wx, T* Wy, int rows, int cols, double tau);
template <typename T> __global__ void compute_Ux_Uy_column_major_order(T* U, T* Ux, T* Uy, int rows, int cols);
template <typename T> __global__ void compute_Ux_Uy_row_major_order(T* U, T* Ux, T* Uy, int rows, int cols);
template <typename T> __global__ void bregman_update(T* b, T* U, T* W, int rows, int cols, T gamma);
template <typename T> __global__ void bregman_update(T* bx, T* Ux, T* Wx, T* by, T* Uy, T* Wy, int n, T gamma);

template <typename T> __global__ void calculate_prd(T* prd_array, T* reduction_result, T* Denom, T aTV, int m, int n, T beta);
template <typename T, typename C> __global__ void set_Numer1_and_Denom1(C* Numer1, T* Denom1, T* abs_fb, C* fft2_complex_data, T thresh, double* maxU, double* minU, int fft2_m, int fft2_n, int n_sqrt);
template <typename T, typename C> __global__ void psf_from_fft(C* v, int N, int n, T* result, T aL1_beta);
template <typename T, typename C> __global__ void add_and_divide_cut_complex(C* Numer1, C* fft2_rhs, T* Denom, int rows, int cols);
template <typename T> __global__ void divide_and_abs_fft2(cufftDoubleComplex * idata, T *odata, T* median_array, int odata_len, int fft2_m, int fft2_n, T divider);
template <typename T> __global__ void order_fft2_data(cufftDoubleComplex * idata, T *odata, int fft2_rows, int fft2_cols);

__global__ void z_subproblem(double * PsiTU, double * d, double * Z, int n, double beta);
__global__ void Wx_Wy_anisotropic(double * Wx, double * Wy, double * Ux, double * Uy, double * bx, double * by, int n, double beta);
//__global__ void Ux_Uy_anisotropic(double * Ux, double * Uy, double * bx, double * by, int n);
__global__ void rhs_aL1(double * rhs, double * dct, int n, double aL1, double beta);
__global__ void update_d(double * d, double * PsiTU, double *Z, int n, double gamma);

__device__ __inline__ cuDoubleComplex getComplexFromCufftMatrix(int x, int y, int n, int m, int cufft_width, int cufft_height, cuDoubleComplex* fft2_matrix);

__device__ __inline__ cuDoubleComplex getComplexFromCufftMatrix(int x, int y, int n, int m, int cufft_width, int cufft_height, cuDoubleComplex* fft2_matrix){
	/*

	DANE TRZYMANE KOLUNOWO !!!

	x, y - wspó³rzêdne pe³nej macierzy, wskazuj¹ na element, do którego chcemy dostaæ indeks w macierzy zwróconej przez cufft fft2
	cufft_width - szerokoœæ macierzy fft2 z cufft
	cufft_height - wysokoœc macierzy fft2 z cufft
	*/

	int index; 
	if (y < cufft_height){

		//dane trzymane kolumnowo
		index = x*cufft_height + y;
		return fft2_matrix[index];
	}
	else{
		// indeks y wychodzi poza macierz skompresowan¹ cufft
		// nale¿y u¿yæ w³aœciwoœci: X(n1, n2) = X*(N1-n1, N2-n2)  gdzie * oznacza sprzê¿enie
		x = (n - x) % n;
		y = (m - y) % m;
		//dane trzymane kolumnowo
		index = x*cufft_height + y;

		// przy takim indeksowaniu bierzemy sprzê¿enie 
		cuDoubleComplex c = fft2_matrix[index];
		c.y = -c.y;
		return c;
	}

}


template<typename T, typename C>
__global__ void add_and_divide_cut_complex(C* Numer1, C* fft2_rhs, T* Denom, int rows, int cols){
	
	// operation: (Numer1 + fft2(rhs))./Denom, 
	// but the result is hold in fft2(rhs) -> fft2_rhs. Later it goes to cufft ifft2
	// Numer1 - complex matrix of size rows x cols (N x N)
	// Denom - real matrix of size rows x cols (N x N)
	// fft2_rhs - complex matrix of size (N/2+1) x N

	// IMPORTANT: Numer1 and Denom are matrices of size rows x cols, but fft2_rhs was computed by cufft in D2Z plan so its total length is N*(N/2+1);

	/*
	Uwaga - funkcja w odró¿nieniu od innych jak set_Denom ... jest uruchamiana w innej konfiguracji Kernela

	poniewa¿ wype³niamy tutaj macierz fft2_rhs (która jest rozmiarnu (N/2+1) x N ) kernel te¿ uruchamiamy w takiej konfiguracji:
	blocks = n_sqrt
	threads = n_sqrt/2+1
	<<<blocks, threads>>>  <<<n_sqrt/2+1, n_sqrt>>>

	czyli mamy skompresowan¹ macierz fft2_rhs - i bierzemy indeksy wg niej, tzn. chodz¹c po odpowiednich elementach z macierz Numer1 i Denom
											  - wymusza to troszeczkê inne indeksowanie ni¿ dotychczas
											  - indeksowanie odwrotne - z macierzy fft2 cufft do pe³nej macierzy

	Numer1 - complex matrix NxN
	Denom - real matrix NxN
	fft2_rhs - complex fft2 matrix (cufft) of size (N/2+1) x N    !!!!!!!!!!!!!

	Dane u³o¿one kolumnowo (w macierzach Numer1, fft2_rhs, Denom)

	Similar to the one-dimensional case, the frequency domain representation of real-valued
	input data satisfies Hermitian symmetry, defined as: X(n1, n2, ..., nd) = X*(N1-n1,N2-n2,...,Nd-nd)
	for two dimensional fft, i.e. fft2 on NxM matrix indexing is the following: X(n,m) = X*(N-n, M-m);
	the length of fft2 done by cufft from NxM is: N*(M/2+1);

	kernel run configuration should be fitted to this size N*(M/2+1)
	*/

	int fft2_rows = rows/2 + 1;
	if (threadIdx.x < fft2_rows && blockIdx.x < cols){


		int cufft_index = threadIdx.x + blockIdx.x*fft2_rows;
		int full_index = threadIdx.x + blockIdx.x*rows;

		// get elements from Numer1 and fft2_rhs
		C t = fft2_rhs[cufft_index];
		C n = Numer1[full_index];

		// add operation -  Numer1 + fft2(rhs)
		t.x += n.x;  
		t.y += n.y;

		// final division - (Numer1 + fft2(rhs))./Denom
		
		// UWAGA CO JESLI JEST DZIELENIE PRZEZ ZERO??
		// TODO - CASE Denom[index] equals 0 - cannot divide by 0
		T divider = Denom[full_index];
		if (divider != 0.0){
			t.x /= divider;
			t.y /= divider;
			fft2_rhs[cufft_index] = t;
		}
	}
}


//version for real numbers
template<typename T>
__global__ void compute_rhs_DxtU_DytU_column_mayor_order(T* bx, T* by, T* Wx, T* Wy, T* RHS, int rows, int cols, double tau){
	 
	/*
	it is assumed that data in every matrix (bx, by, Wx, Wy, RHS) are in column major order, which is typicall for matlab
	kernel configuration: blocks are spanned to cover columns i.e. every block is one-dimensional and may visualized as a column of matrix
	*/

	int index = threadIdx.x+blockIdx.x*rows;

	int colt = rows*(cols-1);
    int rowt = rows-1;

	// predicates depend on block configuration
	// predicate1 - if a thread belogns to the first column (0 indexed)
	// predicate2 - if a thread (cell) is a first thread in a column (0 indexed)
	int index1 = index + (colt * (blockIdx.x == 0)) - (rows*(blockIdx.x != 0));
	int index2 = index + (rowt * (threadIdx.x == 0)) - (1*(threadIdx.x != 0));

	if (blockIdx.x < cols && threadIdx.x < rows){

		RHS[index] = tau*(bx[index] - bx[index1] - Wx[index] + Wx[index1] + by[index] - by[index2] - Wy[index] + Wy[index2]);
	}
}

template<typename T>
__global__ void compute_rhs_DxtU_DytU_row_mayor_order(T* bx, T* by, T* Wx, T* Wy, T* RHS, int rows, int cols, double tau){

	/*
	it is assumed that data in every matrix (bx, by, Wx, Wy, RHS) are in row major order, which is NOT typicall for matlab
	*/
	 
	int index = threadIdx.x+blockIdx.x*cols;
	int colt = rows*(cols-1);

	// predicates depend on block configuration
	int index1 = index + ((cols-1) * (threadIdx.x == 0)) - (1*(threadIdx.x != 0));
	int index2 = index + (colt * (blockIdx.x == 0)) - (cols*(blockIdx.x != 0));

	if (threadIdx.x < cols && blockIdx.x < rows){

		RHS[index] = tau*(bx[index] - bx[index1] - Wx[index] + Wx[index1] + by[index] - by[index2] - Wy[index] + Wy[index2]);
	}
}


template<typename T, typename C>
__global__ void psf_from_fft(C* v, int N, int n, T* result, T aL1_beta){

	/*
	   v - complex vector of size n = (N/2+1)
	   T - real vector of size N

		uwaga: w fft jednowymiarowej te¿ musimy braæ dalsze elementy ze sprzê¿eniem ale tutaj chyba nie ma to znaczenia bo liczymy kwadrat

	   Denom2
				takie same kolumny				    takie same wiersze (odpowiadaj¹ce kolumnom z wczeœniejszego)
	   Denom2 = abs(psf2otf([prd,-prd],[M,N])).^2 + abs(psf2otf([prd;-prd],[M,N])).^2; 
	   mozemy na potrzeby CUDA zastapic to wywolaniem: fft([prd, -prd], n) - kolumny s¹ takie same

	   przyk³adowo dla prd = 0.5, M=5, N=5 w matlabie 

	   psf2otf([prd,-prd],[M,N]) = 
		
		0.0000 + 0.0000i  -0.3455 + 0.4755i  -0.9045 + 0.2939i  -0.9045 - 0.2939i  -0.3455 - 0.4755i
		0.0000 + 0.0000i  -0.3455 + 0.4755i  -0.9045 + 0.2939i  -0.9045 - 0.2939i  -0.3455 - 0.4755i
		0.0000 + 0.0000i  -0.3455 + 0.4755i  -0.9045 + 0.2939i  -0.9045 - 0.2939i  -0.3455 - 0.4755i
		0.0000 + 0.0000i  -0.3455 + 0.4755i  -0.9045 + 0.2939i  -0.9045 - 0.2939i  -0.3455 - 0.4755i
		0.0000 + 0.0000i  -0.3455 + 0.4755i  -0.9045 + 0.2939i  -0.9045 - 0.2939i  -0.3455 - 0.4755i

	   psf2otf([prd;-prd],[M,N]) = 

		0.0000 + 0.0000i   0.0000 + 0.0000i   0.0000 + 0.0000i   0.0000 + 0.0000i   0.0000 + 0.0000i
	   -0.3455 + 0.4755i  -0.3455 + 0.4755i  -0.3455 + 0.4755i  -0.3455 + 0.4755i  -0.3455 + 0.4755i
	   -0.9045 + 0.2939i  -0.9045 + 0.2939i  -0.9045 + 0.2939i  -0.9045 + 0.2939i  -0.9045 + 0.2939i
	   -0.9045 - 0.2939i  -0.9045 - 0.2939i  -0.9045 - 0.2939i  -0.9045 - 0.2939i  -0.9045 - 0.2939i
	   -0.3455 - 0.4755i  -0.3455 - 0.4755i  -0.3455 - 0.4755i  -0.3455 - 0.4755i  -0.3455 - 0.4755i
	
	   fft([prd, -prd], N) = 0.0000 + 0.0000i   0.3455 + 0.4755i   0.9045 + 0.2939i   0.9045 - 0.2939i   0.3455 - 0.4755i
		
	   a w CUDA u¿ywaj¹c cufft otrzymujemy wektor o d³ugoœci n = N/2+1:

	   fft([prd, -prd], n) = 0.0000 + 0.0000i   0.3455 + 0.4755i   0.9045 + 0.2939i

	   Ka¿dy blok wype³nia pamiêæ dzielon¹ pe³n¹ fft:

	   abs(fft([prd, -prd], N)).^2 = 0.0000 + 0.0000i   0.3455 + 0.4755i   0.9045 + 0.2939i   0.9045 - 0.2939i   0.3455 - 0.4755i

	*/
	// blocks are rows which cover the whole length, usually the matrix is of size 512x512, the NVIDIA GPU's allows maximum number
	// of threads per block of 1024
	
	extern __shared__ double v_shared[];

	/*
	nie wiem w jaki sposób to dzia³a³o (nie wywala³ siê  b³¹d, skoro v jest d³ugoœci n = N/2 + 1
	if (threadIdx.x >= n && threadIdx.x < N && blockIdx.x == 0){
		v[threadIdx.x] = v[N-threadIdx.x];
	}
	*/

	if (threadIdx.x < N){
		// nowe indeksowanie
		int cufft_index = threadIdx.x * (threadIdx.x < n) + (N-threadIdx.x)*(threadIdx.x >= n);

		C a = v[cufft_index];
		//real_fft[threadIdx.x] = pow(a.x, 2.0) + pow(a.y, 2.0);
		//v_shared[threadIdx.x] = pow(a.x, 2.0) + pow(a.y, 2.0);
		v_shared[threadIdx.x] = a.x*a.x + a.y*a.y;
	}
	__syncthreads();

	if (blockIdx.x < N && threadIdx.x < N){
		//v_shared[threadIdx.x] = real_fft[threadIdx.x];
		result[threadIdx.x+blockIdx.x*N] += v_shared[threadIdx.x] + v_shared[blockIdx.x] + aL1_beta;
	}
}

template<typename T>
__global__ void compute_wx_wy(T* Ux, T* Uy, T* bx, T* by, T* Wx, T* Wy, int rows, int cols, double tau){

	int index = threadIdx.x + blockIdx.x * cols;

	if (index < rows*cols){

		T xr = Ux[index] + bx[index];
		T yr = Uy[index] + by[index];

		T vr = sqrt(xr*xr+yr*yr);
		if (vr <= tau){
			Wx[index] = 0.0; 
			Wy[index] = 0.0;
        }
        else{
			vr = (vr - tau) / vr;
			Wx[index] = xr*vr; 
			Wy[index] = yr*vr;
		}
	}
}

// FB = fft2(U)/nn; --> w cuda divide_and_abs_fft2
// picks = find(abs(FB)>thresh);
// B = FB(picks); --> B jest typu complex
// Numer1 = zeros(m,n); Numer1(picks) = sqrt(m*n)*B; --> B jest typy complex
// Denom1 = zeros(m,n); Denom1(picks) = 1;
template <typename T, typename C>
__global__ void set_Numer1_and_Denom1(C* Numer1, T* Denom1, T* abs_fb, C* fft2_complex_data, T thresh, double* maxU, double* minU, int fft2_m, int fft2_n, int n_sqrt){

	/*
	UWAGA TU BÊDZIE PROBLEM Z INDEKSOWANIEM FFT2_COMPLEX_DATA - PRZECIEZ TO NIE JEST PELNA MACIERZ MxN

	wywo³anie kernela: <<<N, N>>>
	 
	T - real type (float or double)
	C - complex type (cufftFloatComplex or cufftDoubleComplex)

	Numer1 - pointer to complex array Numer1
	Denom1 - pointer to real array Denom1

	abs_fb				(of size n_sqrt*n_sqrt)				-- abs value of fft2, abs_fb is of size n_sqrt*n_sqrt
	fft2_complex_data   (of size size n_sqrt*(n_sqrt/2+1))  -- fft2 values, fft2_complex_data is of size n_sqrt*(n_sqrt/2+1);

	median_max - median*max(10+k, 10+K);
	variance - value of variance

	fft2_m - 
	fft2_n -- ??
	*/

	int x = threadIdx.x;
	int y = blockIdx.x;
	int out_index = x + y*fft2_m;
	/* indeksowanie stare
	int x2 = (fft2_m - x) % fft2_m;
	int y2 = (fft2_n - y) % fft2_n;

	int cut_cols = fft2_n/2+1;

	int in_index = (x + y*cut_cols)*(x < cut_cols) + (x2 + y2*cut_cols)*(x >= cut_cols);
	*/
	
	double fctr = maxU[0] - minU[0];
	double mn = fft2_m*fft2_n;
//	T thresh = variance[0]/mn;  // pamiêtajmy, ¿e variance to jest suma nie znormalizowana dlatego dzielimy przez mn
//	thresh *= median_max;
	
	if (out_index < mn){
		T b = abs_fb[out_index];
		//C f2 = fft2_complex_data[in_index];
		C f2 = getComplexFromCufftMatrix(y, x, fft2_m, fft2_n, fft2_m, fft2_n/2+1, fft2_complex_data);  // uwaga, tutaj zamieniamy miejscami x i y bo oznaczaj¹ x roœnie w dó³, y w prawo
		
		f2.x /= n_sqrt;
		f2.y /= n_sqrt;

		f2.x /= fctr;
		f2.y /= fctr;
		
		Numer1[out_index].x = (b>thresh)*(sqrt(mn)*f2.x);
		Numer1[out_index].y = (b>thresh)*(sqrt(mn)*f2.y); // ale ale pamietaj, ze indeksowanie to N sprzê¿one --> to ju¿ robi funkcja getComplexFromCufftMatrix
		Denom1[out_index] = (b>thresh)*1.0;
	}
}

template<typename T>
__global__ void compute_Ux_Uy_column_major_order(T* U, T* Ux, T* Uy, int rows, int cols){

	// <<< threads = rows, blocks = cols >>>

	// shuffle instructions ??
	int index = threadIdx.x+blockIdx.x*rows;

	int ux_index = (index + rows) % (rows*cols);
	int uy_index = index + 1 - (rows*(threadIdx.x == (cols-1)));

	if (threadIdx.x < cols && blockIdx.x < rows){

		T u = U[index];
		Ux[index] = U[ux_index] -u;
		Uy[index] = U[uy_index] -u;
	}
}

template<typename T>
__global__ void compute_Ux_Uy_row_major_order(T* U, T* Ux, T* Uy, int rows, int cols){

	// mo¿e t¹ funkcjê rozbiæ na dwa kernele, obliczaj¹ce Ux i Uy i wtedy mo¿na by uzyskaæ memory access coalesced albo shuffle instructions
	//shuffle instructions ??
	int index = threadIdx.x+blockIdx.x*cols;

	// TODO przekopiowane rozwi¹zanie z góry, czy w wersje column i row powinny siê zamieniaæ rows z cols ??? chyba nie!!!
	int ux_index = (threadIdx.x + 1) % cols + blockIdx.x*cols;
	int uy_index = (index + cols) % (rows*cols);

	if (threadIdx.x < rows && blockIdx.x < cols){

		T u = U[index];
		Ux[index] = U[ux_index] -u;
		Uy[index] = U[uy_index] -u;
	}
}

template<typename T>
__global__ void bregman_update(T* b, T* U, T* W, int rows, int cols, T gamma){

	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < (rows*cols); i += blockDim.x*gridDim.x){
		b[i] += gamma*(U[i]-W[i]);
	}
}

template <typename T> 
__global__ void bregman_update(T* bx, T* Ux, T* Wx, T* by, T* Uy, T* Wy, int n, T gamma){

	if (blockIdx.x < gridDim.x/2){
		for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x/2){
			bx[i] += gamma*(Ux[i]-Wx[i]);
		}
	}
	else{
		for (int i = threadIdx.x + (blockIdx.x - gridDim.x/2)*blockDim.x; i < n; i += blockDim.x*gridDim.x/2){
			by[i] += gamma*(Uy[i]-Wy[i]);
		}
	}
}

template <typename T> __global__ void calculate_prd(T* prd_array, T* reduction_result, T* Denom, T aTV, int m, int n, T beta){
	
	T nnz = reduction_result[0];// - Denom[0];
	T new_aTV = nnz / sqrt((double)m*n) * aTV;
	T prd = sqrt((double)new_aTV*beta);

	if (threadIdx.x < 2){
		prd_array[threadIdx.x] = prd * (threadIdx.x == 0) - prd * (threadIdx.x == 1);
	}
}

template <typename T>
__global__ void order_fft2_data(cufftDoubleComplex * idata, T *odata, int fft2_rows, int fft2_cols){

	int difference = fft2_rows - fft2_cols;
	int index = threadIdx.x + blockIdx.x*fft2_rows;
	int index2 = index + index/fft2_cols*difference;//difference*blockIdx.x + difference*(threadIdx.x >= fft2_cols);
	
	odata[index2] = idata[index].x;
}


template <typename T> __global__ void divide_and_abs_fft2(cufftDoubleComplex * U_fft2, T *abs_fft2, T* median_array, int odata_len, int fft2_m, int fft2_n, T denominator){
	
	/*
	funkcja napisana z myœl¹ o wykonaniu operacji dzielenia a potem obliczenia modulu liczby zespolonej, na danych uzyskanych z fft2
	przyk³adowo wykonalismy wczesniej y = fft2(x)
	i teraz chcemy zrobiæ za pomoc¹ tej funkcji: abs(y/n)

	U_fft2 - macierz wynikowa fft2 z cufft (obciêty rozmiar)
	abs_fft2 - macierz rzeczywista (do wpisywania wyniku) rozmiaru NxN (do tego dobrana jest konfiguracja kernela <<<N,N>>>)
	median_array - tablica do która ma wygl¹daæ tak samo jak abs_fft2, a s³u¿yæ bêdzie do obliczania mediany

	odata_len - fft2 array length for cufft = n*(n/2+1);
	fft_m - rozmiar m (wysokoœæ) macierzy abs_fft2
	fft_n - romizar n (szerokoœæ) macierzy abs_fft2

	denominator - dzielnik

	Kontekst dzia³ania oraz konfiguracja kernela:
	- do napisania

	DANE NA JAKICH FUNKCJA OPERUJE:
	- dane s¹ u³o¿one kolumnowo
	| | | | | | | | | |
	| | | | | | | | | |
	| | | | | | | | | |
	| | | | | | | | | |

	width - n
	height - n/2+1
	

	KONFIGURACJA KERNELA:
	<<<N, N>>> dla macierzy o rozmiarze NxN (nie jest to macierz fft2 z cufft)

	bloki to pojedyncza (pionowa) tablica obejmuj¹ca jedn¹ kolumnê macierzy odata

	BLOK w¹tków:

	_
   | |  X    Y -->
   | |  
   | |  |
   | |  v
   | |
   | |
   | |
   | |
   |_|

	co za tym idzie:
	x - roœnie w dó³ (jest wspó³rzêdn¹ w¹tku w bloku)
	y - roœnie w prawo (jest wspó³rzêdn¹ bloku w siatce)

	pierwszym problemem jest odpowiednie indeksowanie, cufft robi fft2 oszczednie, wiec aby zmapowac abs(y/n) na cala macierz trzeba to robiæ z g³ow¹
	*/

	int x = threadIdx.x;
	int y = blockIdx.x;

	int x2 = (fft2_m - x) % fft2_m;		// indeksowanie
	int y2 = (fft2_n - y) % fft2_n;		// indeksowanie

	int cut_cols = fft2_n/2+1;

	int out_index = x + y*fft2_m;  // pytanie czy czymœ to siê ró¿ni od threadIdx.x + blockIdx.x*blockDim.x - ró¿ni siê jak widaæ fft2_m a blockDim.x
	int in_index = (x + y*cut_cols)*(x < cut_cols) + (x2 + y2*cut_cols)*(x >= cut_cols); // ale ale kolego!! dla fft2 nie tylko zmieniamy indeks ale bierzemy 
																						 // sprzê¿enie wartoœci zespolonej !
																						// tylko, ¿e tutaj to nie ma znaczenia - obliczamy modu³ liczby zespolonej

	if(in_index < odata_len){
		cufftDoubleComplex c = U_fft2[in_index];
		c.x /= denominator;				// division
		c.y /= denominator;				// division

		T x_pow = c.x*c.x;
		T y_pow = c.y*c.y;
		T result = sqrt(x_pow+y_pow);	// abs

		abs_fft2[out_index] = result;
		median_array[out_index] = result;
	}

}

__global__ void Wx_Wy_anisotropic(double * Wx, double * Wy, double * Ux, double * Uy, double * bx, double * by, int n, double beta){

	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x){
		Ux[i] += bx[i];
		Uy[i] += by[i];
		double ux = Ux[i];
		double uy = Uy[i];
		double wx = fmax(abs(ux) - 1/beta, 0.0);
		double wy = fmax(abs(uy) - 1/beta, 0.0);

		Wx[i] = wx * ((1.0)*(ux>0.0) + (-1.0)*(ux<0.0));
		Wy[i] = wy * ((1.0)*(uy>0.0) + (-1.0)*(uy<0.0));
	}
}
/*
__global__ void Ux_Uy_anisotropic(double * Ux, double * Uy, double * bx, double * by, int n){

	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x){
		Ux[i] += bx[i];
		Uy[i] += by[i];
	}
}
*/

__global__ void z_subproblem(double * PsiTU, double * d, double * Z, int n, double beta){

	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x){
		
		double dv = d[i];
		PsiTU[i] += dv;
		double p = PsiTU[i];

		double m = fmax(abs(p) - 1/beta, 0.0);
		Z[i] = m * ((1.0)*(p>0.0) + (-1.0)*(p<0.0)) - dv;
	}
}

__global__ void rhs_aL1(double * rhs, double * dct, int n, double aL1, double beta){

	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x){
		rhs[i] += aL1*beta*dct[i];
	}
}

__global__ void update_d(double * d, double * PsiTU, double *Z, int n, double gamma){

	for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x){
		d[i] += gamma * (PsiTU[i] - Z[i]);
	}
}
