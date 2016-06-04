/*
	cuda implementation of RecPF algorithm based on matlab function version

	[U,Out] = RecPF(m,n,aTV,aL1,picks,B,TVtype,opts,PsiT,Psi,URange,uOrg) - deklaracja funkcji w Matlabie

	poni¿ej przyk³adowe wywo³anie funkcji RecPF w matlabie ze skryptu sart_tv
	[UU,Out_RecPF] = RecPF(nn,nn,aTV,aL1,picks,B,2,opts,PsiT,Psi,range(U(:)),U);

	U - macierz
	U(:) - wektor, dodaje kolejno kolumny macierzy U
	range - dla wektora, zwraca ró¿nice miêdzy najbardziej skrajnymi wartoœciami (najmniejsz¹ i najwiêksz¹)
			innymi s³owy, zwraca najwiêksz¹ ró¿nicê miêdzy dowolnymi dwoma elementami

	opis parametrów:
	czy parametr jest skalarem, wektorem, macierz¹, typ danych itp.

	m - nn - wartoœæ typu int
	n - nn - wartoœæ typu int
	aTV - sta³a double 
	aL1 - sta³a double
	picks - wektor zawieraj¹cy indeksy, patrz: picks = find(abs(FB)>thresh);
	B - B = FB(picks); - tworzy wektor B z wartoœciami z macierzy FB odpowiadaj¹cymi indeksom picks
	TVtype -- 2 (isotropic) or 1 (anisotropic) (w przyk³adzie wartoœæ 2)
	opts
	Psit - chyba funkcja
	Psi - chyba funkcja
	URange  - range(U(:))
	uOrg -- (optional) true image - macierz

	-----w sart_tv: ----------
	fb = FB(:);
	U = reshape(xx,nn,nn); 
    FB = fft2(U)/nn;%sqrt(n);
	thresh = var(abs(fb))*median(abs(fb(2:end)))*max(10+k,10+K);%(K-k+1);
    picks = find(abs(FB)>thresh);
	B = FB(picks);
	----------------------------

	tresh - ta wartoœæ jest obliczana w sposób doœæ skomplikowany
	
	picks, B oraz U, które s¹ argumentami w wywo³aniu RecPF
	B = FB(picks); - tworzy wektor B z wartoœciami z macierzy FB odpowiadaj¹cymi indeksom picks
*/



/*
U = zeros(m,n);     % initial U. 
                    % If changing to anything nonzeor, you must change the 
                    % initialization of Ux and Uy below
					*/

#define BATCH_1 1

int m = 5400, n = 2500;
int mn = m*n;
int normalize = 1;
int prd_fft_output_size = m*(n/2+1);

// ----------------------------- initializing part --------------------------------
double *U, *Numer1, *Denom1, *Denom2, *Denom2_work, *prd_fft2;
checkCudaErrors(cudaMalloc((void**)&U, mn*sizeof(double)));
checkCudaErrors(cudaMalloc((void**)&Numer1, mn*sizeof(double)));
checkCudaErrors(cudaMalloc((void**)&Denom1, mn*sizeof(double)));

checkCudaErrors(cudaMalloc((void**)&Denom2, mn*sizeof(double)));
checkCudaErrors(cudaMalloc((void**)&Denom2_work, mn*sizeof(double)));
checkCudaErrors(cudaMalloc((void**)&prd_fft2, (mn+1)*2*sizeof(double))); //mno¿ymy razy 2 bo to dla zespolonych, powinno byæ sizeof(cuDoubleComplex)

double *Ux, *Uy, *bx, *by;
checkCudaErrors(cudaMalloc((void**)&Ux, mn*sizeof(double)));
checkCudaErrors(cudaMalloc((void**)&Uy, mn*sizeof(double)));
checkCudaErrors(cudaMalloc((void**)&bx, mn*sizeof(double)));
checkCudaErrors(cudaMalloc((void**)&by, mn*sizeof(double)));

double *PsiTU, *Z, *d;
if (aL1 > 0){
	checkCudaErrors(cudaMalloc((void**)&PsiTU, mn*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&Z, mn*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d, mn*sizeof(double)));
	
	// memset to zero
	checkCudaErrors(cudaMemset(PsiTU, 0, mn*sizeof(double)));
	checkCudaErrors(cudaMemset(Z, 0, mn*sizeof(double)));
	checkCudaErrors(cudaMemset(d, 0, mn*sizeof(double)));
}

checkCudaErrors(cudaMemset(U, 0, mn*sizeof(double)));
checkCudaErrors(cudaMemset(Numer1, 0, mn*sizeof(double)));
checkCudaErrors(cudaMemset(Denom1, 0, mn*sizeof(double)));
checkCudaErrors(cudaMemset(Ux, 0, mn*sizeof(double)));
checkCudaErrors(cudaMemset(Uy, 0, mn*sizeof(double)));
checkCudaErrors(cudaMemset(bx, 0, mn*sizeof(double)));
checkCudaErrors(cudaMemset(by, 0, mn*sizeof(double)));

if (normalize){

}

// prd = sqrt(aTV*beta);
double prd = sqrt(aTV*beta);

/*
algorytm do obliczania Denom2 = abs(psf2otf([prd,-prd],[m,n])).^2 + abs(psf2otf([prd;-prd],[m,n])).^2; 

abs(psf2otf([prd,-prd],[m,n])).^2 = abs(fft([prd,-prd], n)).^2 (gdzie fft daje tylko wiersz, który nale¿y powieliæ)
abs(psf2otf([prd;-prd],[m,n])).^2 = abs(fft([prd;-prd], m)).^2 (gdzie fft daje tylko kolumnê, któr¹ nale¿y powieliæ)

*/
// Denom2 = abs(psf2otf([prd,-prd],[m,n])).^2 + abs(psf2otf([prd;-prd],[m,n])).^2; % mozemy na potrzeby CUDA zastapic to wywolaniem: fft([prd, -prd], n) - kolumny s¹ takie same

cufftHandle plan1D_n, plan1D_m;
cufftComplex *output_2, *output_1;
cufftReal * input_1, *input_2;

cudaMalloc((void**)&output_1, sizeof(cufftComplex)*(n/2+1)); // a mo¿e BATCH = 2?
cudaMalloc((void**)&output_2, sizeof(cufftComplex)*(m/2+1));
cudaMalloc((void**)&input_1, sizeof(double)*n);
cudaMalloc((void**)&input_2, sizeof(double)*m);

// padding data wih zero [prd,-prd]
cudaMemset(input_1, 0, sizeof(double)*n)
cudaMemset(inptu_2, 0, sizeof(double)*m);
input_1[0] = prd; input_1[1] = -prd;
input_2[0] = prd; input_2[1] = -prd;

// cufft plan
cufftPlan1d(&plan_n, n, CUFFT_RC2, BATCH_1);
cufftPlan2d(plan1D_m, m, CUFFT_RC2, BATCH_1);

// ?? pytanie czy w przypadku tej transformaty mo¿emy j¹ zrobiæ w miejscu??
// cufftResult cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftReal *odata);
cufftExecRC2(plan_n, input_1, output_1);
cufftExecRC2(plan_m, input_2, output_2);

cudaFree(input_1);
cudaFree(input_2);
cudaFree(output_1);
cudaFree(output_2);

// ------------------------------- MAIN LOOP ----------------------------------
int maxItr = opts.maxItr;
int i;
for(i=0; i<max_iter; i++){

	// ---------------- w naszej wersji domyœlnie TV_type jest 2 -------------
	if (TV_type == 1){
		/*
			% anisotropic TV
            Ux = Ux + bx; Uy = Uy + by;      % latest Ux and Uy are already calculated
            Wx = sign(Ux).* max(abs(Ux)-1/beta,0);
            Wy = sign(Uy).* max(abs(Uy)-1/beta,0);
		*/
	}
	else if(TV_type == 2){
		/*
			% isotropic TV
            [Wx, Wy] = Compute_Wx_Wy(Ux,Uy,bx,by,1/beta);
		*/
	}
	else{
		// error
	}

    //   Z-subprolem
    if (aL1 > 0){
        // PsiTU = PsiTU + d;
        // Z = sign(PsiTU).*max(abs(PsiTU)-1/beta,0);
    }
}

// ------------------------------- cleanup part --------------------------------
cudaFree(U);
cudaFree(Numer1);
cudaFree(Denom1);
cudaFree(Denom2);
cudaFree(Ux);
cudaFree(Uy);
cudaFree(bx);
cudaFree(by);

cudaFree(prd_fft2);

if (aL1 > 0){
	cudaFree(PsiTU);
	cudaFree(Z);
	cudaFree(d);
}


__global__ __device__ void fill_after_fft(cufftComplex* v, int N, int n){

	int index = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (index > N && index < n){
		v[index] = v[N-index];
	}
}