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
algorytm do obliczania Denom2 = abs(psf2otf([prd,-prd],[m,n])).^2 + abs(psf2otf([prd;-prd],[m,n])).^2; 

abs(psf2otf([prd,-prd],[m,n])).^2 = abs(fft([prd,-prd], n)).^2 (gdzie fft daje tylko wiersz, który nale¿y powieliæ)
abs(psf2otf([prd;-prd],[m,n])).^2 = abs(fft([prd;-prd], m)).^2 (gdzie fft daje tylko kolumnê, któr¹ nale¿y powieliæ)

*/


