/*=================================================================
* syntax: B = biadjacency_matrix(A,M,K) OR B = biadjacency_matrix(A,M)
*
* BIADJACENCY_MATRIX  - build biadjacancy matrix from assignment image
* 			
* 			Input: 	- A: X-by-Y assignment image
* 					- M: patch size (length of edge)
*                   - K: number of dictionary patches, defaults to max(A(:))
*
* 			Output: - B: XY-by-MMK sparse biadjacency matrix 
*
* 			Author: Vedrana Andersen Dahl, vand@dtu.dk, december 2015.
*=================================================================*/

#include <math.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "mex.h"

// struct containing i and j indices of a sparse matrix element
struct ij
{
    int i,j;
    ij(int i_, int j_) : i(i_), j(j_){}; // constructor
    bool operator<(const ij second) const{ // leq operator needed for sorting
        return (j<second.j) || ((j==second.j) && (i<second.i));
    }    
};

// The gateway mex routine
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Check for proper number of input and output arguments */    
    if ((nlhs != 1) || (nrhs != 2))
        mexErrMsgTxt("Usage: B = BIADJANCENCY_MATRIX(A,M) OR "
                "B = BIADJANCENCY_MATRIX(A,M,K).\n");
        
    /* Read input */
    double *A = mxGetPr(prhs[0]); // assignment image
    int X = mxGetM(prhs[0]); // image size X
	int Y = mxGetN(prhs[0]); // image size Y
    int M = (int)mxGetPr(prhs[1])[0]; // patch size
    int K; // number dict patches
    if (nrhs==3)
        K = (int)mxGetPr(prhs[2])[0]; 
    else{ // assumes number of dict patches is max(A)
        K = 0;
        for (int a=0; a<X*Y; a++)
            if (A[a]>K)
                K = A[a];
    }
    
    /* Compute some useful sizes */
    int c = (M-1)/2; // width of boundary having no assignment 
    int n = X*Y; // number of image pixels
    int m = M*M*K; // number of dict pixels
    int s = (X-M+1)*(Y-M+1)*M*M; // number image-dict links (elements in B)
    
    /* Finding elements of B as row-column indices */
    std::vector<ij> bij;
    bij.reserve(s); 
    int ic,i,j;   
    for (int y=0+c; y<Y-c; y++){ // visiting patches centered around pixels
        for (int x=0+c; x<X-c; x++){
            ic = x+y*X; // central patch pixel
            for (int dy=-c; dy<=c; dy++){ // visiting pixels around central
                for (int dx=-c; dx<=c; dx++){
                    i = (x+dx)+(y+dy)*X;
                    j = (c+dx)+(c+dy)*M+(A[ic]-1)*M*M;
                    bij.push_back(ij(i,j));
                }
            }
        }
    }
    
    /* Sorting elements in bij columnwise */
    std::sort (bij.begin(), bij.end());    
    
    /* Placeholder for output */
    plhs[0] = mxCreateSparseLogicalMatrix(n,m,s); // output mxArray, sparse logical matrix B
    if (plhs[0]==NULL)
        mexErrMsgTxt("Could not allocate enough memory!\n");
    
    /* Access fields of output mxArray via pointers  */
    mwIndex *ir = mxGetIr(plhs[0]); // row index (0 indexed) 
    mwIndex *jc = mxGetJc(plhs[0]); // cumulative number of elements per column 
    mxLogical *pr = mxGetLogicals(plhs[0]); // element values (will be all true)
        
    /* Converting row-column indices into row-cumulative column  */
    int k = 0; // for visiting elements of bij
    jc[0] = 0; // first element of cumulative sum is 0
    for (int bc=0; bc<m; bc++){ // all columns of B        
        jc[bc+1] = jc[bc]; 
        while (k<bij.size() && bij[k].j==bc){
            jc[bc+1]++;
            ir[k] = bij[k].i;
            pr[k] = true;
            k++;
        }
    }
}
 
