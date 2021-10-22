/*=================================================================
* syntax: Q = probability_search_km_tree(I, T, P, b, n); OR Q = search_km_tree(I, T, P, b);
*
* serach_km_tree  - build probability image from dictionary and intensity image
* 			
* 			Input: 	- I: X-by-Y image
* 					- T: MMl-by-K tree matrix where l is the number of layers 
*                        in the image (1 for grayscale and 3 for RGB)
                    - P: MMc-by-K probability matrix where c is the number of 
                         classes
*                   - b: branching factor
*                   - n: normalization (true or false), defaults to false
*
* 			Output: - Q: X-by-Y-by-c probability image
*
* 			Author: Anders Dahl, abda@dtu.dk, February 2016.
*=================================================================*/

#include "mex.h"
#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include <vector>

#include <iostream>
using namespace std;

// struct for the tree
struct tree_st
{
    double *tree_data;
    double *probability_data;
    int n_dim, n_dprob, n_nodes, branch_fac, M, Mh, n_cls;
    vector<int> idx_disp;
};

// struct for image
struct im_st
{
    double *im_data;
    int rows, cols, layers, n_pix;
};

// estimate the distance between a vector in tree given by the node and a 
// patch in the image given by the row and column
double get_dist(vector<double>& patch, tree_st& tree, int& node)
{
    double d = 0, tmp;
    int id_t = tree.n_dim*node;
    
    for ( int i = 0; i < tree.n_dim; i++ ){
        tmp = patch[i] - *(tree.tree_data + id_t);
        id_t++;
        d += tmp*tmp;
    }
    
    return d;
}


// Function for sampling patches from the image into the patch arrays
// inputs reference to the image struct, tree struct, patch struct and position of the sampling coordinate.
// There is no check if the sampling is outside the image
// vector<double> sample_patch(im_st& im, int& M, int r_im, int c_im, bool& normalize)
vector<double> sample_patch(im_st& im, int& M, int r_im, int c_im, bool& normalize)
{
    int id_l, id_r, id_i; // iterators for looking up image data
    int id_p = 0; // iterator for looking up patch data
    double sum_sq = 0, pix_val; // variables for normalization
    int n_dim = M*M*im.layers; // number of dimensions computed here, becasue tree is not included
    vector<double> patch(n_dim);
    int Mh = (M-1)/2;
    
    for ( int l = 0; l < im.layers; l++ ){ // image is sampled by three nested loops (layers, columns, rows)
        id_l = im.n_pix*l;
        for ( int i = c_im-Mh; i <= c_im+Mh; i++ ){
            id_r = id_l + i*im.rows;
            for ( int j = r_im-Mh; j <= r_im+Mh; j++ ){
                id_i = id_r + j;
                pix_val = *(im.im_data + id_i);
                patch[id_p] = pix_val;
                sum_sq += pix_val*pix_val; // sum of squares for normalization
                id_p++;
            }
        }
    }
    if ( normalize ){ // if the patch should be normalized to unit length
        double inv_sq = 1;
        if ( sum_sq > 0 ){
            inv_sq = 1/sqrt(sum_sq); // inverse sum of squares
        }
        for ( int i = 0; i < n_dim; i++ ){
            patch[i] = patch[i]*inv_sq;
        }
    }
    return patch;
}


// The tree search function
int search_tree(im_st& im, tree_st& tree, int& r, int& c, bool& normalize)
{
    int node = 0, node_min = -1, node_min_level, next_node; // variables for searching the tree
    double d_min = 10e100, d, d_min_level; 
    
    vector<double> patch = sample_patch(im, tree.M, c, r, normalize); // get the pixel values in a patch
    while ( node < tree.n_nodes ){ // go through the tree
        if ( *(tree.tree_data + node*tree.n_dim) == -1 ){ // check if node is a leaf-node
            return node_min;
        }
        
        d_min_level = 10e100; // set minimum distance to high value
        for ( int i = 0; i < tree.branch_fac; i++ ){ // go through nodes at level 
            next_node = node + i;
            d = get_dist(patch, tree, next_node);
            
            if ( d < d_min_level ){ // set current node to the minimum distance
                d_min_level = d;
                node_min_level = next_node;
            }
        }
        if ( d_min_level < d_min ){ // set overall minimum distance and minimum node
            d_min = d_min_level;
            node_min = node_min_level;
        }
        node = (node_min_level+1)*tree.branch_fac; // go to the child node
    }
    return node_min;
}

void add_probability(tree_st& tree, double *Q, int& idx, int& node){
    int id_node = node*tree.n_dprob;
    for ( int i = 0; i < tree.idx_disp.size(); i++ ){
        *(Q + idx + tree.idx_disp[i]) += *(tree.probability_data + id_node + i);
    }
}


void add_count(vector<int>& ct_im, int& col, int& row, int& rows, tree_st& tree){
    for ( int i = col-tree.Mh; i <= col+tree.Mh; i++ ){
        for ( int j = row-tree.Mh; j <= row+tree.Mh; j++ ){
            ct_im[i*rows+j]++;
        }
    }
}

// The tree search function applied to the entire image - border is zero and interior is in 1,...,n
void probability_search_image(im_st& im, tree_st& tree, double *Q, bool& normalize)
{
    vector<int> ct_im;
    ct_im.resize(im.rows*im.cols);
    int node;
    int idx = tree.Mh*im.rows; // increase with empty rows at border
    for ( int i = tree.Mh; i < im.cols-tree.Mh; i++ ){
        idx += tree.Mh; // first Mh pixels are border
        for ( int j = tree.Mh; j < im.rows-tree.Mh; j++ ){           
            node = search_tree(im, tree, i, j, normalize); // find assignment
            add_probability(tree, Q, idx, node);
            idx++;
            add_count(ct_im, i, j, im.rows, tree);
        }
        idx += tree.Mh; // last Mh pixels are border
    }
    
    int l_id;
    for ( int j = 0; j < tree.n_cls; j++) {
        l_id = j*im.rows*im.cols;
        for ( int i = 0; i < ct_im.size(); i++ ){
            *(Q+i+l_id) = *(Q+i+l_id)/(double)ct_im[i];
        }
    }
}


// The gateway routine 
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  // input image (I), tree (tree), probabilities (P) and output probabilities (Q)
  double *I, *tree, *P, *Q;
  int b, M, ndim, ndtree, ndprob;
  const int *dim, *dtree, *dprob;
  bool normalize = false;
  /*  Check for proper number of arguments. */
  /* NOTE: You do not need an else statement when using
     mexErrMsgTxt within an if statement. It will never
     get to the else statement if mexErrMsgTxt is executed.
     (mexErrMsgTxt breaks you out of the MEX-file.) 
  */
  if(nrhs < 4 || nrhs > 5) 
    mexErrMsgTxt("Four or five inputs required.");
  if(nlhs != 1) 
    mexErrMsgTxt("One output required.");
    
  // Create a pointer to the input matrix.
  I = mxGetPr(prhs[0]);
  tree = mxGetPr(prhs[1]);
  P = mxGetPr(prhs[2]);
  
  double *bd;
  bd = mxGetPr(prhs[3]);
  b = (int)bd[0];
  
  if ( nrhs == 5 ){
      bool *normalize_d;
      normalize_d = (bool *)mxGetData(prhs[4]);
      normalize = normalize_d[0];
  }
  
  if ( b < 1 )
    mexErrMsgTxt("b must be positive.");
  
  // Get the dimensions of the matrix input.
  ndim = mxGetNumberOfDimensions(prhs[0]);
  if (ndim != 2 && ndim != 3)
    mexErrMsgTxt("probability_search_km_tree only works for 2-dimensional or 3-dimensional images.");

  ndtree = mxGetNumberOfDimensions(prhs[1]);
  if (ndtree != 2)
    mexErrMsgTxt("probability_search_km_tree requires tree to be a matrix (2-dimensional).");

  ndprob = mxGetNumberOfDimensions(prhs[2]);
  if (ndprob != 2)
    mexErrMsgTxt("probability_search_km_tree requires probability matrix to be a matrix (2-dimensional).");

  dim = mxGetDimensions(prhs[0]);
  dtree = mxGetDimensions(prhs[1]);
  dprob = mxGetDimensions(prhs[2]);
  
  if ( dtree[1] != dprob[1] )
    mexErrMsgTxt("Tree matrix and probability matrix must have the same number of columns.");
  
  if ( ndim == 3 )
  {
      M = (int)sqrt((double)dtree[0]/(double)dim[2]);
  }
  else
  {
      M = (int)sqrt((double)dtree[0]);
  }
  
  if ( 1 - (M % 2)  || M < 1)
    mexErrMsgTxt("M must be odd and positive.");
  
  
  // image struct
  im_st Im;
  Im.im_data = I;
  Im.rows = dim[0];
  Im.cols = dim[1];
  if ( ndim == 3 )
  {
      Im.layers = dim[2];
  }
  else
  {
      Im.layers = 1;
  }
  Im.n_pix = Im.rows*Im.cols;

  // tree struct
  tree_st Tree;
  Tree.tree_data = tree;
  Tree.probability_data = P;
  Tree.n_dim = dtree[0];
  Tree.n_nodes = dtree[1];
  Tree.n_dprob = dprob[0];
  Tree.branch_fac = b;
  Tree.M = M;
  Tree.Mh = (int)(0.5*(double)(M-1.0));
  // Number of classes
  Tree.n_cls = dprob[0]/(M*M);
  // Compute relative displacement vector
  Tree.idx_disp.reserve(dprob[0]);
  int id;
  for ( int k = 0; k < Tree.n_cls; k++ ){
      id = -Tree.Mh*Im.rows - Tree.Mh + k*Im.rows*Im.cols;
      for ( int i = 0; i < M; i++ ){
          for ( int j = 0; j < M; j++ ){
              Tree.idx_disp.push_back(id);
              id++;
          }
      id+=Im.rows-M;
      }
  }
//   for ( int i = 0; i < Tree.idx_disp.size(); i++ ){
//       cout << Tree.idx_disp[i] << " ";
//   }
//   cout << endl;

  
  if ( M*M*Im.layers != Tree.n_dim )
    mexErrMsgTxt("Dimensions of the tree and the image does not fit.");
  
  int ndpout[3] = {Im.rows, Im.cols, Tree.n_cls};
  
  // Set the output pointer to the output matrix. Array initialized to zero. 
  plhs[0] = mxCreateNumericArray(3, ndpout, mxDOUBLE_CLASS, mxREAL);
  
  // Create a C pointer to a copy of the output matrix.
  Q = mxGetPr(plhs[0]);
  // Search the tree using the C++ subroutine
  probability_search_image(Im, Tree, Q, normalize);
}

