#include "mex.h"
#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include <vector>

#include <iostream>
using namespace std;

/*
 * This is a MEX-file for MATLAB.
 * 
 * Anders Bjorholm Dahl, 8/12-2015
 *
 *
 */

// struct for the tree
struct tree_st
{
    double *tree_data;
    int n_dim, inv_ndim, n_nodes, branch_fac, M, Mh;
};

// struct for image
struct im_st
{
    double *im_data;
    int rows, cols, layers, n_pix;
};

// estimate the distance between a vector in tree given by the node and a 
// patch in the image given by the row and column
double get_corr(vector<double>& patch, tree_st& tree, int& node)
{
    double corr = 0, tmp;
    int id_t = tree.n_dim*node;
    
    for ( int i = 0; i < tree.n_dim; i++ ){
        corr += patch[i]*(*(tree.tree_data + id_t));
        id_t++;
    }
    
    return corr;
}

// nomalize patch vectors to zero mean and unit Euclidean length
void norm_patch(vector<double>& patch, double& mu, int n_dim){
    
    double sum_sq = 0;
    
    for ( int i = 0; i < n_dim; i++ ){
        patch[i] = patch[i] - mu;
        sum_sq += patch[i]*patch[i];
    }
    
    double inv_sq = 1;
    if ( sum_sq > 0 ){
        inv_sq = 1.0/sqrt(sum_sq); // divide by sum of squares
    }

    for ( int i = 0; i < n_dim; i++ ){
        patch[i] = patch[i]*inv_sq;
    }
}


// Function for sampling patches from the image into the patch arrays
// inputs reference to the image struct, tree struct, patch struct and position of the sampling coordinate.
// There is no check if the sampling is outside the image
// vector<double> sample_patch(im_st& im, int& M, int r_im, int c_im, bool& normalize)
vector<double> sample_patch(im_st& im, int& M, int r_im, int c_im)
{
    int id_l, id_r, id_i; // iterators for looking up image data
    int id_p = 0; // iterator for looking up patch data
    double sum = 0, pix_val, mu; // variables for normalization
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
                sum += pix_val; // sum for estimating the mean
                id_p++;
            }
        }
    }
    
    mu = sum/((double)n_dim); // mean value
    norm_patch(patch, mu, n_dim); // nomalization

    return patch;
}


// The tree search function
int search_tree(im_st& im, tree_st& tree, int& r, int& c)
{
    int node = 0, node_max = -1, node_max_level, next_node; // variables for searching the tree
    double corr_max = -1, corr, corr_max_level; 
    
    vector<double> patch = sample_patch(im, tree.M, c, r); // get the pixel values in a patch
    while ( node < tree.n_nodes ){ // go through the tree
        if ( *(tree.tree_data + node*tree.n_dim) == -1 ){ // check if node is a leaf-node
            return node_max;
        }
        
        corr_max_level = -1; // set correlation to low value
        for ( int i = 0; i < tree.branch_fac; i++ ){ // go through nodes at level 
            next_node = node + i;
            corr = get_corr(patch, tree, next_node);
            
            if ( corr > corr_max_level ){ // set current node to the maximum correlation
                corr_max_level = corr;
                node_max_level = next_node;
            }
        }
        if ( corr_max_level > corr_max ){ // set overall maximum correlation and corresponding node
            corr_max = corr_max_level;
            node_max = node_max_level;
        }
        node = (node_max_level+1)*tree.branch_fac; // go to the child node
    }
    return node_max;
}

// The tree search function applied to the entire image - border is zero and interior is in 1,...,n
void search_image(im_st& im, tree_st& tree, double *A)
{
    int idx = tree.Mh*im.rows; // increase with empty rows at border
    for ( int i = tree.Mh; i < im.cols-tree.Mh; i++ ){
        idx += tree.Mh; // first Mh pixels are border
        for ( int j = tree.Mh; j < im.rows-tree.Mh; j++ ){           
            *(A + idx) = search_tree(im, tree, i, j) + 1; // find assignment
            idx++;
        }
        idx += tree.Mh; // last Mh pixels are border
    }
}


// The gateway routine 
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
  // input image (I), tree (tree) and output assignment (A)
  double *I, *A, *tree;
  int b, M, ndim, ndtree;
  const int *dim, *dtree;
  /*  Check for proper number of arguments. */
  /* NOTE: You do not need an else statement when using
     mexErrMsgTxt within an if statement. It will never
     get to the else statement if mexErrMsgTxt is executed.
     (mexErrMsgTxt breaks you out of the MEX-file.) 
  */
  if(nrhs != 3 ) 
    mexErrMsgTxt("Three inputs required.");
  if(nlhs != 1) 
    mexErrMsgTxt("One output required.");
    
  // Create a pointer to the input matrix.
  I = mxGetPr(prhs[0]);
  tree = mxGetPr(prhs[1]);
  
  double *bd;
  bd = mxGetPr(prhs[2]);
  b = (int)bd[0];
    
  if ( b < 1 )
    mexErrMsgTxt("b must be positive.");
  
  // Get the dimensions of the matrix input.
  ndim = mxGetNumberOfDimensions(prhs[0]);
  if (ndim != 2 && ndim != 3)
    mexErrMsgTxt("search_km_tree only works for 2-dimensional or 3-dimensional images.");

  ndtree = mxGetNumberOfDimensions(prhs[1]);
  if (ndtree != 2)
    mexErrMsgTxt("search_km_tree only works for 2-dimensional tree.");

  dim = mxGetDimensions(prhs[0]);
  dtree = mxGetDimensions(prhs[1]);
  
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
  
  
  // tree struct
  tree_st Tree;
  Tree.tree_data = tree;
  Tree.n_dim = dtree[0];
  Tree.inv_ndim = 1/((double)Tree.n_dim);
  Tree.n_nodes = dtree[1];
  Tree.branch_fac = b;
  Tree.M = M;
  Tree.Mh = (int)(0.5*(double)(M-1.0));
  
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
  
  if ( M*M*Im.layers != Tree.n_dim )
    mexErrMsgTxt("Dimensions of the tree and the image does not fit.");
  
  // Set the output pointer to the output matrix. Array initialized to zero. 
  plhs[0] = mxCreateNumericArray(ndtree, dim, mxDOUBLE_CLASS, mxREAL);
  
  // Create a C pointer to a copy of the output matrix.
  A = mxGetPr(plhs[0]);
  // Search the tree using the C++ subroutine
  search_image(Im, Tree, A);
}

