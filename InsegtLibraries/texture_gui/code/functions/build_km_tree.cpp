/*=================================================================
* syntax: T = build_km_tree(I, M, b, t, L, n); OR T = build_km_tree(I, M, b, t, L);
*
* build_km_tree  - build km-tree matrix from image
* 			
* 			Input: 	- I: X-by-Y intensity image
* 					- M: patch size (length of edge)
*                   - L: number of dictionary layers. This parameter is limited 
*                        such that the average number of patches in a leafnode is 
*                        greater than five
*                   - b: branching factor
*                   - t: number of training patches
*                   - n: normalization (true or false), defaults to false
*
* 			Output: - T: MMl-by-K matrix where l is the number of layers 
*                        in the image (1 for grayscale and 3 for RGB)
*                        and K is the number of nodes in the tree.
*
* 			Author: Anders Dahl, abda@dtu.dk, december 2015.
*=================================================================*/

#include "mex.h"
#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include <vector>
#include <algorithm>

#include <iostream>
using namespace std;

// struct for image
struct im_st
{
    double *im_data; // pointer to image data
    int rows, cols, layers, n_pix; // rows, cols and layers are the image dimensions and n_pix = rows*cols
};

// struct for the tree
struct tree_st
{
    double *tree_data;
    int n_dim, n_nodes, branch_fac, M, Mh;
};

// struct for image patches
struct im_patch
{
    double *patch_data; // pointer to the data
    int idx; // id used for sorting the patches according to the tree
    bool operator<(const im_patch& rhs) const {return idx < rhs.idx;} // operator needed for sorting
};


// Function for sampling patches from the image into the patch arrays
// inputs reference to the image struct, tree struct, patch struct and position of the sampling coordinate.
// There is no check if the sampling is outside the image
void sample_patch(im_st& im, tree_st& tree, im_patch& patch, int r_im, int c_im, bool normalize)
{
    int id_l, id_r, id_i; // iterators for looking up image data
    int id_p = 0; // iterator for looking up patch data
    double sum_sq = 0, pix_val; // variables for normalization
    
    for ( int l = 0; l < im.layers; l++ ){ // image is sampled by three nested loops
        id_l = im.n_pix*l;
        for ( int i = c_im-tree.Mh; i <= c_im+tree.Mh; i++ ){
            id_r = id_l + i*im.rows;
            for ( int j = r_im-tree.Mh; j <= r_im+tree.Mh; j++ ){
                id_i = id_r + j;
                pix_val = *(im.im_data + id_i);
                *(patch.patch_data + id_p) = pix_val;
                sum_sq += pix_val*pix_val;
                id_p++;
            }
        }
    }
    
    if ( normalize ){ // normalization to unit length
        double inv_sq = 1;
        if ( sum_sq > 0 ){
            inv_sq = 1/sqrt(sum_sq); // divide by sum of squares
        }
        for ( int i = 0; i < tree.n_dim; i++ ){
            *(patch.patch_data + i) = (*(patch.patch_data + i))*inv_sq;
        }
    }

    
}


// function for randomly permuted set of indices
// n is the numbers to choose from, n_set is the number of random indices
// that is returned. Returns a vector of indices
vector<int> randperm( int n, int n_set ) {
    if ( n_set > n ){ // check if n_set exceeds n
        n_set = n;
    }
    
    vector<int> rid;
    rid.reserve(n); // vector of indices
    for ( int i = 0; i < n; i++ ){ // set all indices in order
        rid.push_back(i);
    }
    
    int t, id; // place holders for id and temporary number
    int r_max = RAND_MAX; // maximum random number
    for ( int i = 0; i < n_set; i++ ){
        // choose a random number between i and n-1 and swap place between i and id
        if ( LONG_MAX > RAND_MAX && n-i-1>RAND_MAX ){ // not enough with a random integer up til RAND_MAX
            id = ((rand()*(r_max+1)+rand()) % (n-i)) + i; 
        }
        else{
            id = (rand() % (n-i)) + i; 
        }
        t = rid[id];
        rid[id] = rid[i];
        rid[i] = t;
    }
    rid.resize(n_set); // set size to n_set
    return rid;
}


// copy values from a patch array into the tree array at node
void set_values(tree_st& tree, im_patch& patch, int node){
    int idx = tree.n_dim*node;
    for ( int i = 0; i < tree.n_dim; i++ ){
        *(tree.tree_data + idx) = *(patch.patch_data + i);
        idx++;
    }
}

// add values to vector of cluster center points
void add_values(vector<double>& center_sum, im_patch& patch, int id, int n_dim){
    int idx = n_dim*id;
    for ( int i = 0; i < n_dim; i++ ){
        center_sum[idx] += *(patch.patch_data + i);
        idx++;
    }
}

// estimates the squared Euclidian distance between an image patch and a tree node
double get_dist(tree_st& tree, im_patch& patch, int node)
{
    double d = 0, tmp;
    int id = tree.n_dim*node;
    
    for ( int i = 0; i < tree.n_dim; i++ ){
        tmp = *(tree.tree_data + id) - *(patch.patch_data + i);
        d += tmp*tmp;
        id++;
        
    }
    
    return d;
}

// k-means-function taking a reference to the vector of image patches and a
// tree struct as input f and t gives the image patches that should be clustered.
// node is the first node in the tree included in the clustering
void k_means( vector<im_patch>& patches, tree_st& tree, int f, int t, int node )
{
    // vectors holding the sum of values in the cluster and a vector containing the change
    vector<double> centSum(tree.branch_fac*tree.n_dim), centChange(tree.branch_fac);
    vector<int> centCount(tree.branch_fac); // vector for counting the number of points in a cluster
    double min_d, d, tmp;//, diff; // variables for clustering
    // variables for cluster id and index of previous cluseter, which will be overwritten by new cluster id
    int id = 0, id_in = patches[f].idx;
    
    if ( t-f > tree.branch_fac ){ // check if there are enough point to carry out the clustering
        // initialize the center positions
        for (  int i = 0; i < tree.branch_fac; i++ ){
            set_values(tree, patches[f+i], node+i);
        }
        
        // run clutering for 20 iterations - only little change happens after 10 iterations
        for ( int n_run = 0; n_run < 30; n_run++){
            
            for ( int i = f; i < t; i++ ){ // go through the patches from f to t
                min_d = get_dist(tree, patches[i], node); // initially set min distance and id to the first
                id = 0;
                for ( int j = 1; j < tree.branch_fac; j++ ){ // run throgh the other points
                    d = get_dist(tree, patches[i], node + j); // get the distance
                    if ( d < min_d ){ // if the this cluster is closer set this as min distance
                        min_d = d;
                        id = j;
                    }
                }
                add_values(centSum, patches[i], id, tree.n_dim); // add the patch to the closest cluster
                centCount[id]++; // count the extra patch
                // update the idx to the child idx - note that all layers start with idx = 0
                patches[i].idx = (id + id_in*tree.branch_fac);
            }
            
            // update the clusters in the tree and calculate the center change (not used for anything)
            id = node*tree.n_dim;
            int id_c = 0;
            
            for ( int i = 0; i < tree.branch_fac; i++ ){ // go through all new clusters
                if ( centCount[i] == 0 ){
                    centCount[i] = 1;
                }
                for ( int j = 0; j < tree.n_dim; j++ ){ // go through cluster pixels
                    tmp = centSum[id_c]/centCount[i];
                    //diff = (tmp - *(tree.tree_data + id)); // for calculating center change
                    //centChange[i] += diff*diff;
                    *(tree.tree_data + id) = tmp;
                    id_c++;
                    id++;
                }
            }
            
            // set counter and sum to zero
            fill(centSum.begin(), centSum.end(), 0);
            fill(centCount.begin(), centCount.end(),0);
            fill(centChange.begin(), centChange.end(), 0);
        }
    }
}

// runs through the patches vector to find the last element with id
int get_to( vector<im_patch>& patches, int id )
{
    int to = 0;
    for ( int i = 0; i < patches.size(); i++ ){
        if ( patches[i].idx == id ){
            to = i;
        }
    }
    return to+1;
}

// Main function for building the km-tree. Takes the image and tree struct
// and the number of training patches as argument
void build_km_tree ( im_st& im, tree_st& tree, int n_train, bool normalize ) {
    // allocate memory for the image patches
    double* im_patch_data = new double[n_train*tree.M*tree.M*im.layers];
    
    int rows_c = im.rows-tree.M+1, cols_c = im.cols-tree.M+1; // number of rows and cols within sampling area
    int n_im_patches = rows_c*cols_c; // number of pixels in the image for sampling - inside boundary
    
    // checks that the number of training patches is not exceeding the number of pixels in the sample area
    if ( n_im_patches < n_train ){
        n_train = n_im_patches;
    }
    
    vector<int> r_id = randperm(n_im_patches, n_train); // indices of random patches
    
    vector<im_patch> patches; // vector of image patches
    patches.resize(n_train); // allocate memory
    
    int r, c, idx = 0; // variables used for sampling the image
    // sample image patches
    for (int i = 0; i < n_train; i++ )
    {
        c = r_id[i]/rows_c; // column is floored because of int
        r = r_id[i]-c*rows_c; // row is rest after column
        patches[i].idx = 0; // inital id is 0
        patches[i].patch_data = im_patch_data + idx; // pointer to patch memory
        sample_patch(im, tree, patches[i], r + tree.Mh, c + tree.Mh, normalize); // sampel in image with added boundary
        idx += tree.n_dim; // step number of patch pixels forward
    }
    
    // k-means tree
    int n_layer = (int)ceil(log((double)tree.n_nodes)/log((double)tree.branch_fac)); // number of layers in the tree
    int n_in_layer; // number of nodes in layer
    int t, f; // keeps track of patches that belong to a certain cluster
    int node = 0; // node number that will be updated
    
    // go through the layers in the tree
    for (int i = 0; i < n_layer; i++ )
    {
        t = 0; // start at 0
        n_in_layer = pow((double)tree.branch_fac,i); // number of nodes in current layer of the tree
        sort(patches.begin(), patches.end()); // sort the patches according to their current id
        for ( int j = 0; j < n_in_layer; j++ ) // go through the nodes in the layer and cluster that node
        {
            f = t; // patch j from
            t = get_to(patches,j); // patch j to
            // check that the node does not exceed the size of the tree
            if ( node + tree.branch_fac <= tree.n_nodes ){
                k_means( patches, tree, f, t, node );
            }
            else {
                break;
            }
            node += tree.branch_fac; // next node
        }
    }
    
    delete[] im_patch_data; // free up patch memory
}


// The gateway routine
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    // input image (I), patch size (M*M), number of nodes in the tree (n), branching
    // factor (b), and number of training patches (n_train). Outputs the km-tree (tree)
    double *I, *tree; // pointers to image and tree
    int b, M, L, n, ndim, n_train; // variables
    const int *dim; // image dimensinos
    int dtree[2]; // tree dimensions
    bool normalize = false; // decide if vectors of image patches should be normalized to unit length
    
    /*  Check for proper number of arguments. */
    /* NOTE: You do not need an else statement when using
     mexErrMsgTxt within an if statement. It will never
     get to the else statement if mexErrMsgTxt is executed.
     (mexErrMsgTxt breaks you out of the MEX-file.)
     */
    if(nrhs < 5 || nrhs > 6)
        mexErrMsgTxt("Five or six inputs required.");
    if(nlhs != 1)
        mexErrMsgTxt("One output required.");
    
    if ( !mxIsDouble(prhs[0]) )
        mexErrMsgTxt("Image should be double!");
    
    // Create a pointer to the input matrix.
    I = mxGetPr(prhs[0]);
    
    // input passing
    double *Md, *bd, *Ld, *n_train_d;
    Md = mxGetPr(prhs[1]);
    M = (int)Md[0];
    
    bd = mxGetPr(prhs[2]);
    b = (int)bd[0];
    
    // check if number of clusters is smaller than branching factor
    if ( n < b ){
        n = b;
    }
    
    n_train_d = mxGetPr(prhs[3]);
    n_train = (int)n_train_d[0];
    
    // determine number of tree nodes
    Ld = mxGetPr(prhs[4]);
    L = (int)Ld[0]; // layers in tree
    n = 0;
    int n_tmp = 0;
    int max_n = (double)n_train/1.0;
//     int max_n = (double)n_train/5.0;
    for ( int i = 0; i < L; i++ ){
        n_tmp += pow((double)b,(i+1));
        if ( n_tmp > max_n ){
            L = i+1;
            break;
        }
        n = n_tmp;
    }
    printf("Number of nodes in resulting tree: %d in %d layers.\n", n, L);
    
    if ( nrhs == 6 ){
        bool *normalize_d;
        normalize_d = (bool *)mxGetData(prhs[5]);
        normalize = normalize_d[0];
    }
    
    // check input properties
    if ( 1 - (M % 2)  || M < 1)
        mexErrMsgTxt("M must be odd and positive.");
    
    if ( n < 1 )
        mexErrMsgTxt("n must be positive.");
    
    if ( b < 1 )
        mexErrMsgTxt("b must be positive.");
    
    // Get the dimensions of the matrix input.
    ndim = mxGetNumberOfDimensions(prhs[0]);
    if (ndim != 2 && ndim != 3)
        mexErrMsgTxt("search_km_tree only works for 2-dimensional or 3-dimensional images.");
    
    // image dimensions
    dim = mxGetDimensions(prhs[0]);
    
    
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
    
    dtree[0] = Im.layers*M*M;
    dtree[1] = n;
    
    // Set the output pointer to the output matrix. Array initialized to zero.
    plhs[0] = mxCreateNumericArray(2, dtree, mxDOUBLE_CLASS, mxREAL);
    
    // Create a C pointer to a copy of the output matrix.
    tree = mxGetPr(plhs[0]);
    for (int i = 0; i < dtree[0]*dtree[1]; i++ )
        *(tree + i) = -1;
    
    // tree struct
    tree_st Tree;
    Tree.tree_data = tree;
    Tree.n_dim = dtree[0];
    Tree.n_nodes = dtree[1];
    Tree.branch_fac = b;
    Tree.M = M;
    Tree.Mh = (int)(0.5*((double)M-1.0));
    
    // build the km-tree using C++ subroutine
    build_km_tree ( Im, Tree, n_train, normalize );
    
}
