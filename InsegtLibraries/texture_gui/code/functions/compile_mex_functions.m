% compile mex files
mex -largeArrayDims biadjacency_matrix.cpp
mex -compatibleArrayDims build_km_tree.cpp % based on Euclidean distance
mex -compatibleArrayDims search_km_tree.cpp % based on Euclidean distance
mex -compatibleArrayDims build_km_tree_xcorr.cpp % based on normalized cross correlation
mex -compatibleArrayDims search_km_tree_xcorr.cpp % based on normalized cross correlation
mex -compatibleArrayDims probability_search_km_tree.cpp % based on normalized cross correlation

