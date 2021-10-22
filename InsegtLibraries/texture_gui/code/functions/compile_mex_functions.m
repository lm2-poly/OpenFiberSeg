% compile mex files
mex -largeArrayDims biadjacency_matrix.cpp
mex build_km_tree.cpp % based on Euclidean distance
mex search_km_tree.cpp % based on Euclidean distance
mex build_km_tree_xcorr.cpp % based on normalized cross correlation
mex search_km_tree_xcorr.cpp % based on normalized cross correlation
mex probability_search_km_tree.cpp % based on normalized cross correlation

