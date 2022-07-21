## Compile and install ImageMagick.

Option --disable-openmp must be used because openmp is not supported by the nvcc compiler, which is required for CUDA compilation. 

    ./configure --disable-openmp

install ImageMagick and make sure the libraries are in path: for instance. add this to .bashrc:

    export LD_LIBRARY_PATH="/usr/local/lib:$LD_LIBRARY_PATH"

    export LD_LIBRARY_PATH="/usr/local/cuda-11.2/targets/x86_64-linux/include:$LD_LIBRARY_PATH"

