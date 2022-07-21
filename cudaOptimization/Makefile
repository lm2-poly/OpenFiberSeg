all:
	nvcc -g `Magick++-config --cxxflags --cppflags` -O2 -o readTiffAndProcess.out main.cpp tiff.cpp gpuInterface.cu `Magick++-config --ldflags --libs`