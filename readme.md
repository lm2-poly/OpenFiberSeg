# OpenFiberSeg
Automated segmentation tool for the extraction of individual fibers in short fiber reinforced composites


## Install python packages

The packages and recommended versions are found in the requirements.txt . If those conflict with your installation, consider using this project in it's own environment. 

$ python3 -m pip install -r requirements.txt

On windows, OpenCV from pip should work:

$ python3 -m pip install opencv-python

On Linux, you may need to build OpenCV from source. Please see: https://docs.opencv.org/4.5.3/d7/d9f/tutorial_linux_install.html

## Pull datasets from remote repository

In order to keep the main repository (this one) lightweight, the 14Gb of data were put in a separate repository, to allow users to pull it only if needed. 

$ python3 getRemoteData.py

## Pre-process sample data set: PEEK 5 wt.%CF

The parameters for pre-processing of all data sets are in the file preProcessing.py. Executing as-is will process PEEK05 sub folder. Uncomment another line from 53-60 to select another data folder. 

$ python3 preProcessing.py

## Pre-segmentation with Insegt

run script InsegtFiber_3D.m in matlab (tested on R2018a). Again, the sub folder with PEEK05 is pre-selected. Uncomment another line from 35-41 to select another data folder. 

## Processing

Script main.py finds datasets processed with Insegt, but haven't been tracked, and processes them sequentially. 

$ python3 main.py



