# OpenFiberSeg
Automated segmentation tool for the extraction of individual fibers in short fiber reinforced composites


## Install python packages

The packages and recommended versions are found in the requirements.txt . If those conflict with your installation, consider using this project in it's own environment. 

$ python3 -m pip install -r requirements.txt

On windows, OpenCV from pip should work:

$ python3 -m pip install opencv-python

On Linux, you may need to build OpenCV from source. Please see: https://docs.opencv.org/4.5.3/d7/d9f/tutorial_linux_install.html

# Run extraction on the sample data provided

## Pull datasets from remote repository

In order to keep the main repository (this one) lightweight, the 14Gb of data were put in a separate repository, to allow users to pull it only if needed. 

$ python3 getRemoteData.py

## Pre-process sample data set: PEEK 5 wt.%CF

The parameters for pre-processing of all data sets are in the file preProcessing.py. Executing as-is will process PEEK05 sub folder. Uncomment another line from 53-60 to select another data folder. 

$ python3 preProcessing.py

## Pre-segmentation with Insegt

run script InsegtFiber_3D.m in matlab (tested on R2018a). Again, the sub folder with PEEK05 is pre-selected. Uncomment another line from 35-41 to select another data folder. The output of Insegt will be in a folder with the ranges in x, y and z in the folder name, in a second folder with the date and time, so they are not overwritten if Insegt is run again with different parameters. 

## Processing

Script main.py finds datasets processed with Insegt, but haven't been tracked, and processes them sequentially. 

$ python3 main.py

The file PropertyMaps can be visualized with Paraview. 

## Plotting results

Extract fiber statistics and plot results as shown in the journal publication by indicating the correct path on line 29 of fiberStatistics.py. 

example path:
path="./TomographicData/PEEK05/processed_x1-901_y1-871_z1-978/2020-01-01_12h00m00/"

then run:

$ python3 fiberStatistics.py

# Working with your own tomographic data

Place the tiff files of your scans in their own path in:
./TomographicData/<Scan Name>/uCT_RawData

To find the preprocessing parameters, run the GUI application and follow instructions in:

$ python3 preProcessing_GUI.py





















