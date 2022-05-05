#!/usr/bin/python3
# by Facundo Sosa-Rey, 2021. MIT license

from tifffile import TiffFile
import tifffile

import numpy as np

from extractCenterPoints import getTiffProperties

import os

import sys

if len(sys.argv)<2:
    raise RuntimeError("need to pass first argument as folderPath")

commonPath=sys.argv[1]

if __name__ == "__main__":
        
    print("commonPath: \n"+commonPath)

    filesInDir = [f.path for f in os.scandir(commonPath) if f.is_file()]

    indexPerim=None
    indexHist =None
    indexFibers=None
    indexPores=None

    for i,iPath in enumerate(filesInDir):
        if ".tiff" in iPath:

            if "V_hist.tiff" in iPath:
                indexHist=i
            if "V_perim.tiff" in iPath:
                indexPerim=i
            if "V_pores.tiff" in iPath:
                indexPores=i

    ########################################################################################

    print("\tprocessing V_fibers.tiff")

    with TiffFile(os.path.join(commonPath,"V_fibers.tiff")) as tif:
        xRes,unitTiff=getTiffProperties(tif)


        V_fibers=np.array(tif.asarray())

    descriptionStr="{"+"\"shape([x,y,z])\":[{},{},{}]".format(V_fibers.shape[1],V_fibers.shape[2],V_fibers.shape[0])+"}"

    tifffile.imwrite(
        os.path.join(commonPath,"V_fibers.tiff"),
        V_fibers,
        resolution=(xRes,xRes,unitTiff), #yRes==xRes
        compress=True,
        description=descriptionStr
        )

    print("\tprocessing V_fibers.tiff completed")


    ########################################################################################

    print("\tprocessing V_hist.tiff")

    if indexHist is not None:
        with TiffFile(os.path.join(commonPath,"V_hist.tiff")) as tif:
            xRes,unitTiff=getTiffProperties(tif)


            V_hist=np.array(tif.asarray())

        descriptionStr="{"+"\"shape([x,y,z])\":[{},{},{}]".format(V_hist.shape[1],V_hist.shape[2],V_hist.shape[0])+"}"


        print("\tdescription: "+descriptionStr)

        tifffile.imwrite(
            os.path.join(commonPath,"V_hist.tiff"),
            V_hist,
            resolution=(xRes,xRes,unitTiff), #yRes==xRes
            compress=True,
            description=descriptionStr
            )

        print("\tprocessing V_hist.tiff completed")


    ########################################################################################

    #Perim will not be present if not created in preprocessing
    if indexPerim is not None:
        print("\tprocessing V_perim.tiff")


        with TiffFile(os.path.join(commonPath,"V_perim.tiff")) as tif:
            xRes,unitTiff=getTiffProperties(tif)

            V_perim=np.array(tif.asarray())

        descriptionStr="{"+"\"shape([x,y,z])\":[{},{},{}]".format(V_perim.shape[1],V_perim.shape[2],V_perim.shape[0])+"}"

        tifffile.imwrite(
            os.path.join(commonPath,"V_perim.tiff"),
            V_perim,
            resolution=(xRes,xRes,unitTiff), #yRes==xRes
            compress=True,
            description=descriptionStr
            )

        print("\tprocessing V_perim.tiff completed")


    ########################################################################################

    print("\tprocessing V_pores.tiff")

    with TiffFile(os.path.join(commonPath,"V_pores.tiff")) as tif:
        xRes,unitTiff=getTiffProperties(tif)

        V_pores=np.array(tif.asarray())

    descriptionStr="{"+"\"shape([x,y,z])\":[{},{},{}]".format(V_pores.shape[1],V_pores.shape[2],V_pores.shape[0])+"}"

    tifffile.imwrite(
        os.path.join(commonPath,"V_pores.tiff"),
        V_pores,
        resolution=(xRes,xRes,unitTiff), #yRes==xRes
        compress=True,
        description=descriptionStr
        )

    print("\tprocessing V_pores.tiff completed")

    ########################################################################################

    print("\tprocessing V_prob.tiff")

    with TiffFile(os.path.join(commonPath,"V_prob.tiff")) as tif:
        xRes,unitTiff=getTiffProperties(tif)

        V_prob=np.array(tif.asarray())

    descriptionStr="{"+"\"shape([x,y,z])\":[{},{},{}]".format(V_prob.shape[1],V_prob.shape[2],V_prob.shape[0])+"}"

    tifffile.imwrite(
        os.path.join(commonPath,"V_prob.tiff"),
        V_prob,
        resolution=(xRes,xRes,unitTiff), #yRes==xRes
        compress=True,
        description=descriptionStr
        )

    print("\tprocessing V_prob.tiff completed")


