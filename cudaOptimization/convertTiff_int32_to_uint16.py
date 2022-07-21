#!/usr/bin/python3
# by Facundo Sosa-Rey, 2021. MIT license

from tifffile import TiffFile
import tifffile

import numpy as np
import os
import sys

if len(sys.argv)<2:
    raise RuntimeError("need to pass first argument as folderPath")

commonPath=sys.argv[1]

def getTiffProperties(tif,getDescription=False,getDimensions=False):
    
    try:
        # resolution is returned as a ratio of integers. value is in inches by default
        xRes=tif.pages[0].tags['XResolution'].value
        yRes=tif.pages[0].tags['YResolution'].value

        if xRes!=yRes:
            raise ValueError('not implemented for unequal x and y scaling')

        unitEnum=tif.pages[0].tags['ResolutionUnit'].value
        if repr(unitEnum)=='<RESUNIT.CENTIMETER: 3>':
            unitTiff="CENTIMETER"
        elif repr(unitEnum)=='<RESUNIT.INCH: 2>':
            unitTiff="INCH"
        else:
            raise ValueError("not implemented for {}".format(repr(unitEnum)))
    except:
        print("\n\tTiff files do not contain scaling information, assigning default value of 1 micron/pixel")
        xRes=(int(1e4),1)
        unitTiff="CENTIMETER"

    if getDescription:
        descriptionStr=tif.pages[0].tags["ImageDescription"].value

        if getDimensions:
            return xRes,unitTiff,descriptionStr,tif.pages[0].shape
        else:
            return xRes,unitTiff,descriptionStr

    else:
        if getDimensions:
            return tif.pages[0].shape
        else:
            return xRes,unitTiff

if __name__ == "__main__":
        
    print("commonPath: \n"+commonPath)

    ########################################################################################

    print("\tLoading V_fiberMapCombined_postProcessed.tiff")

    with TiffFile(os.path.join(commonPath,"V_fiberMapCombined_postProcessed_noDesc.tiff")) as tif:
        xRes,unitTiff=getTiffProperties(tif)

        V=np.array(tif.asarray())

    # V=V[:32] #HACK keep only a few slices for testing

    maxVal=np.max(V)

    if maxVal>2**16-1:
        raise ValueError("Not implemented for number of fibers above {}".format(2**16-1))

    print("max value in V_fiberMap: {: >8.0f}".format(np.max(V)))

    V_mask=np.zeros(V.shape,np.uint8)

    V_mask[V==-1]=255

    V[V_mask==255]=0

    print("converting to uint16 and writing to disk")

    tifffile.imwrite(
        os.path.join(commonPath,"V_uint16.tiff"),
        V.astype(np.uint16),
        resolution=(xRes,xRes,unitTiff), #yRes==xRes
        compress=True,
        # description=descriptionStr #incompatible with ImageMagick
        )

    tifffile.imwrite(
        os.path.join(commonPath,"V_mask_uint8.tiff"),
        V_mask,
        resolution=(xRes,xRes,unitTiff), #yRes==xRes
        compress=True,
        # description=descriptionStr #incompatible with ImageMagick
        )

    print("Done")



