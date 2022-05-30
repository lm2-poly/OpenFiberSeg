# by Facundo Sosa-Rey, 2021. MIT license

import multiprocessing
from joblib import Parallel, delayed  
import shutil
from random import shuffle
from tifffile import TiffFile
import tifffile
import json
import numpy as np
import os
import matplotlib.pyplot as plt

from combineFunctions import compactifySlice
from extractCenterPoints import getTiffProperties
from outputPropertyMapsRefactored import outputPropertyMap



dataPath="" # path to folder containing PropertyMap.vtk

pathOriginal="" # path to preProcessed folder containing V_original.tiff

# croppedZone={
#   "xMin":200,
#   "xMax":800,
#   "yMin":200,
#   "yMax":800,
#   "zMin":1,
#   "zMax":980
# }

croppedZone=None

randomize_123=False

permutationPaths=["Permutation123/","Permutation132/","Permutation321/"]

pathsToProcess=[
    dataPath  +permutationPaths[0]+"V_hist.tiff",
    dataPath  +permutationPaths[0]+"V_pores.tiff",
    dataPath+"V_fiberMapCombined_randomized.tiff",
    dataPath+"V_fiberMapCombined_postProcessed.tiff",
    dataPath  +permutationPaths[0]+"V_fiberMap.tiff",
    dataPath+"V_fiberMapCombined_randomizedFloat.tiff",
    pathOriginal+"V_original.tiff"
]

print('reading from disk')

with TiffFile(pathsToProcess[2]) as tif:
    xResDS,unitTiff,descriptionStrDS=getTiffProperties(tif,getDescription=True)

    descriptionDictDS=json.loads(descriptionStrDS)

    if "downSamplingFactor" in descriptionDictDS:
        downSamplingFactor=descriptionDictDS["downSamplingFactor"]
    else:
        downSamplingFactor=1

    V_fiberMap_randomized=np.array(tif.asarray(),np.float32)

with TiffFile(pathsToProcess[0]) as tif:
    V_hist=tif.asarray()

with TiffFile(pathsToProcess[1]) as tif:
    V_pores=tif.asarray()

with TiffFile(pathsToProcess[3]) as tif:
    V_fiberMap=tif.asarray()

with TiffFile(pathsToProcess[5]) as tif:
    V_fiberMap_randomizedFloat=tif.asarray()

with TiffFile(pathsToProcess[4]) as tif:
    xRes,unitTiff,descriptionStr=getTiffProperties(tif,getDescription=True)

    descriptionDict=json.loads(descriptionStr.replace("None","[]"))

    V_fiberMap123=tif.asarray()

with TiffFile(pathsToProcess[6]) as tif:
    V_original=tif.asarray()


print('reading from disk complete')

if croppedZone is not None:

    xMin=croppedZone["xMin"]
    xMax=croppedZone["xMax"]
    yMin=croppedZone["yMin"]
    yMax=croppedZone["yMax"]
    zMin=croppedZone["zMin"]
    zMax=croppedZone["zMax"]

    xMinDS=xMin//downSamplingFactor
    xMaxDS=xMax//downSamplingFactor
    yMinDS=yMin//downSamplingFactor
    yMaxDS=yMax//downSamplingFactor
    if zMin is not None:
        zMinDS=zMin//downSamplingFactor
        zMaxDS=zMax//downSamplingFactor


    if zMin is None:
        V_fiberMap                  =V_fiberMap             [:,xMin:xMax,yMin:yMax]
        V_fiberMap123               =V_fiberMap123          [:,xMin:xMax,yMin:yMax]
        V_fiberMap_randomized       =V_fiberMap_randomized  [:,xMinDS:xMaxDS,yMinDS:yMaxDS]
        V_fiberMap_randomizedFloat  =V_fiberMap_randomizedFloat  [:,xMinDS:xMaxDS,yMinDS:yMaxDS]
        V_pores                     =V_pores                [:,xMin:xMax,yMin:yMax]
        V_hist                      =V_hist                 [:,xMin:xMax,yMin:yMax]
        V_original                  =V_original             [:,xMin:xMax,yMin:yMax]
    else:
        V_fiberMap                  =V_fiberMap             [zMin:zMax,xMin:xMax,yMin:yMax]
        V_fiberMap123               =V_fiberMap123          [zMin:zMax,xMin:xMax,yMin:yMax]
        V_fiberMap_randomized       =V_fiberMap_randomized  [zMinDS:zMaxDS,xMinDS:xMaxDS,yMinDS:yMaxDS]
        V_fiberMap_randomizedFloat  =V_fiberMap_randomizedFloat  [zMinDS:zMaxDS,xMinDS:xMaxDS,yMinDS:yMaxDS]
        V_pores                     =V_pores                [zMin:zMax,xMin:xMax,yMin:yMax]
        V_hist                      =V_hist                 [zMin:zMax,xMin:xMax,yMin:yMax]
        V_original                  =V_original             [zMin:zMax,xMin:xMax,yMin:yMax]


plt.figure(figsize=[15,15])
plt.imshow(V_fiberMap_randomized[10//downSamplingFactor,:,:],cmap="gist_stern_r")
plt.title("V_fiberMapRandomized",fontsize=22)

plt.figure(figsize=[15,15])
plt.imshow(V_fiberMap[10,:,:],cmap="gist_stern_r")
plt.title("V_fiberMap",fontsize=22)

plt.figure(figsize=[15,15])
plt.imshow(V_fiberMap123[10,:,:],cmap="gist_stern_r")
plt.title("V_fiberMap123",fontsize=22)

plt.figure(figsize=[15,15])
plt.imshow(V_hist[10,:,:],cmap="ocean")
plt.title("V_hist",fontsize=22)

plt.figure(figsize=[15,15])
plt.imshow(V_pores[10,:,:],cmap="ocean")
plt.title("V_pores",fontsize=22)

plt.show()

print('Writing to disk started')

descriptionDict["croppedZone"]=croppedZone

descriptionStr=json.dumps(descriptionDict)

outputPath=dataPath+"CroppedResults/"

exists = os.path.exists(outputPath)

if not exists:
    os.mkdir(outputPath)

    shutil.copy2("composition_generic.pvsm",outputPath)


tifffile.imwrite(outputPath+'V_fiberMap_randomized_cropped.tiff',
    V_fiberMap_randomized,
    resolution=(xResDS,xResDS,unitTiff),
    description=descriptionStr,
    compress=True
    )

tifffile.imwrite(outputPath+'V_fiberMap_randomizedFloat_cropped.tiff',
    V_fiberMap_randomizedFloat,
    resolution=(xResDS,xResDS,unitTiff),
    description=descriptionStr,
    compress=True
    )

tifffile.imwrite(outputPath+'V_hist_cropped.tiff',
    V_hist,
    resolution=(xRes,xRes,unitTiff),
    description=descriptionStr,
    compress=True
    )

tifffile.imwrite(outputPath+'V_pores_cropped.tiff',
    V_pores,
    resolution=(xRes,xRes,unitTiff),
    description=descriptionStr,
    compress=True
    )

tifffile.imwrite(outputPath+'V_fiberMapCombined_postProcessed_cropped.tiff',
    V_fiberMap,
    resolution=(xRes,xRes,unitTiff),
    description=descriptionStr,
    compress=True
    )

tifffile.imwrite(outputPath+'V_fiberMap123_cropped.tiff',
    V_fiberMap123,
    resolution=(xRes,xRes,unitTiff),
    description=descriptionStr,
    compress=True
    )

tifffile.imwrite(outputPath+'V_original.tiff',
    V_original,
    resolution=(xRes,xRes,unitTiff),
    description=descriptionStr,
    compress=True
    )

if randomize_123:
    print("Randomizing V_fiberMap123")

    listMarkers=np.array(sorted(np.unique(V_fiberMap123)))

    cutoffIndex=np.where(listMarkers>=0)[0][0]

    trackedMarkers=listMarkers[cutoffIndex:]
    rejectedMarkers=listMarkers[:cutoffIndex]

    listMarkersTracked=[val for val in listMarkers if val>=0]# tracked fibers have markers starting at 0

    V_fiberMap123_randomized=V_fiberMap123.copy()

    reassignedMarkers=trackedMarkers.copy()
    #random shuffling of original markers
    shuffle(reassignedMarkers)

    markerLUT={}
    for i,iMark in enumerate(rejectedMarkers):
        markerLUT[iMark]=rejectedMarkers[i]

    for i,iMark in enumerate(trackedMarkers):
        markerLUT[iMark]=reassignedMarkers[i]

    parallelHandle=True

    if parallelHandle:
        num_cores=min(int(multiprocessing.cpu_count()-1),48)
    else:
        num_cores=1

    results = Parallel(n_jobs=num_cores)\
    (delayed(compactifySlice)\
        (
            V_fiberMap123[iSlice],
            markerLUT
        )for iSlice in range(V_fiberMap123.shape[0]) )


    for iSlice,resTuple in enumerate(results):
        V_fiberMap123_randomized[iSlice]=resTuple


    # Conversion to floats makes a different rendering in Paraview
    V_fiberMap123_randomizedFloat=np.array(V_fiberMap123_randomized,np.float32)

    V_fiberMap123_randomizedFloat[V_fiberMap123_randomized==-1]=np.nan

    print("writing randomized V_fiberMap123 to disk...")

    tifffile.imwrite(
        outputPath+'V_fiberMap123_randomized.tiff',
        V_fiberMap123_randomized,
        resolution=(xRes,xRes,unitTiff),
        description=descriptionStr,
        compress=True
    )

    tifffile.imwrite(
        outputPath+'V_fiberMap123_randomizedFloat.tiff',
        V_fiberMap123_randomizedFloat,
        resolution=(xRes,xRes,unitTiff),
        description=descriptionStr,
        compress=True
    )


#make propertyMap for cropped region
outputPropertyMap(
    outputPath,
    parallelHandle=True,
    randomizeFiberMap=False,
    croppedFiles=True,
    forceReprocessing=False
    )

print('Done')
