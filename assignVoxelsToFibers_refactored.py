# by Facundo Sosa-Rey, 2021. MIT license

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from tifffile import TiffFile
import tifffile
import json
import pickle

from trackingFunctions      import assignVoxelsToFibers
from trackingParameters     import getTrackingParams
from centroid               import centroidObj
from extractCenterPoints    import getTiffProperties


from joblib import Parallel, delayed  
import multiprocessing

import os
import time


def interpolateCenterPnt(fibO,z):

    deltaZ=z-fibO.meanPntCloud[2]

    oriVec=fibO.orientationVec
    if oriVec[2]<0.:
        oriVec=-oriVec

    if oriVec[2]>1e-6: # vector in xy plane, no use in finding interpolation 
        scalarFactor=deltaZ/oriVec[2]

        deltaX=oriVec[0]*scalarFactor
        deltaY=oriVec[1]*scalarFactor

        return (fibO.meanPntCloud[0]+deltaX,fibO.meanPntCloud[1]+deltaY)
    else:
        return (np.nan,np.nan )

def assignVoxelsToFibers_Main(
    commonPath,
    permutationPath,
    manualRange=None,
    makePlots=False,
    parallelHandle=True,
    verbose=False,
    addDisksAroundCenterPnts=False,
    textLabels=True 
    ):

    print('\n\tassignVoxelsToFibers() called on dataset:\n{}\n\treading from disk'.format(commonPath))

    tic = time.perf_counter()

    filesInDir = [f.path for f in os.scandir(commonPath+permutationPath) if f.is_file()]
    watershedFound=False
    fiberStructFound=False
    for i,iPath in enumerate(filesInDir):
        if ".tiff" in iPath:
            if "V_hist.tiff" in iPath:
                indexHistTiff=i
            if "V_pores.tiff" in iPath:
                indexPoresTiff=i
            if "V_fibers.tiff" in iPath:
                indexFibersTiff=i
            if "V_voxelMap.tiff" in iPath:
                indexVoxelMap=i
        if "fiberStruct.pickle" in iPath:
            fiberStrucPickle=i
            fiberStructFound=True

        if "watershedCenterPoints.pickle" in iPath:
            indexWaterPickle=i
            watershedFound=True

        if "watershedExtractionStats.json" in iPath:
            indexWaterJson=i

    if not fiberStructFound:
        raise FileNotFoundError("fiberStruct.json not found in \n"+commonPath+permutationPath)

    if not watershedFound:
        raise FileNotFoundError("watershedCenterPoints.json not found in\n"+commonPath+permutationPath)

    with open(filesInDir[indexWaterJson], "r") as f:

        watershedDict  = json.load(f)
        
        startSlice=watershedDict["volumeDescription"]["manualRange"]["start"]
        endSlice=  watershedDict["volumeDescription"]["manualRange"]["end"]  #endSlice is excluded

    with open(filesInDir[indexWaterPickle], "rb") as f:
        watershedData  = pickle.load(f)

    if permutationPath!="Permutation123/":
        with TiffFile(commonPath+"Permutation123/V_fibers_masked.tiff") as tif:
            V_fibers_masked=tif.asarray() #already clipped to exclusiveZone

            if permutationPath=="Permutation132/":
                V_fibers_masked=np.transpose(V_fibers_masked,(2,1,0)) # z,x,y -> y,x,z
            elif permutationPath=="Permutation321/":
                V_fibers_masked=np.transpose(V_fibers_masked,(1,0,2)) # z,x,y -> x,z,y

    else:
        V_fibers_masked=None

    # multithreading parameters
    # num_cores = int(round(multiprocessing.cpu_count()/2)) #crashes on large sets
    num_cores = multiprocessing.cpu_count()-2 #crashes on large sets


    if makePlots:
        with TiffFile(filesInDir[indexFibersTiff]) as tif:
            V_fibers=tif.asarray()
        with TiffFile(filesInDir[indexHistTiff]) as tif:
            V_hist=tif.asarray()
        plt.rcParams.update({'font.size': 26})
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams["font.family"] = "Times New Roman"
        
    else:
        V_fibers=None 


    with TiffFile(filesInDir[indexVoxelMap]) as tif:
        xRes,unitTiff=getTiffProperties(tif)        

        V_voxelMap=tif.asarray()

    with open(filesInDir[fiberStrucPickle], "rb") as f:
        fiberStruct  = pickle.load(f)

    zMin=0
    xOffset =0
    yOffset =0

    if fiberStruct["exclusiveZone"]:
        if permutationPath=="Permutation123/":
            zMin    =fiberStruct["exclusiveZone"]["zMin"]
            zMax    =fiberStruct["exclusiveZone"]["zMax"]
            xOffset =fiberStruct["exclusiveZone"]["xMin"]
            yOffset =fiberStruct["exclusiveZone"]["yMin"]
        elif permutationPath=="Permutation132/":
            zMin    =fiberStruct["exclusiveZone"]["yMin"]
            zMax    =fiberStruct["exclusiveZone"]["yMax"]
            xOffset =fiberStruct["exclusiveZone"]["xMin"]
            yOffset =fiberStruct["exclusiveZone"]["zMin"]
        elif permutationPath=="Permutation321/":
            zMin    =fiberStruct["exclusiveZone"]["xMin"]
            zMax    =fiberStruct["exclusiveZone"]["xMax"]
            xOffset =fiberStruct["exclusiveZone"]["zMin"]
            yOffset =fiberStruct["exclusiveZone"]["yMin"]
        if manualRange is None:
            manualRange=range(zMin,zMax)
        else:
            manualRange=range(max(manualRange.start,zMin),min(manualRange.stop,zMax) )

    # format in a way that is readable for .tiff file properties in file explorer
    # manual range may be different than previous
    descriptionStr="{"+"\"shape([z,x,y])\":[{},{},{}],\n\"manualRange\":{},\n\"exclusiveZone\":{}".\
        format(*V_voxelMap.shape,manualRange,fiberStruct["exclusiveZone"])

    toc = time.perf_counter()

    times_assign={"parallelHandle":parallelHandle}

    times_assign["reading from disk only:"]=time.strftime("%Hh%Mm%Ss", time.gmtime(toc-tic))


    print(f"\treading from disk complete in {toc - tic:0.4f} seconds\n")

    ticAssign = time.perf_counter()

    trackedCenterPoints = fiberStruct["trackedCenterPoints"]
    rejectedCenterPoints= fiberStruct["rejectedCenterPoints"]

    allCenterPoints={}



    fiberData={
        "rejected"               :{},
        "LUT_fiberID_to_color"  :fiberStruct["fiberObj_classAttributes"]["LUT_fiberID_to_color"],
        "colors"                :fiberStruct["fiberObj_classAttributes"]["colors"].copy()
    }

    #override colors for plotting here
    fiberData["colors"]["basic"]=(0.,1.,0.)

    fiberData["addDisks_usingInterpolated"]={}

    for fiberID,fibObj in fiberStruct["fiberStruct"].items():
        # fibObj.rejected will be False for fibers that have been added (smartStitched), 
        # but trimmed centroids will remain in rejectedCenterPoints
        # ["listFiberIDs_tracked"] is a sure-fire way of checking if fiber is tracked
        if fiberID in fiberStruct["fiberObj_classAttributes"]["listFiberIDs_tracked"]:
            fiberData["rejected"][fiberID]=False
        else:
            fiberData["rejected"][fiberID]=True

        fiberData["addDisks_usingInterpolated"][fiberID]=True
    
    plottingParams=getTrackingParams(commonPath,"plottingParams",xRes,unitTiff)

    fiberDiameter   =plottingParams["fiberDiameter"]         #used to check whether real or interpolated centroid should be used for addingDisks
    fiberRadius     =int(plottingParams["fiberDiameter"]/2.) #used to create disk SE for adding over fiber centerpoint

    for imSlice in trackedCenterPoints.keys():
        allCenterPoints[imSlice]={
            "x"                     :[],
            "x_interp"              :[],
            "y"                     :[],
            "y_interp"              :[],
            "fiberID"               :[]
            }

        for i,fiberID in enumerate(trackedCenterPoints[imSlice]["fiberID"]):
            x=trackedCenterPoints[imSlice]["x"][i]
            y=trackedCenterPoints[imSlice]["y"][i]
            allCenterPoints[imSlice]["fiberID"] .append(fiberID)
            allCenterPoints[imSlice]["x"]       .append(x)
            allCenterPoints[imSlice]["y"]       .append(y)
            
            x_interp,y_interp=interpolateCenterPnt(fiberStruct["fiberStruct"][fiberID],imSlice)
            
            allCenterPoints[imSlice]["x_interp"].append(x_interp)
            allCenterPoints[imSlice]["y_interp"].append(y_interp)
            
            d=np.linalg.norm([x-x_interp,y-y_interp])
            if d>fiberDiameter or np.isnan(x_interp) or np.isnan(y_interp):
                # if the fiber is not sufficiently straight, use actual centroids for adding disk instead of interpolated centroids
                fiberData["addDisks_usingInterpolated"][fiberID]=False
                print("Distance between interpolated center and (tracked) extracted centroid larger than a fiber diameter, d={:8.4f}, fiberID={}, imSlice={}".format(d,fiberID,imSlice))


        for i,fiberID in enumerate(rejectedCenterPoints[imSlice]["fiberID"]):
            x=rejectedCenterPoints[imSlice]["x"][i]
            y=rejectedCenterPoints[imSlice]["y"][i]
            allCenterPoints[imSlice]["fiberID"] .append(fiberID)
            allCenterPoints[imSlice]["x"]       .append(x)
            allCenterPoints[imSlice]["y"]       .append(y)

            x_interp,y_interp=interpolateCenterPnt(fiberStruct["fiberStruct"][fiberID],imSlice)
            
            allCenterPoints[imSlice]["x_interp"].append(x_interp)
            allCenterPoints[imSlice]["y_interp"].append(y_interp)

            d=np.linalg.norm([x-x_interp,y-y_interp])
            if d>fiberDiameter or np.isnan(x_interp) or np.isnan(y_interp):
                fiberData["addDisks_usingInterpolated"][fiberID]=False



    if parallelHandle:
        makePlots=False

    V_fiberMap=[]


    if manualRange is None:
        nSlices=endSlice-startSlice
        slicesRange=range(nSlices)
    else:
        startSlice=max(startSlice,manualRange.start)
        endSlice=min(endSlice,manualRange.stop)

        nSlices=min(endSlice,manualRange.stop)-startSlice
        slicesRange=range(startSlice,endSlice)

    if makePlots: # not allowed in parallel, would overload figure window generation
        for imSlice in slicesRange:
            figureHandle=plt.figure(figsize=[38,16])
            ax1 = figureHandle.add_subplot(1, 3, 1)
            ax2 = figureHandle.add_subplot(1, 3, 2)
            ax3 = figureHandle.add_subplot(1, 3, 3)

            allCenterPoints[imSlice],fiberMap  = \
                assignVoxelsToFibers(
                    allCenterPoints[imSlice],
                    fiberData,
                    V_voxelMap[imSlice-zMin,:,:],
                    watershedData[imSlice],
                    imSlice,
                    slicesRange,
                    V_hist_slice=V_hist[imSlice],
                    V_fibers_slice=V_fibers[imSlice],
                    V_fibers_masked_slice=V_fibers_masked[imSlice-zMin] if V_fibers_masked is not None else None,
                    axisHandles=[ax1,ax2,ax3],
                    verbose=verbose,
                    addDisksAroundCenterPnts=addDisksAroundCenterPnts,
                    fiberRadius=fiberRadius,
                    xOffset=xOffset,
                    yOffset=yOffset,
                    textLabels=textLabels
                    )

            figureHandle.suptitle("trackedCenterPoints, imSlice={}/{}".format(imSlice,slicesRange),fontsize=24)
            
            figureHandle.tight_layout(pad=5)

            plt.show()

            V_fiberMap.append(fiberMap)

    else:
        if parallelHandle:

            results= Parallel(n_jobs=num_cores)\
                (delayed(assignVoxelsToFibers)\
                    (allCenterPoints[imSlice],
                    fiberData,
                    V_voxelMap[imSlice-zMin,:,:],
                    watershedData[imSlice],
                    imSlice,
                    slicesRange,
                    V_fibers_masked_slice=V_fibers_masked[imSlice-zMin] if V_fibers_masked is not None else None,
                    verbose=verbose,
                    addDisksAroundCenterPnts=addDisksAroundCenterPnts,
                    fiberRadius=fiberRadius,                    
                    xOffset=xOffset,
                    yOffset=yOffset,
                    textLabels=textLabels
                    )for imSlice in slicesRange)

            for resTuple in results:
                
                V_fiberMap.append(resTuple[1])

        else:
            for imSlice in slicesRange:

                allCenterPoints[imSlice],fiberMap  = \
                    assignVoxelsToFibers(
                        allCenterPoints[imSlice],
                        fiberData,
                        V_voxelMap[imSlice-zMin,:,:],
                        watershedData[imSlice],
                        imSlice,
                        slicesRange,
                        V_fibers_masked_slice=V_fibers_masked[imSlice-zMin] if V_fibers_masked is not None else None,
                        verbose=verbose,
                        addDisksAroundCenterPnts=addDisksAroundCenterPnts,
                        fiberRadius=fiberRadius,
                        xOffset=xOffset,
                        yOffset=yOffset,
                        textLabels=textLabels
                        )

                V_fiberMap.append(fiberMap)

    tocAssign = time.perf_counter()

    V_fiberMap=np.array(V_fiberMap,np.int32)

    times_assign["assignVoxelsToFibers call:"]=time.strftime("%Hh%Mm%Ss", time.gmtime(tocAssign-ticAssign))
    print(f"\tassignVoxelsToFibers call complete in {tocAssign-ticAssign:0.4f} seconds\n")

    return V_fiberMap,fiberStruct,xRes,unitTiff,descriptionStr,times_assign
