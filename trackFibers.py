# by Facundo Sosa-Rey, 2021. MIT license

import json
import pickle
import numpy as np
# from random import random as rand
from trackingFunctions import firstPassKNN
from extractCenterPoints import getTiffProperties
from fibers import fiberObj
from matplotlib import pyplot as plt

import os
import time

from joblib import Parallel, delayed  
import multiprocessing

from tifffile import TiffFile

from trackingParameters import getTrackingParams

def plotCandidates(fibObjDown,fibObjUp):
    fig = plt.figure()
    ax = fig.gca(projection='3d')


    x = fibObjDown.x
    y = fibObjDown.y
    z = fibObjDown.z
    ax.plot(x, y, z, label='parametric curve')

    x = fibObjUp.x
    y = fibObjUp.y
    z = fibObjUp.z
    ax.plot(x, y, z, label='parametric curve')
    
    ax.legend()

    plt.show()


###########################################################

num_cores =min(multiprocessing.cpu_count()-2,48) # errors thrown if too many cores are used


def tracking(commonPath,permutationPath,permutationVec,exclusiveZone=None,parallelHandle=False,verboseHandle=False):

    print("\n\n\ttracking() called on dataset: \n{}".format(os.path.join(commonPath,permutationPath)))
    print("\t\treading from disk")
    tic = time.perf_counter()

    filesInDir = [f.path for f in os.scandir(os.path.join(commonPath,permutationPath)) if f.is_file()]
    watershedFound=False
    for i,iPath in enumerate(filesInDir):
        if "V_fibers.tiff" in iPath:
            indexFibersTiff=i

        if "watershedCenterPoints.pickle" in iPath:
            indexWaterPickle=i
            watershedFound=True
            
        if "watershedExtractionStats.json" in iPath:
            indexWaterJson=i

    if not watershedFound:
        raise FileExistsError("\nwatershedCenterPoints.json not found in\n{}\n\tCentroid extraction needs to be done first".format(
            os.path.join(commonPath,permutationPath)))


    filesCommonPath = [f.path for f in os.scandir(commonPath) if f.is_file()]
    indexJson=None
    for i,iPath in enumerate(filesCommonPath):
        if "SegtParams.json" in iPath:
            indexJson=i

    with open(filesInDir[indexWaterPickle], "rb") as f:
        watershedData  = pickle.load(f)

    with open(filesInDir[indexWaterJson], "r") as f:
        watershedDict  = json.load(f)

    # rewrite with formatting, Matlab writes to a single line
    overwriteJson=False
    if indexJson is not None:
        with open(filesCommonPath[indexJson], "r") as f:
            segtParams  = json.load(f)

        #Matlab json is all on a single line, this makes it more human-readable
        with open(filesCommonPath[indexJson], "r") as f:
            if len(f.readlines())==1:
                overwriteJson=True

        if overwriteJson:
            with open(filesCommonPath[indexJson], "w") as f:
                json.dump(segtParams, f, sort_keys=True, indent=4)

    with TiffFile(filesInDir[indexFibersTiff]) as tif:
        xRes,unitTiff=getTiffProperties(tif) 
        V_fibers      =tif.asarray()


    toc = time.perf_counter()
    times_tracking={"parallelHandle:":parallelHandle}

    times_tracking["reading from disk (tracking)"]=time.strftime("%Hh%Mm%Ss", time.gmtime(toc-tic))

    print(f"\treading from disk complete in {toc - tic:0.4f} seconds\n")

    fiberObj.initializeClassAttributes()
    fiberObj.loadSegmentationMask(V_fibers[:,:,:])

    print("\n\t\ttracking begins")

    params=getTrackingParams(commonPath,permutationVec,xRes,unitTiff)

    #unpacking of parameters
    distLateral_knnFirstPass=        params["distLateral_knnFirstPass"]
    processingMinFiberLength=        params["processingMinFiberLength"]
    blindStitching=                  params["blindStitching"]
    tagAngleTooSteep=                params["tagAngleTooSteep"]
    maxSteepnessAngle=               params["maxSteepnessAngle"]
    blindStitchingMaxDistance=       params["blindStitchingMaxDistance"]
    blindStitchingMaxLateralDist=    params["blindStitchingMaxLateralDist"]
    smartStitching=                  params["smartStitching"]
    smartStitchingMaxDistance=       params["smartStitchingMaxDistance"]
    smartStitchingAlignAngle=        params["smartStitchingAlignAngle"]
    smartStitchingMaxLateralDist=    params["smartStitchingMaxLateralDist"]
    smartStitchingMinFibLength=      params["smartStitchingMinFibLength"]
    smartStitchingBackTrackingLimit= params["smartStitchingBackTrackingLimit"]
    collisionDistance=               params["collisionDistance"]
    fillingFraction=                 params["fillingFraction"]
    fillingNumberAlwaysAllowed=      params["fillingNumberAlwaysAllowed"]
    maxTrimPoints=                   params["maxTrimPoints"]

    print("\nLaunching tracking with parameters:\n")
    for key,value in params.items():
        if type(value)==bool:
            print("\t{:<30}\t:{}".format(key,value)) 
        else:
            print("\t{:<30}\t:{: >8.4f}".format(key,value)) 
    print("")

    if smartStitchingBackTrackingLimit>=maxTrimPoints:
        raise ValueError("backtrackLimit={} must be inferior to maxTrimPoints={}, else maxTrimPoint can be reached in normal operation".\
            format(smartStitchingBackTrackingLimit,maxTrimPoints))

    fiberObj.setTrackingParameters(
        collisionDistance,
        fillingFraction,
        fillingNumberAlwaysAllowed,
        maxTrimPoints
        )

    if exclusiveZone is not None:
        fiberObj.setExclusiveZone(exclusiveZone)

        xMin=exclusiveZone["xMin"]
        xMax=exclusiveZone["xMax"]
        yMin=exclusiveZone["yMin"]
        yMax=exclusiveZone["yMax"]
        zMin=exclusiveZone["zMin"]
        zMax=exclusiveZone["zMax"]

        if permutationVec=="132":
            tempMin=yMin
            tempMax=yMax
            yMin=zMin
            yMax=zMax
            zMin=tempMin
            zMax=tempMax

        if permutationVec=="321":
            tempMin=xMin
            tempMax=xMax
            xMin=zMin
            xMax=zMax       
            zMin=tempMin
            zMax=tempMax

        offset=zMin

        # keep only centrePoints inside exclusiveZone
        # centerPoints=[ np.array([ centroid.getPnt() for centroid in ctsInSlice if centroid.getPnt()[0]>xMin and\
        #     centroid.getPnt()[0]<xMax and centroid.getPnt()[1]>yMin and centroid.getPnt()[1]<yMax],float) for ctsInSlice in watershedData[zMin:zMax]]

        centerPoints={}
        for iSlice in range(zMin,zMax):
            centerPoints[iSlice]=[ 
                np.array(centroid.getPnt(),float) for centroid in watershedData[iSlice] if 
                centroid.getPnt()[0]>xMin and
                centroid.getPnt()[0]<xMax and 
                centroid.getPnt()[1]>yMin and 
                centroid.getPnt()[1]<yMax
                ] 
            
    else:
        # keep only centrePoint coordinates (not markers and contours)
        centerPoints={}
        for iSlice in range(len(watershedData)):
            centerPoints[iSlice]=[ np.array( centroid.getPnt(),float) for centroid in watershedData[iSlice]  ] 

        offset=0

    if np.max(V_fibers)!=255:
        print("V_fibers does not contain \"True\" value (255), tracking skipped")
        fiberObj.initTrackedCenterPoints(len(watershedData),offset)
        return {},V_fibers.shape,times_tracking,V_fibers,xRes,unitTiff 


    LUT_id_bottom   =[[] for i in range(len(centerPoints)-1)]
    LUT_id_top      =[[] for i in range(len(centerPoints)-1)]

    print("\tFirst pass knn started")

    ticGlobal=time.perf_counter()

    listSlicesLUT=list(centerPoints.keys())

    listSlicesLUT.remove(max(listSlicesLUT))#avoids access violation when indexing for imSlices and imSlice+1

    if parallelHandle:
        # the larger the dataset, the more the gains from parallel processing will be large. 
        # initialising the multiple python subprocesses has a fixed overhead. 
        results = Parallel(n_jobs=num_cores)\
            (delayed(firstPassKNN)\
                (
                centerPoints[iSlice+1],
                centerPoints[iSlice],
                distLateral_knnFirstPass
                )for iSlice in listSlicesLUT )

        for i,resultTuple in enumerate(results):
            LUT_id_bottom[i]=resultTuple[0]
            LUT_id_top   [i]=resultTuple[1]

        for proc in multiprocessing.active_children():
            # Manual termination of processes to avoid strange race condition at initializeFromLUT
            proc.terminate()
            proc.join()

    else:
        for i,iSlice in enumerate(listSlicesLUT):
            LUT_id_bottom [i],LUT_id_top[i]=firstPassKNN(centerPoints[iSlice+1],centerPoints[iSlice],distLateral_knnFirstPass)
            print("First pass knn on slice: {} completed".format(iSlice))

    toc=time.perf_counter()

    times_tracking["First Pass KNN search"]=time.strftime("%Hh%Mm%Ss", time.gmtime(toc-ticGlobal))

    print('\tFirst pass KNN search completed in {: >.2f}s with parallelHandle={}'.format(toc-ticGlobal,parallelHandle))

    print("\tInitializing fiberObjects from LUT")
    tic=time.perf_counter()

    fiberStruct =fiberObj.initializeFromLUT(centerPoints,listSlicesLUT,LUT_id_bottom,LUT_id_top,offset)
        
    if len(fiberStruct)==0:
        print("no fibers in this region, possibly wrong exclusive zone")
        return fiberStruct,V_fibers.shape,times_tracking,V_fibers,xRes,unitTiff

    toc=time.perf_counter()

    times_tracking["Initializing fiberObjects from LUT"]=time.strftime("%Hh%Mm%Ss", time.gmtime(toc-tic))
    print("\tInitializing fiberObjects from LUT completed in {: >.2f}s".format(toc-tic))

    tic=time.perf_counter()    

    # check that none of the initial segments are too spread out, i.e. centroids too far from principal orientation vector
    # this can happen if two segments from different fibers have endpoints very close, but strong inclination between them

    results=Parallel(n_jobs=num_cores)\
        (
            delayed(fibO.checkSpread)(
                distLateral_knnFirstPass*2,
                verboseHandle
            ) for fibO in fiberStruct.values() 
        )

    # points trimmed in checkSpread need to be removed from trackedCenterPoints
    for pointList in results:
        for point in pointList:
            fiberObj.classAttributes["trackedCenterPoints" ].reject(
                fiberObj.classAttributes["trimmedCenterPoints"],*point)


    if blindStitching:
        blindStitchedListCache_fiberID=\
            fiberObj.blindStitching(
                fiberStruct,
                blindStitchingMaxDistance,
                blindStitchingMaxLateralDist,
                verboseHandle=verboseHandle
                )
    else:
        blindStitchedListCache_fiberID=set([])

    # split fiberStruct between actual fiberObj and segments that were added to others (kept for plotting purposes)
    # smart stitching is only attempted on main fiber objects
    fiberStructMain={}
    fiberStructExtended={}
    for fibID,fObj in fiberStruct.items():
        if fibID in blindStitchedListCache_fiberID:
            fiberStructExtended[fibID]=fObj
        else:
            fiberStructMain[fibID]=fObj

    toc=time.perf_counter()
    times_tracking["BlindStitching only"]=time.strftime("%Hh%Mm%Ss", time.gmtime(toc-tic))
    print("time for blindStitching only: {}".format(toc-tic))

    print("time since tracking began: {}".format(toc-ticGlobal))

    ##################################################################################

    # Least-squared fit of 1st degree polynomial to each of the fibers

    ##################################################################################

    print("\tLeast-squared fit of 1st degree polynomial to each of the fibers")
    tic=time.perf_counter()

    # only principal fiberObj are processed, not the ones added to them in blind stitching
    for fibID,fObj in fiberStructMain.items():

        fObj.processPointCloudToFiberObj(processingMinFiberLength,tagAngleTooSteep,maxSteepnessAngle)

    if len(fiberStructMain)==0:
        raise RuntimeError("No fiberObj were initialized")

    times_tracking["Least-squared fit only"]=time.strftime("%Hh%Mm%Ss", time.gmtime(time.perf_counter()-tic))
    print("Least-squared fit performed in {: >0.4f}s".format(time.perf_counter()-tic))


    ##################################################################################

    #  Smart fiberStitching

    ##################################################################################
    if smartStitching:

        print("\tSmart fiber stitching started")

        fiberStructExtended,times_stitching=fiberObj.smartStitching(
            fiberStructMain,
            smartStitchingMinFibLength,
            smartStitchingMaxDistance,
            smartStitchingMaxLateralDist,
            smartStitchingAlignAngle,
            smartStitchingBackTrackingLimit,
            processingMinFiberLength,
            tagAngleTooSteep,
            maxSteepnessAngle,
            verboseHandle=verboseHandle
            )

        times_tracking.update(times_stitching)

        # add initial segments as new fiberObj, so that they can be plotted seperately (different color)
        for fibID,fib in fiberStructExtended.items():
            fiberStruct[fibID]=fib

    # edge case: if fiberObj with fiberID==0 is rejected, then at assignVoxelsToFiberID, flipping
    # fiberID to -fiberID will have no effect. To allow the convention that only positive fiberID are tracked in
    # fiberMap, reassignID here
    # other edge case: if fiberID is rejected, the assigning marker -1 will cause confusion with background. reassign. 
    for fiberID in [0,1]:
        if fiberID in fiberStruct.keys(): #validation dataset can have only one fiber
            if fiberStruct[fiberID].rejected:
                nextID=max(fiberStruct.keys())+1
                fiberStruct[nextID]=fiberStruct[fiberID]

                # transfer ID of all centerPoints belonging to this fiber
                for iSlice in fiberStruct[fiberID].z:
                    fiberStruct[nextID].transferID(int(iSlice),fiberID,nextID,True) # otherFiberObj.rejected=True, but centerPoints are in "tracked" object

                fiberStruct[nextID].fiberID=nextID

                # update color LUT
                fiberStruct[fiberID].classAttributes["LUT_fiberID_to_color"][nextID]=\
                    fiberStruct[fiberID].classAttributes["LUT_fiberID_to_color"][fiberID]

                del fiberStruct[fiberID].classAttributes["LUT_fiberID_to_color"][fiberID]

                # should not attempt postprocessing on this fiberID, wont be present in voxelMap: remove it
                if fiberID in fiberObj.classAttributes["interpolatedCenters"].keys():
                    del fiberObj.classAttributes["interpolatedCenters"][fiberID]

                del fiberStruct[fiberID]


        

    times_tracking["SmartStitching:"]=time.strftime("%Hh%Mm%Ss", time.gmtime(time.perf_counter()-tic))
    print("\nSmartStitching performed in {: >0.4f}s".format(time.perf_counter()-tic))
    
    print("\nTotal number of fiberObj\t\t: {}".format(len(fiberStructMain)))
    print("Total number of tracked fiberObj\t: {}".format(len(fiberObj.classAttributes["listFiberIDs_tracked"])))


    return fiberStruct,V_fibers.shape,times_tracking,V_fibers,xRes,unitTiff


def saveFiberStruct(commonPath,permutationPath,fiberStructSave,fiberStats):
    ##################################################################################

    # save fiberStruct to JSON

    #################################################################################

    filenameJSONexport  =os.path.join(commonPath,permutationPath,"fiberStats.json")
    filenamePickleExport=os.path.join(commonPath,permutationPath,"fiberStruct.pickle")


    print("\n\ttracking():\n\tWriting output to : \n "+filenameJSONexport)

    with open(filenameJSONexport, "w") as f:
        json.dump(fiberStats, f, sort_keys=False, indent=4)

    with open(filenamePickleExport,"wb") as f:
        pickle.dump(fiberStructSave,f,protocol=pickle.HIGHEST_PROTOCOL)

