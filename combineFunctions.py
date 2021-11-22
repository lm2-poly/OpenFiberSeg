# by Facundo Sosa-Rey, 2021. MIT license

import numpy as np
import os
import pickle
import time

from joblib import Parallel, delayed  
import multiprocessing

from tifffile import TiffFile
from matplotlib import pyplot as plt

from fibers import fiberObj
from extractCenterPoints    import getTiffProperties
from trackingParameters     import getTrackingParams

class LUT_marker():
    def __init__(self):
        self.markers_0={}
        self.markers_1={}

    @staticmethod
    def insertInDict(dictM,m0,m1):
        if m0 in dictM.keys():
            if m1 in dictM[m0].keys():
                dictM[m0][m1]+=1 #increment count
            else:
                dictM[m0][m1]=1
        else:
            dictM[m0]={m1:1}


    def addPair(self,m0,m1):
        LUT_marker.insertInDict(self.markers_0,m0,m1)
        LUT_marker.insertInDict(self.markers_1,m1,m0)

    @staticmethod
    def getMax_fromDict(dictM):
        mMaxAll={}
        for m0 in dictM.keys():
            mMax={
                "fiberID":None,
                "counts" :0
                }

            for m1 in dictM[m0].keys():
                if dictM[m0][m1]>mMax["counts"]:
                    mMax={
                        "fiberID":m1,
                        "counts" :dictM[m0][m1]
                    }
                
            mMaxAll[m0]=mMax

        return mMaxAll

    def getMaxCount(self):
        return LUT_marker.getMax_fromDict(self.markers_0),\
            LUT_marker.getMax_fromDict(self.markers_1)


def findCollisions(V_0,V_1,makeV_collisions=True):

    if V_0.shape != V_1.shape:
        raise ValueError("incompatiple dimensions: {} and {}".\
            format(V_0.shape,V_1.shape))

    mask=np.logical_and(V_0>0,V_1>0)

    z,x,y = np.where(mask)

    if makeV_collisions:
        V_collisions=np.zeros(V_0.shape,np.uint8)
    else:
        V_collisions=None

    listCollidingMarkers={}

    LUT_0_to_1=LUT_marker()


    for i,xx in enumerate(x):
        yy=y[i]
        zz=z[i]
        marker_0=V_0[zz,xx,yy]
        marker_1=V_1[zz,xx,yy]

        LUT_0_to_1.addPair(marker_0,marker_1)

        if makeV_collisions:
            V_collisions[zz,xx,yy]=255

    
    # LUT_0_to_1 contains all the markers from both V_0 and 
    # V_1 that collide with markers from the other, along 
    # with the number of colliding voxels for each pair

    # get the collisions from both side with the highest count
    maxAll_0,maxAll_1=LUT_0_to_1.getMaxCount()

    return maxAll_0,maxAll_1,V_collisions

def addPixelsToImHist(
    imHist_rgb,
    fiberMap,
    fiberStruc:dict=None,
    minLength:float=None,
    cmap:str="Blues",
    compactify:bool=True,
    rejected:bool=False,
    alpha=0.15,
    newRange=[0.4,1.]#how much of the colormap range to occupy
    ):

    if compactify:# only do this if plotting a single slice. else the markers from one slice to another wont correspond
        if rejected:
            markersPresent=[val for val in np.unique(fiberMap) if val<-1. and val!=-999999]
        else:
            markersPresent=[val for val in np.unique(fiberMap) if val>=0]

        compactifyIDs_LUT={}
        for i,fiberID in enumerate(markersPresent):
            if minLength is not None and fiberStruc is not None:
                if fiberStruc[abs(fiberID)].totalLength>minLength:
                    compactifyIDs_LUT[fiberID]=i
                else:
                    compactifyIDs_LUT[fiberID]=-1
            else:
                compactifyIDs_LUT[fiberID]=i

        fiberMap=compactifySlice(fiberMap,compactifyIDs_LUT,rejected=rejected)
    

    oldRange=[-1.,np.max(fiberMap)]

    fiberImg_float=np.interp(np.array(fiberMap,np.float32),oldRange,newRange)

    cm=plt.get_cmap(cmap)
    fiberImg_rgb=cm(fiberImg_float)

    imHist_rgb[fiberMap>=0]=imHist_rgb[fiberMap>=0]*alpha+fiberImg_rgb[fiberMap>=0]*(1.-alpha)

    return sum(sum(fiberMap>=0))

def compactifySlice(slice,compactifyIDs_LUT,rejected=False):
    sliceCompactified=np.ones(slice.shape,np.int32)*-1
    for ix in range(len(slice)):
        for iy in range(len(slice[0])):
            if rejected:
                if -999999<slice[ix,iy]<-1:
                    sliceCompactified[ix,iy]=compactifyIDs_LUT[slice[ix,iy]]
            else:
                if slice[ix,iy]>=0:
                    sliceCompactified[ix,iy]=compactifyIDs_LUT[slice[ix,iy]]
    
    return sliceCompactified

def makePropertySlice(slice,compactifyIDs_LUT):
    sliceProperty=np.ones(slice.shape,np.float32)*-1
    for ix in range(len(slice)):
        for iy in range(len(slice[0])):
            sliceProperty[ix,iy]=compactifyIDs_LUT[slice[ix,iy]]
    
    return sliceProperty

def compactify(commonPath,permutationPaths,parallelHandle=False):
    """this function serve to keep only those fiberID marked as "tracked", eliminating those marked as "rejected",
    also, the fiberIDs form the 3 permutations will be concatenated, i.e. the ids from 132 and 321 will be reassigned to 
    new values such that the fiberIDs in V_fiberMapCompactified are 0,1,2,...,nFibersTotal , with no missing numbers """

    if parallelHandle:
        num_cores=multiprocessing.cpu_count()-1
    else:
        num_cores=1

    ticGlobal=time.perf_counter()

    print("\tLoading permutation 123")

    permutationIndex=0
    permutationPath=permutationPaths[permutationIndex]

    ### permutation123

    filesInDir = [f.path for f in os.scandir(commonPath+permutationPath) if f.is_file()]

    indexFiberMaptiff123=None
    fiberStrucPickle123 =None

    for i,iPath in enumerate(filesInDir):
        if "V_fiberMap_postProcessed.tiff" in iPath:
            indexFiberMaptiff123=i
        if "fiberStruct.pickle" in iPath:
            fiberStrucPickle123=i

    if indexFiberMaptiff123 is None or \
        fiberStrucPickle123 is None:
        raise FileNotFoundError(f"missing files in {commonPath+permutationPath}")


    with TiffFile(filesInDir[indexFiberMaptiff123]) as tif:
        xRes,unitTiff,descriptionStr=getTiffProperties(tif,getDescription=True) 
        V_fiberMap123      =tif.asarray()


    with open(filesInDir[fiberStrucPickle123], "rb") as f:
        fiberStruct123  = pickle.load(f)

        # loading class attributes from 123 allows checking for collisions at filling(),
        # with any of the fibers from 123
        fiberObj.classAttributes=fiberStruct123["fiberObj_classAttributes"]
        listTracked123=list(fiberStruct123["fiberObj_classAttributes"]["listFiberIDs_tracked"])

    exclusiveZone=fiberStruct123["exclusiveZone"]

    color132            =(0.7,0.4,0.1)
    color321            =(0.2,0.7,0.4)
    colorCombination    =(0.1,0.,0.95)
    # colorStitched       =(0.55  ,0.76   ,1.00   ) # cyan

    fiberObj.classAttributes["colors"].update(
        {
            "basic_123"                             :fiberObj.classAttributes["colors"]["basic"],
            "basic_132"                             :color132,
            "basic_321"                             :color321,
            "combined"                              :colorCombination,
            "combined_other"                        :(0.56  ,0.12   ,1.     ),  # purple

            "stitched_smart(added)_transposed"      :(0.55  ,0.76   ,1.00   ),  # cyan
            "stitched_smart(extended)_transposed"   :(0.47  ,0.04   ,0.14   ),  # burgundy
            "backTracking_transposed"               :(0.65  ,0.04   ,1.     ),  # violet
            "stitched(initial)_transposed"          :(1.    ,1.     ,0.     ),  # yellow

            "stitched_smart(added)_lastPass"        :(0.14  ,0.78   ,0.78   ),  # turquoise
            "stitched_smart(extended)_lastPass"     :(0.85  ,0.0    ,0.     ),  # blood red
            "backTracking_lastPass"                 :(0.56  ,0.12   ,1.     ),  # purple
            "stitched(initial)_lastPass"            :(1.    ,0.66   ,0.21   ),  # yellow-orange
        }
    )

    fiberObj.classAttributes["listFiberIDs_tracked"]=set([])

    params=getTrackingParams(commonPath,"secondPass",xRes,unitTiff)

    include123=params["include123"]

    ####################################################################

    ### begin compactification

    nextID=0

    V_fiberMapCompactified=np.ones(V_fiberMap123.shape,np.int32)*-1 #default as -1 for background
    fiberStruct_compactified={}

    compactifyIDs_LUT={}

    if include123:

        for fiberID in listTracked123:
            fiberStruct123["fiberStruct"][fiberID].transpose("123")
            fiberStruct123["fiberStruct"][fiberID].setColor("basic_123")
            # newID containing appropriate suffix, remaped to make a compact representation 
            # (at present, fiberIDs are sparse dut to rejected fibers)

            fiberStruct_compactified[nextID]=fiberStruct123["fiberStruct"][fiberID]
            fiberStruct_compactified[nextID].fiberID=nextID+0.123
            fiberStruct_compactified[nextID].fiberID_previous=fiberID+0.123

            compactifyIDs_LUT[fiberID]=nextID

            nextID+=1

            fiberObj.classAttributes["listFiberIDs_tracked"].add(nextID+0.123)

    
    print("reassign call on 123")

    results = Parallel(n_jobs=num_cores)\
        (delayed(compactifySlice)\
            (
                V_fiberMap123[iSlice],
                compactifyIDs_LUT
            )for iSlice in range(V_fiberMap123.shape[0]) )

    for iSlice,resTuple in enumerate(results):
        V_fiberMapCompactified[iSlice]=resTuple

    toc123=time.perf_counter()

    print("\t permutation123 compactified in ",time.strftime("%Hh%Mm%Ss", time.gmtime(toc123-ticGlobal)))

    ###################################################################

    print("\tLoading permutation 132")

    ### permutation132
    permutationIndex=1
    permutationPath=permutationPaths[permutationIndex]

    filesInDir = [f.path for f in os.scandir(commonPath+permutationPath) if f.is_file()]

    for i,iPath in enumerate(filesInDir):
        if "V_fiberMap_postProcessed.tiff" in iPath:
            indexFiberMaptiff132=i
        if "fiberStruct.pickle" in iPath:
            fiberStrucPickle132=i

    with TiffFile(filesInDir[indexFiberMaptiff132]) as tif:
        V_fiberMap132      =np.transpose(tif.asarray(),(2,1,0))

    with open(filesInDir[fiberStrucPickle132], "rb") as f:
        fiberStruct132  = pickle.load(f)

        listTracked132=fiberStruct132["fiberObj_classAttributes"]["listFiberIDs_tracked"]

        V_temp=np.ones(V_fiberMap132.shape,np.int32)*-1 #default as -1 for background

        fibers132={}
        for fiberID in listTracked132:
            fiberStruct132["fiberStruct"][fiberID].transpose("132")
            fiberStruct132["fiberStruct"][fiberID].setColor("basic_132")

            # newID containing appropriate suffix, remaped to make a compact representation 
            # (at present, fiberIDs are sparse dut to rejected fibers)

            fiberStruct_compactified[nextID]=fiberStruct132["fiberStruct"][fiberID]
            fiberStruct_compactified[nextID].fiberID=nextID+0.132
            fiberStruct_compactified[nextID].fiberID_previous=fiberID+0.132

            fibers132[nextID]=fiberStruct_compactified[nextID]

            compactifyIDs_LUT[fiberID]=nextID

            nextID+=1

            fiberObj.classAttributes["listFiberIDs_tracked"].add(nextID+0.132)

    print("reassign call on 132")

    results = Parallel(n_jobs=num_cores)\
        (delayed(compactifySlice)\
            (
                V_fiberMap132[iSlice],
                compactifyIDs_LUT
            )for iSlice in range(V_fiberMap132.shape[0]) )

    for iSlice,resTuple in enumerate(results):
        V_temp[iSlice]=resTuple

    V_temp[V_temp==0]=-1 #this is a hack, there are fiberIDs==0 present where fibers from 123 are present. TODO: elucidate 
    V_fiberMapCompactified[V_temp!=-1]=V_temp[V_temp!=-1]
    V_fiberMap132=V_temp # needs to be compactified as well so collision check with 321 is consistent with new fiberIDS

    toc132=time.perf_counter()

    print("\t permutation132 compactified in ",time.strftime("%Hh%Mm%Ss", time.gmtime(toc132-toc123)))

    ###################################################################

    print("\tLoading permutation 321")

    ### permutation321
    permutationIndex=2
    permutationPath=permutationPaths[permutationIndex]

    filesInDir = [f.path for f in os.scandir(commonPath+permutationPath) if f.is_file()]
    
    for i,iPath in enumerate(filesInDir):
        if "V_fiberMap_postProcessed.tiff" in iPath:
            indexFiberMaptiff321=i
        if "fiberStruct.pickle" in iPath:
            fiberStrucPickle321=i

    with TiffFile(filesInDir[indexFiberMaptiff321]) as tif:
        V_fiberMap321      =np.transpose(tif.asarray(),(1,0,2))

    with open(filesInDir[fiberStrucPickle321], "rb") as f:
        fiberStruct321  = pickle.load(f)

        listTracked321=fiberStruct321["fiberObj_classAttributes"]["listFiberIDs_tracked"]

        V_temp=np.ones(V_fiberMap132.shape,np.int32)*-1 #default as -1 for background

        fibers321={}
        for fiberID in listTracked321:
            fiberStruct321["fiberStruct"][fiberID].transpose("321")
            fiberStruct321["fiberStruct"][fiberID].setColor("basic_321")

            # newID containing appropriate suffix, remaped to make a compact representation 
            # (at present, fiberIDs are sparse dut to rejected fibers)

            fiberStruct_compactified[nextID]=fiberStruct321["fiberStruct"][fiberID]
            fiberStruct_compactified[nextID].fiberID=nextID+0.321
            fiberStruct_compactified[nextID].fiberID_previous=fiberID+0.321

            fibers321[nextID]=fiberStruct_compactified[nextID]

            compactifyIDs_LUT[fiberID]=nextID

            nextID+=1

            fiberObj.classAttributes["listFiberIDs_tracked"].add(nextID+0.321)

    results = Parallel(n_jobs=num_cores)\
        (delayed(compactifySlice)\
            (
                V_fiberMap321[iSlice],
                compactifyIDs_LUT
            )for iSlice in range(V_fiberMap321.shape[0]) )

    for iSlice,resTuple in enumerate(results):
        V_temp[iSlice]=resTuple

    V_temp[V_temp==0]=-1 #hack: there are regions labelled as 0, should not happen, as fiberID==0 will always be in permutation123 TODO, elucidate
    V_fiberMapCompactified[V_temp!=-1]=V_temp[V_temp!=-1]
    V_fiberMap321=V_temp # needs to be compactified as well so collision check with 321 is consistent with new fiberIDS

    toc321=time.perf_counter()

    print("\t permutation321 compactified in ",time.strftime("%Hh%Mm%Ss", time.gmtime(toc321-toc132)))

    print("\t total compactification in in ",time.strftime("%Hh%Mm%Ss", time.gmtime(toc321-ticGlobal)))

    return V_fiberMapCompactified,fiberStruct_compactified,V_fiberMap132,V_fiberMap321,fibers132,fibers321,xRes,unitTiff,descriptionStr,exclusiveZone