# by Facundo Sosa-Rey, 2021. MIT license

##############################################################################

###   Volumetric processing of fibers

##############################################################################

import time
import os
from tifffile import TiffFile
import pickle
from random import shuffle

from skimage import morphology
from scipy import ndimage
import numpy as np

from extractCenterPoints    import getTiffProperties
from visualisationTool      import addPlaneWidgets
from combineFunctions       import findCollisions
from fibers                 import fiberObj
from combineFunctions       import compactifySlice,makePropertySlice

from mayavi import mlab

from joblib import Parallel, delayed  
import multiprocessing


color0=tuple([float(val)/255. for val in [ 24, 120,250] ])
color1=tuple([float(val)/255. for val in [ 25,155, 50] ])
color2=tuple([float(val)/255. for val in [ 77, 217,155] ])
color3=tuple([float(val)/255. for val in [ 255,179, 25] ])


colorCollisions=tuple([float(val)/255. for val in [ 207, 27,25] ])

def paddingOfVolume(V,radiusZ,radiusX,radiusY,paddingValue=255):

    paddedV_perim = np.zeros((V.shape[0]+2*radiusZ,V.shape[1]+2*radiusX,V.shape[2]+2*radiusY),np.uint8)
    
    #fibers connected to top or bottom of volume must be connected with True value, else would be trimmed
    for i in range(radiusZ):
        paddedV_perim[i,   radiusX:-radiusX,radiusY:-radiusY] = V[ 0,:,:].copy()
        paddedV_perim[-i-1,radiusX:-radiusX,radiusY:-radiusY] = V[-1,:,:].copy()

    #interior region
                                               #included:-(excluded)   
    paddedV_perim[radiusZ:-radiusZ,radiusX:-radiusX,radiusY  :-radiusY    ] = V

    return paddedV_perim
    
def volumetricGapFilling(
    fiberID,
    operationTuple,
    V_thisMarkerOnly,
    SE_radius,
    useInclinedCylinderSE,
    makePlotsIndividual,
    V_hist=None,
    engine=None,
    articlePlots=False
    ):

    lenChain=max(1,operationTuple[0]) # can be 0 in case of backtracking, but should be at least 1
    oriVec=operationTuple[1]
    angle=operationTuple[2]

    # outer convention is [z,x,y]. avoiding transposition except for plotting, for performance purposes
    z,x,y=np.where(V_thisMarkerOnly==255)

    # if a fiber is strongly inclined, it is more efficient to process it after transposition,
    # such that the principal direction is closer to z direction
    transposeBool=None
    if angle>50.:
        
        if abs(oriVec[0])>abs(oriVec[1]): #oriVec is denoted (x,y,z)
            transposeBool="xz"
            temp=z
            z=x
            x=temp
            oriVec=oriVec[[2,1,0]] #transposed to (z,y,x)
            oldShape=V_thisMarkerOnly.shape
            newShape=[oldShape[1],oldShape[0],oldShape[2]] # was denoted (z,x,y), transposed to (x,z,y)
            V_thisMarkerOnly=np.zeros(newShape,np.uint8)
        else:
            transposeBool="yz"
            temp=z
            z=y
            y=temp
            oriVec=oriVec[[0,2,1]] #transposed to (x,z,y)
            oldShape=V_thisMarkerOnly.shape
            newShape=[oldShape[2],oldShape[1],oldShape[0]] # was denoted (z,x,y), transposed to (y,x,z) 
            V_thisMarkerOnly=np.zeros(newShape,np.uint8)

        V_thisMarkerOnly[z,x,y]=255

    xMin=min(x)
    yMin=min(y)
    zMin=min(z)

    xMax=max(x)
    yMax=max(y)
    zMax=max(z)

    xMean=np.mean([xMin,xMax])
    yMean=np.mean([yMin,yMax])
    zMean=np.mean([zMin,zMax])

    SE_size=SE_radius*2+1

    SE_ball3D=morphology.ball(SE_radius, dtype=np.uint8)*255

    SE_ball3D_opening=morphology.ball(SE_radius-1, dtype=np.uint8)*255


    if useInclinedCylinderSE:
        SE_disk=morphology.disk(SE_radius, dtype=np.uint8)*255

        lengthTranslation=lenChain+SE_radius*2
        #extend orientation vector so it reaches across delta z = lengthTranslation
        vectorTranslation=np.array([round(val) for val in oriVec*lengthTranslation/oriVec[2]],np.int)

        offsetX=int(round(vectorTranslation[0]/2))
        offsetY=int(round(vectorTranslation[1]/2))
        
        sizeX=2*abs(offsetX)+2*SE_radius
        sizeY=2*abs(offsetY)+2*SE_radius

        sizeZ=abs(vectorTranslation[2])+2*SE_radius

        #odd size is required for structuring element
        if sizeX%2==0:
            sizeX+=1
        if sizeY%2==0:
            sizeY+=1 
        if sizeZ%2==0:
            sizeZ+=1 

        posX=np.array([round(val)for val in np.linspace(-offsetX,offsetX,sizeZ)],np.int)
        posY=np.array([round(val)for val in np.linspace(-offsetY,offsetY,sizeZ)],np.int)

        SE_rod3D=np.zeros((sizeZ,sizeX,sizeY),np.uint8)

        #middle position, counting from 0:sizeX (sizeX excluded)
        midX=int((sizeX-1)/2)
        midY=int((sizeY-1)/2)

        for i in range(SE_rod3D.shape[0]):
            SE_rod3D[
                i,
                midX+posX[i]-SE_radius:midX+posX[i]-SE_radius+SE_size,
                midY+posY[i]-SE_radius:midY+posY[i]-SE_radius+SE_size]=SE_disk

        paddingSizeX=sizeX
        paddingSizeY=sizeY
        paddingSizeZ=sizeZ

    else:
        #use vertical rod instead of inclined cylinder: bad results for inclined fibers
        SE_rod3D=np.zeros((lenChain+2*SE_radius,SE_ball3D.shape[0],SE_ball3D.shape[1]),np.uint8)

        for i in range(lenChain):
            SE_rod3D[i:SE_size+i,:,:][SE_ball3D==255]=255

        paddingSizeX=paddingSizeY=paddingSizeZ=SE_radius # to avoid artifacts on corners after opening
    
    
    if makePlotsIndividual:
        mlab.figure(figure="Structuring element, closing, fiberID={}".format(fiberID),size=(1200,1050),bgcolor=(1.,1.,1.))

        transposedSE_rod3D=np.transpose(SE_rod3D,(1,2,0))

        if not articlePlots:
            addPlaneWidgets(transposedSE_rod3D,engine,axOnly="z_axes") ## article

        srcRod = mlab.pipeline.scalar_field(transposedSE_rod3D)
        mlab.pipeline.iso_surface(srcRod, contours=[255], opacity=0.5, color=(0.,0.,0.))

        mlab.outline(color=(0,0,0))

        mlab.figure(figure="Structuring element, opening, fiberID={}".format(fiberID),size=(1200,1050),bgcolor=(1.,1.,1.))

        transposedSE_ball3D_opening=np.transpose(SE_ball3D_opening,(1,2,0))

        if not articlePlots:
            addPlaneWidgets(transposedSE_ball3D_opening,engine,axOnly="z_axes") ## article

        srcBall= mlab.pipeline.scalar_field(transposedSE_ball3D_opening)
        mlab.pipeline.iso_surface(srcBall, contours=[255], opacity=0.5, color=(0.,0.,0.))

        mlab.outline(color=(0,0,0))



    # padding on all sides is necessary or else ball SE cannot reach pixels close to boundary
    paddedV_thisMarkerOnly=paddingOfVolume(V_thisMarkerOnly,paddingSizeZ,paddingSizeX,paddingSizeY)

    V_sliceMarker=paddedV_thisMarkerOnly[zMin:zMax+2*paddingSizeZ,xMin:xMax+2*paddingSizeX,yMin:yMax+2*paddingSizeY]


    if makePlotsIndividual:
        mlab.figure(figure="V_sliceMarker, fiberID={}".format(fiberID),size=(1200,1050),bgcolor=(1.,1.,1.))

        srcFiber  = mlab.pipeline.scalar_field(
            np.transpose(
                V_sliceMarker[
                    paddingSizeZ:-paddingSizeZ,
                    paddingSizeX:-paddingSizeX,
                    paddingSizeY:-paddingSizeY
                    ],
                (1,2,0)
            ) 
        )

        mlab.pipeline.iso_surface(srcFiber, contours=[255], opacity=0.45, color=color0)

        if articlePlots:
            import tifffile
            tifffile.imwrite(
                "/home/facu/Phd_Private/Redaction/OpenSeg/PostProcessing/Single/V_fiberID{}_before.tiff".format(fiberID),
                V_sliceMarker,
                compress=True
                )

        else:
            addPlaneWidgets( ###article
                np.transpose(V_hist,(1,2,0)),
                engine, 
                widgetLUT="black-white",
                axOnly="z_axes"
                )
        
        mlab.outline(color=(0,0,0))


    tic = time.perf_counter()



    print("\tvolumetric closing started for fiberID={} on {}, centerOfMass x={: >4.0f}, y={: >4.0f}, z={: >4.0f}".\
        format(fiberID,multiprocessing.current_process().name,xMean,yMean,zMean))

    try:
        V_sliceMarker_closed=np.array(ndimage.binary_closing(V_sliceMarker,SE_rod3D),np.uint8)*255
    except MemoryError:
        print("Encountered: MemoryError, continuing without performing closing on marker={}".format(fiberID))
        V_sliceMarker_closed=V_sliceMarker

    paddedV_thisMarkerOnly[zMin:zMax+2*paddingSizeZ,xMin:xMax+2*paddingSizeX,yMin:yMax+2*paddingSizeY]=V_sliceMarker_closed

    toc = time.perf_counter()

    print("\t\tvolumetric closing completed in {: >4.4f}s for fiberID={} on {}".format(toc-tic,fiberID,multiprocessing.current_process().name))        

    #############################################
    # # # opening

    print("\tvolumetric opening started for fiberID={} on {}, centerOfMass x={: >4.0f}, y={: >4.0f}, z={: >4.0f}".\
        format(fiberID,multiprocessing.current_process().name,xMean,yMean,zMean))

    try:
        V_sliceMarker_closed_opened=np.array(ndimage.binary_opening(V_sliceMarker_closed,SE_ball3D_opening),np.uint8)*255
    except MemoryError:
        print("Encountered: MemoryError, continuing without performing opening on marker={}".format(fiberID))
        V_sliceMarker_closed_opened=V_sliceMarker_closed

    toc = time.perf_counter()

    print("\t\tvolumetric opening completed in {: >4.4f}s for fiberID={} on {}".format(toc-tic,fiberID,multiprocessing.current_process().name))        

    if makePlotsIndividual:
        mlab.figure(figure="V_sliceMarker_closed, fiberID={}".format(fiberID),size=(1200,1050),bgcolor=(1.,1.,1.))

        if not articlePlots:
            #overlay SE in plot ## article
            V_sliceMarker_closed[
                paddingSizeZ:SE_rod3D.shape[0]+paddingSizeZ,
                paddingSizeX:SE_rod3D.shape[1]+paddingSizeX,
                paddingSizeY:SE_rod3D.shape[2]+paddingSizeY]=SE_rod3D

        srcFiber = mlab.pipeline.scalar_field(
            np.transpose(
                V_sliceMarker_closed[
                paddingSizeZ:-paddingSizeZ,
                paddingSizeX:-paddingSizeX,
                paddingSizeY:-paddingSizeY
                ],(1,2,0)
                )
            )

        if articlePlots:
            tifffile.imwrite(
                "/home/facu/Phd_Private/Redaction/OpenSeg/PostProcessing/Single/V_fiberID{}_closed.tiff".format(fiberID),
                V_sliceMarker_closed,
                compress=True
                )    

        mlab.pipeline.iso_surface(srcFiber, contours=[255], opacity=0.5, color=color1)

        transposedV_sliceHist=np.transpose(V_hist,(1,2,0))

        mlab.outline(color=(0,0,0))
        
        ### article
        if not articlePlots:
            addPlaneWidgets(transposedV_sliceHist,engine, widgetLUT="black-white",axOnly="z_axes")

        #################################
        ### opening

        mlab.figure(figure="V_sliceMarker_closed_opened, fiberID={}".format(fiberID),size=(1200,1050),bgcolor=(1.,1.,1.))

        if not articlePlots:
            # overlay SE in plot ### article
            V_sliceMarker_closed_opened[
                paddingSizeZ:SE_ball3D_opening.shape[0]+paddingSizeZ,
                paddingSizeX:SE_ball3D_opening.shape[1]+paddingSizeX,
                paddingSizeY:SE_ball3D_opening.shape[2]+paddingSizeY]=SE_ball3D_opening

        srcFiber = mlab.pipeline.scalar_field(
            np.transpose(
                V_sliceMarker_closed_opened[
                paddingSizeZ:-paddingSizeZ,
                paddingSizeX:-paddingSizeX,
                paddingSizeY:-paddingSizeY
                ],(1,2,0)
                )
            )

        mlab.pipeline.iso_surface(srcFiber, contours=[255], opacity=0.5, color=color3)

        transposedV_sliceHist=np.transpose(V_hist,(1,2,0))

        mlab.outline(color=(0,0,0))
        
        ###article
        if not articlePlots:
            addPlaneWidgets(transposedV_sliceHist,engine, widgetLUT="black-white",axOnly="z_axes")

        if articlePlots:
            mlab.outline(color=(0,0,0))
            engine=mlab.get_engine()
            for iScene in range(5):#[2,3,4]:
                scene = engine.scenes[iScene]
                scene.scene.camera.position = [230.7885982150969, -55.77080802359471, 70.10653478028216]
                scene.scene.camera.focal_point = [13.5, 22.0, 56.0]
                scene.scene.camera.view_angle = 30.0
                scene.scene.camera.view_up = [-0.060090652925034176, 0.013151557327761474, 0.9981062819013302]
                scene.scene.camera.clipping_range = [185.46468568513774, 289.3690443494203]
                scene.scene.camera.compute_view_plane_normal()
                scene.scene.render()

            tifffile.imwrite(
                "/home/facu/Phd_Private/Redaction/OpenSeg/PostProcessing/Single/V_fiberID{}_after.tiff".format(fiberID),
                V_sliceMarker_closed_opened,
                compress=True
                ) 

        mlab.show()
    
    # if fiber connects with first or last imSlice, it is copied in the z direction,
    # must be erased before transfering
    V_sliceMarker_closed_opened[:paddingSizeZ ,:,:]=0
    V_sliceMarker_closed_opened[-paddingSizeZ:,:,:]=0

    zNew,xNew,yNew=np.where(V_sliceMarker_closed_opened==255)

    xNew+=xMin-paddingSizeX
    yNew+=yMin-paddingSizeY
    zNew+=zMin-paddingSizeZ

    if transposeBool is not None:
        if transposeBool=="xz":
            temp=zNew
            zNew=xNew
            xNew=temp
        elif transposeBool=="yz":
            temp=zNew
            zNew=yNew
            yNew=temp

    return zNew,xNew,yNew


def parallelGapFilling(
    fiberID,
    operationTuple,
    V_fiberMap,
    makePlotAll,
    makePlotsIndividual,
    V_hist=None,
    engine=None,
    checkCollision=True,
    SE_radius=4,
    useInclinedCylinderSE=True,
    articlePlots=False
    ):
        fiberID=int(round(fiberID))

        if articlePlots:
            V_fiberMap[0:100]=-1 ## article

        zBefore,xBefore,yBefore=np.where(V_fiberMap==fiberID)

        xMin=min(xBefore)
        xMax=max(xBefore)

        yMin=min(yBefore)
        yMax=max(yBefore)

        zMin=min(zBefore)
        zMax=max(zBefore)

        #create smaller volume where fiber is present (will be padded inside volumetricGapFilling)
        V_thisMarkerOnly=np.zeros((zMax-zMin+1,xMax-xMin+1,yMax-yMin+1),np.uint8)

        V_thisMarkerOnly[zBefore-zMin,xBefore-xMin,yBefore-yMin]=255

        if makePlotsIndividual:
            V_hist_thisMarkerOnly=V_hist[zMin:zMax,xMin:xMax,yMin:yMax]
        else:
            V_hist_thisMarkerOnly=None

        if 255 not in V_thisMarkerOnly:
            raise ValueError(f"fiberID:{fiberID} not found in V_fiberMap")

        if makePlotAll:
            oldVoxels=(zBefore,xBefore,yBefore)
        else:
            oldVoxels=None

        zNew,xNew,yNew=volumetricGapFilling(
            fiberID,
            operationTuple,
            V_thisMarkerOnly,
            SE_radius=SE_radius,
            useInclinedCylinderSE=useInclinedCylinderSE,
            makePlotsIndividual=makePlotsIndividual,
            V_hist=V_hist_thisMarkerOnly,
            engine=engine,
            articlePlots=articlePlots
            )

        newVoxels={fiberID:{
            "zNew"      :zNew+zMin,
            "xNew"      :xNew+xMin,
            "yNew"      :yNew+yMin
            }
        }


        if checkCollision:
            V_NewVoxels=np.zeros(V_fiberMap.shape,np.uint32)

            #reassign voxels added by morphological operations for collision detection
            V_NewVoxels[zNew+zMin,xNew+xMin,yNew+yMin]=fiberID

            # maxAll_old contains collisions from V_newVoxels to global V_fiberMap
            # maxAll_new contains collisions from V_fiberMap to V_new: will only have self-references in this case 
            maxAll_old,maxAll_new=findCollisions(V_fiberMap,V_NewVoxels,makeV_collisions=False)[:2]

            #remove self-referenced, false collision
            collisions={key:val for key,val in maxAll_old.items() if key!=fiberID}

            collisionsDict={}

            if maxAll_old:
                collisionsDict[fiberID]={
                        "collisions":collisions,
                        "newVoxels" :newVoxels[fiberID]
                    }
        else: 
            collisionsDict=None

        return oldVoxels,newVoxels,collisionsDict


def collisionDetectionWrapper(
    postProcessQueue,
    minCountsCombination,
    angleCombineDEG,
    oriVecAll,
    fiberStruct,
    V_fiberMap,
    fiberStruct_combined,
    V_hist=None,
    makePlotsIndividual=False,
    makePlotAll=False,
    parallelHandle=False
    ):

    newVoxels={}
    collisionsDict={}

    if makePlotsIndividual or makePlotAll:
        engine=mlab.get_engine()
    else:
        engine=None

    if makePlotAll:        
        V_before=np.zeros(V_fiberMap.shape,np.uint8)
        V_after =np.zeros(V_fiberMap.shape,np.uint8)
        V_collisions=np.zeros(V_fiberMap.shape,np.uint8)
    else:
        V_before=V_after=V_collisions=None
    
    if parallelHandle:
        num_cores=int(multiprocessing.cpu_count()/3) # may cause memory overload if too many processes are used simultaneously
        makePlotsIndividual=False
        V_hist_parallel=None # shouldn't be sent if not used for plotting
        engine_parallel=None # cant be sent in a parallel call, un-pickleable
    else:
        num_cores=1
        V_hist_parallel=V_hist
        engine_parallel=engine

    results= Parallel(n_jobs=num_cores)\
        (delayed(parallelGapFilling)\
            (fiberID,
            operationTuple,
            V_fiberMap,
            makePlotAll,
            makePlotsIndividual,
            V_hist_parallel,
            engine_parallel,
            )for fiberID,operationTuple in postProcessQueue)

    for resTuple in results:
        if makePlotAll:
            zBefore,xBefore,yBefore=resTuple[0]
            V_before[zBefore,xBefore,yBefore]=255

        newVoxels       .update(resTuple[1])
        collisionsDict  .update(resTuple[2])


    combineLUT={}
    combinedAfterGapFilling=set([])
    combinedPreviously=set([fiberID for fiberID in fiberStruct_combined.keys()])

    for fiberID,collisions in collisionsDict.items():
        for fiberID_other,collision in collisions["collisions"].items():
            if collision["counts"]>minCountsCombination:

                oriVecSelf =oriVecAll[fiberID]
                oriVecOther=oriVecAll[fiberID_other]

                angle=np.degrees(np.arccos(np.dot(oriVecSelf,oriVecOther)))

                if angle>90:
                    angle=180-angle

                if angle<angleCombineDEG:

                    # if this fiber already has been combined to another, combine to that other one
                    if fiberID in combineLUT.keys():
                        fiberID=combineLUT[fiberID]
                    
                    # if the other fiber already has been combined to another, combine to that other one
                    if fiberID_other in combineLUT.keys():
                        fiberID_other=combineLUT[fiberID_other]

                    if fiberID!=fiberID_other: #otherwise, it means the combination has already been performed

                        combinedAfterGapFilling.add(fiberID)
                        fiberStruct_combined[fiberID]=fiberStruct[fiberID]
                        
                        if fiberID_other in combinedAfterGapFilling:
                            combinedAfterGapFilling.remove(fiberID_other)
                        
                        fiberID_otherSuffix=fiberID_other+fiberStruct[fiberID_other].suffix

                        if fiberID_otherSuffix in fiberObj.classAttributes["listFiberIDs_tracked"]:
                            # if fiber is combined more than once
                            fiberObj.classAttributes["listFiberIDs_tracked"].remove(fiberID_otherSuffix)

                        # add otherFib's voxels to this one:
                        V_fiberMap[V_fiberMap==fiberID_other]=fiberID
                        if makePlotAll:
                            V_collisions[V_fiberMap==fiberID_other]=255

                        #combine fiberObj
                        fiberStruct[fiberID].combine(fiberStruct[fiberID_other])
                        fiberStruct[fiberID].setColor("combined")
                        fiberStruct[fiberID_other].setColor("combined_other")

                        combineLUT[fiberID_other]=fiberID
                        
                        fiberStruct[fiberID_other].tags.add("combined_postProcessing")

                    

    for key,fiber in fiberStruct.items():
        if fiber.colorLabel=="combined":
            raiseError=False
            if int(key) not in combinedAfterGapFilling and \
                int(key not in combinedPreviously):
                print("key:",key,"\tfiberID:",fiber.fiberID)
                raiseError=True
            if raiseError:
                raise RuntimeError("not labelling correctly")


    #last Pass of postProcessing for the fibers which were combined (there can remain gaps)

    postProcessQueue_index={}
    
    for index,dataTuple in enumerate(postProcessQueue):
        fiberID=dataTuple[0]

        postProcessQueue_index[int(fiberID)]=index

    postProcessQueueSecondPass=[
        (
            fiberID,
            postProcessQueue[postProcessQueue_index[fiberID] ][1] #(lenChain, oriVec, angle)
        ) for fiberID in combinedAfterGapFilling]

    #TODO if only a section of V_fiberMap is passed to each parallel process, much less memory requirement. 
    # to that end, could use a function to truncate V_fiberMap to where fiberID is present, and keep track of the x,y,z offsets,
    # and reflect those in results
    results= Parallel(n_jobs=num_cores)\
        (delayed(parallelGapFilling)\
            (fiberID,
            operationTuple,
            V_fiberMap,
            makePlotAll,
            makePlotsIndividual,
            V_hist_parallel,
            engine_parallel
            )for fiberID,operationTuple in postProcessQueueSecondPass)

    for resTuple in results:
        if makePlotAll:
            zBefore,xBefore,yBefore=resTuple[0]
            V_before[zBefore,xBefore,yBefore]=255

        newVoxels       .update(resTuple[1])
        collisionsDict  .update(resTuple[2])

    # Assign all new voxels to global fiberMap
    for fiberID in newVoxels.keys():
        zNew=newVoxels[fiberID]["zNew"]
        xNew=newVoxels[fiberID]["xNew"]
        yNew=newVoxels[fiberID]["yNew"]

        V_fiberMap[zNew,xNew,yNew]=fiberID
    
        if makePlotAll:
            V_after[zNew,xNew,yNew]=255

    if makePlotAll:
        mlab.figure(figure="Before/After",size=(1200,1050),bgcolor=(0.1,0.1,0.1))

        srcBefore =     mlab.pipeline.scalar_field(np.transpose(V_before,       (1,2,0)) )
        srcAfter  =     mlab.pipeline.scalar_field(np.transpose(V_after ,       (1,2,0)) )
        srcCollisions = mlab.pipeline.scalar_field(np.transpose(V_collisions,   (1,2,0)) )

        mlab.pipeline.iso_surface(srcBefore,        contours=[255], opacity=0.8, color=color1)
        mlab.pipeline.iso_surface(srcAfter ,        contours=[255], opacity=0.8, color=color2)
        mlab.pipeline.iso_surface(srcCollisions,    contours=[255], opacity=0.8, color=colorCollisions)

        V_planeWidget=np.transpose(V_hist,(1,2,0))

        addPlaneWidgets(V_planeWidget,engine, widgetLUT="black-white",axOnly="z_axes")

        mlab.outline()

        mlab.show()

    return collisionsDict,V_fiberMap


def randomizeVoxels(V_fiberMap,listMarkers,parallelHandle=True):

    V_fiberMap_randomized=V_fiberMap.copy()

    reassignedMarkers=listMarkers.copy()
    #random shuffling of original markers
    shuffle(reassignedMarkers)
    
    markerLUT={}
    for i,iMark in enumerate(listMarkers):
        markerLUT[iMark]=reassignedMarkers[i]

    if parallelHandle:
        num_cores=multiprocessing.cpu_count()-1 
    else:
        num_cores=1

    results = Parallel(n_jobs=num_cores)\
    (delayed(compactifySlice)\
        (
            V_fiberMap[iSlice],
            markerLUT
        )for iSlice in range(V_fiberMap.shape[0]) )

    for iSlice,resTuple in enumerate(results):
        V_fiberMap_randomized[iSlice]=resTuple

    return V_fiberMap_randomized

def makePropertyMap(V_fiberMap,marker_to_propertyLUT,parallelHandle=True):

    V_fiberMap_property=np.empty(V_fiberMap.shape,np.float32)
    
    if parallelHandle:
        num_cores=multiprocessing.cpu_count()-1 #will cause memory overload for large sets if too many cores used 
    else:
        num_cores=1

    results = Parallel(n_jobs=num_cores)\
    (delayed(makePropertySlice)\
        (
            V_fiberMap[iSlice],
            marker_to_propertyLUT
        )for iSlice in range(V_fiberMap.shape[0]) )

    for iSlice,resTuple in enumerate(results):
        V_fiberMap_property[iSlice]=resTuple

    return V_fiberMap_property


def postProcessingOfFibers(
        commonPath,
        permutationPath,
        SE_radius=4,
        useInclinedCylinderSE=True,
        makePlotsIndividual=False,
        makePlotAll=False,
        randomize=False,
        parallelHandle=False,
        postProcessAllFibers=False,
        articlePlots=False,
        article_savePlotALL=False,
        exclusiveFibers=None #list of fibers which will be postprocessed. useful for debugging
    ):
    
    if makePlotsIndividual or makePlotAll:
        engine = mlab.get_engine()
    else:
        engine=None

    print('\n\tpostProcessingOfFibers() called on dataset:\n {}\n\treading from disk'.format(commonPath))

    tic = time.perf_counter()

    with TiffFile(commonPath+permutationPath+"V_fiberMap.tiff") as tif:
        print("\tloading: \n"+commonPath+permutationPath+"V_fiberMap.tiff")
        xRes,unitTiff,descriptionStr=getTiffProperties(tif,getDescription=True)

        V_fiberMap=tif.asarray()

    with TiffFile(commonPath+permutationPath+"V_pores.tiff") as tif:
        print("\tloading: \n"+commonPath+permutationPath+"V_pores.tiff")
        V_pores=tif.asarray()   

    try:
        with TiffFile(commonPath+permutationPath+"V_perim.tiff") as tif:
            print("\tloading: \n"+commonPath+permutationPath+"V_perim.tiff")
            V_perim=tif.asarray()   
    except:
        V_perim=None

    if makePlotsIndividual or makePlotAll:
        with TiffFile(commonPath+permutationPath+"V_hist.tiff") as tif:
            print("\tloading: \n"+commonPath+permutationPath+"V_hist.tiff")
            V_hist=tif.asarray()

    else:
        V_hist=None

    if makePlotAll:        
        V_before=np.zeros(V_fiberMap.shape,np.uint8)
        V_after =np.zeros(V_fiberMap.shape,np.uint8)
    else:
        V_before=V_after=None

    if permutationPath!="Permutation123/":

        filesInDir123 = [f.path for f in os.scandir(commonPath+"Permutation123/") if f.is_file()]
        indexFibers_mask=None
        for i,iPath in enumerate(filesInDir123):
            if "V_fibers_masked.tiff" in iPath:
                indexFibers_mask=i

        if indexFibers_mask is None:
            raise RuntimeError("Can't find V_fibers_masked.tiff in \n{}".\
                format(commonPath+"Permutation123/"))

        with TiffFile(filesInDir123[indexFibers_mask]) as tif:
            print("\tloading: \n"+filesInDir123[indexFibers_mask])
            if permutationPath=="Permutation132/":
                transposeTuple=(2,1,0) # [z,x,y]->[y,x,z]
            elif permutationPath=="Permutation321/":
                transposeTuple=(1,0,2) # [z,x,y]->[x,z,y]

            V_fibers_masked=np.array(np.transpose(tif.asarray()/255,transposeTuple),np.uint8)

        if V_fiberMap.shape!=V_fibers_masked.shape:
            raise ValueError("V_fiberMap.tiff and V_fibers_masked.tiff are of incompatible shapes")

    with open(commonPath+permutationPath+"fiberStruct.pickle" , "rb") as f:
        fiberStruct  = pickle.load(f)

    exclusiveZone=fiberStruct["exclusiveZone"]

    if len(exclusiveZone)>0:
        if exclusiveZone is not None:
            if permutationPath=="Permutation123/":
                zMin=exclusiveZone["zMin"]
                zMax=exclusiveZone["zMax"]
                xMin=exclusiveZone["xMin"]
                xMax=exclusiveZone["xMax"]
                yMin=exclusiveZone["yMin"]
                yMax=exclusiveZone["yMax"]
            elif permutationPath=="Permutation132/":
                zMin=exclusiveZone["yMin"]
                zMax=exclusiveZone["yMax"]
                xMin=exclusiveZone["xMin"]
                xMax=exclusiveZone["xMax"]
                yMin=exclusiveZone["zMin"]
                yMax=exclusiveZone["zMax"]
            elif permutationPath=="Permutation321/":
                zMin=exclusiveZone["xMin"]
                zMax=exclusiveZone["xMax"]
                xMin=exclusiveZone["zMin"]
                xMax=exclusiveZone["zMax"]
                yMin=exclusiveZone["yMin"]
                yMax=exclusiveZone["yMax"]

        V_pores=V_pores[zMin:zMax,xMin:xMax,yMin:yMax]
        if V_perim is not None:
            V_perim=V_perim[zMin:zMax,xMin:xMax,yMin:yMax]
    else:
        exclusiveZone=None

    toc = time.perf_counter()
    print(f"\treading from disk complete in {toc - tic:0.4f} seconds\n")

    times_postProc={}
    times_postProc["reading from disk only:"]=time.strftime("%Hh%Mm%Ss", time.gmtime(toc-tic))

    ticPost = time.perf_counter()

    postProcessQueue=[]

    if postProcessAllFibers:
        doNotPostProcess={"initial_stitched_segment","stitched_blind(added)","stitched_smart(added)"}
        for fiberID,fibObj in fiberStruct["fiberStruct"].items():
            if fiberID not in fiberStruct["fiberObj_classAttributes"]["interpolatedCenters"].keys() and\
                not fibObj.tags.intersection(doNotPostProcess):
                    fiberStruct["fiberObj_classAttributes"]["interpolatedCenters"][fiberID]=[5]

    # only fiberObj that have interuptions (centroids are added by filling), and 
    # that are not rejected are postProcessed
    for fiberID,interpolationChains in fiberStruct["fiberObj_classAttributes"]["interpolatedCenters"].items():

        fibO=fiberStruct["fiberStruct"][fiberID]

        skip=True  
        if exclusiveFibers is not None:
            if fiberID in exclusiveFibers:
                skip=False
        else:
            skip=False

        if not skip:

            fiberObj.initializeClassAttributes(savedAttributes=fiberStruct["fiberObj_classAttributes"])

            #fiberObj that were added to another at blindStitching() or smartStitching wont be processed (starting fiberObj will)
            doNotPostProcess={"stitched_blind(added)","stitched_smart(added)"}
            if not fibO.tags.intersection(doNotPostProcess):
                if not fibO.rejected:
                    oriVec=fibO.orientationVec
                    oriVec/=np.linalg.norm(oriVec)
                    #if there is more than one interpolation chain, keep the longest to create Structuring Element
                    if len(interpolationChains)>1:
                        # listLengths=[len(chain) for chain in interpolationChains]
                        # pos=listLengths.index(max(listLengths))
                        pos=interpolationChains.index(max(interpolationChains))
                    else:
                        pos=0
                    
                    angle=np.degrees(np.arccos(np.dot(oriVec,[0.,0.,1.])))

                    postProcessQueue.append(
                        (
                            fiberID,
                            (interpolationChains[pos], oriVec,angle)
                        )
                    )

    ########################################
    ### 
    ###  in V_fiberMap:
    ###  markers>=0 are "tracked" fiberIDs
    ###  marker==-1 is background
    ###  markers<-1 are "rejected" fiberIDs
    ###  markers==-999999 are unmatched
    ### 
    ########################################
  
    newVoxels={}

    if parallelHandle:
        num_cores=int(multiprocessing.cpu_count()*2/3) # may cause memory overload if too many processes are used simultaneously on large datasets
        makePlotsIndividual=False
        V_hist_parallel=None # shouldn't be sent if not used for plotting
        engine_parallel=None # cant be sent in a parallel call, un-pickleable
    else:
        num_cores=1
        V_hist_parallel=V_hist
        engine_parallel=engine

    results= Parallel(n_jobs=num_cores)\
    (delayed(parallelGapFilling)\
        (fiberID,
        operationTuple,
        V_fiberMap,
        makePlotAll,
        makePlotsIndividual,
        V_hist_parallel,
        engine_parallel,
        checkCollision=False, #it is unlikely to encounter large collisions at this point. will be checked at combinePermutations()
        SE_radius=SE_radius,
        useInclinedCylinderSE=useInclinedCylinderSE,
        articlePlots=articlePlots
        )for fiberID,operationTuple in postProcessQueue)

    for resTuple in results:
        if makePlotAll:
            zBefore,xBefore,yBefore=resTuple[0]
            V_before[zBefore,xBefore,yBefore]=255

        newVoxels       .update(resTuple[1])
   
    # Assign all new voxels to global fiberMap
    for fiberID in newVoxels.keys():
        zNew=newVoxels[fiberID]["zNew"]
        xNew=newVoxels[fiberID]["xNew"]
        yNew=newVoxels[fiberID]["yNew"]

        V_fiberMap[zNew,xNew,yNew]=fiberID
    
        if makePlotAll:
            V_after[zNew,xNew,yNew]=255

    #prevent spillover to pores and perim
    V_fiberMap[V_pores==255]=-1
    if V_perim is not None:
        V_fiberMap[V_perim==255]=-1

    if permutationPath!="Permutation123/":
        # V_fibers_masked==1 where there is a fiber present from permutation123
        # collisions prevented here
        V_fiberMap[V_fibers_masked==1]=-1 # marker==-1 is background
        if makePlotAll:
            V_after_masked=V_after.copy()
            V_after_masked[V_fibers_masked==1]=0

    if makePlotAll:
        mlab.figure(figure="Before/After",size=(1200,1050),bgcolor=(0.1,0.1,0.1))

        srcBefore = mlab.pipeline.scalar_field(np.transpose(V_before,(1,2,0)) )
        srcAfter  = mlab.pipeline.scalar_field(np.transpose(V_after ,(1,2,0)) )

        mlab.pipeline.iso_surface(srcBefore, contours=[255], opacity=1.0, color=color3)
        mlab.pipeline.iso_surface(srcAfter , contours=[255], opacity=0.8, color=color0)
    
        if permutationPath!="Permutation123/":
            srcAfter_masked  = mlab.pipeline.scalar_field(np.transpose(V_after_masked ,(1,2,0)) )
            mlab.pipeline.iso_surface(srcAfter_masked , contours=[255], opacity=0.8, color=(0.9,0.20,0.1))

            V_planeWidget=np.transpose(V_fibers_masked,(1,2,0))

        else:
            V_planeWidget=np.transpose(V_hist,(1,2,0))

        if exclusiveZone:
            rangeOutline=[-exclusiveZone["xMin"],0,-exclusiveZone["yMin"],0,-exclusiveZone["zMin"],0.]
        else:
            rangeOutline=None
        
        addPlaneWidgets(V_planeWidget,engine, widgetLUT="black-white",axOnly="z_axes",rangeOutline=rangeOutline)

        mlab.outline()

        if article_savePlotALL:
            import tifffile
            tifffile.imwrite(
                "/home/facu/Phd_Private/Redaction/OpenSeg/PostProcessing/All/V_before.tiff",
                V_before,
                resolution=(xRes,xRes,unitTiff),
                compress=True
                )

            tifffile.imwrite(
                "/home/facu/Phd_Private/Redaction/OpenSeg/PostProcessing/All/V_after.tiff",
                V_after,
                resolution=(xRes,xRes,unitTiff),
                compress=True
                )


    tocPost=time.perf_counter()

    print(f"\tpostProcessingOfFibers call complete in {tocPost-ticPost:0.4f} seconds\n")
    times_postProc["postProcessingOfFibers only:"]=time.strftime("%Hh%Mm%Ss", time.gmtime(tocPost-ticPost))

    ###############################################################################################

    # voxelReassignment to make it easier to identify different fibers:

    ##############################################################################################

    if randomize:

        listMarkers=np.unique(V_fiberMap)
        listMarkers=[val for val in listMarkers if val>=0]# tracked fibers have markers starting at 0

        for fiberID in fiberStruct["fiberObj_classAttributes"]["listFiberIDs_tracked"]:
            if fiberID not in listMarkers:
                print("in listFiberIDs_tracked, not in listMarkers",fiberID) 

        for fiberID in listMarkers:
            if fiberID not in fiberStruct["fiberObj_classAttributes"]["listFiberIDs_tracked"]:
                print("in listMarkers, not in listFiberIDs_tracked",fiberID) 

        print("\trandom shuffling of markers for rendering purposes started")

        ticRandomize=time.perf_counter()
        
        V_fiberMap_randomized=randomizeVoxels(V_fiberMap,listMarkers)

        tocRandomize=time.perf_counter()

        times_postProc["random shuffling in: "]=time.strftime("%Hh%Mm%Ss", time.gmtime(tocRandomize-ticRandomize))

    else:
        V_fiberMap_randomized=None


    if permutationPath=="Permutation123/":

        ticMakeMask=time.perf_counter()

        V_fibers_masked=np.zeros(V_fiberMap.shape,np.uint8)

        # in V_fibers_masked, mark pixels that were not tracked as False. 
        # "Tracked" pixels will then be removed from the V_fibers.tiff in 
        # other permutations. This is so extractCenterPoints() only finds centroids in regions not 
        # already containing a fiber from permutation 123, as in V_fibers[V_fibers_masked==1]=0. line 293

        V_fibers_masked[V_fiberMap>=0]=255 # markers>0 are "tracked", background(matrix) has marker==-1. markers>-1 are untracked(-999999) or rejected

        tocMakeMask=time.perf_counter()

        print(f"making fibermask in: {tocMakeMask-ticMakeMask:0.4f} seconds\n")

        times_postProc["making mask in: "]=time.strftime("%Hh%Mm%Ss", time.gmtime(tocMakeMask-ticMakeMask))
    else:
        V_fibers_masked=None

    return V_fiberMap,V_fiberMap_randomized,V_fibers_masked,xRes,unitTiff,descriptionStr,times_postProc

