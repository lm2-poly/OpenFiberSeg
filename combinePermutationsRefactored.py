# by Facundo Sosa-Rey, 2021. MIT license

import multiprocessing
from tifffile.tifffile import imwrite
from extractCenterPoints    import getTiffProperties
from fibers                 import fiberObj
from trackingFunctions      import fiberPloting
from trackingParameters     import getTrackingParams
from combineFunctions       import compactifySlice,findCollisions,compactify
from postProcessing         import collisionDetectionWrapper

from tifffile import TiffFile,imwrite
import pickle
from joblib import Parallel, delayed  
import os
import time
import numpy as np


def combinePermutations(
    commonPath,
    permutationPaths,
    makePlot=False,
    parallelHandle=True
    ):

    permutationIndex=0

    permutationPath=permutationPaths[permutationIndex]

    print("\n\tcombinePermutation() called on dataset: \n{}".format(commonPath))
    print("\t\treading from disk")

    tic=time.perf_counter()

    ###################################################################

    ### check if compactification has already been performed

    filesInDir = [f.path for f in os.scandir(commonPath) if f.is_file()]

    indexFiberMapTiffCompactified   =fiberStrucCompactifiedPickle    =None
    indexFiberMapTiffCompactified132=indexFiberMapTiffCompactified321=None

    for i,iPath in enumerate(filesInDir):
        if "V_fiberMapCompactified.tiff" in iPath:
            indexFiberMapTiffCompactified=i
        if "fiberStruct_compactified.pickle" in iPath:
            fiberStrucCompactifiedPickle=i
        if "V_fiberMap132_Compactified.tiff" in iPath:
            indexFiberMapTiffCompactified132=i
        if "V_fiberMap321_Compactified.tiff" in iPath:
            indexFiberMapTiffCompactified321=i


    if indexFiberMapTiffCompactified is None:
        #first time through
        print("\tCompactifying data...")

        V_fiberMapCompactified,\
        fiberStruct_compactified,\
        V_fiberMap132,\
        V_fiberMap321,\
        fibers132,\
        fibers321,\
        xRes,\
        unitTiff,\
        descriptionStr,\
        exclusiveZone=compactify(commonPath,permutationPaths,parallelHandle=parallelHandle)

        print(f"\tWriting compactified fiberMap to disk at\n {commonPath}V_fiberMapCompactified.tiff")

        imwrite(
            os.path.join(commonPath,'V_fiberMapCompactified.tiff'),
            V_fiberMapCompactified,
            resolution=(xRes,xRes,unitTiff),
            description=descriptionStr,
            compress=True
            )

        imwrite(
            os.path.join(commonPath,'V_fiberMap132_Compactified.tiff'),
            V_fiberMap132,
            resolution=(xRes,xRes,unitTiff),
            description=descriptionStr,
            compress=True
            )

        imwrite(
            os.path.join(commonPath,'V_fiberMap321_Compactified.tiff'),
            V_fiberMap321,
            resolution=(xRes,xRes,unitTiff),
            description=descriptionStr,
            compress=True
            )

        fiberStructPickle={
            "fiberStructCompactified" :fiberStruct_compactified,
            "fiberObj_classAttributes":fiberObj.classAttributes, #otherwise class attributes are not pickled
            "fibers132"               :fibers132,               # these have the new IDs (compactified)
            "fibers321"               :fibers321,
            "exclusiveZone"           :exclusiveZone
        }

        with open(os.path.join(commonPath,"fiberStruct_compactified.pickle"),"wb") as f:
            pickle.dump(fiberStructPickle,f,protocol=pickle.HIGHEST_PROTOCOL)

    else:
        print("\tFound previously compactified data, loading...")

        with TiffFile(filesInDir[indexFiberMapTiffCompactified]) as tif:
            xRes,unitTiff,descriptionStr=getTiffProperties(tif,getDescription=True) 
            V_fiberMapCompactified      =tif.asarray()

        #load fiberMap from other permutations, to check for collisions. 
        #these fiberMaps have the compactified (dense, not spare) fiberIDs
        with TiffFile(filesInDir[indexFiberMapTiffCompactified132]) as tif:
            V_fiberMap132     =tif.asarray()

        with TiffFile(filesInDir[indexFiberMapTiffCompactified321]) as tif:
            V_fiberMap321     =tif.asarray()

        with open(filesInDir[fiberStrucCompactifiedPickle], "rb") as f:
            fiberStruct_compactified_all  = pickle.load(f)

        fiberStruct_compactified=fiberStruct_compactified_all["fiberStructCompactified"]

        fibers132=fiberStruct_compactified_all["fibers132"]
        fibers321=fiberStruct_compactified_all["fibers321"]

        # loading class attributes from 123 allows checking for collisions at filling(),
        # with any of the fibers from 123
        fiberObj.classAttributes=fiberStruct_compactified_all["fiberObj_classAttributes"]

        exclusiveZone=fiberStruct_compactified_all["exclusiveZone"]

        del fiberStruct_compactified_all


    ### permutation123

    filesInDir = [f.path for f in os.scandir(os.path.join(commonPath,permutationPath)) if f.is_file()]
    watershedFound=False

    if makePlot:
        print("\tLoading V_hist for plotting purposes")
        indexHistTiff123    =None
        for i,iPath in enumerate(filesInDir):
            if "V_hist.tiff" in iPath:
                indexHistTiff123=i

        if indexHistTiff123 is None :
            raise FileNotFoundError(f"missing files in {os.path.join(commonPath,permutationPath)}")

        with TiffFile(filesInDir[indexHistTiff123]) as tif:
            V_hist=tif.asarray()/255
    else:
        V_hist=None
            
    params=getTrackingParams(commonPath,"secondPass",xRes,unitTiff)

    smartStitchingMinFibLength          =params["smartStitchingMinFibLength"]
    smartStitchingMaxDistance           =params["smartStitchingMaxDistance"]
    smartStitchingMaxLateralDist        =params["smartStitchingMaxLateralDist"]
    smartStitchingAlignAngle            =params["smartStitchingAlignAngle"]
    smartStitchingBackTrackingLimit     =params["smartStitchingBackTrackingLimit"]
    processingMinFiberLength            =params["processingMinFiberLength"]
    tagAngleTooSteep                    =params["tagAngleTooSteep"]
    maxSteepnessAngle                   =params["maxSteepnessAngle"]

    # include123                          =params["include123"]
    doSecondPass                        =params["doSecondPass"]
    doLastPass                          =params["doLastPass"]
    verboseHandle                       =params["verboseHandle"]
    preventSelfStitch           =params["preventSelfStitch_123_123"]

    toc=time.perf_counter()

    print("\t\treading from disk complete in {: >6.2f} s".format(toc-tic))


    ####################################################################################

    ### check for collisions between fibers in 132 and 321

    ####################################################################################

    # by design, there can't be any collisions from V_fiberMap123-> no centroid allowed in regions masked by fiber detections in 123


    maxAll132,maxAll321,V_collisions=findCollisions(V_fiberMap132,V_fiberMap321)


    #########################################################

    ### create new fiberObj from colliding voxels

    minCountCombination=100
    angleCombineDEG=30. #max angle difference to allow combination of fibers from different permutations

    offset_nextAvailableID=1

    doCombineFibers=True

    fiberStruct_combined={}

    if doCombineFibers:

        for fiberID132,dictF in maxAll132.items():
            fiberID321=dictF["fiberID"]

            counts=dictF["counts"]

            if fiberID132==5594:
                print()

            # Only combine fibers if minimal number of pixels are interfering
            if counts>minCountCombination:
                fibObj321=fibers321[fiberID321]

                oriVec321=fibObj321.orientationVec/np.linalg.norm(fibObj321.orientationVec)

                oriVec132=fibers132[fiberID132].orientationVec/np.linalg.norm(fibers132[fiberID132].orientationVec)

                angle=np.degrees(np.arccos(np.dot(oriVec132,oriVec321)))

                if angle>90:
                    angle=180-angle

                if angle<angleCombineDEG:

                    # keep uncombined Obj for plotting purposes
                    newFibObj=fibers132[fiberID132].copy()
                    nextID= len(fibers132)+offset_nextAvailableID
                    offset_nextAvailableID+=1
                    fibers132[nextID]=newFibObj 


                    fibers132[fiberID132].combine(fibObj321)
                    fibers132[fiberID132].setColor("combined")

                    fiberStruct_combined[fiberID132]=fibers132[fiberID132]
                    
                    if fiberID321 in fiberStruct_compactified.keys():
                        fiberStruct_compactified.pop(fiberID321)

                    #reassign voxels
                    V_fiberMapCompactified[V_fiberMapCompactified==fiberID321]=fiberID132

        for fiberID321,dictF in maxAll321.items():
            fiberID132=dictF["fiberID"]

            counts=dictF["counts"]

            if counts>minCountCombination:
                fibObj321=fibers321[fiberID321]

                oriVec321=fibObj321.orientationVec/np.linalg.norm(fibObj321.orientationVec)

                oriVec132=fibers132[fiberID132].orientationVec/np.linalg.norm(fibers132[fiberID132].orientationVec)

                angle=np.degrees(np.arccos(np.dot(oriVec132,oriVec321)))

                if angle>90:
                    angle=180-angle

                if angle<angleCombineDEG:

                    if fiberID132 in fiberStruct_combined.keys() and \
                        fiberID321 not in fiberStruct_combined[fiberID132].combinedWith:

                        fiberStruct_combined[fiberID132].combine(fibObj321)
                        fiberStruct_combined[fiberID132].setColor("combined")

                        # in the event of a second combination to the same fiberObj in 132, no need to delete again
                        if fiberID321 in fiberStruct_compactified.keys():
                            fiberStruct_compactified.pop(fiberID321)

                        #reassign voxels
                        V_fiberMapCompactified[V_fiberMapCompactified==fiberID321]=fiberID132

                    elif fiberID132 not in fiberStruct_combined.keys():
                        # edge case where the fiber132 had a more numerous match, which failed the combination conditions,
                        # but another fiber321 still passes combination requirements with it

                        # keep uncombined Obj for plotting purposes
                        newFibObj=fibers132[fiberID132].copy()
                        nextID= len(fibers132)+offset_nextAvailableID
                        offset_nextAvailableID+=1
                        fibers132[nextID]=newFibObj 


                        fibers132[fiberID132].combine(fibObj321)
                        fibers132[fiberID132].setColor("combined")

                        fiberStruct_combined[fiberID132]=fibers132[fiberID132]
                        
                        if fiberID321 in fiberStruct_compactified.keys():
                            fiberStruct_compactified.pop(fiberID321)

                        #reassign voxels
                        V_fiberMapCompactified[V_fiberMapCompactified==fiberID321]=fiberID132


    ##################################################################################

    #  Smart fiberStitching

    ##################################################################################

    fiberObj.classAttributes["collisionDistance"]   =params["collisionDistance"]
    fiberObj.classAttributes["maxTrimPoints"]       =params["maxTrimPoints"]

    #reinitialise for second postProcessing step
    fiberObj.classAttributes["interpolatedCenters"]={}

    if doSecondPass:

        print("\tSmart fiber stitching started")

        fiberStructExtended,times_stitching=fiberObj.smartStitching(
            fiberStruct_compactified,
            smartStitchingMinFibLength,
            smartStitchingMaxDistance,
            smartStitchingMaxLateralDist,
            smartStitchingAlignAngle,
            smartStitchingBackTrackingLimit,
            processingMinFiberLength,
            tagAngleTooSteep,
            maxSteepnessAngle,
            verboseHandle=verboseHandle,
            checkIfInSegt=False,
            createNewPoints=False,
            stitchingType="smart_transposed",
            preventSelfStitch=preventSelfStitch
            )

        if doLastPass:
            fiberStruct_lastPass={fiberID:fib for \
                fiberID,fib in fiberStruct_compactified.items() 
                if not fib.addedTo and fib.suffix!=0.123}

            fiberStructExtended2,times_stitching=fiberObj.smartStitching(
                fiberStruct_lastPass,
                smartStitchingMinFibLength,
                smartStitchingMaxDistance,
                smartStitchingMaxLateralDist,
                smartStitchingAlignAngle,
                smartStitchingBackTrackingLimit,
                processingMinFiberLength,
                tagAngleTooSteep,
                maxSteepnessAngle,
                verboseHandle=verboseHandle,
                checkIfInSegt=False,
                createNewPoints=False,
                stitchingType="smart_lastPass",
                preventSelfStitch=preventSelfStitch # Only fibers from different permutations are allowed to be stitched at this stage 
                )
                
            for fib in fiberStructExtended2.values():
                if not fib.addedTo:
                    fiberID_toKeep=fib.fiberID
                    marker_toKeep   =int(fiberID_toKeep)
                    for fiberID_toReplace in fib.extendedBy:

                        marker_toReplace=int(fiberID_toReplace)
                        #no need to do this one in parallel, as typically very few fibers are concerned
                        V_fiberMapCompactified[V_fiberMapCompactified==marker_toReplace]=marker_toKeep


        # add initial segments as new fiberObj, so that they can be plotted seperately (different color)
        for fibID,fib in fiberStructExtended.items():
            fiberStruct_compactified[fibID]=fib

        # Reassign markers in V_fiberMapCombined with the new stitched fibers

        print("\tReassign markers in V_fiberMapCombined with the new stitched fibers")

        currentMarkers=np.unique(V_fiberMapCompactified)
        #by default, each marker points to itself
        compactifyIDs_LUT={m:m for m in currentMarkers}

        for fib in fiberStructExtended.values():
            if not fib.addedTo:
                fiberID_toKeep=fib.fiberID
                marker_toKeep   =int(fiberID_toKeep)
                for fiberID_toReplace in fib.extendedBy:

                    marker_toReplace=int(fiberID_toReplace)
                    compactifyIDs_LUT[marker_toReplace]=marker_toKeep

                    # V_fiberMapCompactified[V_fiberMapCompactified==marker_toReplace]=marker_toKeep

        if parallelHandle:
            num_cores=min(int(multiprocessing.cpu_count())-2,48)
        else:
            num_cores=1

        results = Parallel(n_jobs=num_cores)\
            (delayed(compactifySlice)\
                (
                    V_fiberMapCompactified[iSlice],
                    compactifyIDs_LUT
                )for iSlice in range(V_fiberMapCompactified.shape[0]) )

        for iSlice,resTuple in enumerate(results):
            V_fiberMapCompactified[iSlice]=resTuple

        print("\tReassign markers finished in V_fiberMapCombined")


    ########################################################################
    ###
    ### PostProcessing: gap filling due to stitching
    ###
    ########################################################################

    postProcessQueue=[]

    for fiberID,interpolationChains in fiberObj.classAttributes["interpolatedCenters"].items():

        #fiberObj that were added to another at smartStitching wont be processed (only starting fiberObj will)
        if not fiberStruct_compactified[int(fiberID)].addedTo:

            oriVec=fiberStruct_compactified[int(fiberID)].orientationVec
            oriVec/=np.linalg.norm(oriVec)
            #if there is more than one interpolation chain, keep the longest to create Structuring Element
            if len(interpolationChains)>1:
                # listLengths=[len(chain) for chain in interpolationChains]
                # pos=listLengths.index(max(listLengths))
                pos=interpolationChains.index(max(interpolationChains))
            else:
                pos=0
            
            angle=np.degrees(np.arccos(np.dot(oriVec,[0.,0.,1.])))
            
            # costly to evaluate, used in debugging
            # numPixels=np.count_nonzero(V_fiberMapCompactified==int(fiberID))
            # print('Post-Processqueue, adding: fiberID: {} ,angle: {: >8.4f}, numPixels: {: >8.0f}, length: {: >8.4f}'.format(int(fiberID),angle,numPixels,interpolationChains[pos]))
            print('Post-Processqueue, adding: fiberID: {} ,angle: {: >8.4f}, length: {: >8.4f}'.format(int(fiberID),angle,interpolationChains[pos]))

            postProcessQueue.append(
                (
                    fiberID,
                    (interpolationChains[pos], oriVec, angle)
                ) 
            )


    oriVecAll={}
    for fiberID,fib in fiberStruct_compactified.items():
        if "oriVec_normalized" in fib.__dir__():
            oriVecAll[int(fiberID)]=fib.oriVec_normalized
        else:
            fib.oriVec_normalized=fib.orientationVec/np.linalg.norm(fib.orientationVec)
            oriVecAll[int(fiberID)]=fib.oriVec_normalized

    collisionsDict,V_fiberMap=collisionDetectionWrapper(
        postProcessQueue,
        minCountCombination,
        angleCombineDEG,
        oriVecAll,
        fiberStruct_compactified,
        V_fiberMapCompactified,
        fiberStruct_combined,
        V_hist,
        makePlotsIndividual=False,#only implemented for parallelHandle==False, will be overriden internally otherwise
        makePlotAll=False,
        parallelHandle=parallelHandle
        )

    print(f"Writing to disk: {commonPath}V_fiberMapCombined_postProcessed.tiff")

    imwrite(
        os.path.join(commonPath,'V_fiberMapCombined_postProcessed.tiff'),
        V_fiberMapCompactified,
        resolution=(xRes,xRes,unitTiff),
        description=descriptionStr,
        compress=True
        )


    fiberStructPickle={ #saved to binary
        "fiberStruct"           :fiberStruct_compactified,
        "fiberObj_classAttributes":fiberObj.classAttributes, #otherwise class attributes are not pickled,
        "exclusiveZone"         :exclusiveZone
    }

    with open(os.path.join(commonPath,"fiberStruct_final.pickle"),"wb") as f:
        pickle.dump(fiberStructPickle,f,protocol=pickle.HIGHEST_PROTOCOL)

    if makePlot:

        import cameraConfig
        from mayavi import mlab
        from visualisationTool      import makeLegend


        rangeOutline=[0.,V_hist.shape[1],0.,V_hist.shape[2],0.,V_hist.shape[0] ]

        cameraConfigKey="dynamic"
        cameraConfig.createCamViewFromOutline(rangeOutline,"132",cameraConfigKey)


        engine=mlab.get_engine()

        mlab.figure(figure="Combined Permutations",size=(1200,1050),bgcolor=(0.9,0.9,0.9))

        srcHist=mlab.pipeline.scalar_field(np.transpose(V_hist,(1,2,0)))

        ipw=mlab.pipeline.image_plane_widget(srcHist,
                                    plane_orientation='z_axes',
                                    slice_index=0,
                                )

        ipw.module_manager.scalar_lut_manager.lut_mode = "black-white"

        mlab.outline()

        axes = mlab.axes(color=(0., 0., 0.), nb_labels=10)
        axes.title_text_property.color = (0., 0., 0.)
        axes.title_text_property.font_family = 'times'

        axes.label_text_property.color = (0., 0., 0.)
        axes.label_text_property.font_family = 'times'

        axes.axes.font_factor=0.65

        color132            =(0.7,0.4,0.1)
        color321            =(0.2,0.7,0.4)
        colorCollisions     =(1.,0.1,0.1)
        colorCombination    =(0.1,0.,0.95)

        V_fiberMap132_mask=np.zeros(V_fiberMap132.shape,np.uint8)
        V_fiberMap132_mask[V_fiberMap132>=0]=255

        V_fiberMap321_mask=np.zeros(V_fiberMap321.shape,np.uint8)
        V_fiberMap321_mask[V_fiberMap321>=0]=255

        srcFiberMap132=mlab.pipeline.scalar_field(np.transpose(V_fiberMap132_mask,(1,2,0)))

        mlab.pipeline.iso_surface(srcFiberMap132,contours=[255], opacity=0.2, color=color132 )

        srcFiberMap321=mlab.pipeline.scalar_field(np.transpose(V_fiberMap321_mask,(1,2,0)))

        mlab.pipeline.iso_surface(srcFiberMap321,contours=[255], opacity=0.2, color=color321 )

        srcCollisions=mlab.pipeline.scalar_field(np.transpose(V_collisions,(1,2,0)))

        mlab.pipeline.iso_surface(srcCollisions,contours=[255], opacity=0.8, color=colorCollisions)

        if fiberStruct_compactified:
            fib0=list(fiberStruct_compactified.keys())[0]
            numFibersTracked=len(fiberStruct_compactified[fib0].classAttributes["listFiberIDs_tracked"])
        else:
            numFibersTracked=0

        params={
            "drawJaggedLines" :True,
            "drawCenterLines" :False,
            "drawEllipsoids"  :False,
            "addText"         :True,
            "fiberDiameter"   :10.
        }

        for fibID,fib in fiberStruct_compactified.items(): 

            if fib.colorLabel!="basic_123":

                fiberPloting(fib,fibID,len(fiberStruct_compactified),
                    numFibersTracked,
                    engine,
                    params,
                    scale=2.
                    ) 

        V_fibersShape=V_fiberMap132_mask.shape

        rangeOutline=[0.,V_fibersShape[0],0.,V_fibersShape[1],0.,V_fibersShape[2] ]

        makeLegend(rangeOutline)

        mlab.show(stop=False)

    print("\tDone")




