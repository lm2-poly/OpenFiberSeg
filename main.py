# by Facundo Sosa-Rey, 2021. MIT license


from trackingParameters import getTrackingParams
from extractCenterPoints import extractCenterPoints, checkIfFilesPresent
from trackFibers import tracking, saveFiberStruct
from assignVoxelsToFibers_refactored import assignVoxelsToFibers_Main
from postProcessing import postProcessingOfFibers
from combinePermutationsRefactored import combinePermutations
from outputPropertyMapsRefactored import outputPropertyMap

from tifffile import TiffFile
import tifffile
import json
import pickle
import subprocess
import os
import numpy as np

from fibers import fiberObj
import time


def getCommonPaths(rootPath):
    # finds all files in rootPath that have been preprocessed and pre-segmented with INSEGT, but not tracked, postprocessed and and combined
    cmd = ["find", rootPath, "-name", "SegtParams.json", "-type", "f"]
    systemCall = subprocess.run(cmd, stdout=subprocess.PIPE)

    directories = systemCall.stdout.decode("utf-8").split("SegtParams.json\n")

    unProcessedDirectories = []

    for dir in directories:
        if dir:
            cmd = ["find", dir, "-name", "PropertyMaps.vtk", "-type", "f"]
            systemCall = subprocess.run(cmd, stdout=subprocess.PIPE)
            if len(systemCall.stdout) == 0:
                unProcessedDirectories.append(dir)

    unProcessedDirectories.sort()

    return unProcessedDirectories


cmd = "hostname"

# returns output as byte string
hostnameStr = subprocess.check_output(cmd).decode("utf-8").split("\n")[0]

# using decode() function to convert byte string to string
print('Current hostname is :', hostnameStr)

##############################################################################

# Directories for pre segmented data

####################################################################makePlots##

dataPath="./TomographicData/"


# find directories with dataset processed with Insegt, but not tracked 
# (will have SegtParams.json file, but not PropertyMaps.vtk, final output)
directories = getCommonPaths(dataPath)

if directories:
    print("\n\n\tFound unprocessed datasets at these locations:\n")
    for dir in directories:
        print(dir)
else:
    print("\n\n\tNo unprocessed datasets at specified location\n")

exclusiveZone=None # will process entire dataset

# example of exclusive zone: processing will only occur in volume defined by these coordinates.
# produces truncated volumes as output tiff files.
# exclusiveZone={
#   "xMin":300,
#   "xMax":601,
#   "yMin":300,
#   "yMax":641,
#   "zMin":0,
#   "zMax":600
# }



##############################################################################
permutationPaths = ["Permutation123/", "Permutation132/", "Permutation321/"]
permutationVecAll = ["123", "132", "321"]

permutationIndices = [0, 1, 2]

for commonPath in directories:
    print("\nBatch mode, entering:\n ", commonPath)
    for permutationIndex in permutationIndices:

        permutationVec = permutationVecAll[permutationIndex]

        print("\n initiating Processing on permutation:{}".format(permutationVec))

        ##############################################################################

        # extraction of centerpoints with the watershed transform

        # for every step, intermediate results are writen to disk. wont be redone if files found

        ##############################################################################

        doExtraction = checkIfFilesPresent(
            commonPath+permutationPaths[permutationIndex],
            'V_voxelMap.tiff',
            "watershedExtractionStats.json",
            "watershedCenterPoints.pickle",
        )

        if doExtraction:

            ticTotal_extraction = time.perf_counter()

            V_voxels, watershedData, xRes, unitTiff, descriptionDict, times_centroids =\
                extractCenterPoints(
                    commonPath,
                    permutationVec,
                    plotCentroids               =False,
                    plotInitialAndResults       =False,
                    plotEveryIteration          =False,
                    plotWatershedStepsGlobal    =False,
                    plotWatershedStepsMarkersOnly=False,
                    plotConvexityDefects        =False,
                    plotExpansion               =False, # won't plot if useProbabilityMap==False in trackingParams.json
                    plotOverlayFrom123          =False,
                    # manualRange                     =range(10),
                    exclusiveZone               =exclusiveZone,
                    parallelHandle              =True
                )

            # write results to disk
            saveExtractionData = True
            if saveExtractionData:

                times_centroids["Entire extractCenterPoints procedure:"] = time.strftime(
                    "%Hh%Mm%Ss", time.gmtime(time.perf_counter()-ticTotal_extraction))

                print("\n\textractCenterPoints():\n\tWriting output to : \n " +
                    commonPath+permutationPaths[permutationIndex]+'V_voxelMap.tiff')

                tifffile.imwrite(
                    commonPath +
                    permutationPaths[permutationIndex]+'V_voxelMap.tiff',
                    V_voxels,
                    resolution=(xRes, xRes, unitTiff),
                    description=descriptionDict["descriptionStr"],
                    compress=True
                )

                watershedDict = {
                    "times_centroids": times_centroids,
                    "volumeDescription": descriptionDict,
                }

                pathCenterPointStats = commonPath + \
                    permutationPaths[permutationIndex] + \
                    "watershedExtractionStats.json"
                with open(pathCenterPointStats, "w") as f:
                    json.dump(watershedDict, f, sort_keys=False, indent=4)

                pathCenterPointsData = commonPath + \
                    permutationPaths[permutationIndex] + \
                    "watershedCenterPoints.pickle"
                with open(pathCenterPointsData, "wb") as f:
                    pickle.dump(watershedData, f,
                                protocol=pickle.HIGHEST_PROTOCOL)

        ##############################################################################

        # tracking of fibers from centerPoints

        ##############################################################################

        doTracking = checkIfFilesPresent(
            commonPath+permutationPaths[permutationIndex],
            "fiberStats.json",
            "fiberStruct.pickle"
        )

        if doTracking:

            ticTracking = time.perf_counter()

            fiberStruct, V_fibersShape, times_tracking, V_fibers, xRes, unitTiff = tracking(
                commonPath,
                permutationPaths[permutationIndex],
                permutationVec,
                exclusiveZone=exclusiveZone,
                parallelHandle=True,
                verboseHandle=False
            )

            tocTracking = time.perf_counter()

            times_tracking["Entire tracking procedure:"] = time.strftime(
                "%Hh%Mm%Ss", time.gmtime(tocTracking-ticTracking))
            print("\n\n\ttime for entire tracking procedure: {: >.2f}s \n".format(
                tocTracking-ticTracking))

            saveTrackingData = True
            if saveTrackingData:

                fiberStructPickle = {  # saved to binary
                    "fiberStruct": fiberStruct,
                    "fiberObj_classAttributes": fiberObj.classAttributes,# otherwise class attributes are not pickled
                    "trackingTimes": times_tracking,
                    "exclusiveZone": fiberObj.getExclusiveZone(),
                    "trackedCenterPoints": fiberObj.getTrackedCenterPoints(),
                    "rejectedCenterPoints": fiberObj.getRejectedCenterPoints()
                }

                fiberStats = {  # saved to human readable .json
                    "trackingTimes": times_tracking,
                    "numberOfFiberObj (total)": len(fiberStruct),
                    "numberOfFiberObj (tracked)": len(fiberObj.classAttributes["listFiberIDs_tracked"]),
                    "exclusiveZone": fiberObj.getExclusiveZone()
                }

                saveFiberStruct(
                    commonPath,
                    permutationPaths[permutationIndex],
                    fiberStructPickle,
                    fiberStats
                )

            plotTracking = False
            if plotTracking:
                import cameraConfig

                if exclusiveZone is None:
                    rangeOutline = [0., V_fibersShape[1], 0.,
                                    V_fibersShape[2], 0., V_fibersShape[0]]
                else:
                    rangeOutline = [
                        exclusiveZone["xMin"],
                        exclusiveZone["xMax"],
                        exclusiveZone["yMin"],
                        exclusiveZone["yMax"],
                        exclusiveZone["zMin"],
                        exclusiveZone["zMax"]
                    ]

                    xMin = exclusiveZone["xMin"]
                    xMax = exclusiveZone["xMax"]
                    yMin = exclusiveZone["yMin"]
                    yMax = exclusiveZone["yMax"]

                plottingParams = getTrackingParams(
                    commonPath, "plottingParams", xRes, unitTiff)

                plottingParams["panningPlane"] = True
                plottingParams["planeWidgets"] = True
                plottingParams["staticCam"] = False

                # plottingParams["cameraConfigKey"]="manual_0"

                if plottingParams["cameraConfigKey"] == "dynamic":
                    cameraConfig.createCamViewFromOutline(
                        rangeOutline, permutationVec, "dynamic")

                from visualisationTool import makeVisualisation

                filesInDir = [f.path for f in os.scandir(
                    commonPath+permutationPaths[permutationIndex]) if f.is_file()]
                for i, iPath in enumerate(filesInDir):
                    if ".tiff" in iPath:
                        if "V_hist.tiff" in iPath:
                            indexHistTiff = i
                        if "V_pores.tiff" in iPath:
                            indexPoresTiff = i

                if plottingParams["planeWidgets"]:
                    with TiffFile(filesInDir[indexHistTiff]) as tif:
                        if exclusiveZone is None:
                            V_hist = np.transpose(tif.asarray(), (1, 2, 0))/255
                        else:
                            rangeSlice = range(
                                exclusiveZone["zMin"], exclusiveZone["zMax"])
                            offset = exclusiveZone["zMin"]
                            nSlices = len(rangeSlice)

                            temp = tif.pages[0].asarray()[xMin:xMax, yMin:yMax]

                            V_hist = np.empty((nSlices, *temp.shape), np.uint8)

                            for imSlice in rangeSlice:
                                V_hist[imSlice-offset] = tif.pages[imSlice].asarray()[xMin:xMax,
                                                                                    yMin:yMax]

                            V_hist = np.transpose(V_hist, (1, 2, 0))
                else:
                    V_hist = None

                if plottingParams["plotPorosityMask"]:
                    with TiffFile(filesInDir[indexPoresTiff]) as tif:
                        if exclusiveZone is None:
                            # load entire volume
                            V_porosity = np.transpose(
                                tif.asarray(), (1, 2, 0))/255
                        else:
                            # load partial volume
                            temp = tif.pages[0].asarray()[xMin:xMax, yMin:yMax]
                            V_porosity = np.empty(
                                (nSlices, *temp.shape), np.uint8)

                            for imSlice in rangeSlice:
                                V_porosity[imSlice-offset] = tif.pages[imSlice].asarray()[
                                    xMin:xMax, yMin:yMax]

                        V_porosity = np.transpose(V_porosity, (1, 2, 0))
                else:
                    V_porosity = None

                makeVisualisation(fiberStruct, V_porosity, V_hist,
                                rangeOutline, plottingParams, widgetLUT="black-white")

        ##############################################################################

        # assign voxels to tracked fibers V_voxelMap

        ##############################################################################

        doAssignment = checkIfFilesPresent(
            commonPath+permutationPaths[permutationIndex],
            "V_fiberMap.tiff"
        )

        if doAssignment:

            ticAssign = time.perf_counter()

            V_fiberMap,\
                fiberStruct,\
                xRes, unitTiff,\
                descriptionStr,\
                times_assign = assignVoxelsToFibers_Main(
                    commonPath,
                    permutationPaths[permutationIndex],
                    # manualRange=range(208+300,510),
                    makePlots=False,
                    parallelHandle=True,
                    verbose=False,
                    addDisksAroundCenterPnts=True
                )

            saveAssignmentData = True

            if saveAssignmentData:

                tocAssign = time.perf_counter()

                times_assign["Total assignVoxelsToFibers procedure"] = time.strftime(
                    "%Hh%Mm%Ss", time.gmtime(tocAssign-ticAssign))

                fiberStructPickle = {  # saved to binary
                    "fiberStruct": fiberStruct["fiberStruct"],
                    "fiberObj_classAttributes": fiberStruct["fiberObj_classAttributes"], # otherwise class attributes are not pickled
                    "trackingTimes": fiberStruct["trackingTimes"],
                    "assignmentTimes": times_assign,
                    "exclusiveZone": fiberStruct["exclusiveZone"],
                    "trackedCenterPoints": fiberStruct["trackedCenterPoints"],
                    "rejectedCenterPoints": fiberStruct["rejectedCenterPoints"]
                }

                fiberStats = {  # saved to human-readable JSON
                    "trackingTimes": fiberStruct["trackingTimes"],
                    "assignmentTimes": times_assign,
                    "numberOfFiberObj (total)": len(fiberStruct["fiberStruct"]),
                    "numberOfFiberObj (tracked)": len(fiberStruct["fiberObj_classAttributes"]["listFiberIDs_tracked"]),
                    "exclusiveZone": fiberStruct["exclusiveZone"],
                }

                saveFiberStruct(
                    commonPath,
                    permutationPaths[permutationIndex],
                    fiberStructPickle,
                    fiberStats
                )

                print("\n\tassignVoxelsToFibers():\n\tWriting tiff file to : \n " +
                    commonPath+permutationPaths[permutationIndex]+'V_fiberMap.tiff')

                tifffile.imwrite(commonPath+permutationPaths[permutationIndex]+'V_fiberMap.tiff',
                                V_fiberMap,
                                resolution=(xRes, xRes, unitTiff),
                                description=descriptionStr,
                                compress=True
                                )

        ##############################################################################

        # Volumetric post-processing of fibers

        ##############################################################################

        makePlotAll = False

        doPostProcessing = checkIfFilesPresent(
            commonPath+permutationPaths[permutationIndex],
            "V_fiberMap_postProcessed.tiff"
        )

        if doPostProcessing:
            V_fiberMap,\
                V_fiberMap_randomized,\
                V_fibers_masked,\
                xRes, unitTiff,\
                descriptionStr,\
                times_postProc = postProcessingOfFibers(
                    commonPath,
                    permutationPaths[permutationIndex],
                    SE_radius=4,
                    useInclinedCylinderSE=True,# cant be done in parallel, will be over-riden inside function
                    makePlotsIndividual=False,# cant be done in parallel, will be over-riden inside function
                    makePlotAll=makePlotAll,
                    parallelHandle=True,  # requires large amounts of RAM
                    postProcessAllFibers=True
                )

            savePostProcessingData = True
            if savePostProcessingData:

                print("\n\tpostProcessingOfFibers():\n\tWriting tiff file to : \n " +
                    commonPath+permutationPaths[permutationIndex]+'V_fiberMap_postProcessed.tiff')

                tifffile.imwrite(
                    commonPath +
                    permutationPaths[permutationIndex] +
                    'V_fiberMap_postProcessed.tiff',
                    V_fiberMap,
                    resolution=(xRes, xRes, unitTiff),
                    description=descriptionStr,
                    compress=True
                )

                if V_fiberMap_randomized is not None:
                    print("\n\tpostProcessingOfFibers():\n\tWriting tiff file to : \n " +
                        commonPath+permutationPaths[permutationIndex]+'V_fiberMap_randomized.tiff')

                    tifffile.imwrite(commonPath+permutationPaths[permutationIndex]+'V_fiberMap_randomized.tiff',
                                    V_fiberMap_randomized,
                                    resolution=(xRes, xRes, unitTiff),
                                    description=descriptionStr,
                                    compress=True
                                    )

                if permutationVec == "123":
                    print("\n\tpostProcessingOfFibers():\n\tWriting tiff file to : \n " +
                        commonPath+permutationPaths[permutationIndex]+'V_fibers_masked.tiff')

                    tifffile.imwrite(commonPath+permutationPaths[permutationIndex]+'V_fibers_masked.tiff',
                                    V_fibers_masked,
                                    resolution=(xRes, xRes, unitTiff),
                                    description=descriptionStr,
                                    compress=True
                                    )

                with open(commonPath+permutationPaths[permutationIndex]+'postProcStats.json', "w") as f:
                    json.dump(times_postProc, f, sort_keys=False, indent=4)

            if makePlotAll:
                from mayavi import mlab
                mlab.show()

    #########################################################################

    # Combination of processed data from all permuted referentials

    #########################################################################

    doCombination = checkIfFilesPresent(
        commonPath,
        "fiberStruct_final.pickle",
        "V_fiberMapCombined_postProcessed.tiff",
    )
    if doCombination:
        combinePermutations(commonPath, permutationPaths, parallelHandle=True)

    doOutputVTK = checkIfFilesPresent(
        commonPath,
        "PropertyMaps.vtk",
        "V_fiberMapCombined_randomized.tiff",
        "V_fiberMapCombined_randomizedFloat.tiff"
    )

    if doOutputVTK:
        outputPropertyMap(
            commonPath,
            parallelHandle=False,
            randomizeFiberMap=True)



print("\n\tDone ")
