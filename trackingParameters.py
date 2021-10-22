# by Facundo Sosa-Rey, 2021. MIT license

import numpy as np
import os
import json

##########################################################

# Rendering and tracking parameters

# many dimensions are in MICRONS, are converted to pixel counts in call to convertMicrons_to_Pixels()


trackingAndPlottingParams={"123":{},"132 and 321":{},"plottingParams":{}, "extractionParams":{}}

trackingAndPlottingParams["extractionParams"]={
    "dilatePores"           :True,
    "dilationRadius_pores"  :5,     # pixels
    "dilationRadius_perim"  :8,     # pixels
    "initialWaterLevel"     :0.8,   # pixels: if <1., will preserve small regions, which are sometimes erroneous
    "waterLevelIncrements"  :0.1,   # pixels
    "convexityDefectDist"   :1.2,
    "checkConvexityAndSplit":True,
    "useProbabilityMap"     :True
}

trackingAndPlottingParams["plottingParams"]={
    "fiberDiameter"             :7.5,               #microns     
    "plotPorosityMask"          :True,
    "addText"                   :True,
    "drawCenterLines"           :False,
    "drawJaggedLines"           :True,
    "markMissedCtrPnts"         :False,#computationally expensive
    "drawEllipsoids"            :False,#computationally expensive,
    "planeWidgets"              :True,
    "panningPlane"              :False,
    "plotRejectedFibers"        :False,
    "plotOnlyStitchedFibers"    :False,
    "plotOnlyTrackedFibers"     :False,
    "cameraConfigKey"           :"dynamic",
    "staticCam"                 :True
}

if not  trackingAndPlottingParams["plottingParams"]["plotPorosityMask"] and\
    not trackingAndPlottingParams["plottingParams"]["addText"]          and\
    not trackingAndPlottingParams["plottingParams"]["drawCenterLines"]  and\
    not trackingAndPlottingParams["plottingParams"]["drawJaggedLines"]  and\
    not trackingAndPlottingParams["plottingParams"]["markMissedCtrPnts"]and\
    not trackingAndPlottingParams["plottingParams"]["drawEllipsoids"]   and\
    not trackingAndPlottingParams["plottingParams"]["planeWidgets"]     and\
    not trackingAndPlottingParams["plottingParams"]["panningPlane"]:
    raise ValueError("Bad visualization config, rendering will crash")


trackingAndPlottingParams["123"]={
    "tagAngleTooSteep"              :True,
    "maxSteepnessAngleDEG"          :50.,
    "distLateral_knnFirstPass"      :3,             # microns
    "processingMinFiberLength"      :10,            # pixels, below this length, fiber obj marked as rejected in processPointCloudToFiberObj()
    "blindStitching"                :True,
    "blindStitchingMaxDistance"     :20.,           # microns
    "blindStitchingMaxLateralDist"  :8.,            # microns
    "smartStitching"                :True,
    "smartStitchingMaxDistance"     :50.,           # microns
    "smartStitchingAlignAngleDEG"   :15.,
    "smartStitchingMaxLateralDist"  :5.,            # microns
    "smartStitchingMinFibLength"    :5.,            # pixels (has more to do with the number of centroids than physical length. too short and the orientationVec is meaningless)
    "smartStitchingBackTrackingLimit":4.,           # pixels
    "collisionDistance"             :10.,           # microns
    "fillingFraction"               :0.5,           # fraction of centroids to be added in stitching gap at smartStitching by 
                                                    # fiberObj.filling() that are allowed to be ouside of V_fibers mask
    "fillingNumberAlwaysAllowed"    :35,            # number of centroids to be allowed to be added in stitching gap at smartStitching by 
                                                    # fiberObj.filling() without checking if ouside of V_fibers mask
    "maxTrimPoints"                 :25,            # maximum number of points (in pixels) that are allowed to be trimmed 
                                                    # (SVD processing in processPointCloudToFiberObj leaves some endpoints 
                                                    # after tip of main vector, can cause problems at smartStitching).
}

if trackingAndPlottingParams["123"]["smartStitchingBackTrackingLimit"]>=trackingAndPlottingParams["123"]["maxTrimPoints"]:
    raise ValueError(
        "backtrackLimit={} must be strictly inferior to maxTrimPoints={}, else maxTrimPoint can be reached in normal operation".\
        format(
            trackingAndPlottingParams["123"]["smartStitchingBackTrackingLimit"],
            trackingAndPlottingParams["123"]["maxTrimPoints"]
            )
        )


trackingAndPlottingParams["132 and 321"]={
    "tagAngleTooSteep"                  :True,
    "maxSteepnessAngleDEG"              :50.,           # degrees, will be converted to radians
    "distLateral_knnFirstPass"          :3. ,           # microns
    "processingMinFiberLength"          :20,            # pixels
    "blindStitching"                    :True,
    "blindStitchingMaxDistance"         :5.,            # microns
    "blindStitchingMaxLateralDist"      :3.,            # microns
    "smartStitching"                    :True,
    "smartStitchingMaxDistance"         :80,            # microns
    "smartStitchingAlignAngleDEG"       :20.,           # degrees, will be converted to radians
    "smartStitchingMaxLateralDist"      :8.,            # microns
    "smartStitchingMinFibLength"        :trackingAndPlottingParams["123"]["smartStitchingMinFibLength"], #pixels
    "smartStitchingBackTrackingLimit"   :10,    #trackingAndPlottingParams["123"]["smartStitchingBackTrackingLimit"], #pixels
    "collisionDistance"                 :4. ,                   # microns
    "fillingFraction"                   :trackingAndPlottingParams["123"]["fillingFraction"],
    "fillingNumberAlwaysAllowed"       :trackingAndPlottingParams["123"]["fillingNumberAlwaysAllowed"],
    "maxTrimPoints"                     :trackingAndPlottingParams["123"]["maxTrimPoints"],
}

trackingAndPlottingParams["secondPass"]={
    "smartStitchingMinFibLength"        :0.,                # pixels        
    "smartStitchingMaxDistance"         :120.,              # microns
    "smartStitchingMaxLateralDist"      :10.,               # microns   
    "smartStitchingAlignAngleDEG"       :20.,               # degrees, will be converted to radians
    "smartStitchingBackTrackingLimit"   :40.,               # pixels
    "processingMinFiberLength"          :0.,                # pixels        
    "tagAngleTooSteep"                  :False,
    "maxSteepnessAngle"                 :None,
    "collisionDistance"                 :1.5,               # microns
    "maxTrimPoints"                     :30,                # pixels
    "include123"                        :True,
    "doSecondPass"                      :True,
    "doLastPass"                        :True,
    "verboseHandle"                     :True,
    "preventSelfStitch_123_123"         :True
}

if trackingAndPlottingParams["132 and 321"]["smartStitchingBackTrackingLimit"]>=trackingAndPlottingParams["132 and 321"]["maxTrimPoints"]:
    raise ValueError(
        "backtrackLimit={} must be strictly inferior to maxTrimPoints={}, else maxTrimPoint can be reached in normal operation".\
        format(
            trackingAndPlottingParams["132 and 321"]["smartStitchingBackTrackingLimit"],
            trackingAndPlottingParams["132 and 321"]["maxTrimPoints"]
            )
        )

def convertMicrons_to_Pixels(params,key,xRes=None,unitTiff=None):
    if unitTiff=="INCH":
        pixelSize_micron=xRes[1]/xRes[0]*0.0254*1e6
    elif  unitTiff=="CENTIMETER":
        pixelSize_micron=xRes[1]/xRes[0]*0.01*1e6
    elif unitTiff is not None:
        pixelSize_micron=None 
        raise ValueError("other units values not implemented in getTiffProperties")

    if xRes is not None:
        if pixelSize_micron<0.1:
            raise ValueError("trackingParameters will cause tracking to fail for such small pixel size. ")

        if pixelSize_micron>10:
            print("\n\trescaling to one micron/pixel, or tracking parameters won't be of any use at present scale of {: >6.3f} micron/pixel".format(pixelSize_micron))
            pixelSize_micron=1.


    if key=="secondPass":
        params["smartStitchingMaxDistance"]     /=pixelSize_micron                 
        params["smartStitchingMaxLateralDist"]  /=pixelSize_micron                 
        params["smartStitchingBackTrackingLimit"]/=pixelSize_micron                   
        params["collisionDistance"]             /=pixelSize_micron 

        # if a second file is loaded from disk, params is still present from previous run
        if "smartStitchingAlignAngle" not in params.keys():
            params["smartStitchingAlignAngle"]=np.radians(params["smartStitchingAlignAngleDEG"])
            del params["smartStitchingAlignAngleDEG"]  #makes it impossible to use wrong one

    elif key=="plottingParams":
        params["fiberDiameter"]                 /=pixelSize_micron

    elif key in ["123","132 and 321"]:
        params["distLateral_knnFirstPass"]      /=pixelSize_micron
        params["blindStitchingMaxDistance"]     /=pixelSize_micron
        params["blindStitchingMaxLateralDist"]  /=pixelSize_micron
        params["smartStitchingMaxDistance"]     /=pixelSize_micron
        params["smartStitchingMaxLateralDist"]  /=pixelSize_micron
        params["collisionDistance"]             /=pixelSize_micron

        # if a second file is loaded from disk, params is still present from previous run
        if "maxSteepnessAngle" not in params.keys():
            params["maxSteepnessAngle"]=np.radians(params["maxSteepnessAngleDEG"])
            del params["maxSteepnessAngleDEG"] #makes it impossible to use wrong one
        
        if "smartStitchingAlignAngle" not in params.keys():
            params["smartStitchingAlignAngle"]=np.radians(params["smartStitchingAlignAngleDEG"])
            del params["smartStitchingAlignAngleDEG"] #makes it impossible to use wrong one
    elif key=="extractionParams":
        pass #param in pixels not microns
    else:
        raise ValueError("not implemented for key={}".format(key))

    return params

def getTrackingParams(commonPath,key,xRes=None,unitTiff=None):

    if key in ["132","321"]:
        key="132 and 321"

    filesCommonPath = [f.path for f in os.scandir(commonPath) if f.is_file()]
    indexJson=None
    for i,iPath in enumerate(filesCommonPath):
        if "trackingParams.json" in iPath:
            indexJson=i

    if indexJson is None:
        #create default param file, write to commonPath
        
        with open(commonPath+"trackingParams.json", "w") as f:
            json.dump(trackingAndPlottingParams, f, sort_keys=True, indent=4)

        params=trackingAndPlottingParams

    else:
        #load customized parameters from file
        with open(filesCommonPath[indexJson], "r") as f:
            params=json.load(f)


    if key in ["123","132 and 321"] and params[key]["smartStitchingBackTrackingLimit"]>=trackingAndPlottingParams[key]["maxTrimPoints"]:
        raise ValueError(
            "backtrackLimit={} must be strictly inferior to maxTrimPoints={}, else maxTrimPoint can be reached in normal operation".\
            format(
                params[key]["smartStitchingBackTrackingLimit"],
                params[key]["maxTrimPoints"]
                )
            )

    return convertMicrons_to_Pixels(params[key],key,xRes,unitTiff)






