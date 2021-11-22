# by Facundo Sosa-Rey, 2021. MIT license

import os
import subprocess
import numpy as np
import time
import pickle

from joblib import Parallel, delayed  
import multiprocessing

import tifffile
from tifffile import TiffFile

from fibers import fiberObj
from postProcessing import randomizeVoxels,makePropertyMap
from extractCenterPoints import getTiffProperties

from vtk import vtkStructuredPoints,vtkStructuredPointsWriter,VTK_FLOAT
from vtk.util import numpy_support


def outputPropertyMap(commonPath,parallelHandle=True,randomizeFiberMap=False,croppedFiles=False,makeVTKfiles=True):

    print("\noutputPropertyMap() called on dataset:\n{}".format(commonPath))

    if croppedFiles:
        postProcessedFileName="V_fiberMapCombined_postProcessed_cropped.tiff"
        fiberStructPath=commonPath.split("CroppedResult")[0]
    else:
        postProcessedFileName="V_fiberMapCombined_postProcessed.tiff"
        fiberStructPath=commonPath


    with TiffFile(commonPath+postProcessedFileName) as tif:
        print("\tloading: \n"+commonPath+"V_fiberMapCombined_postProcessed.tiff")
        xRes,unitTiff,descriptionStr=getTiffProperties(tif,getDescription=True)

        if unitTiff=="INCH":
            pixelSize_micron=xRes[1]/xRes[0]*0.0254*1e6
        elif  unitTiff=="CENTIMETER":
            pixelSize_micron=xRes[1]/xRes[0]*0.01*1e6
        else:
            raise ValueError("other units values not implemented in getTiffProperties")

        V_fiberMap=tif.asarray()

    with open(fiberStructPath+"fiberStruct_final.pickle", "rb") as f:
        fiberStruct_all  = pickle.load(f)
        fiberStruct=fiberStruct_all["fiberStruct"]
        exclusiveZone=fiberStruct_all["exclusiveZone"]


    print("\t loading from disk complete")

    ####################################################################################################

    ### make vtk with multi-field data


    if "processedFibers"in fiberStruct_all.keys():
        print("\tload processed fibers from a previous run")

        processedFibers=fiberStruct_all["processedFibers"]
    else:
        print("\tfirst run, re-process all fibers, some of them need their orientationVec updated after combinations")

        processedFibers={}
        for fiberID,fib in fiberStruct.items():
            fib.processPointCloudToFiberObj(0.,False,None,sort=False,doTrimming=False)

            if "oriVec_normalized" not in fib.__dir__():
                oriVec=fib.orientationVec/np.linalg.norm(fib.orientationVec)
            else:
                oriVec=fib.oriVec_normalized

            angle=np.degrees(np.arccos(np.dot(oriVec,[0,0,1])))

            processedFibers[fiberID]={
                "length_inMicrons":fib.totalLength*pixelSize_micron,
                "angle" :angle
            }

    fiberStructPickle={ #saved to binary
        "fiberStruct"               :fiberStruct,
        "fiberObj_classAttributes"  :fiberStruct_all["fiberObj_classAttributes"], #otherwise class attributes are not pickled
        "processedFibers"           :processedFibers,
        "exclusiveZone"             :exclusiveZone
    }

    with open(commonPath+"fiberStruct_final.pickle","wb") as f:
        pickle.dump(fiberStructPickle,f,protocol=pickle.HIGHEST_PROTOCOL)

    if makeVTKfiles:
        print("\tcreating property map for lengths")

        marker_to_lengthLUT={-1:np.nan}
        marker_to_angleLUT ={-1:np.nan}

        for fiberID,fibData in processedFibers.items():

            marker_to_lengthLUT[fiberID]=fibData["length_inMicrons"]
            marker_to_angleLUT [fiberID]=fibData["angle"]

        V_length    =makePropertyMap(V_fiberMap,marker_to_lengthLUT,parallelHandle)

        for proc in multiprocessing.active_children():
            # Manual termination of processes to avoid strange infinite hanging at 0% CPU for the next step, for large datasets
            # print(f"\tforced termination of process {proc.name}")
            proc.terminate()
            proc.join()

        print("\n\tcreating property map for angles")

        V_angleTheta=makePropertyMap(V_fiberMap,marker_to_angleLUT, parallelHandle)

        for proc in multiprocessing.active_children():
            # Manual termination of processes to avoid strange infinite hanging at 0% CPU for the next step, for large datasets
            # print(f"\tforced termination of process {proc.name}")
            proc.terminate()
            proc.join()

        print("\n\tsaving propertyMaps to disk")


        structPoints = vtkStructuredPoints()
        structPoints.SetDimensions(
            V_fiberMap.shape[2],
            V_fiberMap.shape[1],
            V_fiberMap.shape[0]
            )
        structPoints.SetOrigin(0, 0, 0)
        if unitTiff=="INCH":
            structPoints.SetSpacing(xRes[1]/xRes[0]*25.4, xRes[1]/xRes[0]*25.4, xRes[1]/xRes[0]*25.4)
        else:
            #TODO, untested for "CENTIMETER"
            structPoints.SetSpacing(xRes[1]/xRes[0]*10, xRes[1]/xRes[0]*10, xRes[1]/xRes[0]*10)


        fiberIDs_VTK = numpy_support.numpy_to_vtk(num_array=V_fiberMap.ravel(), array_type=VTK_FLOAT)
        fiberIDs_VTK.SetName('FiberID')

        lengthVTK = numpy_support.numpy_to_vtk(num_array=V_length.ravel(), array_type=VTK_FLOAT)
        lengthVTK.SetName('Length (microns)')

        deviationVTK = numpy_support.numpy_to_vtk(num_array=V_angleTheta.ravel(), array_type=VTK_FLOAT)
        deviationVTK.SetName('Deviation (degrees)')

        structPoints.GetPointData().SetScalars(lengthVTK)
        structPoints.GetPointData().AddArray(deviationVTK)
        structPoints.GetPointData().AddArray(fiberIDs_VTK)


        filename = commonPath+"PropertyMaps.vtk"
        writer = vtkStructuredPointsWriter()
        writer.SetFileName(filename)
        writer.SetInputData(structPoints)
        writer.SetFileTypeToBinary()
        writer.Write()



    if randomizeFiberMap:
        print("\n\trandom shuffling of markers for rendering purposes started")
        
        listMarkers=np.unique(V_fiberMap)
        listMarkers=[val for val in listMarkers if val>=0]# tracked fibers have markers starting at 0


        ticRandomize=time.perf_counter()

        V_fiberMap_randomized=randomizeVoxels(V_fiberMap,listMarkers,parallelHandle)

        tocRandomize=time.perf_counter()

        print("random shuffling in: {}".format(time.strftime("%Hh%Mm%Ss", time.gmtime(tocRandomize-ticRandomize))))

        print("saving to disk...")

        # Conversion to floats makes a different rendering in Paraview
        # visualization in Paraview is different if data type is 
        # float rather than int (handling of the np.nan is different)

        V_fiberMap_randomizedFloat=np.array(V_fiberMap_randomized,np.float32)

        V_fiberMap_randomizedFloat[V_fiberMap_randomized==-1]=np.nan

        tifffile.imwrite(
            commonPath+'V_fiberMapCombined_randomized.tiff',
            V_fiberMap_randomized,
            resolution=(xRes,xRes,unitTiff),
            description=descriptionStr,
            compress=True
        )

        tifffile.imwrite(
            commonPath+'V_fiberMapCombined_randomizedFloat.tiff',
            V_fiberMap_randomizedFloat,
            resolution=(xRes,xRes,unitTiff),
            description=descriptionStr,
            compress=True
        )

    print("\toutputPropertyMap done\n\n")

