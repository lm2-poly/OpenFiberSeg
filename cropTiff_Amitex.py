# by Facundo Sosa-Rey, 2021. MIT license

from tifffile import TiffFile
import tifffile

import pickle
import numpy as np
import subprocess

from extractCenterPoints import getTiffProperties

import os
import matplotlib.pyplot as plt
import json

from weightConcToVolumeConc import calculateFractionsFromVolumes

cropV_hist=False
doDownSampling=True



##################################################################

# sampleB1: only microstructure

commonPath="/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Juliette/PublicPaper3/SampleB1/processed_x1-2009_y1-1722_z1-971/2022-06-15_09h31m26/"

dataInput =commonPath
dataOutput=commonPath+"AmitexFiles_microStructure"

from rectPatchDataBase_RVE_sampleB1_manual import rectPatchDataBase_gridRVE_sampleB1 as rectPatchDataBase

manualCropping={k:rectPatchDataBase[k] for k in rectPatchDataBase.keys()}

mesoStructure=False

downSamplingFactor=2

#####################################################################

# sampleF1: only microstructure

# commonPath="/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Juliette/PublicPaper3/SampleF1/processed_x56-1436_y80-1422_z1-970/2022-06-27_13h16m01"

# dataInput =commonPath
# dataOutput=os.path.join(commonPath,"AmitexFiles_microStructure")

# from rectPatchDataBase_RVE_sampleF1_manual import rectPatchDataBase_gridRVE_sampleF1 as rectPatchDataBase

# manualCropping={k:rectPatchDataBase[k] for k in rectPatchDataBase.keys()}

# mesoStructure=False

# downSamplingFactor=2

#####################################################################

# sampleF1 mesostructure_large. porosity from OpenFiberSeg 

# commonPath="/media/facu/Svalbard_WD6TB/TomographicDataSvalbard/Juliette/PublicPaper3/SampleF1/sampleF1_2022-06-22_185257_meso_large/"

# dataInput =commonPath
# dataOutput=os.path.join(commonPath,"AmitexFiles")

# keepOnlyList=[
#     "sampleF1_meso_high_porosity_RVE_008_000",
#     "sampleF1_meso_high_porosity_RVE_008_001",
#     "sampleF1_meso_high_porosity_RVE_008_002",
#     "sampleF1_meso_high_porosity_RVE_008_003",
#     "sampleF1_meso_high_porosity_RVE_008_004",
#     "sampleF1_meso_high_porosity_RVE_008_005",
#     "sampleF1_meso_high_porosity_RVE_008_006",
#     "sampleF1_meso_high_porosity_RVE_008_007",
# ]

# from rectPatchDataBase_RVE_sampleF1_meso import rectPatchDataBase_gridRVE_sampleF1_meso as rectPatchDataBase

# manualCropping={k:rectPatchDataBase[k] for k in keepOnlyList}

# mesoStructure=True

# downSamplingFactor=4

#####################################################################

def cropTiff_Amitex(
    manualCropping,
    dataInput,
    dataOutput,
    makePlots=False,
    doDownSampling=True,
    downSamplingFactor=2,
    mesoStructure=False,
    keepOnlyDownSampled=False,
    cropV_hist=False
    ):
    
    exists = os.path.exists(dataOutput)

    dataOutputCroppedParent=os.path.join(dataOutput,"cropped/")
    dataOutputDownSampledParent=os.path.join(dataOutput,"downSampled/")


    dataOutputCropped    ={
        volumeTag:os.path.join(
            dataOutputCroppedParent,
            volumeTag
            ) for volumeTag in manualCropping.keys()}
    
    dataOutputDownSampled={
        volumeTag:os.path.join(
            dataOutputDownSampledParent,
            volumeTag
            ) for volumeTag in manualCropping.keys()}

    if not exists:
        print("\tCreating output directories at:\n{}".format(dataOutput))
        cmd = ["mkdir", dataOutput]
        systemCall = subprocess.run(cmd, stdout=subprocess.PIPE)
        
        if not keepOnlyDownSampled:
            cmd = ["mkdir", dataOutputCroppedParent]
            systemCall = subprocess.run(cmd, stdout=subprocess.PIPE)
        
            for outpath in dataOutputCropped.values():
                cmd = ["mkdir", outpath]
                systemCall = subprocess.run(cmd, stdout=subprocess.PIPE)

       
        if doDownSampling:
            cmd = ["mkdir", dataOutputDownSampledParent]
            systemCall = subprocess.run(cmd, stdout=subprocess.PIPE)
        
            for outpath in dataOutputDownSampled.values():
                cmd = ["mkdir",outpath ]
                systemCall = subprocess.run(cmd, stdout=subprocess.PIPE) 
        

    if mesoStructure:
        # no fibers in mesostructure scans
        fiberStructOutput={"fiberStruct":{}}
        
        filenameList={
            "V_fibers":"V_fiberMapCombined_postProcessed",
            "V_pores":"V_pores",
            "V_perim":"V_perim",
        }

        descriptionDict={"mesoStructure":True}
        descriptionStrOutput=str(descriptionDict)

    else:

        filenameList={
            "V_fibers":"V_fiberMapCombined_postProcessed",
            "V_pores":"Permutation123/V_pores",
            "V_perim":"Permutation123/V_perim",
            }

        with open(os.path.join(dataInput,"fiberStruct_final.pickle"), "rb") as f:
            fiberStruct  = pickle.load(f)

    if cropV_hist:
        if mesoStructure:
            filenameList["V_hist"]="V_hist"
        else:
            filenameList["V_hist"]="Permutation123/V_hist"

    exclusizeZone_present=False

    ##########################################################à

    # load volumes

    V_all_input={}

    for V_type,filename in filenameList.items():

        if mesoStructure and V_type=="V_fibers":
            continue # there are no fibers for volumes describing microstructures

        if os.path.exists(os.path.join(dataInput,filename+".tiff")):

            print('\n\treading from disk: {}'.format(os.path.join(dataInput,filename+".tiff")))

            with TiffFile(os.path.join(dataInput,filename+".tiff")) as tif:
                xRes,unitTiff,descriptionStr,dim=getTiffProperties(tif,getDescription=True,getDimensions=True)

                V_all_input[V_type]=tif.asarray()
                
                if "exclusiveZone" in descriptionStr:
                    tempStr=descriptionStr.split("exclusiveZone\":")[1]
                    tempStr=tempStr.replace("\'","\"").replace("__postProcessed","").replace("None","[]")[:-1]
                    print("tempStr",tempStr)

                    try:
                        #this is required due to changes in the description str across versions. new files should have a correct descrition
                        exclusiveZone_fromFile=json.loads(tempStr) if tempStr !="None" else None
                    except:
                        try:
                            #screwup with old versions, missing }
                            exclusiveZone_fromFile=json.loads(tempStr+"}")
                        except:
                            exclusiveZone_fromFile=None

                    if exclusiveZone_fromFile:
                        exclusizeZone_present=True

                        zMin=exclusiveZone_fromFile["zMin"]
                        zMax=exclusiveZone_fromFile["zMax"]
                        xMin=exclusiveZone_fromFile["xMin"]
                        xMax=exclusiveZone_fromFile["xMax"]
                        yMin=exclusiveZone_fromFile["yMin"]
                        yMax=exclusiveZone_fromFile["yMax"]
                    else:
                        exclusizeZone_present=False

                    descriptionStr=descriptionStr.replace("range","").replace("\'","\"").replace("\n","").replace("(","[").replace(")","]").replace("__postProcessed","").replace("None","[]")[:-1]+"}"
                    
                    print("descriptionStr",descriptionStr)

                    try:
                        #this is required due to changes in the description str across versions. new files should have a correct descrition
                        descriptionDict=json.loads(descriptionStr)
                        descriptionDict["manualRange"]=tuple(descriptionDict["manualRange"])
                    except:
                        descriptionDict={}
                        descriptionDict["manualRange"]=None
            
        else:
            if V_type=="V_perim":
                print("\nFile not found:{}".format(os.path.join(dataInput,filename+".tiff")))

                print("Creating empty V_perim file\n")

                V_all_input["V_perim"]=np.zeros(V_all_input["V_pores"].shape,np.uint8)
            else:
                raise IOError("{} not found at {}".format(filename,dataInput))

        if V_type in ["V_pores","V_perim","V_hist"]:
            # cropping on V_pores,V_perim and V_hist due to exclusive zone (file on disk are of the total size)
            # V_fiberMap is already cropped to exclsive zone
            if exclusizeZone_present: 
                V_all_input[V_type]=V_all_input[V_type][zMin:zMax,xMin:xMax,yMin:yMax]

    if mesoStructure:
        #create a V_fibers array and set all pixels to matrix value (-1)
        V_all_input["V_fibers"]=np.ones(V_all_input["V_pores"].shape,np.int8)*-1 

    print('reading from disk complete')

    outputDict={}

    for volumeTag in dataOutputCropped:

        print("\n\tEntering volumeTag:\t{}".format(volumeTag))

        V_all={}
        V_downSampled_all={}

        descriptionDict["manualCropping"]=manualCropping[volumeTag]

        descriptionStrOutput=str(descriptionDict)

        for V_type,filename in filenameList.items():

            #  manual cropping inside exclusive zone
            if manualCropping[volumeTag] is not None:
                if manualCropping[volumeTag]["zMax"]=="all":
                    V_all[V_type]=V_all_input[V_type][:,
                        manualCropping[volumeTag]["xMin"]:manualCropping[volumeTag]["xMax"],
                        manualCropping[volumeTag]["yMin"]:manualCropping[volumeTag]["yMax"]
                        ]
                else:
                    V_all[V_type]=V_all_input[V_type][
                        manualCropping[volumeTag]["zMin"]:manualCropping[volumeTag]["zMax"],
                        manualCropping[volumeTag]["xMin"]:manualCropping[volumeTag]["xMax"],
                        manualCropping[volumeTag]["yMin"]:manualCropping[volumeTag]["yMax"]
                        ]
            else:
                V_all[V_type]=V_all_input[V_type] 


        ################################################################################

        ### checking consistency in segmentation


        test=np.logical_and(V_all["V_perim"]==255,V_all["V_fibers"]>-1)
        if np.any(test):
            V_all["V_perim"][test]=0
        test=np.logical_and(V_all["V_pores"]==255,V_all["V_fibers"]>-1)
        if np.any(test):
            V_all["V_pores"][test]=0
        test=np.logical_and(V_all["V_perim"]==255,V_all["V_pores"]>-1)
        if np.any(test):
            V_all["V_perim"][test]=0

        V_matrix=np.ones(V_all["V_fibers"].shape,np.int8)

        fibersVolumeFraction,\
            meanFiberFrac,\
            stdFiberFrac,\
            errorFiberFrac,\
            fibersMatrixVolumeFraction,\
            meanFiberMatrixFrac,\
            stdFiberMatrixFrac,\
            errorFiberMatrixFrac,\
            poresVolumeFraction,\
            meanPoresFrac,\
            stdPoresFrac,\
            errorPoresFrac=calculateFractionsFromVolumes(V_matrix, V_all["V_fibers"], V_all["V_perim"], V_all["V_pores"])

        fibersInCroppedVolume=np.unique(V_all["V_fibers"])

        if not mesoStructure:
            fiberStructOutput={
                "fiberStruct":{
                    fiberID:fiberObj for fiberID,fiberObj in fiberStruct["fiberStruct"].items() if fiberID in fibersInCroppedVolume
                    }
                }

        dataDict={
            "meanFiberFrac"         :meanFiberFrac,
            "stdFiberFrac"          :stdFiberFrac,
            "errorFiberFrac"        :errorFiberFrac,
            "meanFiberMatrixFrac"   :meanFiberMatrixFrac,
            "stdFiberMatrixFrac"    :stdFiberMatrixFrac,
            "errorFiberMatrixFrac"  :errorFiberMatrixFrac,
            "meanPoresFrac"         :meanPoresFrac,
            "stdPoresFrac"          :stdPoresFrac,
            "errorPoresFrac"        :errorPoresFrac,
            "fibersVolumeFraction"  :fibersVolumeFraction,
            "fibersMatrixVolumeFraction":fibersMatrixVolumeFraction,
            "poresVolumeFraction"   :poresVolumeFraction,
        }

        if not keepOnlyDownSampled:
            with open(os.path.join(dataOutputCropped[volumeTag],"fiberVolumeFractions.json"),"w") as f:
                json.dump(dataDict,f, sort_keys=False, indent=4)

            with open(os.path.join(dataOutputCropped[volumeTag],"fiberStruct_AMITEX.pickle"), "wb") as f:
                pickle.dump(fiberStructOutput,f,protocol=pickle.HIGHEST_PROTOCOL)

        outputDict[volumeTag]={
            "downSamplingFactor":downSamplingFactor,
            "doDownSampling":doDownSampling,
            "cropped":{
                "meanFiberFraction      ":"{: >8.4%}".format(meanFiberFrac),
                "meanFiberMatrixFraction":"{: >8.4%}".format(meanFiberMatrixFrac)
            }
        }

        print('Writing to disk started')

        for V_type,V in V_all.items():

            if makePlots:

                plt.figure(num="{}, first data slice".format(V_type))

                slice=V[0,:,:].copy()

                slice[slice>245]=245
                slice[slice==-1]=-10
                slice+=10

                plt.imshow(slice)

                plt.figure(num="{}, last data slice".format(V_type))

                slice=V[-1,:,:].copy()

                slice[slice>245]=245
                slice[slice==-1]=-10
                slice+=10

                plt.imshow(slice)

                plt.show()

            if doDownSampling:

                _z=np.array([val*downSamplingFactor for val in range(int(V.shape[0]/downSamplingFactor))])
                _x=np.array([val*downSamplingFactor for val in range(int(V.shape[1]/downSamplingFactor))])
                _y=np.array([val*downSamplingFactor for val in range(int(V.shape[2]/downSamplingFactor))])

                _zz,_xx,_yy=np.meshgrid(_z,_x,_y,indexing='ij')

                V_downSampled_all[V_type]=V[_zz,_xx,_yy]

            filename=filenameList[V_type]

            if "Permutation123" in filename:
                filename=filename.split("Permutation123/")[1]

            if not keepOnlyDownSampled:
                tifffile.imwrite(os.path.join(dataOutputCropped[volumeTag],filename+"_cropped.tiff") ,
                    V,
                    resolution=(xRes,xRes,unitTiff),
                    compress=True,
                    description=descriptionStrOutput
                    )

            if doDownSampling:

                descriptionDict["downSamplingFactor"]=downSamplingFactor if doDownSampling else None

                descriptionStrOutput=str(descriptionDict)

                # print("description:{}".format(descriptionStrOutput))
                
                tifffile.imwrite(os.path.join(dataOutputDownSampled[volumeTag],filename+"_downsampled_by_{}.tiff".format(downSamplingFactor)),
                    V_downSampled_all[V_type],
                    resolution=(xRes,xRes,unitTiff),
                    compress=True,
                    description=descriptionStrOutput
                    )

                with open(os.path.join(dataOutputDownSampled[volumeTag],"fiberStruct_AMITEX.pickle"), "wb") as f:
                    pickle.dump(fiberStructOutput,f,protocol=pickle.HIGHEST_PROTOCOL)

        if doDownSampling:
            #redo volume fraction calculations after downsampling

            V_matrix=np.ones(V_downSampled_all["V_fibers"].shape,np.int8)

            print("\n\tAfter downsampling by {}:".format(downSamplingFactor))

            fibersVolumeFraction,\
                meanFiberFrac,\
                stdFiberFrac,\
                errorFiberFrac,\
                fibersMatrixVolumeFraction,\
                meanFiberMatrixFrac,\
                stdFiberMatrixFrac,\
                errorFiberMatrixFrac,\
                poresVolumeFraction,\
                meanPoresFrac,\
                stdPoresFrac,\
                errorPoresFrac=calculateFractionsFromVolumes(
                    V_matrix, 
                    V_downSampled_all["V_fibers"], 
                    V_downSampled_all["V_perim"], 
                    V_downSampled_all["V_pores"]
                    )

            dataDict={
                "meanFiberFrac"         :meanFiberFrac,
                "stdFiberFrac"          :stdFiberFrac,
                "errorFiberFrac"        :errorFiberFrac,
                "meanFiberMatrixFrac"   :meanFiberMatrixFrac,
                "stdFiberMatrixFrac"    :stdFiberMatrixFrac,
                "errorFiberMatrixFrac"  :errorFiberMatrixFrac,
                "meanPoresFrac"         :meanPoresFrac,
                "stdPoresFrac"          :stdPoresFrac,
                "errorPoresFrac"        :errorPoresFrac,
                "fibersVolumeFraction"  :fibersVolumeFraction,
                "fibersMatrixVolumeFraction":fibersMatrixVolumeFraction,
                "poresVolumeFraction"   :poresVolumeFraction,
            }


            outputDict[volumeTag]["downSampled"]={
                    "meanFiberFraction      ":"{: >8.4%}".format(meanFiberFrac),
                    "meanFiberMatrixFraction":"{: >8.4%}".format(meanFiberMatrixFrac)
                }

            with open(os.path.join(dataOutputDownSampled[volumeTag],"fiberVolumeFractions.json"),"w") as f:
                json.dump(dataDict,f, sort_keys=False, indent=4)

    print('Done cropping for Amitex')

    return outputDict


if __name__=="__main__":

    # to enable calling from this script
    outputDict=cropTiff_Amitex(
        manualCropping,
        dataInput,
        dataOutput,
        makePlots=False,
        mesoStructure=mesoStructure,
        doDownSampling=doDownSampling,
        downSamplingFactor=downSamplingFactor,
        keepOnlyDownSampled=False,
        cropV_hist=cropV_hist
        )

    croppingStatsDict={
        "manualCropping":manualCropping,
        "commonPath":commonPath,
    }

    croppingStatsDict.update(outputDict)

    with open(os.path.join(dataOutput,"croppingStats.json"), "w") as f:
        json.dump(croppingStatsDict, f, sort_keys=False, indent=4)