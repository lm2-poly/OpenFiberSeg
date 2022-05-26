# by Facundo Sosa-Rey, 2021. MIT license

import re
import numpy as np
import os
from datetime import date
import cv2 as cv
from tifffile import TiffFile

from extractCenterPoints import getTiffProperties
from preProcessingFunctions import histEqu_CannyDetection,imshowoverlay,contourDetection,paddingOfVolume

from skimage import morphology

from scipy import ndimage

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing

import json

# leave at True here, set to false in a particular scan's parameters if required
findExternalPerimeter=True
findPores=True

##########################################

plotThresholding                =True
plotCanny_perimeterDetection    =True
plotOpening_perimeterDetection  =True #volumetric processing

plotCannyEdgeDetection          =True
plotFloodFilling                =True
plotOpening_Closing_pores       =True #volumetric processing

parallelHandle=True

savePreprocessingData=True

#multithreading parameters
if parallelHandle:
    num_cores =min(multiprocessing.cpu_count()-2,20)
else:
    num_cores=1

dilatePerimOverPores=True #leave at True for the majority of cases: prevents the outer boundary (perimeter) to be encircled by a closed contour, 
    # and thus the entire volume be labelled as "pore". For some rare cases, can be better to deactivate it, to capture the pores in contact with 
    # the boundary (especially when the scanning cylinder is entirely contained in the solid. )

# scanName='Loic_PEEK_05/'
# scanName='Loic_PEEK_10/'
# scanName='Loic_PEEK_15/'
# scanName='Loic_PEEK_20/'
# scanName='Loic_PEEK_25/'
# scanName='Loic_PEEK_30/'
# scanName='Loic_PEEK_35/'
scanName='Loic_PEEK_40/'

formatStr="{:0>4.0f}.tiff" #default, do not touch (change below if necessary, for each scan)
      
if scanName=="Loic_PEEK_05/":
    
    commonPath='./TomographicData/PEEK05/'
    # which pixels to include in the processing
    pixelRangeX=[60, 960]
    pixelRangeY=[60, 930]
    iFirst=1   # first slice
    iLast= 980 #  last slice
    
    # thresholds for the Canny algorithm
    Canny_valLow_perimeter =30
    Canny_valHigh_perimeter=50
    Canny_sigma_perimeter=1.0

    Canny_valLow_pores=60 #60
    Canny_valHigh_pores=180 #180
    Canny_sigma_pores=3.0

    
    # value in original data to use in thresholding for perimeter detection
    thresholding_valPerim=50

    SE_Canny_dilation_diameter=15
    SE_edges_diameter=9
    SE_fills_diameter=5
    SE_large_diameter=15 

    SE_perim_3DOpening_radius=3

    SE_pores3d_radiusOpening=1
    SE_pores3d_radiusClosing=3  

if scanName=="Loic_PEEK_10/":
    
    commonPath='./TomographicData/PEEK10/'
    # which pixels to include in the processing
    pixelRangeX=[60, 960]
    pixelRangeY=[60, 930]
    iFirst=1  # first slice
    iLast= 980# last slice
    
    # thresholds for the Canny algorithm
    Canny_valLow_perimeter =30
    Canny_valHigh_perimeter=50
    Canny_sigma_perimeter=1.0

    Canny_valLow_pores=60
    Canny_valHigh_pores=180
    Canny_sigma_pores=3.0

    
    # value in original data to use in thresholding for perimeter detection
    thresholding_valPerim=50 


    SE_Canny_dilation_diameter=15
    SE_edges_diameter=9
    SE_fills_diameter=5
    SE_large_diameter=15 

    SE_perim_3DOpening_radius=3

    SE_pores3d_radiusOpening=1
    SE_pores3d_radiusClosing=3  

if scanName=="Loic_PEEK_15/": 
    
    commonPath='./TomographicData/PEEK15/'
    # which pixels to include in the processing
    pixelRangeX=[60, 960]
    pixelRangeY=[60, 930]
    iFirst=1   # first slice
    iLast= 978 # last slice
    
    # thresholds for the Canny algorithm
    Canny_valLow_perimeter =30
    Canny_valHigh_perimeter=50
    Canny_sigma_perimeter=1.0

    Canny_valLow_pores=60
    Canny_valHigh_pores=180
    Canny_sigma_pores=3.0

    
    # value in original data to use in thresholding for perimeter detection
    thresholding_valPerim=60    

    SE_Canny_dilation_diameter=15
    SE_edges_diameter=9
    SE_fills_diameter=5
    SE_large_diameter=15 

    SE_perim_3DOpening_radius=3

    SE_pores3d_radiusOpening=1
    SE_pores3d_radiusClosing=3 

if scanName=="Loic_PEEK_20/":
    
    commonPath='./TomographicData/PEEK20/'
    # which pixels to include in the processing
    pixelRangeX=[60, 960]
    pixelRangeY=[60, 930]
    iFirst=1   # first slice
    iLast= 978 # last slice
    
    # thresholds for the Canny algorithm
    Canny_valLow_perimeter =30
    Canny_valHigh_perimeter=50
    Canny_sigma_perimeter=1.0

    Canny_valLow_pores=60
    Canny_valHigh_pores=180
    Canny_sigma_pores=3.0

    # value in original data to use in thresholding for perimeter detection
    thresholding_valPerim=30 

    SE_Canny_dilation_diameter=15
    SE_edges_diameter=9
    SE_fills_diameter=5
    SE_large_diameter=15 

    SE_perim_3DOpening_radius=3

    SE_pores3d_radiusOpening=1
    SE_pores3d_radiusClosing=3 

if scanName=="Loic_PEEK_25/":
    
    commonPath='./TomographicData/PEEK25/'
    # which pixels to include in the processing
    pixelRangeX=[60, 960]
    pixelRangeY=[60, 930]
    iFirst=1   # first slice
    iLast= 976 # last slice
    
    # thresholds for the Canny algorithm
    Canny_valLow_perimeter =30
    Canny_valHigh_perimeter=50
    Canny_sigma_perimeter=1.0

    Canny_valLow_pores=100
    Canny_valHigh_pores=180
    Canny_sigma_pores=3.0

    # value in original data to use in thresholding for perimeter detection
    thresholding_valPerim=60 

    SE_Canny_dilation_diameter=15
    SE_edges_diameter=9
    SE_fills_diameter=5
    SE_large_diameter=15 

    SE_perim_3DOpening_radius=3

    SE_pores3d_radiusOpening=1
    SE_pores3d_radiusClosing=3

if scanName=="Loic_PEEK_30/":

    commonPath='./TomographicData/PEEK30/'
    # which pixels to include in the processing
    pixelRangeX=[60, 960]
    pixelRangeY=[60, 930]
    iFirst=1   # first slice
    iLast= 978 #  last slice
    
    # thresholds for the Canny algorithm
    Canny_valLow_perimeter =30
    Canny_valHigh_perimeter=50
    Canny_sigma_perimeter=1.0

    Canny_valLow_pores=100
    Canny_valHigh_pores=180
    Canny_sigma_pores=3.0

    
    # value in original data to use in thresholding for perimeter detection
    thresholding_valPerim=50 

    SE_Canny_dilation_diameter=15
    SE_edges_diameter=9
    SE_fills_diameter=5
    SE_large_diameter=15 

    SE_perim_3DOpening_radius=3

    SE_pores3d_radiusOpening=1
    SE_pores3d_radiusClosing=3      
        
if scanName=="Loic_PEEK_35/":
    
    commonPath='./TomographicData/PEEK35/'
    # which pixels to include in the processing
    pixelRangeX=[60, 960]
    pixelRangeY=[60, 930]
    iFirst=1   # first slice
    iLast= 978 #  last slice
    
    # thresholds for the Canny algorithm
    Canny_valLow_perimeter =30
    Canny_valHigh_perimeter=50
    Canny_sigma_perimeter=1.0

    Canny_valLow_pores=100
    Canny_valHigh_pores=200
    Canny_sigma_pores=3.0

    
    # value in original data to use in thresholding for perimeter detection
    thresholding_valPerim=60 

    SE_Canny_dilation_diameter=15
    SE_edges_diameter=9
    SE_fills_diameter=5
    SE_large_diameter=15 

    SE_perim_3DOpening_radius=3

    SE_pores3d_radiusOpening=1
    SE_pores3d_radiusClosing=3  

if scanName=="Loic_PEEK_40/":
    
    commonPath='./TomographicData/PEEK40/'
    # which pixels to include in the processing
    pixelRangeX=[60, 960]
    pixelRangeY=[60, 930]
    iFirst=1   # first slice
    iLast= 980 #  last slice
    
    # thresholds for the Canny algorithm
    Canny_valLow_perimeter =30
    Canny_valHigh_perimeter=50
    Canny_sigma_perimeter=1.0

    Canny_valLow_pores=100
    Canny_valHigh_pores=200
    Canny_sigma_pores=3.0

    
    # value in original data to use in thresholding for perimeter detection
    thresholding_valPerim=120 

    SE_Canny_dilation_diameter=15
    SE_edges_diameter=9
    SE_fills_diameter=5
    SE_large_diameter=15 

    SE_perim_3DOpening_radius=3

    SE_pores3d_radiusOpening=1
    SE_pores3d_radiusClosing=3  

#####################################################################################

if (iLast-iFirst)>40:
    if any([
        plotThresholding,
        plotCanny_perimeterDetection,
        plotOpening_perimeterDetection,
        plotCannyEdgeDetection,
        plotFloodFilling,
        plotOpening_Closing_pores
        ]):
        raise ValueError("These parameters will cause the production of too many figures. Set plotting to False to continue.")


######################################################

pathRawData="uCT_RawData"

while pathRawData in commonPath:
    # commonPath should not point to uCT_RawData but to the parent folder
    commonPath=os.path.dirname(commonPath)

currentPath=os.path.join(commonPath,pathRawData)

if not os.path.exists(currentPath):
    raise IOError("The path provided does not exist: {}".format(currentPath))

######################################################

# get filelist and pixel size

tiffFilesInDir=[f.name for f in os.scandir(currentPath) if f.is_file() and "Header.txt" not in f.name]

matchObj=re.match(r"(.*)([0-9]{4}).tiff",tiffFilesInDir[0])

prefix=matchObj.group(1)

filenameList    ={}

for tF in tiffFilesInDir:
    matchObj=re.match(r"(.*)([0-9]{4}).tiff",tF)

    if matchObj is not None:
        filenameList[int(matchObj.group(2))]=tF


with TiffFile(os.path.join(currentPath,filenameList[iFirst])) as tif:
    xRes,unitTiff=getTiffProperties(tif)

    if pixelRangeX is None or pixelRangeY is None:
        firstImage=tif.asarray()

        pixelRangeX=[1,firstImage.shape[0]]
        pixelRangeY=[1,firstImage.shape[1]]
        

print("PreProcessing on files in:\n"+commonPath+pathRawData)

useTimeStamp=False # adds date to Folder name, to redo preprocessing and keep old files. Useful for debugging

if useTimeStamp:
    today   = date.today()
    dateStr = today.strftime("%b-%d-%Y")

    outputFolderName="preProcessed_"+dateStr
else:
    outputFolderName="preProcessed"


offset=iFirst

nData=iLast-iFirst+1

if pixelRangeX[0]<1:
    raise ValueError(f"First pixel position must be at least 1, given: {pixelRangeX[0]}")

if pixelRangeY[0]<1:
    raise ValueError(f"First pixel position must be at least 1, given: {pixelRangeY[0]}")

if pixelRangeX[0]>pixelRangeX[1]:
    raise ValueError(f"Pixel position must be in increasing order, given: {pixelRangeX}")

if pixelRangeY[0]>pixelRangeY[1]:
    raise ValueError(f"Pixel position must be in increasing order, given: {pixelRangeY}")


numPixX=pixelRangeX[1]-pixelRangeX[0]+1
numPixY=pixelRangeY[1]-pixelRangeY[0]+1

pixXvec=[val for val in range(pixelRangeX[0],pixelRangeX[1]+1)]
pixYvec=[val for val in range(pixelRangeY[0],pixelRangeY[1]+1)]

if numPixX<1 or numPixY <1:
    raise ValueError ("pixelRange must be in increasing order")

# pre-allocation
V_original= np.zeros((numPixX,numPixY,nData),np.uint8)
V_hist  = np.zeros((numPixX,numPixY,nData),np.uint8)


filename    ={}

filename={imSlice:prefix+formatStr.format(imSlice) for imSlice in range(iFirst,iLast+1) }

with TiffFile(os.path.join(commonPath,pathRawData,filename[iFirst])) as tif:
    xRes,unitTiff=getTiffProperties(tif)

# structuring elements for the different morphological operations

V_perim=np.zeros((numPixX,numPixY,nData),np.uint8)
if findExternalPerimeter:
    SE_perim = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(SE_Canny_dilation_diameter, SE_Canny_dilation_diameter))
    SE_perim[:, 0]=SE_perim[ 0,:]
    SE_perim[:,-1]=SE_perim[-1,:]
else:
    SE_perim=None

if findPores:
    SE_edges = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(SE_edges_diameter, SE_edges_diameter))
    SE_edges[:, 0]=SE_edges[ 0,:]
    SE_edges[:,-1]=SE_edges[-1,:]

    SE_fills = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(SE_fills_diameter, SE_fills_diameter))
    SE_fills[:, 0]=SE_fills[ 0,:]
    SE_fills[:,-1]=SE_fills[-1,:]

    SE_large = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(SE_large_diameter, SE_large_diameter))
    SE_large[:, 0]=SE_large[ 0,:]
    SE_large[:,-1]=SE_large[-1,:]
else:
    SE_edges=None
    SE_fills=None
    SE_large=None


edgesPores      = np.zeros((numPixX,numPixY,nData),np.uint8)
V_pores         = np.zeros((numPixX,numPixY,nData),np.uint8)


print('\n\tPre-allocation completed')


# if the output directory doesn't exist, it is created here:

filesInDir = [f.path for f in os.scandir(commonPath) if f.is_dir()]

if os.path.join(commonPath,outputFolderName) not in filesInDir:
    os.mkdir(os.path.join(commonPath,outputFolderName))


print('\n\tHistogram equalization and Canny edge detection started')

results = Parallel(n_jobs=num_cores)\
    (delayed(histEqu_CannyDetection)\
        (os.path.join(commonPath,pathRawData,filename[imSlice]),
        imSlice,iFirst,iLast,pixelRangeX,pixelRangeY,
        findExternalPerimeter,
        findPores,
        thresholding_valPerim,
        Canny_sigma_perimeter,
        Canny_valLow_perimeter,
        Canny_valHigh_perimeter,
        SE_perim,
        Canny_sigma_pores,    
        Canny_valLow_pores,    
        Canny_valHigh_pores,
        plotCanny_perimeterDetection=plotCanny_perimeterDetection,
        plotCannyEdgeDetection=plotCannyEdgeDetection,
        plotThresholding=plotThresholding
        )for imSlice in range(iFirst,iLast+1) )

#unpacking of results from parallel execution
for imSlice,resultTuple in enumerate(results):
    V_original      [:,:,imSlice]=resultTuple[0]
    V_hist          [:,:,imSlice]=resultTuple[1]

    #no need to overwrite if no detection was attempted
    if findExternalPerimeter:
        V_perim         [:,:,imSlice]=resultTuple[2]
    if findPores:
        edgesPores      [:,:,imSlice]=resultTuple[3]


if findExternalPerimeter:

    print("\tVolumetric processing (perimeter detection): opening...")

    #this step removes false positive: thin regions that spills from the perimeter to inside the filament
    
    SE_ball3D=morphology.ball(SE_perim_3DOpening_radius, dtype=np.uint8)

    paddingSize=SE_Canny_dilation_diameter # to avoid artifacts on corners after opening

    # padding on all sides is necessary because ball SE cannot reach side pixels
    paddedV_perim=paddingOfVolume(V_perim,paddingSize)

    paddedV_perim_opened=np.array(ndimage.binary_opening(paddedV_perim,SE_ball3D),np.uint8)*255

    if plotOpening_perimeterDetection:
        for imSlice in range(iFirst,iLast+1):

            imshowoverlay(
                paddedV_perim[paddingSize:-paddingSize,paddingSize:-paddingSize,imSlice-offset+paddingSize],
                V_hist[:,:,imSlice-offset],
                color=[0,233,77],
                title="paddedV_perim, processed in 2D, imslice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast),
                alpha=0.4
                )

            imshowoverlay(
                paddedV_perim_opened[paddingSize:-paddingSize,paddingSize:-paddingSize,imSlice-offset+paddingSize],
                V_hist[:,:,imSlice-offset],
                color=[255,120,50],
                title="Opening in 3D, imslice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast),
                alpha=0.4
                )

            plt.show()

    # remove padding on all 6 sides of volume
    V_perim=paddedV_perim_opened[paddingSize:-paddingSize,paddingSize:-paddingSize,paddingSize:-paddingSize]

    if dilatePerimOverPores:  
        for imSlice in range(iFirst,iLast+1):
            # use the perimeter mask to remove edges on the perimeter so that they are
            # not taken to be pores at the stage of closing contours
            # Needs to be done after volumetric opening so spillover from sample perimeter
            # doesn't contaminate edges of pores
            temp=V_perim[:,:,imSlice-offset].copy()
            temp=cv.dilate(temp,SE_edges)

            edgesPoresSlice=edgesPores[:,:,imSlice-offset]
            edgesPoresSlice[temp==255]=0

            edgesPores[:,:,imSlice-offset]=edgesPoresSlice
    


print('\tHistogram equalization and Canny edge detection complete')

################################################################

###     Closing of contours

################################################################

if findPores:
    print('\n\tContour detection (floodfill) started')


    results = Parallel(n_jobs=num_cores)\
        (delayed(contourDetection)\
            (edgesPores[:,:,imSlice-offset],
                V_hist[:,:,imSlice-offset],
                imSlice,iFirst,iLast,
                SE_fills,SE_edges,SE_large,
                plotFloodFilling=plotFloodFilling
            )for imSlice in range(iFirst,iLast+1) )

    #unpacking of results from parallel execution
    for imSlice,resultTuple in enumerate(results):
        V_pores[:,:,imSlice]=resultTuple



    print("\tVolumetric processing (porosity detection): opening...")

    #opening step removes false positive: thin regions that are not pores

    paddingSize=max(SE_pores3d_radiusOpening,SE_pores3d_radiusClosing) # to avoid artifacts on corners after opening and closing

    SE_ball3D_opening=morphology.ball(SE_pores3d_radiusOpening, dtype=np.uint8)

    paddedV_pores=paddingOfVolume(V_pores,paddingSize,paddingValue=0)

    temp=np.array(ndimage.binary_opening(paddedV_pores,SE_ball3D_opening),np.uint8)*255

    print("\tVolumetric processing (porosity detection): closing...")

    # this step fill missing detections, where a region is between two labelled halves of a pore

    SE_ball3D_closing=morphology.ball(SE_pores3d_radiusClosing, dtype=np.uint8)

    paddedV_pores_opened_closed=np.array(ndimage.binary_closing(temp,SE_ball3D_closing),np.uint8)*255


    if plotOpening_Closing_pores:
        for imSlice in range(iFirst,iLast+1):

            imshowoverlay(
                V_pores[:,:,imSlice-offset],
                V_hist[:,:,imSlice-offset],
                color=[0,233,77],
                title="Processing in 2D, imslice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast),
                alpha=0.4
                )

            imshowoverlay(
                paddedV_pores_opened_closed[paddingSize:-paddingSize,paddingSize:-paddingSize,imSlice-offset+paddingSize],
                V_hist[:,:,imSlice-offset],
                color=[255,120,50],
                title="closind in 3D, imslice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast),
                alpha=0.4
                )

            plt.show()

    V_pores=paddedV_pores_opened_closed[paddingSize:-paddingSize,paddingSize:-paddingSize,paddingSize:-paddingSize]

    if findExternalPerimeter:
        V_pores[V_perim==255]=0

    print('\tContour detection (floodfill) completed')

    ##########################################

    # Calculate porosity

    shapeV=V_original.shape

    totalPixels=shapeV[0]*shapeV[1]*shapeV[2]

    if findExternalPerimeter:
        volumeInFilament=totalPixels-np.count_nonzero(V_perim)
    else: # cropped inside filament
        volumeInFilament=totalPixels

    volumeInPores=np.count_nonzero(V_pores)

    porosity=volumeInPores/volumeInFilament

    print("Total porosity detected in volume: {: >8.2f}%".format(porosity*100.))

    ##########################################
else:
    porosity=0.


import tifffile

if savePreprocessingData:

    print('\n\tWriting to disk started')
    
    print("\n\t\tWriting output to : \n\t{}".format(os.path.join(commonPath,outputFolderName)))

    descriptionStr="{"+"\"shape([x,y,z])\":[{},{},{}]".format(*(V_original.shape))+"}"

    print("\t\twriting V_original.tiff")

    tifffile.imwrite(
        os.path.join(commonPath,outputFolderName,'V_original.tiff'),
        np.transpose(V_original,(2,0,1)),
        resolution=(xRes,xRes,unitTiff),
        compress=True,
        description=descriptionStr
        )

    print("\t\twriting V_hist.tiff")

    tifffile.imwrite(
        os.path.join(commonPath,outputFolderName,'V_hist.tiff'),
        np.transpose(V_hist,(2,0,1)),
        resolution=(xRes,xRes,unitTiff),
        compress=True,
        description=descriptionStr
        )

    print("\t\twriting V_pores.tiff")

    tifffile.imwrite(
        os.path.join(commonPath,outputFolderName,'V_pores.tiff'),
        np.transpose(V_pores,(2,0,1)),
        resolution=(xRes,xRes,unitTiff),
        compress=True,
        description=descriptionStr
        )

    if findExternalPerimeter:
        #the next step (InsegtFibre_3D) allows for V_perim to be absent

        print("\t\twriting V_perim.tiff")

        tifffile.imwrite(
            os.path.join(commonPath,outputFolderName,'V_perim.tiff'),
            np.transpose(V_perim,(2,0,1)),
            resolution=(xRes,xRes,unitTiff),
            compress=True,
            description=descriptionStr
            )

    params={
        "porosity"                  :porosity,
        "SE_edges_radius"           :SE_edges_diameter,
        "SE_fills_radius"           :SE_fills_diameter,
        "SE_large_radius"           :SE_large_diameter,
        "SE_perim_radius"           :SE_Canny_dilation_diameter,
        "SE_perim_3DOpening_radius" :SE_perim_3DOpening_radius,
        "SE_pores3d_radiusClosing"  :SE_pores3d_radiusClosing,
        "SE_pores3d_radiusOpening"  :SE_pores3d_radiusOpening,
        "threshValPerim"            :thresholding_valPerim,
        "threshHigh_perimeter"      :Canny_valHigh_perimeter,
        "threshLow_perimeter"       :Canny_valLow_perimeter,
        "threshHigh_pores"          :Canny_valHigh_pores,
        "threshLow_pores"           :Canny_valLow_pores,
        "sigma_perimeter"           :Canny_sigma_perimeter,
        "sigma_pores"               :Canny_sigma_pores,
        "pixelRangeX"               :pixelRangeX,
        "pixelRangeY"               :pixelRangeY,
        "iFirst"                    :iFirst,
        "iLast"                     :iLast,
        "commonPath"                :commonPath,
    }

    with open(os.path.join(commonPath,outputFolderName,'preProcessingParams.json'), "w") as f:
        json.dump(params, f, sort_keys=False, indent=4)

    print('\tWriting to disk completed')

