# by Facundo Sosa-Rey, 2021. MIT license

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from tifffile import TiffFile,imwrite
import subprocess
import pickle
import json
import os

from skimage import morphology
from scipy import ndimage

from trackingFunctions      import watershedTransform,paddingOfImage
from trackingParameters     import getTrackingParams
from preProcessingFunctions import paddingOfVolume,find

from centroid import centroidObj, most_frequent

# from trackingParameters import getBlobDetector

from joblib import Parallel, delayed  
import multiprocessing

import time

#multithreading parameters

# openCV already does multithreading, but only goes to 100% of all CPU for a brief moment. 
num_coresWatershed = int(multiprocessing.cpu_count()*2/3)#using all cpu can cause crashes for large datasets

num_coresCentroid  = int(multiprocessing.cpu_count()*2/3)#using all cpu can cause crashes for large datasets

def checkIfFilesPresent(path,*files):

    allFilesFound=True

    for file in files:
        findList=find(path,file)

        if len(findList)==0:            
            allFilesFound=False

    return not allFilesFound

def getTiffProperties(tif,getDescription=False,getDimensions=False):
    
    try:
        # resolution is returned as a ratio of integers. value is in inches by default
        xRes=tif.pages[0].tags['XResolution'].value
        yRes=tif.pages[0].tags['YResolution'].value

        if xRes!=yRes:
            raise ValueError('not implemented for unequal x and y scaling')

        unitEnum=tif.pages[0].tags['ResolutionUnit'].value
        if repr(unitEnum)=='<RESUNIT.CENTIMETER: 3>':
            unitTiff="CENTIMETER"
        elif repr(unitEnum)=='<RESUNIT.INCH: 2>':
            unitTiff="INCH"
        else:
            raise ValueError("not implemented for {}".format(repr(unitEnum)))
    except:
        print("\n\tTiff files do not contain scaling information, assigning default value of 1 micron/pixel")
        xRes=(int(1e4),1)
        unitTiff="CENTIMETER"

    if getDescription:
        descriptionStr=tif.pages[0].tags["ImageDescription"].value

        if getDimensions:
            return xRes,unitTiff,descriptionStr,tif.pages[0].shape
        else:
            return xRes,unitTiff,descriptionStr

    else:
        if getDimensions:
            return tif.pages[0].shape
        else:
            return xRes,unitTiff


def getSlice(V,i,xRange,yRange):
    # V is none for V_hist when plotting of watershed steps == false
    if V is not None:
        if xRange is not None:
            return V[i,xRange[0]:xRange[1],yRange[0]:yRange[1]]
        else:    
            return V[i,:,:]
    else:
        return None

def addPixelsToImHist(imHist_rgb,fiberMap,cmap:str="Blues"):

    oldRange=[-1.,np.max(fiberMap)]
    newRange=[0.2,1.]

    fiberImg_float=np.interp(np.array(fiberMap,np.float32),oldRange,newRange)

    cm=plt.get_cmap(cmap)
    fiberImg_rgb=cm(fiberImg_float)


    oldRange=[0. ,1.]
    newRange=[0.,0.5]#keep hist on the dark side
    imHist_rgb_untouched=np.interp(imHist_rgb,oldRange,newRange)

    alpha=0.02
    imHist_rgb[fiberMap>=0]=imHist_rgb[fiberMap>=0]*alpha+fiberImg_rgb[fiberMap>=0]*(1.-alpha)

    imHist_rgb[fiberMap==2]=imHist_rgb_untouched[fiberMap==2] #background value is fuly transparent

def addPixelsToImHist_singleColor(
    imHist_rgb,
    fiberMapBefore,
    fiberMapAfter,
    colorBefore,
    colorAfter):




    oldRange=[0. ,1.]
    newRange=[0.5,1.] #keep hist on the dark side
    imHist_rgb=np.interp(imHist_rgb,oldRange,newRange)

    alpha=0.2

    x,y=np.where(fiberMapAfter>2)
    for ix,iy in zip(x,y):
        imHist_rgb[ix,iy,:]=np.array((*colorAfter,1.-alpha))      

    x,y=np.where(fiberMapBefore>2)
    for ix,iy in zip(x,y):
        imHist_rgb[ix,iy,:]=np.array((*colorBefore,1.-alpha))

    return imHist_rgb



def watershedParallel(
    currentSlice,
    currentHist,
    imSlice,
    nSlices,
    initialWaterLevel,
    waterLevelIncrements,
    convexityDefectDist,
    currentPoresSlice,
    currentProbSlice                =None,
    checkConvexityAndSplit          =True,
    plotInitialAndResults           =False,
    plotEveryIteration              =False,
    plotWatershedStepsGlobal        =False,
    plotWatershedStepsMarkersOnly   =False,
    openingBeforeWaterRising        =False,
    plotConvexityDefects            =False,
    plotCentroids                   =False,
    plotExpansion                   =False,
    doNotPltShow                    =False,
    figsize                         =[8,8],
    legendFontSize                  =32
    ):

    print("\twatershed transform started on slice {}/{}, on {}".format(imSlice,nSlices,multiprocessing.current_process().name))

    tic = time.perf_counter()

    voxelMap_slice, grayImg, img, paddingWidth,connectedMap,dist_transform = \
        watershedTransform(
            currentSlice,
            currentHist,
            imSlice,
            initialWaterLevel,
            waterLevelIncrements,
            convexityDefectDist,
            checkConvexityAndSplit          =checkConvexityAndSplit,
            plotInitialAndResults           =plotInitialAndResults,
            plotEveryIteration              =plotEveryIteration,
            plotWatershedStepsGlobal        =plotWatershedStepsGlobal,
            plotWatershedStepsMarkersOnly   =plotWatershedStepsMarkersOnly,
            openingBeforeWaterRising        =openingBeforeWaterRising,
            plotConvexityDefects            =plotConvexityDefects,
            doNotPltShow                    =doNotPltShow,
            figsize                         =figsize,
            legendFontSize                  =legendFontSize           
            )

    if currentProbSlice is not None:
        # in voxelMap: background is 2, outline is -1, sometimes -2 for intersecting outlines
        # to expand voxels according to probability map, 
        # we need "sure" background==-1
        # unknown ==0,
        # seeds starting at 3 
        
        currentProbSlice_=currentProbSlice.copy() #Parallel doesn't allow modifying inputs

        #use mid range value for padding, according to datatype
        if currentProbSlice_.dtype==np.uint32:
            paddingValue=2**15-1 #previous versions used uint32, with no performance improvement
        elif currentProbSlice_.dtype==np.uint8:
            paddingValue=2**7-1

        currentProbSlice_[currentPoresSlice==1]=paddingValue
        
        paddingWidth=5

        # pad image by repeating pixels on edge, so as to not introduce artifacts upon computing the Laplacian
        currentProbSlice_padded=cv.copyMakeBorder(currentProbSlice_, 
            paddingWidth, paddingWidth, paddingWidth, paddingWidth, cv.BORDER_REPLICATE) 

        maskMarkers=np.zeros(voxelMap_slice.shape,np.uint8)
        maskMarkers[voxelMap_slice>2]=1

        laplacian=cv.Laplacian(cv.GaussianBlur(currentProbSlice_padded, ksize=(3,3), sigmaX=0,sigmaY=0),cv.CV_64F)

        mask=np.zeros(laplacian.shape,np.uint8)

        meanLaplacian=np.mean(laplacian[paddingWidth*2:-paddingWidth*2,paddingWidth*2:-paddingWidth*2])
        stdLaplacian=np.std(laplacian[paddingWidth*2:-paddingWidth*2,paddingWidth*2:-paddingWidth*2])

        mask[laplacian<meanLaplacian-stdLaplacian*1.5]=255

        mask_removingMarkers=np.logical_and(voxelMap_slice>=3,mask==255)

        threshMap=np.ones(voxelMap_slice.shape,np.int32)*-2

        threshMap[mask==255]=-1
        threshMap[mask_removingMarkers]=voxelMap_slice[mask_removingMarkers]
        
        threshMap+=1

        # hack to keep background marker at 2, needs to be not in contact with border
        threshMap[2,2]=2

        threshMapBefore=threshMap.copy()

        img[mask==255]=255

        voxelMap_sliceExpanded=cv.watershed(img,threshMap)
    
        voxelMap_sliceExpanded[voxelMap_sliceExpanded==-1]=2

        #fix a bug where sometimes a label is given to a pixel outside the original mask
        voxelMap_slice[img[:,:,0]==0]=2 #background marker

        #restore background value
        backgroundmost_frequent=most_frequent(list(voxelMap_sliceExpanded.ravel()))
        if backgroundmost_frequent!=2:
            voxelMap_sliceExpanded[voxelMap_sliceExpanded==backgroundmost_frequent]=2

        if plotExpansion:

            plt.rcParams.update({'font.size': 22})
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams["font.family"] = "Times New Roman"

            bottomOffset=0.015
            titleHeight =0.04

            nCol=4
            nRow=1

            leftOffset=0.04

            width= (1./(nCol)-leftOffset)
            height=(1./nRow-2.*bottomOffset-titleHeight)

            left=  [val+leftOffset   for val in np.linspace(leftOffset,1.-leftOffset,nCol+1)]
            bottom=[val+bottomOffset for val in np.linspace(1.-1./nRow-bottomOffset,bottomOffset,nRow)]

            listAx=[]
            fig=plt.figure(figsize=[18,4.5],num="Validation_all_imSlice={}__".format(imSlice))

            for iRow in range(nRow):
                for iCol in range(nCol):
                    listAx.append(fig.add_axes([left[iCol],bottom[iRow],width,height]))

            listAx[0].set_title("threshMap")
            listAx[0].imshow(threshMapBefore[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth])

            listAx[1].set_title("voxelMap_sliceExpanded")
            listAx[1].imshow(voxelMap_sliceExpanded[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth])

            listAx[2].set_title("currentProbSlice")
            listAx[2].imshow(currentProbSlice,cmap="gray")

            listAx[3].set_title("img")
            listAx[3].imshow(img[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth])

            fig=plt.figure(figsize=[18,4.5],num="Validation_all_imSlice={}".format(imSlice))

            listAx=[]
            for iRow in range(nRow):
                for iCol in range(nCol):
                    listAx.append(fig.add_axes([left[iCol],bottom[iRow],width,height]))

            listAx[0].set_title("Laplacian")
            mappableHist=listAx[0].imshow(laplacian[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth])

            cax = fig.add_axes([0.01, bottomOffset, 0.005, height])
            plt.colorbar(mappableHist,cax=cax)

            listAx[1].set_title("mask")
            listAx[1].imshow(mask[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth],cmap="gray")

            currentHist     =paddingOfImage(currentHist)

            imHist_rgb=plt.cm.gray(currentHist)

            addPixelsToImHist(imHist_rgb,voxelMap_slice,cmap="Oranges")

            listAx[2].set_title("before expansion")
            listAx[2].imshow(imHist_rgb[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth])

            imHist_rgb=plt.cm.gray(currentHist)

            addPixelsToImHist(imHist_rgb,voxelMap_sliceExpanded,cmap="viridis")

            listAx[3].set_title("after expansion")
            listAx[3].imshow(imHist_rgb[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth])

            imHist_rgb=plt.cm.gray(currentHist)

            imHist_rgb=addPixelsToImHist_singleColor(
                imHist_rgb,
                voxelMap_slice,
                voxelMap_sliceExpanded,
                colorBefore =(1.0,0.33,0.0),
                colorAfter  =(0.0,0.24,0.7)
                )


            plt.figure(figsize=figsize,num="Combined before_after expansion")
            plt.title("Combined")
            plt.imshow(imHist_rgb[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth])

            patches=[
                mpatches.Patch(color=(1.0,0.33,0.0), label="INSEGT output" ),
                mpatches.Patch(color=(0.0,0.24,0.7), label="Expanded regions")
            ]

            # put those patched as legend-handles into the legend
            plt.legend(handles=patches,fontsize=legendFontSize,framealpha=1.)

            if not doNotPltShow:
                plt.tight_layout()
                plt.show()

        voxelMap_slice = voxelMap_sliceExpanded

    
    # remove padding of image
    voxelMap_slice = voxelMap_slice[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth]

    if plotCentroids:
        img[img==255]=100 # greyValue Fibers
        img[img==0]=30  # greyValue Matrix

        img=img[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth]
    else:
        img=None

    toc = time.perf_counter()
    print("\t\twatershed transform performed on slice {} in {} on {}".format(
        imSlice,
        time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)),
        multiprocessing.current_process().name
        )
    )


    return voxelMap_slice,img

def centroidExtractParallel(
    sliceNumber,
    nSlices,
    voxelMapSlice,
    xOffset,
    yOffset,
    plotCentroids   =False,
    img             =None,
    doNotPltShow    =False,
    figsize         =[8,8],
    legendFontSize  =22,
    scatterPntSize  =40
    ):

    print("\tcentroid detection, slice {}/{}, on {}".format(sliceNumber,nSlices,multiprocessing.current_process().name))

    tic = time.perf_counter()

    # loop over the unique labels returned by the Watershed algorithm
    if plotCentroids:
        watershedCentroids=[]
        watershedMarkers  =[]

    centroids=[]

    markerList=[int(marker) for marker in np.unique(voxelMapSlice)] # int32 is not JSON serializable


    if plotCentroids:
        plt.figure(figsize=figsize,num="Centroids")
        plt.imshow(img)
        plt.title("segt data with centroids, slice number={}".format(sliceNumber),fontsize=legendFontSize)
        plt.tight_layout()


    
    for marker in markerList:
        # if the label is 2, we are examining the 'background', if it's -1 it's the contours of the watershed segmentation
        # so simply ignore them. 0 is also encountered when two contours share a pixel
        if marker in [-1,0,2]:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the (binary) mask
        mask = np.zeros(voxelMapSlice.shape, dtype="uint8")
        mask[voxelMapSlice == marker] = 255

        # detect contours in the mask and check if single connected region is present
        cnt = cv.findContours(mask, cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE)[-2]

        if type(cnt)==tuple:
            #OpenCV version 4.5.4 returns tuples instead of list
            cnt=list(cnt)

        if len(cnt)>1:
            print("\t\tMarker is more than one connected region, slice number={}, marker={}".format(sliceNumber,marker))
            for iContour,contourObj in enumerate(cnt):
                print("\t\t\tcontour #{}, area: {}".format(iContour,cv.contourArea(contourObj)))
            cnt = [max(cnt, key=cv.contourArea)] #needs to be a list to work with centroidObj

        # extract statistical moments from connected region
        m=cv.moments(mask,binaryImage=True)

        y=m["m10"]/m["m00"] #permuted to conform with global convention
        x=m["m01"]/m["m00"]

        # construct centroid object
        centroids.append(centroidObj(x+xOffset,y+yOffset,cnt)) 
        centroids[-1].setMarker(marker) 
        
        # x and y are flipped to conform to standard in the rest of the project, where voxelMap[x,y].
        # imshow() transposes the image, so text() must be added as text(y,x) elsewhere but here


        if plotCentroids and x!=1.0 and y != 1.0: # hack was added to point (1,1) to deal with spillover from some markers mistaken as background in watershedTransform output
            watershedCentroids.append([float(x),float(y)]) #index -1(last) is used to allow manualRange
            watershedMarkers  .append(marker)


    if plotCentroids:
        for i in range(len(watershedCentroids)):
            xWater=watershedCentroids[i][0]
            yWater=watershedCentroids[i][1]
        
            # x y are transposed in imshow() convention
            if i==0:
                plt.scatter(yWater,xWater,s=scatterPntSize,c='red',label="Centroids")
            else:
                plt.scatter(yWater,xWater,s=scatterPntSize,c='red')

        plt.legend(fontsize=legendFontSize,loc=1,facecolor="w",framealpha=1.)
        plt.xlim([0,img.shape[1]-1])#imshow inverts x and y coordinates
        plt.ylim([img.shape[0]-1,0])
        if not doNotPltShow:
            plt.show()

    toc = time.perf_counter()
    print("\t\tcentroid detection on slice {} in {} on {}".format(
        sliceNumber,
        time.strftime("%Hh%Mm%Ss", time.gmtime(toc - tic)),
        multiprocessing.current_process().name
        )
    )


    return centroids

def addFlaggedPixelsToImg(img, flaggedPixels,color,alpha):
    for iFlag in flaggedPixels:
        x=iFlag[0]
        y=iFlag[1]
        img[x,y,:]=np.array(np.array(img[x,y,:],np.float)*(1.-alpha)+np.array(color,np.float)*alpha,np.uint8)

def imshowoverlay(binaryMap,grayImg_hist, title=None,color=[200,20,50],makePlot=True,alpha=0.7,xOffset=0,yOffset=0):
    x,y=np.where(binaryMap==1)
    flaggedPixels=[]
    for iPix in range(len(x)):
        flaggedPixels.append((x[iPix]+xOffset,y[iPix]+yOffset))

    # attenuate grayImg_hist to make it more readable
    
    oldRange=[0,1]
    newRange=[0,120]
    
    grayImg_hist=np.array(np.round(np.interp(grayImg_hist,oldRange,newRange)),np.uint8)
   
    imgComp = np.stack([grayImg_hist,grayImg_hist,grayImg_hist],axis=2)

    addFlaggedPixelsToImg(imgComp,flaggedPixels,color=color,alpha=alpha)

    if makePlot:
        plt.figure(figsize=[8,8])
        plt.imshow(imgComp,cmap="ocean")
        plt.title(title,fontsize=28)
        plt.tight_layout()

    return imgComp


def extractCenterPoints(
        commonPath,
        permutationVec,
        plotCentroids                   =False,
        plotInitialAndResults           =False,
        plotEveryIteration              =False,
        plotWatershedStepsGlobal        =False,
        plotWatershedStepsMarkersOnly   =False,
        openingBeforeWaterRising        =True,
        plotConvexityDefects            =False,
        manualRange                     =None,
        exclusiveZone                   =None,
        parallelHandle                  =False,
        plotOverlayFrom123              =False,
        plotExpansion                   =False
        ):


    permutationPaths=["Permutation123/","Permutation132/","Permutation321/"]

    if permutationVec=="123":        
        pathVolumes=os.path.join(commonPath,permutationPaths[0])
    elif permutationVec=="132":      
        pathVolumes=os.path.join(commonPath,permutationPaths[1])
    elif permutationVec=="321":      
        pathVolumes=os.path.join(commonPath,permutationPaths[2])

    print("\n\n\textractCenterPoints() called on dataset:\n {}".format(pathVolumes))
    print("\tReading from disk started")

    params=getTrackingParams(commonPath,"extractionParams")

    checkConvexityAndSplit  =params["checkConvexityAndSplit"]
    convexityDefectDist     =params["convexityDefectDist"]
    dilatePores             =params["dilatePores"]
    dilationRadius_perim    =params["dilationRadius_perim"]
    dilationRadius_pores    =params["dilationRadius_pores"]
    initialWaterLevel       =params["initialWaterLevel"]
    waterLevelIncrements    =params["waterLevelIncrements"]
    useProbabilityMap       =params["useProbabilityMap"]

    print("\nLaunching extraction with parameters:\n")
    for key,value in params.items():
        if type(value)==bool:
            print("\t{:<25}\t:{}".format(key,value)) 
        else:
            print("\t{:<25}\t:{: >8.4f}".format(key,value)) 
    print("")

    xRange=None
    yRange=None
    xMin=0
    yMin=0

    if exclusiveZone is not None:
        if permutationVec=="123":
            zMin=exclusiveZone["zMin"]
            zMax=exclusiveZone["zMax"]
            xMin=exclusiveZone["xMin"]
            xMax=exclusiveZone["xMax"]
            yMin=exclusiveZone["yMin"]
            yMax=exclusiveZone["yMax"]
        elif permutationVec=="132":
            zMin=exclusiveZone["yMin"]
            zMax=exclusiveZone["yMax"]
            xMin=exclusiveZone["xMin"]
            xMax=exclusiveZone["xMax"]
            yMin=exclusiveZone["zMin"]
            yMax=exclusiveZone["zMax"]
        elif permutationVec=="321":
            zMin=exclusiveZone["xMin"]
            zMax=exclusiveZone["xMax"]
            xMin=exclusiveZone["zMin"]
            xMax=exclusiveZone["zMax"]
            yMin=exclusiveZone["yMin"]
            yMax=exclusiveZone["yMax"]

        if manualRange is None:
            manualRange=range(zMin,zMax)
        else:
            manualRange=range(max(manualRange.start,zMin),min(manualRange.stop,zMax) )

        xRange=(xMin,xMax)
        yRange=(yMin,yMax)
    else:
        zMin=0 #for passing slice of V_pores

    if manualRange is not None:
        offset=max(manualRange.start,zMin)
    else:
        offset=zMin

    tic = time.perf_counter()

    if permutationVec!="123":
        with TiffFile(os.path.join(pathVolumes,"V_fibers.tiff")) as tif:
            V_fibers=np.array(tif.asarray()/255,np.uint8)

        # centroid detection only allowed where there hasn't
        # already been a fiber found in permutation123
        path_to_fiberVolume=os.path.join(commonPath,"Permutation123/V_fibers_masked.tiff")
    else:
        path_to_fiberVolume=os.path.join(pathVolumes,"V_fibers.tiff")

    with TiffFile(path_to_fiberVolume) as tif:

        xRes,unitTiff=getTiffProperties(tif)

        V_fibers_masked=np.array(tif.asarray()/255,np.uint8)

        if permutationVec=="123":
            V_fibers=V_fibers_masked
        else:

            if permutationVec=="132":
                V_fibers_masked=np.transpose(V_fibers_masked,(2,1,0)) # z,x,y -> y,x,z
                colorOverlay=[102,204,255]
            elif permutationVec=="321":
                V_fibers_masked=np.transpose(V_fibers_masked,(1,0,2)) # z,x,y -> x,z,y
                colorOverlay=[182,104,155],

            if permutationVec!="123" and plotOverlayFrom123:
                if manualRange is None:
                    nSlices=range(V_fibers_masked.shape[0])
                else:
                    nSlices=manualRange
                for imSlice in nSlices:

                    imshowoverlay(
                        V_fibers_masked[imSlice-zMin,:,:],
                        V_fibers[imSlice,:,:], 
                        title="imSlice={}/{}".format(imSlice,V_fibers_masked.shape[0]),
                        color=colorOverlay,
                        makePlot=True,
                        alpha=0.7,
                        xOffset=xMin,
                        yOffset=yMin
                        )

                    plt.show()
            
            if exclusiveZone:
                V_fibers_segment=V_fibers[zMin:zMax,xMin:xMax,yMin:yMax]
                V_fibers_segment[V_fibers_masked==1]=0

                V_fibers[zMin:zMax,xMin:xMax,yMin:yMax]=V_fibers_segment

            else:
                V_fibers[V_fibers_masked==1]=0

    dilationParams_current={"dilationRadius_perim":dilationRadius_perim,"dilationRadius_pores":dilationRadius_pores}
    
    print("\tLoading V_pores...")
    V_pores_loaded_bool=False
    
    if dilatePores:
        # no need to redo if already present on disk, at correct exclusiveZone and radii
        try:
            with TiffFile(os.path.join(commonPath,permutationPaths[0],"V_pores_dilated.tiff")) as tif:
                descriptionStr_dilatedPores=getTiffProperties(tif,getDescription=True)[2] 

            dilatedPoresFileFound=True
            tempStr=descriptionStr_dilatedPores.split("exclusiveZone\":")[1]
            tempStr=tempStr.replace("\'","\"")

        except:
            # V_pores_dilated.tiff not present: create it (dilatePores==True)
            dilatedPoresFileFound=False
            descriptionStr_dilatedPores=""
            tempStr="Not found"


        if tempStr !="Not found":
            if "dilationRadius" in descriptionStr_dilatedPores:
                tempStr,dilationParamsStr=tempStr.split(",\n")
                dilationParams_fromFile=json.loads(dilationParamsStr.split("dilationParams\":")[1])
            else:
                raise ValueError("Missing info in Tiff file description")

            exclusiveZone_fromFile=json.loads(tempStr) if tempStr !="None" else None

        else:
            exclusiveZone_fromFile=None
            dilationParams_fromFile=dilationParams_current #to make reverse compatible #TODO remove

        if dilatedPoresFileFound and exclusiveZone_fromFile==exclusiveZone and dilationParams_fromFile==dilationParams_current:
            print("\tV_pores_dilated found on disk at correct exclusiveZone and dilation parameters, loading...")
            with TiffFile(os.path.join(commonPath,permutationPaths[0],"V_pores_dilated.tiff")) as tif:
                V_pores=np.array(tif.asarray()/255,np.uint8)
                V_pores_loaded_bool=True
            
            if permutationVec != "123":
                if permutationVec=="132":
                    V_pores=np.transpose(V_pores,(2,1,0)) # z,x,y -> y,x,z
                elif permutationVec=="321":
                    V_pores=np.transpose(V_pores,(1,0,2)) # z,x,y -> x,z,y

            #update V_fibers to reflect the effect of pore and perim dilation
            if exclusiveZone:
                V_fibers[zMin:zMax,xMin:xMax,yMin:yMax][V_pores==1]=0
            else:
                V_fibers[V_pores==1]=0
            
            dilatePores=False 

        else:
            if dilatedPoresFileFound:
                if exclusiveZone_fromFile!=exclusiveZone:
                    print("exclusive zones do not match the one previously used, redo pore dilation")
                if dilationParams_fromFile!=dilationParams_current:
                    print("dilation parameters do not match the one previously used, redo pore dilation")


    try:
        with TiffFile(os.path.join(pathVolumes,"V_perim.tiff")) as tif:
            V_perim=np.array(tif.asarray()/255,np.uint8)
    except:
        V_perim=None    # wont exist if not created in preprocessing


    if not V_pores_loaded_bool: 
        # in the case where pore dilation is not used  (dilatePores=false in trackingParams.json)
        # we need to load it here
        with TiffFile(os.path.join(pathVolumes,"V_pores.tiff")) as tif:
            V_pores=np.array(tif.asarray()/255,np.uint8)
        if exclusiveZone:
            # V_pores_dilated will already be at correct truncation with regards to exclusiveZone
            V_pores=V_pores[zMin:zMax,xMin:xMax,yMin:yMax]

    if exclusiveZone and V_perim is not None:
        V_perim=V_perim[zMin:zMax,xMin:xMax,yMin:yMax]
    

    if dilatePores:
        if 1 in V_pores: # no need to process if no pores are present

            paddingSize=dilationRadius_pores*2
            SE_ball3D=morphology.ball(dilationRadius_pores, dtype=np.uint8)

            paddedV_pores=paddingOfVolume(V_pores,paddingSize,paddingValue=0)

            print("\tDilation of V_pores...")
            paddedV_pores=np.array(ndimage.binary_dilation(paddedV_pores,SE_ball3D),np.uint8)

            V_pores=paddedV_pores[paddingSize:-paddingSize,paddingSize:-paddingSize,paddingSize:-paddingSize]


            if V_perim is not None:
                # processing in 2D is much faster, and for perim, there is no advantage to going 3D: perimeter is smooth and normally continuous across entire Z direction 
                paddingSize=dilationRadius_perim*2

                SE_disk = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(dilationRadius_perim*2, dilationRadius_perim*2))

                print("\tDilation of V_perim...")
                for imSlice in range(V_perim.shape[0]):
                    temp=paddingOfImage(V_perim[imSlice,:,:],paddingSize)

                    temp=cv.dilate(temp,SE_disk)
                    V_perim[imSlice,:,:]=temp[paddingSize:-paddingSize,paddingSize:-paddingSize]

                V_pores[V_perim==1]=1

            descriptionStr=\
                "{"+"\"shape([z,x,y])\":[{},{},{}],\n\"manualRange\":{},\n\"exclusiveZone\":{},\n\"dilationParams\":{}"\
                .format(*V_pores.shape,manualRange,exclusiveZone,dilationParams_current)

            imwrite(
                os.path.join(pathVolumes,'V_pores_dilated.tiff'),
                V_pores*255,
                resolution=(xRes,xRes,unitTiff),
                description=descriptionStr,
                compress=True
                )

            #update V_fibers to reflect the effect of pore and perim dilation
            if exclusiveZone:
                V_fibers[zMin:zMax,xMin:xMax,yMin:yMax][V_pores==1]=0
            else:
                V_fibers[V_pores==1]=0

    if useProbabilityMap:
        print("\tLoading V_prob...")
        with TiffFile(os.path.join(pathVolumes,"V_prob.tiff")) as tif:
            V_prob=np.array(tif.asarray())
    
    else:
        V_prob=None


    if manualRange is None:
        V_voxels=np.ones(V_fibers.shape,np.int32 )*2 # initialize to background value
        manualRangeDict={"start":0,"end":V_fibers.shape[0]} # all slices, full width in x and y

    else:
        if exclusiveZone is None and manualRange is not None:
            xMin=0
            xMax=V_fibers.shape[1]
            yMin=0
            yMax=V_fibers.shape[2]
            zMin=manualRange.start
            zMax=manualRange.stop

        V_voxels=np.ones((zMax-zMin,xMax-xMin,yMax-yMin),np.int32 )*2 # initialize to background value
        manualRangeDict={"start":manualRange.start,"end":manualRange.stop}
    
    # format in a way that is readable for .tiff file properties in file explorer
    descriptionStr="{"+"\"shape([z,x,y])\":[{},{},{}],\n\"manualRange\":{},\n\"exclusiveZone\":{}".format(V_voxels.shape[0],V_voxels.shape[1],V_voxels.shape[2],manualRange,exclusiveZone)

    descriptionDict={
        "shape([z,x,y])":V_voxels.shape,
        "manualRange":manualRangeDict,
        "descriptionStr":descriptionStr,
        "exclusiveZone":exclusiveZone
        }

    if plotInitialAndResults or plotExpansion:
        with TiffFile(os.path.join(pathVolumes,"V_hist.tiff")) as tif:
            V_hist=np.array(tif.asarray(),np.uint8)
    else:
        V_hist=None

    toc = time.perf_counter()
    
    times_centroids={"parallelHandle":parallelHandle}

    times_centroids["readFromDisk, extractCenterPoints:"]=time.strftime("%Hh%Mm%Ss", time.gmtime(toc-tic))

    print(f"\tReading from disk complete in {toc - tic:0.4f} seconds\n")

    if plotCentroids:
        watershedCentroids  =[]
        watershedMarkers    =[]

        # 3 channel images for plotting only
        if exclusiveZone is None:
            if manualRange is None:
                V_img=np.empty((V_fibers.shape[0],V_fibers.shape[1],V_fibers.shape[2],3),np.uint8 )
            else:
                V_img=np.empty((len(manualRange),V_fibers.shape[1],V_fibers.shape[2],3),np.uint8 )
        else:
            V_img=np.empty((zMax-zMin,xMax-xMin,yMax-yMin,3),np.uint8 )



    skipList=[]
    for i,imSlice in enumerate(V_fibers):
        if 1 not in imSlice:
            skipList.append(i)

    if manualRange is None:
        nSlices=[i for i in range(V_fibers.shape[0]) if i not in skipList]
    else:
        nSlices=[i for i in manualRange if i not in skipList]

    ticWatershed=time.perf_counter()

    # check if previous run was not completed
    doWatershed=checkIfFilesPresent(pathVolumes,'V_voxelMapTEMP.pickle')

    if doWatershed:

        if parallelHandle:

            results = Parallel(n_jobs=num_coresWatershed)\
                (delayed(watershedParallel)\
                    (
                    getSlice(V_fibers,imSlice,xRange,yRange),
                    getSlice(V_hist,imSlice,xRange,yRange),
                    imSlice,
                    range(min(nSlices),max(nSlices)),
                    initialWaterLevel,
                    waterLevelIncrements,
                    convexityDefectDist,
                    V_pores[imSlice-zMin], 
                    checkConvexityAndSplit          =checkConvexityAndSplit,  
                    currentProbSlice                =getSlice(V_prob,imSlice,xRange,yRange),
                    plotInitialAndResults           =plotInitialAndResults,
                    plotEveryIteration              =plotEveryIteration,
                    plotWatershedStepsGlobal        =plotWatershedStepsGlobal,
                    plotWatershedStepsMarkersOnly   =plotWatershedStepsMarkersOnly,
                    openingBeforeWaterRising        =openingBeforeWaterRising,
                    plotConvexityDefects            =plotConvexityDefects,
                    plotCentroids                   =plotCentroids,
                    plotExpansion                   =plotExpansion
                    )for imSlice in nSlices )

            print("\n\t\tunpacking results from all processes")

            for imSlice,resultTuple in zip(nSlices,results):

                V_voxels[imSlice-offset,:,:]=resultTuple[0]    # voxelMap_slice

                if plotCentroids:
                    V_img[imSlice-offset,:,:,:]=resultTuple[1] # img

            #save temporary file, as the next step sometimes hangs forever
            print("\tSaving temporary file to disk")

            pathCenterPointsData=os.path.join(pathVolumes,"V_voxelMapTEMP.pickle")
            with open(pathCenterPointsData, "wb") as f:
                pickle.dump(V_voxels, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:

            for imSlice in nSlices: 

                currentSlice=getSlice(V_fibers,imSlice,xRange,yRange)
                currentHist =getSlice(V_hist,imSlice,xRange,yRange)
                currentProb =getSlice(V_prob,imSlice,xRange,yRange)
                
                voxelMap,img = watershedParallel(
                        currentSlice,
                        currentHist,
                        imSlice,
                        range(min(nSlices),max(nSlices)),
                        initialWaterLevel,
                        waterLevelIncrements,
                        convexityDefectDist,
                        V_pores[imSlice-zMin],#V_pores is truncated to exclusiveZone already
                        checkConvexityAndSplit          =checkConvexityAndSplit,
                        currentProbSlice                =currentProb,
                        plotInitialAndResults           =plotInitialAndResults,
                        plotEveryIteration              =plotEveryIteration,
                        plotWatershedStepsGlobal        =plotWatershedStepsGlobal,
                        plotWatershedStepsMarkersOnly   =plotWatershedStepsMarkersOnly,
                        openingBeforeWaterRising        =openingBeforeWaterRising,
                        plotConvexityDefects            =plotConvexityDefects,
                        plotCentroids                   =plotCentroids,
                        plotExpansion                   =plotExpansion
                        )

                V_voxels[imSlice-offset,:,:]=voxelMap

                if plotCentroids:

                    V_img[imSlice-offset,:,:,:]=img
    else: #watershed was performed before, but could not proceed with following steps
        print("\ttemporaty file was found in disk, resuming from completed watershed transform...\n")
        with open(os.path.join(pathVolumes,'V_voxelMapTEMP.pickle'), "rb") as f:
            V_voxels  = pickle.load(f)       


    tocWatershed=time.perf_counter()

    times_centroids["watershedTransform"]=time.strftime("%Hh%Mm%Ss", time.gmtime(tocWatershed-ticWatershed))


    ###########################################################################################

    ### Centroid identification for each labeled region on the watershed output

    ###########################################################################################
    
    if manualRange is None:
        nSlices=range(V_voxels.shape[0])
    else:
        nSlices=manualRange

    ticCentroid=time.perf_counter()

    centroids={}

    if parallelHandle:
        # for large dataset, there is a strange bug where these processes 
        # hang at 0% cpu forever, without crashing. setting timeout to 10 min 
        # make this obvious, without solving the underlying cause
        # TODO remove this warning if manually teminating previous processes solved the problem
        centroidsTemp = Parallel(n_jobs=num_coresCentroid,timeout=600.)\
            (delayed(centroidExtractParallel)\
                (imSlice,nSlices,V_voxels[imSlice-offset,:,:],xMin,yMin,
                plotCentroids,
                img=V_img[imSlice-offset,:,:,:] if plotCentroids else None
                )for imSlice in nSlices )

        centroids={imSlice:centroidsTemp[i] for i,imSlice in enumerate(nSlices)}

    else:
        for imSlice in nSlices: 
            centroids[imSlice]=centroidExtractParallel(
                imSlice,
                nSlices,
                V_voxels[imSlice-offset,:,:],
                xMin,
                yMin,
                plotCentroids,
                img=V_img[imSlice-offset,:,:,:] if plotCentroids else None,
                )

    tocCentroid=time.perf_counter()
    times_centroids["centroidDetection only:"]=time.strftime("%Hh%Mm%Ss", time.gmtime(tocCentroid-ticCentroid))

    #delete temporary file
    deleteTempFile=not(checkIfFilesPresent(pathVolumes,'V_voxelMapTEMP.pickle'))
    if deleteTempFile:
        os.remove(os.path.join(pathVolumes,"V_voxelMapTEMP.pickle"))

    return V_voxels,centroids,xRes,unitTiff,descriptionDict,times_centroids

