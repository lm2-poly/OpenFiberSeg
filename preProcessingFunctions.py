# by Facundo Sosa-Rey, 2021. MIT license

from logging import root
import cv2 as cv
from tifffile import TiffFile
import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.patches as mpatches


from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk


from skimage import feature,morphology
from trackingFunctions import addFlaggedPixelsToImg,paddingOfImage


import os

def count_Tiff_Files(path, extension='.tiff'):
    nbFiles = 0
    fileList = os.listdir(path)
    length = len(fileList)
    for i in range(length):
        if os.path.splitext(fileList[i])[1] == extension:
            nbFiles += 1
    return(nbFiles)

def imshowoverlay(
    binaryMap,
    grayImg_hist,
    title=None,
    figureName=None,
    color=[200,20,50],
    makePlot=True,
    alpha=0.7, 
    withGUI=False,
    axisFill=False,
    figsize=[8,8]
    ):

    x,y=np.where(binaryMap==255)
    flaggedPixels=[]
    for iPix in range(len(x)):
        flaggedPixels.append((x[iPix],y[iPix]))

    # attenuate grayImg_hist to make it more readable
    
    oldRange=[0,255]
    newRange=[0,120]
    
    grayImg_hist=np.array(np.round(np.interp(grayImg_hist,oldRange,newRange)),np.uint8)

    imgComp = np.stack([grayImg_hist,grayImg_hist,grayImg_hist],axis=2)

    addFlaggedPixelsToImg(imgComp,flaggedPixels,color=color,alpha=alpha)

    if makePlot:
        if withGUI:
            fig = plt.figure(figsize=[6,6],num=figureName)
            plt.title(title,fontsize=10)
        else:
            plt.figure(figsize=figsize,num=figureName)
            plt.title(title,fontsize=28)

        plt.imshow(imgComp,cmap="ocean")
        if axisFill:
            plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
            plt.gca().set_position([0,0,1,1])
        else:
            plt.tight_layout()

    if withGUI:
        return imgComp, fig
    else:
        return imgComp

def imshowoverlay_RGB(
    binaryMap,
    grayImg_histRGB, 
    title=None,
    figureName=None,
    color=[200,20,50],
    makePlot=True,
    alpha=0.7, 
    withGUI=False,
    axisFill=False,
    figsize=[8,8]
    ):
    x,y=np.where(binaryMap==255)
    flaggedPixels=[]
    for iPix in range(len(x)):
        flaggedPixels.append((x[iPix],y[iPix]))

    imgTemp=grayImg_histRGB.copy()

    addFlaggedPixelsToImg(imgTemp,flaggedPixels,color=color,alpha=alpha)

    if makePlot:
        if withGUI:
            fig = plt.figure(figsize=[6,6],num=figureName)
            plt.title(title,fontsize=10)
        else:
            plt.figure(figsize=figsize,num=figureName)
            plt.title(title,fontsize=28)

        plt.imshow(imgTemp,cmap="ocean")
        if axisFill:
            plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
            plt.gca().set_position([0,0,1,1])
        else:
            plt.tight_layout()

    if withGUI:
        return imgTemp, fig
    else:
        return imgTemp

def histEqu_CannyDetection(filePath,
    imSlice,iFirst,iLast,
    pixelRangeX,pixelRangeY,
    findExternalPerimeter,
    findPores,
    thresholding_valPerim,
    Canny_sigma_perimeter,Canny_valLow_perimeter,Canny_valHigh_perimeter,SE_perim,
    Canny_sigma_pores,    Canny_valLow_pores,    Canny_valHigh_pores,
    plotCanny_perimeterDetection=False,
    plotCannyEdgeDetection=False,
    plotThresholding=False,
    withGUI=False):

    print("\t\thistEqu_CannyDetection(): imSlice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast) )

    with TiffFile(filePath) as tif:
        im=tif.asarray()

        # scaling to uint8 if necessary
        if im.dtype!=np.uint8:
            if im.dtype==np.uint16:
                im=np.array(tif.asarray()/65535*255,np.uint8)   # scaling from uin16 to uint8         
            else:
                raise TypeError("not implemented for dtype={}".format(im.dtype))
            
        

    im=im[pixelRangeX[0]-1:pixelRangeX[1],pixelRangeY[0]-1:pixelRangeY[1]]

    im_hist      =cv.equalizeHist(im)

    if findExternalPerimeter:

        retval, imgThresh	=	cv.threshold(im	, thresholding_valPerim , 255,cv.THRESH_BINARY)

        # padding is required so Canny edges do not reach image boundary when dilated,
        # and prevent floodfill to be complete
        padPerim=40
        imgThresh = paddingOfImage(imgThresh,paddingWidth=padPerim)

        if plotThresholding:
            if withGUI:
                fig_tresholding = plt.figure(figsize=[6,6])
                plt.title("Thresholding on original data, imslice={: >4.0f}".format(imSlice), fontsize=10)
            else:
                plt.figure(figsize=[16,16])
                plt.title("Thresholding on original data, imslice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast), fontsize=28)

            plt.imshow(imgThresh,cmap="binary")
            plt.tight_layout()


            if withGUI:
                fig_originalData= plt.figure(figsize=[6,6])
                plt.title("Original data, imslice={: >4.0f}".format(imSlice), fontsize=10)
            else:
                plt.figure(figsize=[16,16])
                plt.title("Original data, imslice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast), fontsize=28)
            
            plt.imshow(im,cmap="binary_r")
            plt.tight_layout()

            if not withGUI:
                plt.show()


        edgesPerimeter = feature.canny(imgThresh,sigma=Canny_sigma_perimeter, low_threshold=Canny_valLow_perimeter, high_threshold=Canny_valHigh_perimeter)

        edgesPerimeter = np.array(edgesPerimeter,np.uint8)*255

        # dilate edges to close the outer contour as much as possible
        perimeter=cv.dilate(edgesPerimeter,SE_perim)

        filledSlice=perimeter.copy()
        
        retval, filledSlice, mask, rect=cv.floodFill(filledSlice, mask=None, seedPoint=(0,0), newVal=255)

        # remove original edges from filled region
        filledSlice[perimeter==255]=0

        # dilate filled region to reach size of edges before edge dilation
        dilatedFilledSlice = cv.dilate(filledSlice,SE_perim)

        im_perim=dilatedFilledSlice[padPerim:-padPerim,padPerim:-padPerim]

        if plotCanny_perimeterDetection:
            if withGUI:
                imgTemp, fig_OutsidePerim = imshowoverlay(
                    dilatedFilledSlice[padPerim:-padPerim,padPerim:-padPerim],
                    im_hist, 
                    title="Labelling of outside perimeter, imSlice={: >4.0f}".format(imSlice),
                    color=[40,240,80],
                    makePlot=True,
                    alpha=0.4,
                    withGUI=True)
    
                temp, fig_Labelling=imshowoverlay_RGB(
                    edgesPerimeter[padPerim:-padPerim,padPerim:-padPerim],
                    imgTemp, 
                    title="Labelling of outside perimeter+Canny edge detection, imSlice={: >4.0f}".format(imSlice),
                    color=[255,40,40],
                    alpha=1.0,
                    withGUI=True)

            else:
                imgTemp=imshowoverlay(
                    dilatedFilledSlice[padPerim:-padPerim,padPerim:-padPerim],
                    im_hist, 
                    title="Labelling of outside perimeter, imSlice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast),
                    color=[40,240,80],
                    makePlot=True,
                    alpha=0.4)

                imshowoverlay_RGB(
                    edgesPerimeter[padPerim:-padPerim,padPerim:-padPerim],
                    imgTemp, 
                    title="Labelling of outside perimeter+Canny edge detection, imSlice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast),
                    color=[255,40,40],
                    alpha=1.0)

            patches=[
                mpatches.Patch(color=(255/255,40/255,40/255,1.), label="Canny: Perimeter Edge" ),
                mpatches.Patch(color=(40/255,240/255,80/255,0.4), label="Perimeter" ),
            ]
            
            if withGUI:
                # put those patched as legend-handles into the legend
                plt.legend(handles=patches,fontsize=8,framealpha=1.)
            else:
                # put those patched as legend-handles into the legend
                plt.legend(handles=patches,fontsize=26,framealpha=1.)                
                plt.show()

    else:
        im_perim=np.zeros(im.shape,np.uint8) # no detected perimeter

    if findPores:
        edgesPores = feature.canny(
            im_hist,
            sigma=Canny_sigma_pores, 
            low_threshold=Canny_valLow_pores, 
            high_threshold=Canny_valHigh_pores
            )

        edgesPores = np.array(edgesPores,np.uint8)*255
    else:
        edgesPores = np.zeros(im_hist.shape,np.uint8)


    if plotCannyEdgeDetection:
        if withGUI:
            temp, fig_ApplyPorosity = imshowoverlay(
                edgesPores,im_hist, 
                title="Output of Canny edge detection, applied to porosity, imSlice={: >4.0f}".format(imSlice),
                color=[240,240,40],
                alpha=0.9,
                withGUI=True)
        else:
            imshowoverlay(
                edgesPores,im_hist, 
                title="Output of Canny edge detection, applied to porosity, imSlice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast),
                color=[240,240,40],
                alpha=0.9
                )
        
        if not withGUI:
            plt.show()
            
    # Returning the different graph according to the one chosen with the GUI procedure
    if withGUI and plotCanny_perimeterDetection and not findPores and not plotCannyEdgeDetection:
        return (fig_tresholding, fig_originalData, fig_OutsidePerim, fig_Labelling)
    elif withGUI and plotCannyEdgeDetection and findPores and not plotCanny_perimeterDetection:
        return (fig_ApplyPorosity), im, im_hist, im_perim, edgesPores 
    elif withGUI and plotCanny_perimeterDetection and findPores and plotCannyEdgeDetection:
        return (fig_tresholding, fig_originalData, fig_OutsidePerim, fig_Labelling, fig_ApplyPorosity), im, im_hist, im_perim, edgesPores
    else:
        return im,im_hist,im_perim,edgesPores


def contourDetection(
    edgesPores,
    V_hist,
    imSlice,iFirst,iLast,
    SE_fills,
    SE_edges,
    SE_large,
    plotFloodFilling,
    withGUI=False):

    print("\t\tcontourDetection(): imSlice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast) )

    filledSlice=edgesPores.copy()
    # retval, image, mask, rect	=	cv.floodFill(	image, mask, seedPoint, newVal
    retval, filledSlice, mask, rect=cv.floodFill(filledSlice, mask=None, seedPoint=(0,0), newVal=255)

    filledSlice=cv.bitwise_not(filledSlice)

    edgesPores=edgesPores.copy()

    dilatedFilledSlice = cv.dilate(filledSlice,SE_fills)

    edgesPores[dilatedFilledSlice==255]=0 #remove edges that have already been filled, so they dont get distorted by dilation

    if plotFloodFilling:
        if withGUI:
            imgTemp=imshowoverlay(edgesPores,V_hist,color=[255,50,50],makePlot=False)

            temp, fig_firstPass = imshowoverlay_RGB(
                filledSlice,
                imgTemp, 
                title="First pass floodfill, imSlice={}".format(imSlice),
                color=[255,250,50],
                alpha=0.6,
                withGUI=True)
        else:
            imgTemp=imshowoverlay(edgesPores,V_hist,color=[255,50,50],makePlot=False)

            imshowoverlay_RGB(
                filledSlice,
                imgTemp, 
                title="First pass floodfill, imSlice={}".format(imSlice),
                color=[255,250,50],
                alpha=0.6
                )

            plt.show(block=False)


    edgesPores2=cv.dilate(edgesPores,SE_edges)

    # find coordinates inside a sufficiently large blob to start the filling from, so
    # edges that touch border of image are also filled (would not be filled if started at (0,0))
    y,x=np.where(cv.erode(filledSlice,SE_large)==255)

    if len(x)>0: # handles case where no pore is detected
        retval, filledSlice2, mask, rect=cv.floodFill(edgesPores2, mask=None, seedPoint=(x[0],y[0]), newVal=255)

        filledSlice2=cv.dilate(cv.bitwise_not(filledSlice2),SE_edges)

        dilatedFilledSlice2 = cv.dilate(filledSlice2,SE_fills)

        edgesPores[dilatedFilledSlice2==255]=0

        filledSlice=np.array(np.logical_or(filledSlice,filledSlice2),np.uint8)*255

    if plotFloodFilling:
        if withGUI:
            imgTemp=imshowoverlay(edgesPores,V_hist,color=[255,50,50],makePlot=False)

            temp, fig_MultiPass = imshowoverlay_RGB(
                filledSlice,
                imgTemp, 
                title="Multi-pass floodfill, imSlice={: >4.0f}".format(imSlice),
                color=[255,250,50],
                alpha=0.6,
                withGUI=True)

        else:
            imgTemp=imshowoverlay(edgesPores,V_hist,color=[255,50,50],makePlot=False)

            imshowoverlay_RGB(
                filledSlice,
                imgTemp, 
                title="Multi-pass floodfill, imSlice={: >4.0f},  in range ({: >4.0f}/{: >4.0f})".format(imSlice,iFirst,iLast),
                color=[255,250,50],
                alpha=0.6
                )

        if len(x)>0:
            plt.scatter(x[0],y[0],s=100,c='yellow',label="floodFillind start point")
            if withGUI:
                plt.legend(fontsize=8)
            else:
                plt.legend(fontsize=28)

        if not withGUI:
            plt.show()

    if withGUI and plotFloodFilling:
        return (fig_firstPass, fig_MultiPass), filledSlice
    else:
        return (filledSlice)


def paddingOfVolume(V,radius,paddingValue=255):

    paddedV_perim = np.ones((V.shape[0]+2*radius,V.shape[1]+2*radius,V.shape[2]+2*radius),np.uint8)*paddingValue
    
    for i in range(radius):
        paddedV_perim[radius:-radius,radius:-radius,i   ] = V[:,:, 0].copy()
        paddedV_perim[radius:-radius,radius:-radius,-i-1  ] = V[:,:,-1].copy()

                                               #included:-(excluded)   
    paddedV_perim[radius:-radius,radius:-radius,radius  :-radius    ] = V

    return paddedV_perim