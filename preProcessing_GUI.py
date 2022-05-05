###################################################
# Program to create GUI for the preProcessing.py
# Made by : Jean-Christophe Fronteddu, Facundo Sosa-Rey
# MIT license, 2021
###################################################

from tkinter import *  
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk

import numpy as np
import cv2 as cv
import tifffile
import os
import json

from datetime import date
from os import path
from matplotlib import pyplot as plt
from tifffile import TiffFile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage
from skimage import morphology

from preProcessingFunctions import histEqu_CannyDetection,imshowoverlay,contourDetection,paddingOfVolume,count_Tiff_Files
from extractCenterPoints import getTiffProperties


# Creating GUI window to select the folder path
#-----------------------------------------------
rootPathSelect = Tk()
rootPathSelect.title("Choose directory")
rootPathSelect.geometry('900x400')
rootPathSelect.config(bd=8)

figsize=[12,12] #manual setting of the figsize

# Creating the function to choose a directory
def choose_directory():
    entry.delete(0,END)
    global directory_path
    global params
    directory_path = filedialog.askdirectory(initialdir="./TomographicData")
    entry.insert(0, directory_path)
    if os.path.exists(directory_path+"/PreProcessingParams.json"):
        with open(directory_path+"/PreProcessingParams.json", "r") as f:
            params=json.load(f)
    else:
        #default parameters
        params={
            "ThresholdPerimeter":   50, 
            "PerimeterHigh":    50, 
            "PerimeterLow": 30, 
            "PerimeterSigma":   1., 
            "PoreHigh": 180, 
            "PoreLow":  60,
            "PoreSigma":    3,
            "SE_Canny_dilation_diameter":15,
            "SE_edges_diameter":9,
            "SE_fills_diameter":5,
            "SE_large_diameter":15,
            "SE_perim_3DOpening_radius":3,
            "SE_pores3d_radiusOpening":1,
            "SE_pores3d_radiusClosing":3 
        }


# Creating the function for the ok button when it is clicked
def ok_function():
    try:
        'directory_path' in globals()
        if path.isdir(directory_path):
            global valid
            valid = True
            entry.delete(0,END)
            rootPathSelect.destroy()
        else:
            messagebox.showerror("Error", "You have to enter a valid path")
            valid = False
    except:
        messagebox.showerror("Error", "You have to enter a valid path")
        valid = False



# Creating the function for the cancel button when it is clicked
def cancel_function():
    global valid
    valid = False
    entry.delete(0,END)
    rootPathSelect.destroy()
    if 'directory_path' in globals():
        global directory_path
        del directory_path

# Creating LABEL
info_lbl = Label(rootPathSelect, text="Please enter the directory path containing the sample tiff files")
info_lbl.grid(row=0, column=0, columnspan=2, sticky='W')
label = Label(rootPathSelect, text="path :", font=30)
label.grid(row=1, column=0, sticky='W')

# Creating ENTRY for the PATH
entry = Entry(rootPathSelect, width=50)
entry.grid(row=1, column=1, pady=10, sticky='E')

# Creating BUTTONS
browse_button = Button(rootPathSelect, text="Browse", width=50, height=2, font=30, command=choose_directory)
browse_button.grid(row=2, column=0, columnspan=2, sticky='W')
ok_button = Button(rootPathSelect, text="Ok", width=10, height=2, font=30, command=ok_function)
ok_button.grid(row=3, column=0, columnspan=2, sticky='W')
cancel_button = Button(rootPathSelect, text="Cancel", width=10, height=2, font=30, command=cancel_function)
cancel_button.grid(row=3, column=1, sticky='E')

rootPathSelect.mainloop()

#-----------------------------------------

if not valid:
    messagebox.showerror("Error", "No path were chosen")

# Creating GUI window to enter parameter for the program
# Creating the root window
root = Tk()
root.title("Enter Parameters values")
root.config(bd=5)

# Getting screen width and height for display
width = root.winfo_screenwidth()
height = root.winfo_screenheight()

# Setting the window size
root.geometry("%dx%d" % (width, height))

# Functions
#---------------------------------------------------------
# Creating the function for when the launch 2D button is clicked
def launch2d_command():        
    Graph2D = launch2d_thread_fct()
    show2dGraph(Graph2D)           

# Creating the function for when the launch 3D button is clicked
def launch3d_command():
    Graph3D = launch3d_thread_fct()
    show3dGraph(Graph3D)


def launch2d_thread_fct():
    global skip
    global result
    global filledSlice
    global V
    global V_hist
    global V_perim
    global V_pores
    global findExternalPerimeter
    global findPores
    global iFirst
    skip = False

    verify_number()
    verify_entries()

    # Deactivate Launch 2D and 3D button
    launch2d_btn.config(state=DISABLED)
    launch3d_btn.config(state=DISABLED)
    outPut_Data_btn.config(state=DISABLED)

    # Calling the function to verify and retrieve the parameters values
    values, advanced_values, nbSlice, dimension = validate_AllParam_command(launch=True)
    if valid:
        # Slice intervall
        iFirst  =nbSlice[0]
        iLast   =nbSlice[1]

        # Dimension
        pixelRangeX =[dimension[2], dimension[3]]
        pixelRangeY =[dimension[0], dimension[1]]
        
        # Creating the variable needed for the next step in the program
        nData=iLast-iFirst+1
        numPixX=pixelRangeX[1]-pixelRangeX[0]+1
        numPixY=pixelRangeY[1]-pixelRangeY[0]+1
        edgesPores=np.empty((numPixX,numPixY,nData),np.uint8)
        V         =np.zeros((numPixX,numPixY,nData),np.uint8)
        V_pores   =np.empty((numPixX,numPixY,nData),np.uint8)
        V_hist    =np.empty((numPixX,numPixY,nData),np.uint8)
        V_perim   =np.empty((numPixX,numPixY,nData),np.uint8)

        # Calling the function to make the Graphs
        findExternalPerimeter =findExternalPerimeter_check
        findPores             =findPores_check

        if findExternalPerimeter and not findPores:
            # Calling the function to make all the calculation to generates the graph
            result, fig_CannyDetection = calculate2dGraph(values, advanced_values, nbSlice, dimension,
                    edgesPores, V, V_pores, V_hist, V_perim, findExternalPerimeter, findPores,figsize=figsize)
            return (fig_CannyDetection)
        elif not findExternalPerimeter and findPores:
            # Calling the function to make all the calculation to generates the graph
            result, fig_ContourDetection = calculate2dGraph(values, advanced_values, nbSlice, dimension,
                    edgesPores, V, V_pores, V_hist, V_perim, findExternalPerimeter, findPores,figsize=figsize)
            return (fig_ContourDetection)
        elif findExternalPerimeter and findPores:
            # Calling the function to make all the calculation to generates the graph
            result, fig_CannyDetection, fig_ContourDetection = calculate2dGraph(values, advanced_values, nbSlice, dimension,
                    edgesPores, V, V_pores, V_hist, V_perim, findExternalPerimeter, findPores,figsize=figsize)
            return (fig_CannyDetection, fig_ContourDetection)                               
                

def launch3d_thread_fct():
    global skip
    global findExternalPerimeter
    global findPores
    skip = False
    
    # Deactivate Launch 2D and 3D button
    launch2d_btn.config(state=DISABLED)
    launch3d_btn.config(state=DISABLED)
    outPut_Data_btn.config(state=DISABLED)

    # Calling the function to verify and retrieve all the parameters
    values, advanced_values, nbSlice, dimension = validate_AllParam_command(launch=True)

    # Calling the function to make the Graphs
    findExternalPerimeter =findExternalPerimeter_check
    findPores             =findPores_check

    if valid:
        if 'launch2d' not in globals():
            global launch2d
            launch2d = False                
        if launch2d:
            global result
            edgesPores = result[0]
            V_pores = result[1]
            V_hist = result[2]
            V_perim = result[3]
            launch2d = False

            # Calling the function to calculate the 3D graph
            fig_volumetricProcessing = calculate3dGraph(values, advanced_values, nbSlice, dimension,
                    edgesPores, V_pores, V_hist, V_perim, findExternalPerimeter, findPores,figsize=figsize)

                            

        elif not launch2d:
            # Slice intervall
            iFirst  =nbSlice[0]
            iLast   =nbSlice[1]

            #Dimension
            pixelRangeX =[dimension[2], dimension[3]]
            pixelRangeY =[dimension[0], dimension[1]]
            
            nData=iLast-iFirst+1
            numPixX=pixelRangeX[1]-pixelRangeX[0]+1
            numPixY=pixelRangeY[1]-pixelRangeY[0]+1
            edgesPores=np.empty((numPixX,numPixY,nData),np.uint8)
            V         =np.zeros((numPixX,numPixY,nData),np.uint8)
            V_pores   =np.empty((numPixX,numPixY,nData),np.uint8)
            V_hist    =np.empty((numPixX,numPixY,nData),np.uint8)
            V_perim   =np.empty((numPixX,numPixY,nData),np.uint8)

            result = calculate2dGraph(values, advanced_values, nbSlice, dimension,
                    edgesPores, V, V_pores, V_hist, V_perim, findExternalPerimeter, findPores,
                    show2d=False,figsize=figsize)

            edgesPores = result[0]
            V_pores = result[1]
            V_hist = result[2]
            V_perim = result[3]

            # Calling the function to make the Graphs
            findExternalPerimeter =findExternalPerimeter_check
            findPores             =findPores_check

            fig_volumetricProcessing = calculate3dGraph(values, advanced_values, nbSlice, dimension,
                    edgesPores, V_pores, V_hist, V_perim, findExternalPerimeter, findPores,figsize=figsize)
        
        return fig_volumetricProcessing                                       

def calculate2dGraph(values, advanced_values, nbSlice, dimension, 
        edgesPores, V, V_pores, V_hist, V_perim, findExternalPerimeter, findPores, 
        show2d=True,figsize=[8,8]):

    # Slice intervall
    iFirst  =nbSlice[0]
    iLast   =nbSlice[1]
    offset = iFirst

    #Dimension
    pixelRangeX =[dimension[2], dimension[3]]
    pixelRangeY =[dimension[0], dimension[1]]

    # Parameter from Values 
    Canny_valLow_perimeter  =values[2]        
    Canny_valHigh_perimeter =values[1]    
    Canny_sigma_perimeter   =values[3]     
    Canny_valLow_pores      =values[5]          
    Canny_valHigh_pores     =values[4]        
    Canny_sigma_pores       =values[6]       

    # value in original data to use in thresholding for perimeter detection
    thresholding_valPerim   =values[0]       

    # Parameter from Advance values
    SE_Canny_dilation_diameter    =advanced_values[0]            
    SE_edges_diameter             =advanced_values[1]            
    SE_fills_diameter             =advanced_values[2]            
    SE_large_diameter             =advanced_values[3]            

    # Calling the function to make the Graphs
    plotThresholding                =True
    plotCannyEdgeDetection          =True
    plotFloodFilling                =True
    plotCanny_perimeterDetection    =True

    dilatePerimOverPores=True #leave at True for the majority of cases: prevents the outer boundary (perimeter) to be encircled by a closed contour, 
        # and thus the entire volume be labelled as "pore". For some rare cases, can be better to deactivate it, to capture the pores in contact with 
        # the boundary (especially when the scanning cylinder is entirely contained in the solid. )

    if findExternalPerimeter:
        SE_Canny_dilation = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize=(SE_Canny_dilation_diameter, SE_Canny_dilation_diameter))
        SE_Canny_dilation[:, 0]=SE_Canny_dilation[ 0,:]
        SE_Canny_dilation[:,-1]=SE_Canny_dilation[-1,:]
    else:
        SE_Canny_dilation=None

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
    
    fig_CannyDetection = []
    for imSlice in range(iFirst, iLast+1):
        fig_CannyDetection_imSlice,\
        V[:,:,imSlice-offset],\
        V_hist[:,:,imSlice-offset],\
        V_perim[:,:,imSlice-offset],\
        edgesPores[:,:,imSlice-offset]=histEqu_CannyDetection(
                directory_path+'/'+filename[imSlice],
                imSlice,iFirst,iLast,pixelRangeX,pixelRangeY,
                findExternalPerimeter,
                findPores,
                thresholding_valPerim,
                Canny_sigma_perimeter,Canny_valLow_perimeter,Canny_valHigh_perimeter,SE_Canny_dilation,
                Canny_sigma_pores,    Canny_valLow_pores,    Canny_valHigh_pores,
                plotCanny_perimeterDetection=plotCanny_perimeterDetection,
                plotCannyEdgeDetection=plotCannyEdgeDetection,
                plotThresholding=plotThresholding,
                withGUI=True,
                figsize=figsize
                ) 
        fig_CannyDetection.append(fig_CannyDetection_imSlice)
        plt.close()

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

    if findPores:
        fig_ContourDetection = []
        filledSlice = []
        for imSlice in range(iFirst, iLast+1):
            fig_ContourDetection_imSlice, filledSlice_imSlice=contourDetection(edgesPores[:,:,imSlice-offset],
                V_hist[:,:,imSlice-offset],
                imSlice,iFirst,iLast,
                SE_fills,
                SE_edges,
                SE_large,
                plotFloodFilling,
                withGUI=True,
                figsize=figsize)
            fig_ContourDetection.append(fig_ContourDetection_imSlice)
            filledSlice.append(filledSlice_imSlice)
            plt.close()
        for imSlice,resultTuple in enumerate(filledSlice):
            V_pores[:,:,imSlice]=resultTuple
    
    if show2d and findExternalPerimeter and not findPores:
        return (edgesPores, V_pores, V_hist, V_perim), fig_CannyDetection
    elif show2d and not findExternalPerimeter and findPores:
        return (edgesPores, V_pores, V_hist, V_perim), fig_ContourDetection
    elif show2d and findExternalPerimeter and findPores:
        return  (edgesPores, V_pores, V_hist, V_perim), fig_CannyDetection, fig_ContourDetection
    elif  not show2d:
        return (edgesPores, V_pores, V_hist, V_perim)

def calculate3dGraph(values, advanced_values, nbSlice, dimension,
        edgesPores, V_pores, V_hist, V_perim, findExternalPerimeter, findPores,figsize=[8,8]):
    # Slice intervall
    iFirst  =nbSlice[0]
    iLast   =nbSlice[1]
    offset = iFirst        
    
    # Parameter from Advance values
    SE_Canny_dilation_diameter             =advanced_values[0]            
    SE_perim_3DOpening_radius   =advanced_values[4]  
    SE_pores3d_radiusOpening    =advanced_values[5]   
    SE_pores3d_radiusClosing    =advanced_values[6]

        # Calling the function to make the Graphs
    plotOpening_perimeterDetection  =True #volumetric processing
    plotOpening_Closing_pores       =True #volumetric processing

    if findExternalPerimeter:
        #this step removes false positive: thin regions that spills from the perimeter to inside the sample
        SE_ball3D=morphology.ball(SE_perim_3DOpening_radius, dtype=np.uint8)
        paddingSize=SE_Canny_dilation_diameter # to avoid artifacts on corners after opening

        # padding on all sides is necessary because ball SE cannot reach side pixels
        paddedV_perim=paddingOfVolume(V_perim,paddingSize)
        paddedV_perim_opened=np.array(ndimage.binary_opening(paddedV_perim,SE_ball3D),np.uint8)*255

        if plotOpening_perimeterDetection: #volumetric processing
            fig_paddedV = []
            fig_Opening3d = []
            for imSlice in range(iFirst, iLast+1):
                temp, fig_paddedV_imSlice = imshowoverlay(
                    paddedV_perim[paddingSize:-paddingSize,paddingSize:-paddingSize,imSlice-offset+paddingSize],
                    V_hist[:,:,imSlice-offset],
                    color=[0,233,77],
                    title="paddedV_perim, processed in 2D, imslice={: >4.0f}".format(imSlice),
                    alpha=0.4,
                    withGUI=True,
                    figsize=figsize
                    )

                temp, fig_Opening3d_imSlice = imshowoverlay(
                    paddedV_perim_opened[paddingSize:-paddingSize,paddingSize:-paddingSize,imSlice-offset+paddingSize],
                    V_hist[:,:,imSlice-offset],
                    color=[255,120,50],
                    title="Opening in 3D, imslice={: >4.0f}".format(imSlice),
                    alpha=0.4,
                    withGUI=True,
                    figsize=figsize
                    ) 

                fig_paddedV.append(fig_paddedV_imSlice)
                fig_Opening3d.append(fig_Opening3d_imSlice)
                plt.close()

    if findPores:

        paddingSize=max(SE_pores3d_radiusOpening,SE_pores3d_radiusClosing) # to avoid artifacts on corners after opening and closing
        SE_ball3D_opening=morphology.ball(SE_pores3d_radiusOpening, dtype=np.uint8)
        paddedV_pores=paddingOfVolume(V_pores,paddingSize,paddingValue=0)
        temp=np.array(ndimage.binary_opening(paddedV_pores,SE_ball3D_opening),np.uint8)*255

        SE_ball3D_closing=morphology.ball(SE_pores3d_radiusClosing, dtype=np.uint8)
        paddedV_pores_opened_closed=np.array(ndimage.binary_closing(temp,SE_ball3D_closing),np.uint8)*255
        
        if plotOpening_Closing_pores: #volumetric processing
            fig_process2D = []
            fig_Closing3d = []
            for imSlice in range(iFirst, iLast+1):
                temp, fig_process2D_imSlice = imshowoverlay(
                        V_pores[:,:,imSlice-offset],
                        V_hist[:,:,imSlice-offset],
                        color=[0,233,77],
                        title="Processing in 2D, imslice={: >4.0f}".format(imSlice),
                        alpha=0.4,
                        withGUI=True,
                        figsize=figsize
                        )

                temp, fig_Closing3d_imSlice = imshowoverlay(
                        paddedV_pores_opened_closed[paddingSize:-paddingSize,paddingSize:-paddingSize,imSlice-offset+paddingSize],
                        V_hist[:,:,imSlice-offset],
                        color=[255,120,50],
                        title="closind in 3D, imslice={: >4.0f}".format(imSlice),
                        alpha=0.4,
                        withGUI=True,
                        figsize=figsize
                        )

                fig_process2D.append(fig_process2D_imSlice)
                fig_Closing3d.append(fig_Closing3d_imSlice)
                plt.close()

        if findExternalPerimeter and not findPores:
            return (fig_paddedV, fig_Opening3d)
        elif not findExternalPerimeter and findPores:
            return (fig_process2D, fig_Closing3d)
        elif findExternalPerimeter and findPores:
            return (fig_paddedV, fig_Opening3d, fig_process2D, fig_Closing3d)

def show2dGraph(Graph2D):
    fig_CannyDetection = Graph2D[0]
    fig_ContourDetection = Graph2D[1]
    graph_canvas.delete('all')
    # Creating the place to show the graph on the window
    # Creating Scrollbar for the graph
    if 'scrollbar' not in globals():
        graph_scrollbarY = Scrollbar(graph_frame, orient="vertical", command=graph_canvas.yview)
        graph_scrollbarY.pack(side="right", fill=Y)

        graph_scrollbarX = Scrollbar(graph_frame, orient="horizontal", command=graph_canvas.xview)
        graph_scrollbarX.pack(side="bottom", fill=X)

        # Configure Canvas
        graph_canvas.configure(xscrollcommand=graph_scrollbarX.set,yscrollcommand=graph_scrollbarY.set)

        graph_canvas.configure(scrollregion=graph_canvas.bbox("all"))

        # Creating scrollbar variable to indicate that the scrollbar has been created
        global scrollbar
        scrollbar = True


    # Creating a second graph frame to hold the new window
    graph2_frame = Frame(graph_canvas)
    graph2_frame.pack(fill=BOTH, expand=1)

    iFirst = nbSlice[0]
    iLast = nbSlice[1]

    for imSlice in range(iFirst, iLast+1):
        # Creating the new window to display with the scroll
        graph_canvas.create_window((0,0), window=graph2_frame, anchor='nw')

        # Input Data
        canvas = FigureCanvasTkAgg(fig_CannyDetection[imSlice-iFirst][1], graph2_frame)
        canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

        # Threshold
        canvas = FigureCanvasTkAgg(fig_CannyDetection[imSlice-iFirst][0], graph2_frame)
        canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)


        if findExternalPerimeter:
            # # perimeter alone
            # canvas = FigureCanvasTkAgg(fig_CannyDetection[imSlice-iFirst][2], graph2_frame)
            # canvas.get_tk_widget().grid(row=2, column=0, padx=5, pady=5)

            # Perimeter detection
            canvas = FigureCanvasTkAgg(fig_CannyDetection[imSlice-iFirst][3], graph2_frame)
            canvas.get_tk_widget().grid(row=0, column=2, padx=5, pady=5)
            
        if findPores:
            # Canny contours for pores
            canvas = FigureCanvasTkAgg(fig_CannyDetection[imSlice-iFirst][4], graph2_frame)
            canvas.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)

            # First Pass Floodfill
            canvas = FigureCanvasTkAgg(fig_ContourDetection[imSlice-iFirst][0], graph2_frame)
            canvas.get_tk_widget().grid(row=1, column=1, padx=5, pady=5)

            # Multipass floodfill
            canvas = FigureCanvasTkAgg(fig_ContourDetection[imSlice-iFirst][1], graph2_frame)
            canvas.get_tk_widget().grid(row=1, column=2, padx=5, pady=5)                                        

        canvas.draw() 

        if imSlice == iLast:
            messagebox.showinfo("Information", "Click on launch 3D to start the volumetric processing")
            next_btn.config(state=DISABLED)
            skip_btn.config(state=DISABLED)
            launch2d_btn.config(state=NORMAL)
            launch3d_btn.config(state=NORMAL)                        
        elif imSlice < iLast and not skip:
            messagebox.showinfo("Information", "Click on Next to show the imSlice %d / %d graph or click on Skip to jump to the last imSlice" % (imSlice+1, iLast))
            skip_btn.config(state=NORMAL)
            next_btn.config(state=NORMAL)
            launch2d_btn.config(state=NORMAL)
            # Creating a Wait for the button next to be clicked
            next_btn.wait_variable(click)

def show3dGraph(fig_volumetricProcessing):        
    iFirst = nbSlice[0]
    iLast = nbSlice[1]

    graph_canvas.delete('all')
        
    if 'scrollbar' not in globals():
        graph_scrollbarY = Scrollbar(graph_frame, orient="vertical", command=graph_canvas.yview)
        graph_scrollbarY.pack(side="right", fill=Y)

        graph_scrollbarX = Scrollbar(graph_frame, orient="horizontal", command=graph_canvas.xview)
        graph_scrollbarX.pack(side="bottom", fill=X)

        # Configure Canvas
        graph_canvas.configure(xscrollcommand=graph_scrollbarX.set,yscrollcommand=graph_scrollbarY.set)

        graph_canvas.configure(scrollregion=graph_canvas.bbox("all"))

        # Creating scrollbar variable to indicate that the scrollbar has been created
        global scrollbar
        scrollbar = True
        
    # Creating a second graph frame to hold the new window

    graph2_frame = Frame(graph_canvas)
    graph2_frame.pack(fill=BOTH, expand=1)
    
    for imSlice in range(iFirst, iLast+1):
        if findExternalPerimeter and not findPores:
            # Creating the new window to display with the scroll
            graph_canvas.create_window((0,0), window=graph2_frame, anchor='nw')
            canvas = FigureCanvasTkAgg(fig_volumetricProcessing[0][imSlice-iFirst], graph2_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
            canvas = FigureCanvasTkAgg(fig_volumetricProcessing[1][imSlice-iFirst], graph2_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)
        elif not findExternalPerimeter and findPores:
            canvas = FigureCanvasTkAgg(fig_volumetricProcessing[2][imSlice-iFirst], graph2_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
            canvas = FigureCanvasTkAgg(fig_volumetricProcessing[3][imSlice-iFirst], graph2_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)                 
        elif findExternalPerimeter and findPores:
            graph_canvas.create_window((0,0), window=graph2_frame, anchor='nw')
            canvas = FigureCanvasTkAgg(fig_volumetricProcessing[0][imSlice-iFirst], graph2_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
            canvas = FigureCanvasTkAgg(fig_volumetricProcessing[1][imSlice-iFirst], graph2_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)
            canvas = FigureCanvasTkAgg(fig_volumetricProcessing[2][imSlice-iFirst], graph2_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)
            canvas = FigureCanvasTkAgg(fig_volumetricProcessing[3][imSlice-iFirst], graph2_frame)
            canvas.draw()
            canvas.get_tk_widget().grid(row=1, column=1, padx=5, pady=5)
        if imSlice == 3:
            messagebox.showinfo("Information", "Click on launch 2D to restart the Canny process or click on OutPut Data to create final graph")
            next_btn.config(state=DISABLED)
            skip_btn.config(state=DISABLED)
            launch2d_btn.config(state=NORMAL)
            launch3d_btn.config(state=NORMAL)
            outPut_Data_btn.config(state=NORMAL)                    
        elif imSlice < 3 and not skip:
            messagebox.showinfo("Information", "Click on Next to show the imSlice %d graph or click on Skip to jump to the last imSlice" % (imSlice+1))
            skip_btn.config(state=NORMAL)
            next_btn.config(state=NORMAL)
            launch3d_btn.config(state=NORMAL)
            # Creating a Wait for the button next to be clicked
            next_btn.wait_variable(click)  

def outPut_Date_command():
    nbSlice = validate_AllParam_command(launch=True, OutPutData=True)
    iFirst = nbSlice[0]
    iLast = nbSlice[1]
    response = messagebox.askquestion("Warning", "Are you sure you want to outPut data from the %d slice to the %d slice" % (iFirst, iLast))
    
    if valid and response == 'yes':
        today   = date.today()
        dateStr = today.strftime("%b-%d-%Y")
        outputFolderName="preProcessed_"+dateStr
        descriptionStr="{"+"\"shape([x,y,z])\":[{},{},{}]".format(*(V.shape))+"}"
        
        # if the output directory doesn't exist, it is created here:
        filesInDir = [f.path for f in os.scandir(directory_path) if f.is_dir()]
        if directory_path+outputFolderName not in filesInDir:
            os.mkdir(directory_path+'/'+outputFolderName)

        with TiffFile(directory_path+'/'+filename[iFirst]) as tif:
            xRes,unitTiff=getTiffProperties(tif)

        tifffile.imwrite(
            directory_path+'/'+outputFolderName+'/'+'V_hist.tiff',
            np.transpose(V_hist,(2,0,1)),
            resolution=(xRes,xRes,unitTiff),
            compress=True,
            description=descriptionStr
            )

        tifffile.imwrite(
            directory_path+'/'+outputFolderName+'/'+'V_pores.tiff',
            np.transpose(V_pores,(2,0,1)),
            resolution=(xRes,xRes,unitTiff),
            compress=True,
            description=descriptionStr
            )

        if findExternalPerimeter:
        #the next step (InsegtFibre_3D) allows for V_perim to be absent
            tifffile.imwrite(
                directory_path+'/'+outputFolderName+'/'+'V_perim.tiff',
                np.transpose(V_perim,(2,0,1)),
                resolution=(xRes,xRes,unitTiff),
                compress=True,
                description=descriptionStr
                )            

def skip_command():
    global skip
    skip = True
    messagebox.showinfo("Information", "Click on Next to show the last imSlice graph")

def verify_number():
    global valid
    try:
        # Verify all Parameter entered
        int(ThresholdPerimeter_entry.get())
        int(PerimeterHigh_entry.get())
        int(PerimeterLow_entry.get())
        float(PerimeterSigma_entry.get())
        int(PoreHigh_entry.get())
        int(PoreLow_entry.get())
        float(PoreSigma_entry.get())
        # Verify all Advance Parameter entered
        int(SE_Canny_dilation_diameter_entry.get())
        int(SE_edges_diameter_entry.get())
        int(SE_fills_diameter_entry.get())
        int(SE_large_diameter_entry.get())
        int(SE_perim_3DOpening_radius_entry.get())
        # Verify number of slice entered
        if mode_nbSlice == 0:
            int(firstSlice_entry.get())
            int(lastSlice_entry.get())
        # Verify X and Y value for section entered
        if mode_sampleSize == 1:
            int(x_from_entry.get())
            int(x_to_entry.get())
            int(y_from_entry.get())
            int(y_to_entry.get())
        # Value returned to ensure all entries are integers
        valid = True

    except:
        messagebox.showerror("Error", "You have to enter a number")
        valid = False

# Creating the function to verify that the values entered are in the different intervals
def verify_entries(launch=False): 
    global values
    global advanced_values
    global nbSlice
    global valid       
    values=[int(ThresholdPerimeter_entry.get()), 
            int(PerimeterHigh_entry.get()), 
            int(PerimeterLow_entry.get()), 
            float(PerimeterSigma_entry.get()), 
            int(PoreHigh_entry.get()), 
            int(PoreLow_entry.get()),
            float(PoreSigma_entry.get())]
    advanced_values=[int(SE_Canny_dilation_diameter_entry.get()),
                    int(SE_edges_diameter_entry.get()),
                    int(SE_edges_diameter_entry.get()),
                    int(SE_large_diameter_entry.get()),
                    int(SE_perim_3DOpening_radius_entry.get()),
                    int(SE_pores3d_radiusOpening_entry.get()),
                    int(SE_pores3d_radiusClosing_entry.get())]
    nbSlice = [int(firstSlice_entry.get()), int(lastSlice_entry.get())]
    nbTiffFiles = count_Tiff_Files(directory_path)
    if values[0] < 0 or values[0] > 255:
        messagebox.showerror("Error", "Threshold Perimeter : %d You have to enter a number between [0, 255]" %(values[0]))
        valid = False
    elif values[1] < 0 or values[1] > 255:
        messagebox.showerror("Error", "Perimeter high : %d You have to enter a number between [0, 255]" %(values[1]))
        valid = False
    elif values[2] < 0 or values[2] > 255:
        messagebox.showerror("Error", "Perimeter low : %d You have to enter a number between [0, 255]" %(values[2]))
        valid = False        
    elif values[1] <= values[2]:
        messagebox.showerror("Error", "Perimeter high : %d needs to be greater than Perimeter low : %d" %(values[1], values[2]))
        valid = False
    elif values[3] < 0 or values[3] > 10:
        messagebox.showerror("Error", "Perimeter Sigma : %d You have to enter a number between [0, 10]" %(values[3]))
        valid = False
    elif values[4] < 0 or values[4] > 255:
        messagebox.showerror("Error", "Pore high : %d You have to enter a number between [0, 255]" %(values[4]))
        valid = False
    elif values[5] < 0 or values[5] > 255:
        messagebox.showerror("Error", "Pore low : %d You have to enter a number between [0, 255]" %(values[5]))
        valid = False
    elif values[4] <= values[5]:
        messagebox.showerror("Error", "Pores high : %d needs to be greater than Pores low : %d" %(values[4], values[5]))
        valid = False
    elif values[6] < 0 or values[6] >= 10: 
        messagebox.showerror("Error", "Pore Sigma : %d You have to enter a number between [0, 10]" %(values[6]))
        valid = False
    elif advanced_values[0] < 0 or advanced_values[0] > 100:
        messagebox.showerror("Error", "SE perimeter diameter : %d You have to enter a number between [0, 100]" %(advanced_values[0]))
        valid = False
    elif advanced_values[1] < 0 or advanced_values[1] > 100:
        messagebox.showerror("Error", "SE edges diameter : %d You have to enter a number between [0, 100]" %(advanced_values[1]))
        valid = False
    elif advanced_values[2] < 0 or advanced_values[2] > 100:
        messagebox.showerror("Error", "SE fills diameter : %d You have to enter a number between [0, 100]" %(advanced_values[2]))
        valid = False
    elif advanced_values[3] < 0 or advanced_values[3] > 100:
        messagebox.showerror("Error", "SE large diameter : %d You have to enter a number between [0, 100]" %(advanced_values[3]))
        valid = False
    elif advanced_values[4] < 0 or advanced_values[4] > 100:
        messagebox.showerror("Error", "SE perimeter 3d opening radius : %d You have to enter a number between [0, 100]" %(advanced_values[4]))
        valid = False
    elif advanced_values[5] < 0 or advanced_values[4] > 100:
        messagebox.showerror("Error", "SE pores 3d opening radius : %d You have to enter a number between [0, 100]" %(advanced_values[5]))
        valid = False
    elif advanced_values[6] < 0 or advanced_values[4] > 100:
        messagebox.showerror("Error", "SE pores 3d closing radius : %d You have to enter a number between [0, 100]" %(advanced_values[6]))
        valid = False
    elif nbSlice[0] <= 0:
        messagebox.showerror("Error", "You entered %d for first slice. It needs to be greater than 0" %(nbSlice[0]))
    elif nbSlice[1] > nbTiffFiles:
        messagebox.showerror("Error", "You entered %d for last slice. It needs to be smaller than %d" %(nbSlice[1], nbTiffFiles))        
    elif nbSlice[1] <= nbSlice[0]:
        messagebox.showerror("Error", "Last slice : %d needs to be greater than first slice : %d" %(nbSlice[1], nbSlice[0])) 
    else:        
        valid = True

        params={
            "ThresholdPerimeter":   values[0], 
            "PerimeterHigh":    values[1], 
            "PerimeterLow": values[2], 
            "PerimeterSigma":   values[3], 
            "PoreHigh": values[4], 
            "PoreLow":  values[5],
            "PoreSigma":    values[6],
            "SE_Canny_dilation_diameter":advanced_values[0],
            "SE_edges_diameter":advanced_values[1],
            "SE_edges_diameter":advanced_values[2],
            "SE_large_diameter":advanced_values[3],
            "SE_perim_3DOpening_radius":advanced_values[4],
            "SE_pores3d_radiusOpening":advanced_values[5],
            "SE_pores3d_radiusClosing":advanced_values[6] 
        }
        with open(directory_path+"/PreProcessingParams.json", "w") as f:
            params=json.dump(params, f, sort_keys=True, indent=4)

        if launch:
            return values, advanced_values, nbSlice

# Creating the function that verifies the validity of the Radio choice entries
def verify_radioChoice(launch=False):
    global valid
    global dimension
    global filename
    global nbSlice

    # Slice intervall
    nbSlice = [int(firstSlice_entry.get()), int(lastSlice_entry.get())]
    iFirst = nbSlice[0]
    iLast = nbSlice[1]
    # Getting tiff dimension
    formatStr="{:0>4.0f}.tiff" 
    filename={}
    filename={imSlice:formatStr.format(imSlice) for imSlice in range(iFirst,iLast+1) }
    with TiffFile(directory_path+'/'+filename[iFirst]) as tif:
        dim = getTiffProperties(tif, getDimensions=True)
        dimension = [1, dim[1], 1, dim[0]]    
    if mode_sampleSize.get() == 0:
        valid = True
        return dimension        
    elif mode_sampleSize.get() == 1:
        dim = [int(y_from_entry.get()), int(y_to_entry.get()),
                int(x_from_entry.get()), int(x_to_entry.get())]
        if dimension[3] < dim[3] or dimension[1] < dim[1]:
            messagebox.showerror("Error", "You entered : %dx%d It need smaller than %dx%d" %(dim[3], dim[1], dimension[3],dimension[1]))
            valid = False
        elif 0 >= dim[2] or 0 >= dim[0]:
            messagebox.showerror("Error", "You entered x from : %d and y from : %d They need to be greater than 0" %(dim[2], dim[0]))
            valid= False
        elif dim[2] >= dim[3] or dim[0] >= dim[1]:
            messagebox.showerror("Error", "You entered  x: %d < %d] and y: %d < %d They need to be valid intervals" %(dim[2], dim[3], dim[0], dim[1]))
            valid = False
        elif (dim[2] < dim[3] and dim[0] < dim[1] and dimension[3] >= dim[3]
            and dimension[1] >= dim[1] and 0 < dim[2] and 0 < dim[0]):
            dimension = dim
            valid = True
            if launch:
                return dimension
    
# Creating the function for when the reset button is clicked
def validate_AllParam_command(launch=False, OutPutData = False):
    if launch:
        verify_number()
        if valid:
            values, advanced_values, nbSlice =  verify_entries(launch=True)
            if valid:
                dimension = verify_radioChoice(launch=True)
                if OutPutData:
                    return nbSlice
                elif not OutPutData:
                    return values, advanced_values, nbSlice, dimension
    elif not launch and not OutPutData:
        verify_number()
        if valid:
            verify_entries()
            if valid:
                verify_radioChoice()
                if valid:
                    messagebox.showinfo("Validation", "All parameter are valid")

# Creating the function for when the cancel button is clicked
def cancel_command():
    root.destroy()

# Creating the function for the nbSlice radio button
def radioChoice_nbSlice():
    if mode_nbSlice.get() == 0: # radio button Custom is chosen
        radio_nbSlice_frame.grid()
    elif mode_nbSlice.get() == 1: # radio button All is chosen
        radio_nbSlice_frame.grid_remove()            

# Creating the function for sampleSize radio button
def radioChoice_sampleSize():
    if mode_sampleSize.get() == 0: # radio button All is chosen
        radio_sampleSize_frame.grid_remove()            
    elif mode_sampleSize.get() == 1: # radio button Custom is chosen
        radio_sampleSize_frame.grid()

# Creating the function for when the Advance button is clicked
def advanced_command():
    Advanced_Parameter_frame.grid()

# Creating the function for when the Hide button is clicked
def hide_command():
    Advanced_Parameter_frame.grid_remove()

# Creating the function for when the Ok button is clicked
def validate_AdvancedParam_command():
    global advanced_values
    global valid
    # Verify that Entries are numbers
    try:
        int(SE_Canny_dilation_diameter_entry.get())
        int(SE_edges_diameter_entry.get())
        int(SE_fills_diameter_entry.get())
        int(SE_large_diameter_entry.get())
        int(SE_perim_3DOpening_radius_entry.get())
        valid = True
    
    except:
        messagebox.showerror("Error", "You have to enter a number")
        valid = False
                
    if valid:
        advanced_values = [int(SE_Canny_dilation_diameter_entry.get()),
                        int(SE_edges_diameter_entry.get()),
                        int(SE_edges_diameter_entry.get()),
                        int(SE_large_diameter_entry.get()),
                        int(SE_perim_3DOpening_radius_entry.get())]
        if advanced_values[0] < 0 or advanced_values[0] > 255:
            messagebox.showerror("Error", "For Pore low you have to enter a number between [0, 255]")
            valid = False
        elif advanced_values[1] < 0 or advanced_values[1] > 255:
            messagebox.showerror("Error", "For Pore low you have to enter a number between [0, 255]")
            valid = False
        elif advanced_values[2] < 0 or advanced_values[2] > 255:
            messagebox.showerror("Error", "For Pore low you have to enter a number between [0, 255]")
            valid = False
        elif advanced_values[3] < 0 or advanced_values[3] > 255:
            messagebox.showerror("Error", "For Pore low you have to enter a number between [0, 255]")
            valid = False
        elif advanced_values[4] < 0 or advanced_values[4] > 255:
            messagebox.showerror("Error", "For Pore low you have to enter a number between [0, 255]")
            valid = False
        else:
            messagebox.showinfo("Validation", "All advanced parameters values are valid")
            valid = True

def info_request_command(infoRequest):
    if infoRequest == 1:
        messagebox.showinfo("Information", "Threshold Perimeter : greyscale value, serve to create a binary map (black/white) on which the Canny detection of the perimeter will occur.")
    elif infoRequest == 2:
        messagebox.showinfo("Information", "Perimeter: Canny HIGH level, gradients in the image above this threshold will be selected for linked contours.")
    elif infoRequest == 3:
        messagebox.showinfo("Information", "Perimeter: Canny Low level, gradients in the image above this threshold but below the HIGH level are selected only if neighboring another pixels already selected.")
    elif infoRequest == 4:
        messagebox.showinfo("Information", "Perimeter: smoothing SIGMA value: Standard deviation of the Gaussian filter applied on binary mapping of image.")
    elif infoRequest == 5:
        messagebox.showinfo("Information", "Pores: Canny HIGH level, gradients in the image above this threshold will be selected for linked contours.")
    elif infoRequest == 6:
        messagebox.showinfo("Information", "Pores: Canny Low level, gradients in the image above this threshold but below the HIGH level are selected only if neighboring another pixels already selected.")
    elif infoRequest == 7:
        messagebox.showinfo("Information", "Pores smoothing SIGMA value: Standard deviation of the Gaussian filter applied on input image.")
    elif infoRequest == 8:
        messagebox.showinfo("Information", "SE Canny dilation diameter: After Canny edge detection, the edges found are dilated so some of them can be closed if a few missing pixels are present.")
    elif infoRequest == 9:
        messagebox.showinfo("Information", "SE edges diameter: dilates the perimeter mask by a structuring element (SE) of this diameter to remove edges so that they are not taken to be pores at the stage of closing contours. Needs to be done after volumetric opening so spillover from sample perimeter doesn't contaminate edges of pores")
    elif infoRequest == 10:
        messagebox.showinfo("Information", "SE fills diameter: After first pass floddfill, remove edges that have already been filled, so they dont get distorted by dilation")
    elif infoRequest == 11:
        messagebox.showinfo("Information", "SE large radius: After first pass floodfill, erosion is performed by structuring element of this size to find a large pore to set a seed point in.")
    elif infoRequest == 12:
        messagebox.showinfo("Information", "SE perimeter 3d opening radius: This step removes false positive on the perimeter detection: thin regions that spills from the perimeter to inside the sample")
    elif infoRequest == 13:
        messagebox.showinfo("Information", "SE pores 3d opening radius: Removes small regions that cannot contain the SE. These are false detections of pores")
    elif infoRequest == 14:
        messagebox.showinfo("Information", "SE pores 3d closing radius: Add thin slices inside pore bodies that were missed in the Canny detection. SE that cannot fit into the thin region is added.")

#-----------------------------------------------------
#ALL THE CODE FOR THE GUI WINDOW
# Creating FRAME
checkbtn_frame = Frame(root)
checkbtn_frame.grid(row=8, column=0, columnspan=4, sticky='W')

radio_nbSlice_frame = Frame(root)
radio_nbSlice_frame.grid(row=12, column=1, columnspan=3, sticky='W')

radio_sampleSize_frame = Frame(root)
radio_sampleSize_frame.grid(row=14, column=1, rowspan=2, columnspan=2, sticky='W')
radio_sampleSize_frame.grid_remove()

Advanced_Parameter_frame = Frame(root)
Advanced_Parameter_frame.grid(row=18, column=0, rowspan=6, columnspan=4, sticky='W')
Advanced_Parameter_frame.grid_remove()

Button_frame = Frame(root)
Button_frame.place(relx=0.40, rely=0.92, relwidth=0.60, relheight=0.08)
    
graph_frame = Frame(root)
graph_frame.place(relx=0.25, rely=0, relheight=0.9, relwidth=0.65)

# Creating Canvas
graph_canvas = Canvas(graph_frame)
graph_canvas.pack(side=LEFT, fill=BOTH, expand=1)

# Creating LABELS to hold the variable name
ThresholdPerimeter_lbl = Label(root, text="Threshold Perimeter")
ThresholdPerimeter_lbl.grid(row=0, column=0, sticky='W')
PerimeterHigh_lbl = Label(root, text="Perimeter: Canny \"HIGH\"")
PerimeterHigh_lbl.grid(row=1, column=0, sticky='W')
PerimeterLow_lbl = Label(root, text="Perimeter: Canny \"LOW\"")
PerimeterLow_lbl.grid(row=2, column=0, sticky='W')
PerimeterSigma_lbl = Label(root, text="Perimeter: Canny smoothing SIGMA value")
PerimeterSigma_lbl.grid(row=3, column=0, sticky='W')
PoreHigh_lbl = Label(root, text="Pores: Canny \"HIGH\" level")
PoreHigh_lbl.grid(row=4, column=0, sticky='W')
PoreLow_lbl = Label(root, text="Pores: Canny \"LOW\" level")
PoreLow_lbl.grid(row=5, column=0, sticky='W')
PoreSigma_lbl = Label(root, text="Pores: Canny smoothing SIGMA value")
PoreSigma_lbl.grid(row=6, column=0, sticky='W')

# Creating LABEL to hold information about the variable
TP_info_lbl = Label(root, text="Value between [0, 255]")
TP_info_lbl.grid(row=0, column=2, sticky='W')
PeriH_info_lbl = Label(root, text="Value between [0, 255]")
PeriH_info_lbl.grid(row=1, column=2, sticky='W')
PeriL_info_lbl = Label(root, text="Value between [0, 255]")
PeriL_info_lbl.grid(row=2, column=2, sticky='W')
PeriS_info_lbl = Label(root, text="Value between [0, 10]")
PeriS_info_lbl.grid(row=3, column=2, sticky='W')
PoreH_info_lbl = Label(root, text="Value between [0, 255]")
PoreH_info_lbl.grid(row=4, column=2, sticky='W')
PoreL_info_lbl = Label(root, text="Value between [0, 255]")
PoreL_info_lbl.grid(row=5, column=2, sticky='W')
PoreS_info_lbl = Label(root, text="Value between [0, 10]")
PoreS_info_lbl.grid(row=6, column=2, sticky='W')

# Creating LABEL for the check option for FindExternalPerimeter + FindPores
FPP_info_lbl = Label(root, text="Please check the box if you want to do the procedure")
FPP_info_lbl.grid(row=7, column=0, columnspan=3, sticky='W')

# Creating LABEL for number of slice to be use
nbSlice_info_lbl = Label(root, text="Please choose if you want all the slice or a custom intervall to be use")
nbSlice_info_lbl.grid(row=10, column=0, columnspan=3, sticky='W')
firstSlice_lbl = Label(radio_nbSlice_frame, text="First :")
firstSlice_lbl.grid(row=0, column=0, sticky='W')
lastSlice_lbl = Label(radio_nbSlice_frame, text="Last :")
lastSlice_lbl.grid(row=0, column=2, sticky='W')

# Creating ENTRY for number of slice selection
firstSlice_entry = Entry(radio_nbSlice_frame, width=5)
firstSlice_entry.grid(row=0, column=1, padx=5, pady=5, sticky='W')
lastSlice_entry = Entry(radio_nbSlice_frame, width=5)
lastSlice_entry.grid(row=0, column=3, padx=5, pady=5, sticky='W')

# Creating preset value for the Slice interval
firstSlice_entry.insert(0, "1") 
lastSlice_entry.insert(0, "3")

# Creating LABEL for radio button
sampleSize_info_lbl = Label(root, text="Please choose if you want all the sample to be use or a custom rectangular section")
sampleSize_info_lbl.grid(row=13, column=0, columnspan=3, sticky='W')

# Creating LABEL for Custom option
x_lbl = Label(radio_sampleSize_frame, text="x :")
x_lbl.grid(row=0, column=0, sticky='W')
y_lbl = Label(radio_sampleSize_frame, text="y :")
y_lbl.grid(row=1, column=0, sticky='W')
x_from_lbl = Label(radio_sampleSize_frame, text="from")
x_from_lbl.grid(row=0, column=1)
x_to_lbl = Label(radio_sampleSize_frame, text="to")
x_to_lbl.grid(row=0, column=3)
y_from_lbl = Label(radio_sampleSize_frame, text="from")
y_from_lbl.grid(row=1, column=1)
y_to_lbl = Label(radio_sampleSize_frame, text="to")
y_to_lbl.grid(row=1, column=3)

# Creating LABEL for advance button
advance_lbl = Label(root, text="Please click on Advanced to access the advanced parameters")
advance_lbl.grid(row=16, column=0, columnspan=3, sticky='W')

# Creating LABEL for advance parameter
SE_Canny_dilation_diameter_lbl = Label(Advanced_Parameter_frame, text="SE Canny dilation diameter")
SE_Canny_dilation_diameter_lbl.grid(row=0, column=0, sticky='W')
SE_edges_diameter_lbl = Label(Advanced_Parameter_frame, text="SE edges diameter")
SE_edges_diameter_lbl.grid(row=1, column=0, sticky='W')
SE_fills_diameter_lbl = Label(Advanced_Parameter_frame, text="SE fills diameter")
SE_fills_diameter_lbl.grid(row=2, column=0, sticky='W')
SE_large_diameter_lbl = Label(Advanced_Parameter_frame, text="SE large diameter")
SE_large_diameter_lbl.grid(row=3, column=0, sticky='W')
SE_perim_3DOpening_radius_lbl = Label(Advanced_Parameter_frame, text="SE perim 3D opening radius")
SE_perim_3DOpening_radius_lbl.grid(row=4, column=0, sticky='W')
SE_pores3d_radiusOpening_lbl = Label(Advanced_Parameter_frame, text="SE pores 3D opening radius")
SE_pores3d_radiusOpening_lbl.grid(row=5, column=0, sticky='W')
SE_pores3d_radiusClosing_lbl = Label(Advanced_Parameter_frame, text="SE pores 3D closing radius")
SE_pores3d_radiusClosing_lbl.grid(row=6, column=0, sticky='W')

# Creating LABEL to hold information about the advance parameter
SE_Canny_dilation_diameter_info_lbl = Label(Advanced_Parameter_frame, text="Value between [0, 255]")
SE_Canny_dilation_diameter_info_lbl.grid(row=0, column=2, sticky='W')
SE_edges_diameter_info_lbl = Label(Advanced_Parameter_frame, text="Value between [0, 255]")
SE_edges_diameter_info_lbl.grid(row=1, column=2, sticky='W')
SE_fills_diameter_info_lbl = Label(Advanced_Parameter_frame, text="Value between [0, 255]")
SE_fills_diameter_info_lbl.grid(row=2, column=2, sticky='W')
SE_large_diameter_info_lbl = Label(Advanced_Parameter_frame, text="Value between [0, 255]")
SE_large_diameter_info_lbl.grid(row=3, column=2, sticky='W')
SE_perim_3DOpening_radius_info_lbl = Label(Advanced_Parameter_frame, text="Value between [0, 255]")
SE_perim_3DOpening_radius_info_lbl.grid(row=4, column=2, sticky='W')
SE_pores3d_radiusOpening_info_lbl = Label(Advanced_Parameter_frame, text="Value between [0, 255]")
SE_pores3d_radiusOpening_info_lbl.grid(row=5, column=2, sticky='W')
SE_pores3d_radiusClosing_info_lbl = Label(Advanced_Parameter_frame, text="Value between [0, 255]")
SE_pores3d_radiusClosing_info_lbl.grid(row=6, column=2, sticky='W')

# Creating ENTRIES for Parameter
ThresholdPerimeter_entry = Entry(root, width=7)
ThresholdPerimeter_entry.grid(row=0, column=1, padx=5, pady=5, sticky='W')
PerimeterHigh_entry = Entry(root, width=7)
PerimeterHigh_entry.grid(row=1, column=1, padx=5, pady=5, sticky='W')
PerimeterLow_entry = Entry(root, width=7)
PerimeterLow_entry.grid(row=2, column=1, padx=5, pady=5, sticky='W')
PerimeterSigma_entry = Entry(root, width=7)
PerimeterSigma_entry.grid(row=3, column=1, padx=5, pady=5, sticky='W')
PoreHigh_entry = Entry(root, width=7)
PoreHigh_entry.grid(row=4, column=1, padx=5, pady=5, sticky='W')
PoreLow_entry = Entry(root, width=7)
PoreLow_entry.grid(row=5, column=1, padx=5, pady=5, sticky='W')
PoreSigma_entry = Entry(root, width=7)
PoreSigma_entry.grid(row=6, column=1, padx=5, pady=5, sticky='W')


# Creating preset value for Parameter values
ThresholdPerimeter_entry.insert(0, params["ThresholdPerimeter"])
PerimeterHigh_entry.     insert(0, params["PerimeterHigh"])
PerimeterLow_entry.      insert(0, params["PerimeterLow"])
PerimeterSigma_entry.    insert(0, params["PerimeterSigma"])
PoreHigh_entry.          insert(0, params["PoreHigh"])
PoreLow_entry.           insert(0, params["PoreLow"])
PoreSigma_entry.         insert(0, params["PoreSigma"])

# Creating ENTRY for Advanced parameters
SE_Canny_dilation_diameter_entry = Entry(Advanced_Parameter_frame, width=7)
SE_Canny_dilation_diameter_entry.grid(row=0, column=1, padx=5, pady=5, sticky='W')
SE_edges_diameter_entry = Entry(Advanced_Parameter_frame, width=7)
SE_edges_diameter_entry.grid(row=1, column=1, padx=5, pady=5, sticky='W')
SE_fills_diameter_entry = Entry(Advanced_Parameter_frame, width=7)
SE_fills_diameter_entry.grid(row=2, column=1, padx=5, pady=5, sticky='W')
SE_large_diameter_entry = Entry(Advanced_Parameter_frame, width=7)
SE_large_diameter_entry.grid(row=3, column=1, padx=5, pady=5, sticky='W')
SE_perim_3DOpening_radius_entry = Entry(Advanced_Parameter_frame, width=7)
SE_perim_3DOpening_radius_entry.grid(row=4, column=1, padx=5, pady=5, sticky='W')
SE_pores3d_radiusOpening_entry = Entry(Advanced_Parameter_frame, width=7)
SE_pores3d_radiusOpening_entry.grid(row=5, column=1, padx=5, pady=5, sticky='W')
SE_pores3d_radiusClosing_entry = Entry(Advanced_Parameter_frame, width=7)
SE_pores3d_radiusClosing_entry.grid(row=6, column=1, padx=5, pady=5, sticky='W')        

# Creating preset values for Advanced Parameters
SE_Canny_dilation_diameter_entry.insert(0, params["SE_Canny_dilation_diameter"])
SE_edges_diameter_entry.insert(0, params["SE_edges_diameter"])
SE_fills_diameter_entry.insert(0, params["SE_fills_diameter"])
SE_large_diameter_entry.insert(0, params["SE_large_diameter"])
SE_perim_3DOpening_radius_entry.insert(0, params["SE_perim_3DOpening_radius"])
SE_pores3d_radiusOpening_entry. insert(0, params["SE_pores3d_radiusOpening"])
SE_pores3d_radiusClosing_entry. insert(0, params["SE_pores3d_radiusClosing"])

# Creating ENTRY for Custom option
x_from_entry = Entry(radio_sampleSize_frame, width=5)
x_from_entry.grid(row=0, column=2, padx=5, pady=5, sticky='W')
x_to_entry = Entry(radio_sampleSize_frame, width=5)
x_to_entry.grid(row=0, column=4, padx=5, pady=5, sticky='W')    
y_from_entry = Entry(radio_sampleSize_frame, width=5)
y_from_entry.grid(row=1, column=2, padx=5, pady=5, sticky='W')
y_to_entry = Entry(radio_sampleSize_frame, width=5)
y_to_entry.grid(row=1, column=4, padx=5, pady=5, sticky='W')

#Creating preset values for the Custom section
x_from_entry.insert(0, "60")
x_to_entry.insert(0, "960")
y_from_entry.insert(0, "30")
y_to_entry.insert(0, "960")

# Creating Checkbutton
findExternalPerimeter_check = BooleanVar()
findPores_check = BooleanVar()
#TODO: uncheck has no effect
FEP_checkbtn= Checkbutton(checkbtn_frame, text="Find External Perimeter", variable=findExternalPerimeter_check, onvalue=True, offvalue=False)
FEP_checkbtn.grid(row=8, column=0, padx=10, sticky='W')
FEP_checkbtn.select()
#TODO: uncheck has no effect
FP_checkbtn= Checkbutton(checkbtn_frame, text="Find Pores", variable=findPores_check, onvalue=True, offvalue=False)
FP_checkbtn.grid(row=8, column=1, padx=10, sticky='W')
FP_checkbtn.select()

#Creating Radio button
mode_nbSlice = IntVar()
mode_nbSlice.set(0)
All_nbSlice_radbtn = Radiobutton(root, text="All", variable=mode_nbSlice, value=1, command=lambda: radioChoice_nbSlice())
All_nbSlice_radbtn.grid(row=11, column=0, sticky='W')
Custom_nbSlice_radbtn = Radiobutton(root, text="Custom", variable=mode_nbSlice, value=0, command=lambda: radioChoice_nbSlice())
Custom_nbSlice_radbtn.grid(row=12, column=0, sticky='W')

#Creating Radio button
mode_sampleSize = IntVar()
mode_sampleSize.set(0)
All_sampleSize_radbtn = Radiobutton(root, text="All", variable=mode_sampleSize, value=0, command=lambda: radioChoice_sampleSize())
All_sampleSize_radbtn.grid(row=14, column=0, sticky='W')
Custom_sampleSize_radbtn = Radiobutton(root, text="Custom", variable=mode_sampleSize, value=1, command=lambda: radioChoice_sampleSize())
Custom_sampleSize_radbtn.grid(row=15, column=0, sticky='W')

# Creating INFO button for Parameter
info_Request1_btn = Button(root, text="i", width=2, command=lambda: info_request_command(1))
info_Request1_btn.grid(row=0, column=3, padx=2.5, sticky='W')
info_Request2_btn = Button(root, text="i", width=2, command=lambda: info_request_command(2))
info_Request2_btn.grid(row=1, column=3, padx=2.5, sticky='W')
info_Request3_btn = Button(root, text="i", width=2, command=lambda: info_request_command(3))
info_Request3_btn.grid(row=2, column=3, padx=2.5, sticky='W')
info_Request4_btn = Button(root, text="i", width=2, command=lambda: info_request_command(4))
info_Request4_btn.grid(row=3, column=3, padx=2.5, sticky='W')
info_Request5_btn = Button(root, text="i", width=2, command=lambda: info_request_command(5))
info_Request5_btn.grid(row=4, column=3, padx=2.5, sticky='W')                
info_Request6_btn = Button(root, text="i", width=2, command=lambda: info_request_command(6))
info_Request6_btn.grid(row=5, column=3, padx=2.5, sticky='W')
info_Request7_btn = Button(root, text="i", width=2, command=lambda: info_request_command(7))
info_Request7_btn.grid(row=6, column=3, padx=2.5, sticky='W')

# Creating info for Advance Parameter
info_Request8_btn = Button(Advanced_Parameter_frame, text="i", width=2, command=lambda: info_request_command(8))
info_Request8_btn.grid(row=0, column=3, padx=5, sticky='E')
info_Request9_btn = Button(Advanced_Parameter_frame, text="i", width=2, command=lambda: info_request_command(9))
info_Request9_btn.grid(row=1, column=3, padx=5, sticky='E')
info_Request10_btn = Button(Advanced_Parameter_frame, text="i", width=2, command=lambda: info_request_command(10))
info_Request10_btn.grid(row=2, column=3, padx=5, sticky='E')
info_Request11_btn = Button(Advanced_Parameter_frame, text="i", width=2, command=lambda: info_request_command(11))
info_Request11_btn.grid(row=3, column=3, padx=5, sticky='E')
info_Request12_btn = Button(Advanced_Parameter_frame, text="i", width=2, command=lambda: info_request_command(12))
info_Request12_btn.grid(row=4, column=3, padx=5, sticky='E')                
info_Request13_btn = Button(Advanced_Parameter_frame, text="i", width=2, command=lambda: info_request_command(13))
info_Request13_btn.grid(row=5, column=3, padx=5, sticky='E')
info_Request14_btn = Button(Advanced_Parameter_frame, text="i", width=2, command=lambda: info_request_command(14))
info_Request14_btn.grid(row=6, column=3, padx=5, sticky='E')

# Creating BUTTONS for advanced parameter
advanced_btn = Button(root, text="Advanced", width=10, command=advanced_command)
advanced_btn.grid(row=17, column=0, pady=5, sticky='W')
validate_btn = Button(Advanced_Parameter_frame, text="Validate", width=31, command=validate_AdvancedParam_command)
validate_btn.grid(row=7, column=0, columnspan=4, pady=5, sticky='W')
hide_btn = Button(Advanced_Parameter_frame, text="Hide", width=31, command=hide_command)
hide_btn.grid(row=7, column=0, columnspan=4, pady=5, sticky='E')

# Creating BUTTON at the bottom of the screen
validateAll_btn = Button(Button_frame, text="Validate Parameters", width=20, height=2, command=validate_AllParam_command)
validateAll_btn.grid(row=0, column=0)
launch2d_btn = Button(Button_frame, text="Launch 2D processing", width=20, height=2, command=lambda: launch2d_command())
launch2d_btn.grid(row=0, column=1)
launch3d_btn = Button(Button_frame, text="Launch 3D processing", width=20, height=2, state=DISABLED, command=lambda: launch3d_command())
launch3d_btn.grid(row=0, column=2)
click = IntVar() # variable to make the program wait for the button Next to be clicked
next_btn = Button(Button_frame, text="Next", width=12, height=2, state=DISABLED, command=lambda: click.set(1))
next_btn.grid(row=0, column=3) 
skip_btn = Button(Button_frame, text="Go to last", width=15, height=2, state=DISABLED, command=skip_command)
skip_btn.grid(row=0, column=4)
outPut_Data_btn = Button(Button_frame, text="Process All Data", width=15, height=2, state=DISABLED, command=outPut_Date_command)
outPut_Data_btn.grid(row=0, column=5)   
cancel_btn = Button(Button_frame, text="Cancel", width=12, height=2, command=cancel_command)
cancel_btn.grid(row=0, column=6)

root.mainloop()



