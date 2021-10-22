# by Facundo Sosa-Rey, 2021. MIT license

from random import random as rand
from trackingFunctions import fiberPloting

import mayavi.mlab as mlab
from tvtk.api import tvtk
from mayavi.sources.api import BuiltinSurface,ParametricSurface
import vtk
from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
import cameraConfig
import numpy as np

from mayavi.modules.image_plane_widget import ImagePlaneWidget


from mayavi.modules.surface import Surface

from fibers import fiberObj

def makeLegend(rangeOutline,plotRejectedFibers=False,greyValue=0.8):

    #####################################

    # pseudo Legend object

    #####################################

    legendPlaneObj = BuiltinSurface(source='plane')

    legendPlane = mlab.pipeline.surface(legendPlaneObj)

    legendLabels=fiberObj.getLegendLabels(plotRejectedFibers)
    legendLabels=list(legendLabels)
    legendLabels.sort()

    heightOneEntry=10
    totalHeight=heightOneEntry*(1+len(legendLabels))

    totalWidth=int(3.5*max([len(iLabel) for iLabel in legendLabels]))

    zOffset=40

    legendExtent=[
        rangeOutline[0],
        rangeOutline[0]+totalWidth,
        rangeOutline[2],rangeOutline[2]+.5, #y thickness
        rangeOutline[5]+zOffset,
        rangeOutline[5]+zOffset+totalHeight]

    mlab.outline(extent=legendExtent,color=(0.,0.,0.))

    imSlice = np.ones((totalWidth,totalHeight))*greyValue


    gridLegend = vtk.vtkImageData()
    gridLegend.SetDimensions((imSlice.shape[1],1,imSlice.shape[0]))

    vtkarr = numpy_to_vtk(imSlice.ravel())
    vtkarr.SetName('legendBackGround')
    gridLegend.GetPointData().AddArray(vtkarr)
    gridLegend.GetPointData().SetActiveScalars('legendBackGround')

    vtexLegend = vtk.vtkTexture()
    vtexLegend.SetInputDataObject(gridLegend)

    # white colormap
    lookUpTableLegend=vtk.vtkLookupTable()
    lookUpTableLegend.SetHueRange(0.,0.)
    lookUpTableLegend.SetNumberOfColors(256)
    lookUpTableLegend.SetSaturationRange(0.,0.)
    lookUpTableLegend.SetValueRange(0.,1.)

    lookUpTableLegend.Build()

    vtexLegend.SetLookupTable(lookUpTableLegend)

    vtexLegend.Update()

    # reposition plane
    legendPlaneObj.data_source.trait_set(
                        origin=(legendExtent[0],legendExtent[2],legendExtent[4]),
                        point1=(legendExtent[1],legendExtent[2],legendExtent[4]), 
                        point2=(legendExtent[0],legendExtent[2],legendExtent[5])
                    ) 


    legendPlane.actor.actor.texture = vtexLegend

    offset=5
    textHandle=mlab.text3d(legendExtent[1]-30,legendExtent[3],legendExtent[5]-offset,
        "Legend",color=(0.,0.,0.),orient_to_camera=False,scale=3,orientation=(90.,180.,0.))


    for iLabel in legendLabels:
        offset+=10
        textHandle=mlab.text3d(legendExtent[1]-20,legendExtent[3],legendExtent[5]-offset,
            iLabel,color=(0.,0.,0.),orient_to_camera=False,scale=3,orientation=(90.,180.,0.))

        mlab.plot3d([legendExtent[1]-5,legendExtent[1]-15],[legendExtent[3],legendExtent[3]],[legendExtent[5]-offset+2,legendExtent[5]-offset+2],
            tube_radius=0.6,color=fiberObj.getColor(iLabel))    


def plotPororityMask(V_porosity,rangeOutline):

    src = mlab.pipeline.scalar_field(V_porosity)
    src.spacing = [1, 1, 1]
    if rangeOutline is None:
        src.origin = [0.,0.,0.]
    else:
        src.origin = [rangeOutline[0],rangeOutline[2],rangeOutline[4]]
    src.update_image_data = True


    #smooting
    # blur = mlab.pipeline.user_defined(src, filter='ImageGaussianSmooth')
    # Extract views of the outside surface. 
    voi = mlab.pipeline.extract_grid(src)

    # Add surface module to the cylinder source
    outer = mlab.pipeline.iso_surface(voi, contours=[250, ],
                                            color=(0.1, 0.1, 0.1),opacity=0.25)

def addPlaneWidgets(V_widget,engine,widgetLUT=None,axOnly=None,rangeOutline=None):
    src1 = mlab.pipeline.scalar_field(V_widget)

    src1.spacing = [1, 1, 1]
    if rangeOutline is None:
        src1.origin = [0.,0.,0.]
    else:
        src1.origin = [rangeOutline[0],rangeOutline[2],rangeOutline[4]]
    
    src1.update_image_data = True

    ipw_z = ImagePlaneWidget()
    engine.add_module(ipw_z)
    if axOnly is None:
        ipw_z.ipw.plane_orientation = 'z_axes'
    else: 
        ipw_z.ipw.plane_orientation = axOnly

    ipw_z.module_manager.scalar_lut_manager.show_scalar_bar = False

    if widgetLUT is not None:
        ipw_z.module_manager.scalar_lut_manager.lut_mode = widgetLUT
    # ipw.module_manager.scalar_lut_manager.lut_mode = 'black-white'
    # ipw.module_manager.scalar_lut_manager.lut_mode = 'binary'
    # ipw.module_manager.scalar_lut_manager.lut.table = lutNumpyArray

    if axOnly is None:
        ipw_x = ImagePlaneWidget()
        engine.add_module(ipw_x)
        ipw_x.ipw.plane_orientation = 'x_axes'
        
        ipw_y = ImagePlaneWidget()
        engine.add_module(ipw_y)
        ipw_y.ipw.plane_orientation = 'y_axes'




##################################################################################

# Visualise fibre tracks

##################################################################################


def makeVisualisation(fiberStruct, V_porosity, V_widget, rangeOutline, params,widgetLUT='jet',makeLegendHandle=True):
                                            # (0.2,0.2,0.2)
    fig = mlab.figure(size=(1200,900),bgcolor=(0.9,0.9,0.9),fgcolor=(1.,1.,1.))

    engine = mlab.get_engine()

    if params["plotOnlyStitchedFibers"]:
        params["plotRejectedFibers"]=False
    
    if fiberStruct:
        numFibersTracked=len(fiberObj.classAttributes["listFiberIDs_tracked"])
    else:
        numFibersTracked=0

    for fibID,fib in fiberStruct.items(): 

        if params["plotRejectedFibers"]: # plot ALL fibers
            fiberPloting(fib,fibID,len(fiberStruct),numFibersTracked,engine,params) 

        elif params["plotOnlyStitchedFibers"]:
            doNotPlotSet={"basic","too short","too steep"}
            if not {fib.colorLabel}.intersection(doNotPlotSet):
                fiberPloting(fib,fibID,len(fiberStruct),numFibersTracked,engine,params) 
        elif params["plotOnlyTrackedFibers"]:
            if fibID in fib.classAttributes["listFiberIDs_tracked"]:
                fiberPloting(fib,fibID,len(fiberStruct),numFibersTracked,engine,params) 
        else:
            if not fib.rejected:
                fiberPloting(fib,fibID,len(fiberStruct),numFibersTracked,engine,params) 

        


    mlab.outline(extent=rangeOutline,color=(.3,.3,.3))

    axes = mlab.axes(color=(0., 0., 0.), nb_labels=11)
    axes.title_text_property.color = (0., 0., 0.)
    axes.title_text_property.font_family = 'times'

    axes.label_text_property.color = (0., 0., 0.)
    axes.label_text_property.font_family = 'times'

    axes.axes.font_factor=0.65

    if makeLegendHandle:
        makeLegend(rangeOutline,params["plotRejectedFibers"])

    if params["plotPorosityMask"]:
        print("Plotting porosity mask")
        plotPororityMask(V_porosity,rangeOutline)

    scene = engine.scenes[0]


    topToBottom=False

    # bluish-greyscale colormap
    lookUpTable=vtk.vtkLookupTable()
    lookUpTable.SetHueRange(0.5,0.6)
    lookUpTable.SetNumberOfColors(256)
    lookUpTable.SetSaturationRange(0.,0.5)
    lookUpTable.SetValueRange(0.,1.)
    lookUpTable.SetAlpha(1.)

    lookUpTable.Build()

    lutNumpyArray=vtk_to_numpy(lookUpTable.GetTable())

    if params["staticCam"]:
        cameraConfig.shiftCamera(params["cameraConfigKey"],scene,0.)

    if params["planeWidgets"]:
        print("adding plane widgets")
        addPlaneWidgets(V_widget,engine,widgetLUT,rangeOutline=rangeOutline)


    if params["panningPlane"]:
        delayVal=100
    else:
        delayVal=50000

    @mlab.animate(delay=delayVal)
    def anim():
        if params["panningPlane"]:
            #    plot plane with CT slice
            planeObj = BuiltinSurface(source='plane')

            plane = mlab.pipeline.surface(planeObj)

        while True:
            if topToBottom:
                imVec=range(V_widget.shape[2]-1,0,-1)
            else:
                imVec=range(V_widget.shape[2])
            for im in imVec:
                if im == 0 and params["drawEllipsoids"] and params["panningPlane"]:  
                    for iFib in range(len(fiberStruct)):
                        if not params["plotRejectedFibers"]:
                            #fiberStruct[iFib]["wireObj"].scene.disable_render = False
                            fiberStruct[iFib].wireObj.visible=False
                            # fiberStruct[iFib]["surfaceObj"].scene.disable_render = False
                            fiberStruct[iFib].surfaceObj.visible=False
                        elif params["plotRejectedFibers"]:
                            #fiberStruct[iFib]["wireObj"].scene.disable_render = False
                            fiberStruct[iFib].wireObj.visible=False
                            # fiberStruct[iFib]["surfaceObj"].scene.disable_render = False
                            fiberStruct[iFib].surfaceObj.visible=False
                            

                print('Updating scene...')      
                if params["panningPlane"]:
                    imSlice = np.transpose(V_widget[:,:,im])
                
                    grid = vtk.vtkImageData()
                    grid.SetDimensions((imSlice.shape[1],imSlice.shape[0],1))
        
                    vtkarr = numpy_to_vtk(imSlice.ravel())
                    vtkarr.SetName('imSlice')
                    grid.GetPointData().AddArray(vtkarr)
                    grid.GetPointData().SetActiveScalars('imSlice')

                    vtex = vtk.vtkTexture()
                    vtex.SetInputDataObject(grid)

                    vtex.SetLookupTable(lookUpTable)

                    vtex.Update()

                    # reposition plane
                    planeObj.data_source.trait_set(
                        origin=(rangeOutline[0],                    rangeOutline[2],                    float(im)),
                        point1=(rangeOutline[0]+imSlice.shape[1],   rangeOutline[2],                    float(im)), 
                        point2=(rangeOutline[0],                    rangeOutline[2]+imSlice.shape[0],   float(im))
                        ) 

                    plane.actor.actor.texture = vtex
            
                    if im%10==0:
                        mlab.text3d(float(imSlice.shape[0])+5,float(imSlice.shape[1])+5,float(im),
                            "{}".format(im),color=(0.,0.,0.))


                    if params["drawEllipsoids"]:
                        for iFib in range(len(fiberStruct)):
                            if not params["plotRejectedFibers"]:
                                if fiberStruct[iFib].startPnt[2]<(float(im)+5) and\
                                    fiberStruct[iFib].endPnt[2]>(float(im)-5):
                                    fiberStruct[iFib].wireObj.visible=True
                                    fiberStruct[iFib].surfaceObj.visible=True
                                else:
                                    fiberStruct[iFib].wireObj.visible=False
                                    fiberStruct[iFib].surfaceObj.visible=False
                            elif params["plotRejectedFibers"]:
                                if fiberStruct[iFib].startPnt[2]<(float(im)+5) and\
                                    fiberStruct[iFib].endPnt[2]>(float(im)-5):
                                    fiberStruct[iFib].wireObj.visible=True
                                    fiberStruct[iFib].surfaceObj.visible=True
                                else:
                                    fiberStruct[iFib].wireObj.visible=False
                                    fiberStruct[iFib].surfaceObj.visible=False


                if not params["staticCam"]:
                    cameraConfig.shiftCamera(params["cameraConfigKey"],scene,im)

                yield



    anim()
    mlab.show(stop=False)