# by Facundo Sosa-Rey, 2021. MIT license

from matplotlib import patches
from scipy.spatial import KDTree as KDTree
import numpy as np


import viscid as vs
import cv2 as cv

from skimage import morphology

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches

from centroid import centroidObj

from random import shuffle

import multiprocessing


def unique(myList):
    #elements added to set only once, removes repetitions
    return np.array(list(set(myList)))

def randomizeVoxelSlice(V_fiberMap,listMarkers,exclusionList):

    V_fiberMap_randomized=V_fiberMap.copy()

    reassignedMarkers=listMarkers.copy()
    #random shuffling of original markers
    shuffle(reassignedMarkers)
    
    markerLUT={}
    for i,iMark in enumerate(listMarkers):
        markerLUT[iMark]=reassignedMarkers[i]

    V_fiberMap_randomized=compactifySlice(
            V_fiberMap,
            markerLUT,
            exclusionList
        )

    return V_fiberMap_randomized

def compactifySlice(slice,compactifyIDs_LUT,exclusionList):
    sliceCompactified=np.ones(slice.shape,np.int32)*-1
    for ix in range(len(slice)):
        for iy in range(len(slice[0])):
            if slice[ix,iy] not in exclusionList:
                sliceCompactified[ix,iy]=compactifyIDs_LUT[slice[ix,iy]]
    
    return sliceCompactified

def stitchingRanking(presentMatchDataTuple,candidateMatchDataTuple):
    if len(presentMatchDataTuple)==2: #blindStitching
        presentMatchDistance    =presentMatchDataTuple  [0]
        candidateMatchDistance  =candidateMatchDataTuple[0]
        
        keepNewMatch=False
        if candidateMatchDistance<presentMatchDistance:
            keepNewMatch=True

        elif candidateMatchDistance==presentMatchDistance:
            #edge case where total dist is the same
            presentLateralDist   =presentMatchDataTuple  [1]
            candidateLateralDist =candidateMatchDataTuple[1]

            if candidateLateralDist<presentLateralDist:
                keepNewMatch=True

        return keepNewMatch

    if len(presentMatchDataTuple)==4: #smartStitching

        presentLength_up   = presentMatchDataTuple    [2]
        candidateLength_up = candidateMatchDataTuple[2]

        presentAngleOrienVec   = np.degrees(np.arccos(  presentMatchDataTuple[3]))
        candidateAngleOrienVec = np.degrees(np.arccos(candidateMatchDataTuple[3]))
        
        keepNewMatch=False
        if candidateLength_up>presentLength_up:
            if presentAngleOrienVec==candidateAngleOrienVec:
                #edge case where stitching occurs twice from same fiber pairs, (both pairs of endpoints pass stitching criteria)
                presentMatchDistance    =presentMatchDataTuple  [0]
                candidateMatchDistance  =candidateMatchDataTuple[0]
                if candidateMatchDistance<presentMatchDistance:
                    keepNewMatch=True

            else:
                #the angles between segments can be increase by as much as 5 degrees, if the candidate extension is longer than the current
                if -5.<(presentAngleOrienVec-candidateAngleOrienVec)<5.: 
                    keepNewMatch=True


        elif candidateAngleOrienVec<presentAngleOrienVec:
            keepNewMatch=True


        return keepNewMatch

class knn3D:
    #this class store the KD tree for the blind stitching procedure
    def __init__(self,topCenters):
        # build kd-tree
        self.topCenters=topCenters
        self.tree_down=KDTree(topCenters)

    def query(self,
        bottomCenters,
        distTotal,
        distLateral=None,
        angleMax=None,
        backTrackingLimit=0.,
        lengths=None,
        k=1,
        suffixCheck=None
        ):
        """search the kd-tree, return correspondance between  bottomCenters[id_bottom] to topCenters[id_top]
        that satisfy
        :total (Euclidian) distance<distTotal ,
        :lateral (in plane) distance<distLateral (if present),
        :angle between segments<angleMax,
        :backtracking distance<backTrackingLimit

        For smartStitching, lengths are passed to rank order the candidate matches
        
        For stitching across combinations, suffix in [0.123,0.132,0.321] are passed 
        to prevent matching from the same permutation.
        """
    
        # id_down: topCenter[id_down[i]] is closest to bottomCenters[i], with distance d_down[i]
        [d_down, id_down] = self.tree_down.query(bottomCenters,k=k)

        # build kd-tree
        tree_up = KDTree(bottomCenters)
        # search the kd-tree

        # id_up: bottomCenter[id_up[i]] is closest to topCenters[i], with distance d_up[i]
        [d_up, id_up] = tree_up.query(self.topCenters,k=k)

        matches={}
        matches["listCentersTop"]={}

        matchesBackup={}
        matchesBackup["listCentersTop"]={}

        if suffixCheck is not None:
            in123_bottom=suffixCheck[0]
            in123_top   =suffixCheck[1]
            in132_bottom=suffixCheck[2]
            in132_top   =suffixCheck[3]
            in321_bottom=suffixCheck[4]
            in321_top   =suffixCheck[5]

        if k!=1:
            d_downList=[]
            id_downList=[]
            d_upList  =[]
            id_upList  =[]
            for iCandidateDown in range(len(id_down)):

                if any(d_down[iCandidateDown]<distTotal):

                    if suffixCheck is None:
                        # check if below max distance and prevent self-matching
                        id_downList=[val for val in id_down[iCandidateDown][d_down[iCandidateDown]<distTotal] if val!=iCandidateDown]
                    else:
                        # check if below max distance and prevent self-matching
                        id_downList=[
                            val for val in id_down[iCandidateDown][d_down[iCandidateDown]<distTotal]\
                                if val!=iCandidateDown and \
                                not np.logical_and(in123_bottom[iCandidateDown],in123_top[val]) and \
                                not np.logical_and(in132_bottom[iCandidateDown],in132_top[val]) and \
                                not np.logical_and(in321_bottom[iCandidateDown],in321_top[val])      
                            ]

                    # bidirectional match is required:
                    for iCandidateUp in id_downList:

                        # if match exist from A->B, B->A will also work, to the possible detriment of B->C 
                        # (can happen in combinePermutations)
                        if not(iCandidateUp in matches.keys() and matches[iCandidateUp][0]==iCandidateDown):

                            id_upList=id_up[iCandidateUp][d_up[iCandidateUp]<distTotal]

                            if any(id_upList==iCandidateDown):
                                successfulMatch=False
                                #check if backtracking lmit is not exceeded:
                                if self.topCenters[iCandidateUp][2]-bottomCenters[iCandidateDown][2]>-backTrackingLimit:
                                    if angleMax is None:
                                        # blindStitching
                                        # check lateralDist
                                        lateralDist=np.sqrt(
                                            (bottomCenters[iCandidateDown][0]-self.topCenters[iCandidateUp][0])**2+\
                                            (bottomCenters[iCandidateDown][1]-self.topCenters[iCandidateUp][1])**2)
                                        if  lateralDist<distLateral:
                                            successfulMatch=True
                                    else:
                                        #smartStitching
                                        endPntBottom=np.array(bottomCenters[iCandidateDown][:3])
                                        startPntTop=np.array(self.topCenters[iCandidateUp][:3])
                                        oriVecBottom=np.array(bottomCenters[iCandidateDown][3:])

                                        connectVec=startPntTop-endPntBottom
                                        normConnectVec=np.linalg.norm(connectVec)
                                        connectVec/=normConnectVec
                                        
                                        theta=np.arccos(np.dot(oriVecBottom,connectVec))
                                        lateralDist=np.sin(theta)*normConnectVec
                                        if  lateralDist<distLateral:

                                            cosAnglesOrientationVecs=np.dot(
                                                bottomCenters[iCandidateDown][3:],
                                                self.topCenters[iCandidateUp][3:])
                                            if abs(cosAnglesOrientationVecs)>np.cos(angleMax):
                                                successfulMatch=True
                                                
                                    if successfulMatch:
                                        #successful match
                                        matchDistance=d_up[iCandidateUp ][np.where(id_up[iCandidateUp ]==iCandidateDown)][0]

                                        if lengths==None: #blindStitching
                                            dataTuple=(matchDistance,lateralDist)
                                        else:
                                            dataTuple=(matchDistance,lateralDist,lengths[iCandidateUp],cosAnglesOrientationVecs)

                                        keepNewMatch=True

                                        if iCandidateDown in matches.keys():
                                            # if match already exist for this candidateDown, keep the match which has better ranking 
                                            keepNewMatch=stitchingRanking(matches[iCandidateDown][1],dataTuple )

                                            previousMatchUp=matches[iCandidateDown][0]

                                            if keepNewMatch:
                                                # if match already exist for this candidateUp, keep the match which has better ranking
                                                if iCandidateUp in matches["listCentersTop"].keys():
                                                    previousMatchDown=matches["listCentersTop"][iCandidateUp]
                                                    
                                                    keepNewMatch=stitchingRanking(matches[previousMatchDown][1],dataTuple)

                                                    if keepNewMatch:
                                                        # remove BOTH previous matches
                                                        if previousMatchDown in matchesBackup.keys():
                                                            matchesBackup[previousMatchDown].append(matches.pop(previousMatchDown))
                                                        else:
                                                            matchesBackup[previousMatchDown]=[matches.pop(previousMatchDown)]

                                                        matchesBackup["listCentersTop"][iCandidateUp]   =matches["listCentersTop"].pop(iCandidateUp)   #remove previous iCandidateDown
                                                        matchesBackup["listCentersTop"][previousMatchUp]=matches["listCentersTop"].pop(previousMatchUp)#remove previous iCandidateUp
                                                else:
                                                    #remove previous match
                                                    if iCandidateDown in matchesBackup.keys():
                                                        matchesBackup[iCandidateDown].append(matches[iCandidateDown])
                                                    else:
                                                        matchesBackup[iCandidateDown]=[matches[iCandidateDown]]

                                                    # matchesBackup[iCandidateDown]=matches[iCandidateDown]
                                                    matchesBackup["listCentersTop"][previousMatchUp]=matches["listCentersTop"].pop(previousMatchUp)

                                        else:

                                            # if match already exist for this candidateUp, keep the match which has minimal distance 
                                            if iCandidateUp in matches["listCentersTop"].keys():
                                                previousMatch=matches["listCentersTop"][iCandidateUp]

                                                keepNewMatch=stitchingRanking(matches[previousMatch][1],dataTuple)

                                                if keepNewMatch:
                                                    #remove previous match
                                                    if previousMatch in matchesBackup.keys():
                                                        matchesBackup[previousMatch].append(matches.pop(previousMatch))
                                                    else:
                                                        matchesBackup[previousMatch]=[matches.pop(previousMatch)]

                                                    # matchesBackup[previousMatch]                    =matches.pop(previousMatch)
                                                    matchesBackup["listCentersTop"][iCandidateUp]   =matches["listCentersTop"].pop(iCandidateUp) #remove previous iCandidateDown

                                            #edge case where both pairs of start and endpoints are a positive match (probably due to backtracking)
                                            if iCandidateUp in matches.keys():
                                                keepNewMatch=stitchingRanking(matches[iCandidateUp][1],dataTuple )

                                                previousMatchDown=matches[iCandidateUp][0]

                                                if keepNewMatch:
                                                    previousMatchUp=iCandidateUp

                                                    if iCandidateUp in matches["listCentersTop"].keys():
                                                        # if reverse match already exist for this candidateUp, check against that one
                                                        # if ranks higher, this reverse match will be poped and replaced
                                                        previousMatchUp=matches["listCentersTop"][iCandidateUp]
                                                        previousMatchDown=matches[previousMatchUp][0]
                                                        keepNewMatch=stitchingRanking(matches[previousMatchUp][1],dataTuple )
                                                 
                                                    if keepNewMatch:
                                                        #remove previous match
                                                        if previousMatchUp in matchesBackup.keys():
                                                            matchesBackup[previousMatchUp].append(matches.pop(previousMatchUp))
                                                        else:
                                                            matchesBackup[previousMatchUp]=[matches.pop(previousMatchUp)]

                                                        matchesBackup["listCentersTop"][previousMatchDown]=matches["listCentersTop"].pop(previousMatchDown) #remove previous iCandidateDown

                                        if keepNewMatch:
                                            matches[iCandidateDown]=(iCandidateUp,dataTuple)
                                            matches["listCentersTop"][iCandidateUp]=iCandidateDown

            potentialMatches=list(matchesBackup.keys())
            potentialMatches.remove("listCentersTop")

            #matchesBackup are usefull in the following scenario:
            # first A matched to B
            # then C matched to B (better ranking)
            # then C matched to D

            # in matches, only C to D is kept.
            # but perhaps A and B could still be a good match, 
            # if either wasn't matched to anything
            # this possibility is checked here

            for iCandidateDown in potentialMatches:
                while matchesBackup[iCandidateDown]:
                    iCandidateUp=matchesBackup[iCandidateDown][-1][0]
                    if iCandidateUp in matches["listCentersTop"]:
                        matchesBackup[iCandidateDown].pop()
                    else:
                        matches[iCandidateDown]=matchesBackup[iCandidateDown][-1]
                        matches["listCentersTop"][iCandidateUp]=iCandidateDown
                        matchesBackup[iCandidateDown]=[]



            id_bottom_th=[]
            id_top_th=[]

            for iDown in matches.keys():
                if iDown != "listCentersTop":
                    id_bottom_th.append(iDown)
                    id_top_th.append(matches[iDown][0])
    
            # bottomCenters[id_bottom_th] are closest to topCenters[id_top_th], within max dist
            return id_bottom_th,id_top_th,matches

        # keep matches that are bidirectional
        id_top = np.unique(id_up[id_down])
        id_bottom = id_down[id_top]

        # keep matches that are closer than a distance
        # idMatch = find(d_down[id_top] < dist);
        idMatch = [index for index,val in enumerate(d_down[id_top]) if val<distTotal]
        id_bottom_th = id_top[idMatch]
        id_top_th = id_bottom[idMatch]

        #test to see if more than one match in top
        l_top=list(id_top_th)
        repetitions=[(index,val,d_down[id_bottom_th[index]]) for index,val in enumerate(id_top_th) if l_top.count(val)>1]

        repetitionsDict={}
        rejectPositions=[]

        # if repetitions exist, keep the one at shortest distance, and store indices of rejected ones
        if len(repetitions)>0:
            for index,val,dist in repetitions:
                if val in repetitionsDict.keys():
                    if dist<repetitionsDict[val][1]:
                        rejectPositions.append(repetitionsDict[val][0])
                        repetitionsDict[val]=(index,dist)
                    else:
                        rejectPositions.append(index)
                else:
                    repetitionsDict[val]=(index,dist)

        return id_bottom_th,id_top_th,repetitionsDict,rejectPositions

def knn(topCenters,bottomCenters,dist,returnDist=False):

    # build kd-tree
    tree_down=KDTree(topCenters)

    # search the kd-tree
    
    # id_down: topCenter[id_down[i]] is closest to bottomCenters[i], with distance d_down[i]
    [d_down, id_down] = tree_down.query(bottomCenters)

    # build kd-tree
    tree_up = KDTree(bottomCenters)
    # search the kd-tree

    # id_up: bottomCenter[id_up[i]] is closest to topCenters[i], with distance d_up[i]
    [d_up, id_up] = tree_up.query(topCenters)

    # keep matches that are bidirectional
    id_bottom = np.unique(id_up[id_down])
    id_top = id_down[np.unique(id_up[id_down])]

    # keep matches that are closer than a distance
    # idMatch = find(d_down[id_top] < dist);
    idMatch = [index for index,val in enumerate(d_down[id_bottom]) if val<dist]
    id_bottom_th = id_bottom[idMatch]
    id_top_th = id_top[idMatch]

    if returnDist:
        dist=d_down[id_bottom[idMatch]]
        # bottomCenters[id_bottom_th] are closest to topCenters[id_top_th], within max dist
        return id_bottom_th,id_top_th,dist
    else:
        return id_bottom_th,id_top_th

def firstPassKNN(nextCt,thisCt,distLateral_knn):
    # the query points are the current slice (thisCT)
    # mapped into the next slice
    # thisCT(id_bottom_th)==nextCT(id_top_th)
    if len(nextCt)>0 and len(thisCt)>0: #no need to find neighbours if either is empty
        id_bottom,id_top = knn(nextCt,thisCt,distLateral_knn)
    else:
        id_bottom=[]
        id_top=[]

    return id_bottom,id_top

def greyUint16_to_RGB(imageUint16):
    imageRGB=np.zeros((imageUint16.shape[0],imageUint16.shape[1],3))
    for ix in range(imageUint16.shape[0]):
        for iy in range(imageUint16.shape[1]):
            imageRGB[ix,iy,0]=imageUint16[ix,iy]/2**16
            imageRGB[ix,iy,1]=imageUint16[ix,iy]/2**16
            imageRGB[ix,iy,2]=imageUint16[ix,iy]/2**16

    return imageRGB

def RBGUint8_to_Bool(imageUint8):
    imageRGB=np.zeros((imageUint8.shape[0],imageUint8.shape[1]))
    for ix in range(imageUint8.shape[0]):
        for iy in range(imageUint8.shape[1]):
            imageRGB[ix,iy,0]=imageUint8[ix,iy]/2**8
            imageRGB[ix,iy,1]=imageUint8[ix,iy]/2**8
            imageRGB[ix,iy,2]=imageUint8[ix,iy]/2**8

    return imageRGB



def drawEllipsoid(engine,phi,theta,beta,translationVec,length,radius,representation="surface" ):
    
    #making imports inside function so codebase works even if mayavi is unworkable on some machines 
    from mayavi.sources.api import ParametricSurface
    from mayavi.sources.builtin_surface import BuiltinSurface
    from mayavi.modules.api import Surface
    from mayavi.filters.transform_data import TransformData

    # Add a cylinder builtin source
    ellipsoid_src = BuiltinSurface()
    engine.add_source(ellipsoid_src)
    ellipsoid_src.source = 'superquadric'
    ellipsoid_src.data_source.center = np.array([0.,0.,0.])
    ellipsoid_src.data_source.scale = ([radius,length,radius])
    ellipsoid_src.data_source.phi_resolution=64
    ellipsoid_src.data_source.theta_resolution=64

    # Add transformation filter to translate and then rotate ellipsoid about an axis
    transform_data_filter = TransformData()
    engine.add_filter(transform_data_filter, ellipsoid_src)
    Rt = np.eye(4)
    # in homogeneous coordinates:
    #rotation matrix
    Rt[0:3,0:3] =vs.eul2rot( [phi,theta, beta],'zyz', unit='deg')
    #translation
    Rt[0:3,3]=translationVec

    Rtl = list(Rt.flatten()) # transform the rotation matrix into a list

    transform_data_filter.transform.matrix.__setstate__({'elements': Rtl})
    transform_data_filter.widget.set_transform(transform_data_filter.transform)
    transform_data_filter.filter.update()
    transform_data_filter.widget.enabled = False   # disable the rotation control further.
    
    # Add surface module to the cylinder source
    ellip_surface = Surface()

    engine.add_filter(ellip_surface,transform_data_filter)

    # add color property
    ellip_surface.actor.property.color = (0.0, 0.4, 0.9)
    ellip_surface.actor.property.representation = representation

def drawEllipsoidParametric(engine,phi,theta,beta,translationVec,
    half_length,radius,color,representation="wireframe",opacity=0.1 ):

    #making imports inside function so codebase works even if mayavi is unworkable on some machines 
    from mayavi.sources.api import ParametricSurface
    from mayavi.sources.builtin_surface import BuiltinSurface
    from mayavi.modules.api import Surface
    from mayavi.filters.transform_data import TransformData

    # Add a parametric surface source
    ellipsoid_src = ParametricSurface()
    ellipsoid_src.function = 'ellipsoid'
    engine.add_source(ellipsoid_src)
    
    # Add transformation filter to scale, translate and then rotate ellipsoid about an axis
    transform_data_filter = TransformData()
    engine.add_filter(transform_data_filter, ellipsoid_src)
    Rt = np.eye(4)
    # in homogeneous coordinates:
    # scaling
    scalingMat=np.eye(3)
    scalingMat[0,0]*=radius
    scalingMat[1,1]*=radius
    scalingMat[2,2]*=half_length
    # rotation matrix afterwards
    Rt[0:3,0:3] =np.dot(vs.eul2rot( [phi,theta, beta],'zyz', unit='deg'),scalingMat)
    # translation
    Rt[0:3,3]=translationVec

    Rtl = list(Rt.flatten()) # transform the rotation matrix into a list

    # add transformation matrix to transform filter
    transform_data_filter.transform.matrix.__setstate__({'elements': Rtl})
    transform_data_filter.widget.set_transform(transform_data_filter.transform)
    transform_data_filter.filter.update()
    transform_data_filter.widget.enabled = False   # disable the rotation control further.
    
    # Add surface module to the cylinder source
    ellip_surface = Surface()

    engine.add_filter(ellip_surface,transform_data_filter)

    # add rendering properties
    ellip_surface.actor.property.opacity = opacity
    ellip_surface.actor.property.color = color
    ellip_surface.actor.mapper.scalar_visibility = False # don't colour ellipses by their scalar indices into colour map
    ellip_surface.actor.property.backface_culling = True # gets rid of weird rendering artifact when opacity is < 1
    ellip_surface.actor.property.specular = 0.1
    # ellip_surface.actor.enable_texture=True
    ellip_surface.actor.property.representation = representation
    ellip_surface.scene.disable_render = True

    return ellip_surface

def getPhiThetaFromVec(vec):
    #normalization
    vec/=np.linalg.norm(vec)

    phi=np.arctan2(vec[1],vec[0])
    theta=np.arccos(np.dot([0.,0.,1.],vec ))

    return np.degrees(phi),np.degrees(theta)

def plotEllipsoid(fibObj,fiberDiameter,engine,opacity=0.1,representation="wireframe" ):
    try:
        vec=fibObj.orientationVec
    except:
        fibObj.processPointCloudToFiberObj(minFiberLength=1.,tagAngleTooSteep=False,maxSteepnessAngle=None)
        vec=fibObj.orientationVec


    phi,theta=getPhiThetaFromVec(vec)
    translationVec=fibObj.meanPntCloud
    half_length=fibObj.totalLength/2.
    radius=fiberDiameter/2.
    color=fibObj.color

    return drawEllipsoidParametric(engine,phi,theta,0.,translationVec,half_length,radius,color,representation, opacity )


def fiberPloting(fibObj,iFib,nFib,nFibTracked,engine,params,scale=2.):

    import mayavi.mlab as mlab

    print("Drawing fiber object #{}/{},\t ({} are \"tracked\")".format(iFib,nFib,nFibTracked))

    drawJaggedLines     =params["drawJaggedLines"]
    drawCenterLines     =params["drawCenterLines"]
    drawEllipsoids      =params["drawEllipsoids"]
    addText             =params["addText"]
    fiberDiameter       =params["fiberDiameter"]


    tube_radius=0.4

    if drawJaggedLines:
        if fibObj.addedTo:
            tube_radius=0.5
        mlab.plot3d(fibObj.x,fibObj.y,fibObj.z,
            tube_radius=tube_radius,color=fibObj.color)

    if drawCenterLines:
        pointArray=np.concatenate((np.array(fibObj.startPnt)[:,np.newaxis],
            np.array(fibObj.endPnt)[:,np.newaxis]), axis=1)
        mlab.plot3d(*pointArray,
            tube_radius=tube_radius,color=fibObj.color)    

    if drawEllipsoids:
         fibObj.wireObj=plotEllipsoid(fibObj,fiberDiameter,engine,opacity=0.1,representation='wireframe') 
         fibObj.surfaceObj=plotEllipsoid(fibObj,fiberDiameter,engine,opacity=0.3,representation='surface')

    if addText:
        strText="{}".format(fibObj.fiberID)        

        if "zOffset" in fibObj.__dir__():
            if fibObj.zOffset:
                zOffset=16
        else:
            zOffset=10

        mlab.text3d(fibObj.x[-1],fibObj.y[-1],fibObj.z[-1]+zOffset,
            strText,color=fibObj.color,scale=scale)


def permute3(list3,permuteVec):
    return[list3[i] for i in permuteVec]
 
def makeNegative(im):
    im[im==255]=1
    im[im==0]=255
    im[im==1]=0

# padding

def paddingOfImage(im,paddingWidth=5):
    # along x direction
    padding=np.zeros((paddingWidth,im.shape[1]),np.uint8)
    im=np.concatenate((padding,im, padding),axis=0)
    # along y direction
    padding=np.zeros((im.shape[0],paddingWidth),np.uint8)
    return np.concatenate((padding,im, padding),axis=1)

def paddingOfImage_RGBA(im,paddingWidth=5):
    # along x direction
    padding=np.zeros((paddingWidth,im.shape[1],4),np.uint8)
    im=np.concatenate((padding,im, padding),axis=0)
    # along y direction
    padding=np.zeros((im.shape[0],paddingWidth,4),np.uint8)
    return np.concatenate((padding,im, padding),axis=1)

def removePaddingOfImage(im,paddingWidth=5):
    return im[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth]

from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

class HandlerColormap(HandlerBase):
    # allows use of colorbar as a patch in legend, 
    # taken from https://stackoverflow.com/questions/55501860/how-to-put-multiple-colormap-patches-in-a-matplotlib-legend
    def __init__(self, cmap, num_stripes=8, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_stripes = num_stripes
    def create_artists(self, legend, orig_handle, 
                       xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle([xdescent + i * width / self.num_stripes, ydescent], 
                          width / self.num_stripes, 
                          height, 
                          fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)), 
                          transform=trans)
            stripes.append(s)
        return stripes

# find most frequent element in a list 
def most_frequent(List): 
    return max(set(List), key = List.count) 


def blobDetection(grayImg,imSlice,convexityDefectDist,plotConvexityDefects=False,useNegative=True):
    # Detect blobs by way of convexity defect

    if useNegative:# required for threshold on V_fibers, not on binary mask from voxelMap
        makeNegative(grayImg)

    ret, thresh = cv.threshold(grayImg, 127, 255,0)
    contours,hierarchy = cv.findContours(thresh,2,1)

    if plotConvexityDefects:
        if useNegative==False:
            #use negative only for plot, not convexity testing
            makeNegative(thresh)
        img=cv.cvtColor(thresh,cv.COLOR_GRAY2BGR)

    centroids=[]

    for iCnt,cnt in enumerate(contours):

        hull = cv.convexHull(cnt,returnPoints = False)

        if plotConvexityDefects:
            cv.drawContours(img, [cnt], 0, (165,0,255), 1)

        try:
            defects = cv.convexityDefects(cnt,hull)
        except cv.error as e:
            if e.err!='The convex hull indices are not monotonous, which can be in the case when the input contour contains self-intersections':
                raise RuntimeError("unhandled exception in convexityDefects")
            # The convex hull indices are not monotonous, 
            # which can be in the case when the input contour contains self-intersections in function 'convexityDefects'
            hullLst=list(hull)
            hullLst.sort(key=lambda x:x[0],reverse=True)
            hull=np.array(hullLst)
            defects = cv.convexityDefects(cnt,hull)

        # to avoid flagging the same contour more than once
        flagged=False

        if defects is not None:
            for i in range(defects.shape[0]):
                if flagged:
                    continue
                s,e,f,d = defects[i,0]
                if d/256.0>convexityDefectDist:
                    flagged=True
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    #find centroid of contour
                    m=cv.moments(cnt)
                    if m["m00"]>0.: # avoids colapsed contours with no area
                        x=m["m10"]/m["m00"]
                        y=m["m01"]/m["m00"]

                        # construct centroid object
                        centroids.append(centroidObj(y,x,cnt)) 
                        # x and y are flipped to conform to standard in the rest of the project, where voxelMap[x,y].
                        # imshow() transposes the image, so text() must be added as text(y,x) elsewhere but here

                        if plotConvexityDefects:
                            print("distance: {: >8.3f}\t contour: {}\t Defect_index: {} ##################### FLAGGED".format(d/256.0,iCnt,i))
                            cv.line(img,start,end,[0,255,0],1)

                            cv.circle(img,(int(x),int(y)),20,[0,0,255],1)
                            cv.putText(img, text="{}".format(iCnt), org=(far[0]+10,far[1]-10),# to the right and above
                                fontFace= cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(220,10,40),
                                thickness=2, lineType=cv.LINE_AA) 


    if plotConvexityDefects:
        from matplotlib import pyplot as plt

        plt.figure(figsize=[8,8])
        plt.imshow(img)
        plt.title("convexity defect detection result, imSlice={}".format(imSlice),fontsize=22)
        plt.tight_layout()
        plt.show()

    return centroids

def addPatchToLegend(legend,color,label):
    ax = legend.axes

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(facecolor=color, edgecolor=color))
    labels.append(label)

    legend._legend_box = None
    legend._init_legend_box(handles, labels)
    legend._set_loc(legend._loc)
    legend.set_title(legend.get_title().get_text())


def makeBinaryMap(voxelMap):
    binaryVoxelMap=np.array(voxelMap,np.int16)

    maxVal=np.iinfo(binaryVoxelMap.dtype).max   # ==32767 for signed int16 (wont be a problem unless fiber number exceeds in this slice, which can't be contained in voxelMap anyway)

    binaryVoxelMap[binaryVoxelMap>2]= maxVal    # marker values for individual connected regions
    binaryVoxelMap[binaryVoxelMap<=2]=0         # value for background and outlines
    binaryVoxelMap[binaryVoxelMap==maxVal]=255  # remap for np.uint8

    return np.array(binaryVoxelMap,np.uint8) 

def watershedTransform(imageSliceGray,
    imageHist,
    imSlice,
    initialWaterLevel,
    waterLevelIncrements,
    convexityDefectDist,
    checkConvexityAndSplit          =True,
    plotInitialAndResults           =False,
    plotEveryIteration              =False,
    plotWatershedStepsGlobal        =False,
    plotWatershedStepsMarkersOnly   =False,
    openingBeforeWaterRising        =False,
    plotConvexityDefects            =False,
    paddingWidth                    =5,
    doNotPltShow                    =False,
    figsize                         =[8,8],
    legendFontSize                  =32,
    titleFontSize                   =22,
    textFontSize                    =26
    ):

    # if initialWaterLevel is kept below 1.0 small regions are not erased, but produces many spurious detections
    # cv.watershed is used only to obtain initial labelling (markers) of connected regions. 

    voxelMap_slice, grayImg, img3channel, paddingWidth,seeds,dist_transform=findWatershedMarkers(
        imageSliceGray,
        initialWaterLevel,
        imSlice=imSlice,
        plotEveryStep=plotWatershedStepsGlobal
        )

    binaryVoxelMap=makeBinaryMap(voxelMap_slice)

    if checkConvexityAndSplit:
        #detect blobs that have convexity defects, which need to be split by rising waterLevel (for some datasets this is unneccessary)
        centroids = blobDetection(binaryVoxelMap,imSlice,convexityDefectDist,plotConvexityDefects,useNegative=False)
    else:
        centroids=[]


    seeds=voxelMap_slice
    seeds[1,1]=seeds[0,0] # remove hack at (1,1)

    markerList_notConvex=[]     
    for iC,centroid in enumerate(centroids):
        markerList_notConvex.append(centroid.getMarker(seeds,exclusionList=[-1,2]) )

    markerList_notConvex.sort()

    if plotInitialAndResults:

        plt.rcParams.update({'font.size': 26})
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams["font.family"] = "Times New Roman"

        cmapStr_voxelMap="Blues"
        singleColor_convex=(0.2,0.5,0.7,1.) #use None for color gradient instead

        cmapStr_voxelMap_reProcessed='Wistia'#'YlGn'#'plasma'

        numColors_reProcessed=5

        cmap_rejected=cm.get_cmap('autumn') #Reds
        singleColor_rejected=(0.68,0.11,0.,1.)

        fig=plt.figure(figsize=figsize,num="rejectedSeedsConvexityTest_imSlice{}".format(imSlice))

        imageHist=paddingOfImage(imageHist,paddingWidth)

        voxelImg=overlayVoxelMapToHist_composite(
            voxelMap_slice [paddingWidth:-paddingWidth,paddingWidth:-paddingWidth],
            imageHist[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth],
            exclusionList=[2,0,-1],
            cmapStr=cmapStr_voxelMap,
            singleColor=singleColor_convex
            )

        plt.imshow(voxelImg)


        temp=paddingOfImage_RGBA(voxelImg)

        for i,centroid in enumerate(centroids):
            temp=centroid.addFilledContourToImg(temp,color=(1.,0.,0.,1.))

        imgExtracted=temp[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth,:]

        plt.title("result convexity test, imSlice={}".format(imSlice), fontsize=titleFontSize)
        uniqueLabel=True

        for iC,centroid in enumerate(centroids):
            #x and y are transposed in imshow() convention
            pnt=centroid.getPnt()
            marker=centroid.getMarker()
            if uniqueLabel:#single legend entry
                plt.scatter(pnt[1]-paddingWidth,  pnt[0]-paddingWidth,  s=100,c="red",label="Rejected Blobs (##_##=> markerNum_contourNum)")
                plt.text   (pnt[1]+5-paddingWidth,pnt[0]+5-paddingWidth,s="{}_{}".format(marker,iC),c="red",fontsize=textFontSize)
                uniqueLabel=False
            else:
                plt.scatter(pnt[1]-paddingWidth,  pnt[0]-paddingWidth,  s=100,c="red")
                plt.text   (pnt[1]+5-paddingWidth,pnt[0]+5-paddingWidth,s="{}_{}".format(marker,iC),c="red",fontsize=textFontSize)

        plt.legend(fontsize=legendFontSize,framealpha=1.)
        fig.tight_layout(pad=0.05)

        fig=plt.figure(figsize=figsize,num="resultConvexityTest_imSlice{}".format(imSlice))

        plt.imshow(imgExtracted)

        plt.title("result Convexity Test imSlice={}".format(imSlice), fontsize=titleFontSize)
        fig.tight_layout(pad=0.05)

        cmaps = [plt.cm.get_cmap(cmapStr_voxelMap), cmap_rejected] 

        cmap_labels = [ "Single fiber blobs","Rejected blobs"]
        if singleColor_convex is None:
            # create proxy artists as handles:
            cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
            handler_map = dict(zip(cmap_handles, 
                                [HandlerColormap(cm, num_stripes=32) for cm in cmaps]))
            # cmap_handles[1].set_color(singleColor_rejected)
            # del handler_map[cmap_handles[1]]

            plt.legend(handles=cmap_handles, 
                    labels=cmap_labels, 
                    handler_map=handler_map, 
                    fontsize=legendFontSize,framealpha=1.)

        else:
            # create a patch (proxy artist) for every color
            colors=[singleColor_convex,singleColor_rejected] 
            patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=cmap_labels[i]) ) for i in range(len(cmap_labels)) ]
            # put those patched as legend-handles into the legend
            plt.legend(handles=patches,fontsize=legendFontSize,framealpha=1.)

    for markerVal in markerList_notConvex:
        if plotEveryIteration:
            print("markerVal",markerVal)

        #starting point for seeds for this marker
        binaryMapThisMarker=np.zeros(seeds.shape)

        binaryMapThisMarker[seeds==markerVal]=1 #markerVal True value

        seedsThisMarker = findWatershedMarkers(
            binaryMapThisMarker,
            initialWaterLevel,
            imSlice=imSlice,
            paddingWidth=0,
            plotEveryStep=plotWatershedStepsMarkersOnly,
            fromThresh=True # use binary map from threshold of input instead of eroded sure_fg (foreground) 
                            # as input to find connected regions. So that small regions are not erased
            )[4]

        numBefore=len(np.unique(seedsThisMarker))

        if openingBeforeWaterRising:
            # the effect is negligible. sometimes splits region in two, 
            # but where raising level would probably do it as well

            seedsThisMarker=np.array(seedsThisMarker,np.uint8) #morphologyEx cant use int32

            cnts = cv.findContours(seedsThisMarker, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2]
            numBefore= len(cnts)

            kernel = np.ones((4, 4), np.uint8)

            seedsThisMarkerTrial = cv.morphologyEx(seedsThisMarker, cv.MORPH_OPEN, kernel)

            cnts = cv.findContours(seedsThisMarkerTrial, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[-2]
            numAfter= len(cnts)

            if numAfter>numBefore: # if blob was split by opening
                # this prevents the use of opening if the number of blobs is unchanged
                seedsThisMarker=seedsThisMarkerTrial

        if plotEveryIteration:
            fig=plt.figure(figsize=figsize,num="initial_seeds_map_UNIQUE_marker{}_imSlice{}".format(markerVal,imSlice))
            plt.imshow(seedsThisMarker,cmap="binary")
            plt.title("initial seeds map UNIQUE marker={}, imSlice={}".format(markerVal,imSlice), fontsize=titleFontSize)
            fig.tight_layout(pad=0.05)

            print("numBefore",numBefore)

        keepRising=True
        if waterLevelIncrements<0: #don't do iterative water raising
            keepRising=False

        risingCounter=0 #to avoid getting stuck in while loop
        waterLevel=initialWaterLevel
        while keepRising and risingCounter<12:
            risingCounter+=1
            waterLevel+=waterLevelIncrements

            seedsHighWater = findWatershedMarkers(seedsThisMarker,waterLevel,paddingWidth=0)[4] # get new connectedMap

            seedsHighWater[1,1]=seedsHighWater[0,0] # remove hack at (1,1)
           
            numAfter=len(np.unique(seedsHighWater))

            if plotEveryIteration:
                fig=plt.figure(figsize=figsize,num="waterLevel_raised_seedsMap_marker{}_imSlice{}".format(markerVal,imSlice))
                plt.imshow(seedsHighWater)
                plt.title("waterLevel raised seeds map, marker={}, imSlice={}".format(markerVal,imSlice), fontsize=titleFontSize)
                plt.tight_layout()
                if ~doNotPltShow:
                    plt.show()

                print("numAfter",numAfter)

            if numAfter<numBefore or numAfter==1:
                keepRising=False
            else: # number of connected regions has increased or remained at n>1
                if numAfter>numBefore:
                    #check if a new connected region had convexity defects,if True split it to another while loop
                    
                    newMarkerList=list(np.unique(seedsHighWater))
                    newMarkerList.remove(0) # background value

                    # if there isn't more than one connected region after new waterLevel, no need to check for
                    # convexity defect: the only region present will already go through more cycles
                    if len(newMarkerList)>1: 

                        seeds[seeds==markerVal]=0

                        for newMarker in newMarkerList:

                            # allocate memory for the label region and draw
                            # it on the (binary) mask
                            mask = np.zeros(seedsHighWater.shape, dtype="uint8")
                            mask[seedsHighWater == newMarker] = 255 #True value

                            # detect contours in the mask and check if single connected region is present
                            cnts = cv.findContours(mask, cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)[-2]
                            if len(cnts)>1:
                                print("\t\tMarker is more than one connected region, imSlice={}, marker={}".format(imSlice,marker))
                                for iContour,contourObj in enumerate(cnt):
                                    print("\t\t\tcontour #{}, area: {}".format(iContour,cv.contourArea(contourObj)))
                                cnt = [max(cnt, key=cv.contourArea)] #needs to be a list to work with centroidObj

                            #check if the newly created regions themselves are not convex, create a new marker and process separately
                            centroids = blobDetection(mask,imSlice,convexityDefectDist,plotConvexityDefects=False)
                            
                            if centroids:

                                for centroid in centroids:

                                    newMarkerVal=centroid.getMarker(seedsHighWater,exclusionList=[0],partialMap=True)
                                    
                                    if newMarkerVal is not None:
                                        nMarkers=np.max(seeds)
                                        seeds[seedsHighWater==newMarkerVal]=nMarkers+1
                                        seedsHighWater[seedsHighWater==newMarkerVal]=0

                                        markerList_notConvex.append(nMarkers+1)

                            else:
                                seeds[mask==0]=markerVal
                                                

                numBefore=numAfter
                binaryMapThisMarker=np.zeros(seeds.shape)

                binaryMapThisMarker[seedsHighWater==markerVal]=1#markerVal
                seedsThisMarker=seedsHighWater

        
        #seedThisMarker is the last time the number of connected regions increased -> keep
        seeds[seeds==markerVal]=0 #mark as unknown

        labelsNew=np.delete(np.unique(seedsThisMarker),0) #get unique markers that are not 0 (background)

        for label in labelsNew:
            if label==min(labelsNew): # this markerVal was already present in seeds
                seeds[seedsThisMarker==label]=markerVal
            else:        # all new markerVals that need to be added to seeds 
                newMarker=np.max(seeds)+1
                seeds[seedsThisMarker==label]=newMarker
                markerList_notConvex.append(newMarker)

        if plotEveryIteration:
            fig=plt.figure(figsize=figsize,num="updated_seeds_map_after_rising_waterLevel_to_{: 4.2f}".format(waterLevel-waterLevelIncrements))
            plt.imshow(seeds)
            plt.title("updated seeds map after rising waterLevel to {: 4.2f}".format(waterLevel-waterLevelIncrements), fontsize=titleFontSize)
            plt.tight_layout()
            plt.show()
            fig.clf()

    connectedMap=seeds.copy()

    #add dilation to unknown region so small regions are not eroded by -1
    unknown=np.zeros(seeds.shape,np.uint8)
    unknown[seeds==0]=255

    kernel = np.ones((3,3),np.uint8)
    # sure background area
    unknown = cv.dilate(unknown,kernel,iterations=3)

    backGroundMarker=most_frequent(list(seeds.ravel()))

    for ix in range(seeds.shape[0]):
        for iy in range(seeds.shape[1]):
            if seeds[ix,iy]==backGroundMarker and unknown[ix,iy]==255:
                seeds[ix,iy]=0

    voxelMap_slice = cv.watershed(img3channel,seeds)
    #output of watershed: 
    # usually:
    # background is 2
    # outline is -1, OUTSIDE of the marked region in original input
    # labeled regions range from 3 to n
    # else: handled after plotting to conform to this

    # EDGE case: sometime background has the wrong marker, 
    # however, even for high filling ratio, it will be by far the most common marker, 
    # identify most common marker, change it to 2 if not 2 already
    backGroundMarker=most_frequent(list(voxelMap_slice.ravel()))
    if backGroundMarker !=2:
        if np.max(voxelMap_slice)>3:
            # some validation datasets have only a single fiber (marker==3), 
            # which will occupy the majority of the space, but this is not a problem
            voxelMap_slice[voxelMap_slice==1]=-999999
            voxelMap_slice[voxelMap_slice==backGroundMarker]=2
            voxelMap_slice[voxelMap_slice==-999999]=1

    #fix a bug where sometimes a label is given to a pixel outside the original mask
    voxelMap_slice[grayImg==0]=2 #background marker

    if plotInitialAndResults:
        plt.figure(figsize=figsize,num="voxelMap")
        plt.imshow(voxelMap_slice, cmap="gist_stern_r")

        plt.title("voxelMap",fontsize=titleFontSize)
        plt.tight_layout()

        fig=plt.figure(figsize=figsize,num="voxelMap_after_waterLevel_rising_cycles_imSlice{}".format(imSlice))

        voxelImg=overlayVoxelMapToHist_composite(
            # np.zeros(voxelMap [paddingWidth:-paddingWidth,paddingWidth:-paddingWidth].shape),
            voxelMap_slice [paddingWidth:-paddingWidth,paddingWidth:-paddingWidth],
            imageHist[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth],
            exclusionList=[2,0,-1],
            cmapStr=cmapStr_voxelMap,
            singleColor=singleColor_convex
            )

        markerList_reProcessed=[]
        for m in markerList_notConvex:
            if m in voxelMap_slice:
                markerList_reProcessed.append(m)

        cmap_reProcessed=cm.get_cmap(cmapStr_voxelMap_reProcessed)

        colorsReProcessed=cmap_reProcessed(np.linspace(0,1,6))#len(markerList_reProcessed)))

        shuffle(colorsReProcessed)

        imgExtracted=voxelImg
                
        for i,m in enumerate(markerList_reProcessed):
            x,y=np.where(voxelMap_slice[paddingWidth:-paddingWidth,paddingWidth:-paddingWidth]==m)
            if len(x)>1:
                imgExtracted[x,y,:]=colorsReProcessed[i%numColors_reProcessed]

            x=[]
            y=[]

        plt.imshow(imgExtracted)
        plt.title("voxelMap after waterLevel rising cycles, imSlice={}".format(imSlice), fontsize=titleFontSize)
        fig.tight_layout(pad=0.05)

        cmaps = [plt.cm.get_cmap(cmapStr_voxelMap), cmap_reProcessed] 

        cmap_labels = [ "Single fiber blobs","Re-processed blobs"]
        # create proxy artists as handles:
        cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps]
        handler_map = dict(zip(cmap_handles, 
                            [HandlerColormap(cm, num_stripes=numColors_reProcessed) for cm in cmaps]))

        if singleColor_convex is not None:
            cmap_handles[0].set_color(singleColor_convex)
            del handler_map[cmap_handles[0]]

        plt.legend(handles=cmap_handles, 
                labels=cmap_labels, 
                handler_map=handler_map, 
                fontsize=legendFontSize,
                framealpha=1.)


    if not doNotPltShow and (plotInitialAndResults or plotWatershedStepsGlobal or plotWatershedStepsMarkersOnly ):
        plt.show()


    return voxelMap_slice, grayImg, img3channel, paddingWidth,connectedMap,dist_transform



def findWatershedMarkers(imageSliceGray, 
    waterLevel, 
    imSlice=None,
    paddingWidth=5, 
    plotEveryStep=False,
    fromThresh=False,
    titleFontSize=18
    ): #from_thresh is set to True in the case where NO waterLevel raising is required, i.e.
    #to keep small regions with dist<=1 
    #the paddingWidth argument is set to 0 in the iterative waterRaising algorithm, as padding was already applied


    #scaling to 255 and then padding
    grayImg = np.array(imageSliceGray,np.uint8)
    maxVal=np.max(grayImg)
    if maxVal==1:
        grayImg=grayImg*255
    elif maxVal!=255:
        grayImg[grayImg==maxVal]=255
        # raise ValueError("Binary mask has wrong high value of {}".format(maxVal) )
        
    if paddingWidth!=0:
        grayImg=paddingOfImage(grayImg,paddingWidth)

        if not fromThresh:
            # hack to avoid case where background is confused as a closed contour,
            # occurs if perimeter goes all around the image 
            grayImg[1,1]=255 

    #watershed segmentation requires a three channel image
    img = np.stack([grayImg,grayImg,grayImg],axis=2)

    ret, thresh = cv.threshold(grayImg,0,255,cv.THRESH_BINARY)

    if plotEveryStep:
        figureHandle=plt.figure(figsize=[12,8])
        ax1 = figureHandle.add_subplot(2, 3, 1)
        ax2 = figureHandle.add_subplot(2, 3, 2)
        ax3 = figureHandle.add_subplot(2, 3, 3)
        # ax4 = figureHandle.add_subplot(2, 3, 4)
        ax5 = figureHandle.add_subplot(2, 3, 5)
        ax6 = figureHandle.add_subplot(2, 3, 6)
        figureHandle.tight_layout(pad=3)

        ax1.imshow(thresh,cmap="binary")
        ax1.title.set_text("thresh on original input image, imSlice={}".format(imSlice))
        ax1.title.set_fontsize(titleFontSize)

    # noise removal
    kernel = np.ones((3,3),np.uint8)

    # sure background area
    sure_bg = cv.dilate(thresh,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(thresh,cv.DIST_L2,5)

    #better to set an absolute threshold rather than relative, as if a large region is present
    #the max distance will be larger than usual

    ret, sure_fg = cv.threshold(dist_transform,waterLevel,255,cv.THRESH_BINARY) 

    sure_fg[1,1]=255 #hack to avoid case where background is confused as a closed contour, if perimeter goes all aroung image


    if plotEveryStep:

        ax3.imshow(sure_fg,cmap="binary")
        ax3.title.set_text("sure_fg,waterLevel={}".format(waterLevel))
        ax3.title.set_fontsize(titleFontSize)

        ax2.imshow(dist_transform)
        ax2.title.set_text("dist_transform")
        ax2.title.set_fontsize(titleFontSize)


    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    # Marker labelling
    if fromThresh:
        ret, connectedMap = cv.connectedComponents(thresh,connectivity = 8)
    else:
        ret, connectedMap = cv.connectedComponents(sure_fg,connectivity = 8)

    # substract 1 to all labels so that "sure" background is not 0, but -1
    temp = connectedMap-1

    temp[temp==-1]=-3
    temp+=2

    # Now, mark the region of unknown with zero
    temp[unknown==255] = 0

    if plotEveryStep:

        ax5.imshow(temp,cmap="gist_stern")
        ax5.title.set_text("connectedMap+unkown")
        ax5.title.set_fontsize(titleFontSize)


    voxelMap_slice = cv.watershed(img,temp)
    #output of watershed: 
    # background is 2
    # outline -1, intersecting outline 0
    # labeled regions range from 3 to n

    # EDGE case: sometime background has the wrong marker, 
    # however, even for high filling ratio, it will be by far the most common marker, 
    # identify most common marker, change it to 2 if not 2 already
    backGroundMarker=most_frequent(list(voxelMap_slice.ravel()))
    if backGroundMarker !=2:
        if np.max(voxelMap_slice)>3:
            # some validation datasets have only a single fiber (marker==3), 
            # which will occupy the majority of the space, but this is not a problem
            voxelMap_slice[voxelMap_slice==1]=-999999 #dummy temporary label
            voxelMap_slice[voxelMap_slice==backGroundMarker]=2
            voxelMap_slice[voxelMap_slice==-999999]=1

    #fix a bug where sometimes a label is given to a pixel outside the original mask
    voxelMap_slice[grayImg==0]=2 #background marker

    if plotEveryStep:

        ax6.imshow(voxelMap_slice)
        ax6.title.set_text("voxelMap")
        ax6.title.set_fontsize(titleFontSize)
        

    return voxelMap_slice, grayImg, img, paddingWidth,connectedMap,dist_transform



def addFlaggedPixelsToImg(img, flaggedPixels,color=None,alpha=0.5):
    for iFlag in flaggedPixels:
        x=iFlag[0]
        y=iFlag[1]
        if len(iFlag)==3:
            color=[int(round(val*255)) for val in iFlag[2]]
        img[x,y,:]=np.array(np.array(img[x,y,:],np.float)*(1.-alpha)+np.array(color,np.float)*alpha,np.uint8)

def overlayVoxelMapToHist(voxelMap,imageHist,xOffset=0,yOffset=0,exclusionList=[1,0,-1],scaleInt:int=1):
    #the manipulations here are so the original data (V_hist) are discernible, and the voxel markers remain the original integer label.
    #original data will be negative floats, and voxels positive integers. imshow() does the colormap scaling automatically
    #exclusion list is [1,0,-1] if processing voxels from watershed output ([background, intersection of outline, outline])
    #exclusion list is [2,0,-1] if processing seeds (connected regions) from watershed output

    if type(scaleInt) != type(1):
        scaleInt=int(scaleInt)

    if scaleInt<0:
        scaleInt=1

    voxelsImg=np.array(voxelMap*scaleInt,np.float)#PEEK30, max(voxelMap)=1336

    imgSegment=imageHist[xOffset:xOffset+voxelsImg.shape[0],yOffset:yOffset+voxelsImg.shape[1]]

    rescaleHistFloat=np.max(voxelMap)/np.max(imageHist)/4

    for iMarker in exclusionList: 
        #replace background, outline and intersection of outlines as original image
        voxelsImg[voxelsImg==iMarker*scaleInt]=-rescaleHistFloat*imgSegment[voxelsImg==iMarker*scaleInt]

    return voxelsImg/scaleInt

def overlayVoxelMapToHist_composite(voxelMap,
    imageHist,
    exclusionList=[1,0,-1],
    attenuation=0.4,
    cmapStr="viridis",
    randomize=False,
    singleColor=None):
    #the manipulations here are so the original data (V_hist) are discernible, and the voxel markers remain the original integer label.
    #original data will be negative floats, and voxels positive integers. imshow() does the colormap scaling automatically
    #exclusion list is [1,0,-1] if processing voxels from watershed output ([background, intersection of outline, outline])
    #exclusion list is [2,0,-1] if processing seeds (connected regions) from watershed output

    imageHistRGB=cm.binary_r(imageHist)
    
    imageHistRGB[:,:,:3]=imageHistRGB[:,:,:3]*attenuation

    # to use the full range of the colormap requires to normalize between [0,1]
    cmap=cm.get_cmap(cmapStr)
    norm=plt.Normalize()

    if randomize:
        voxelMap=randomizeVoxelSlice(voxelMap,np.unique(voxelMap),exclusionList)

    if singleColor is None:
        compositeImage=cmap(norm(voxelMap))
    else:
        compositeImage=np.zeros(imageHistRGB.shape)
        compositeImage[voxelMap>1]=singleColor


    for iMarker in exclusionList:
        x,y=np.where(voxelMap==iMarker)

        compositeImage[x,y,:]=imageHistRGB[x,y,:]

    return compositeImage

def assignVoxelsToFibers(
    allCenterPoints,
    fiberData,
    V_voxels,
    watershedData,
    imSlice,
    slicesRange,
    V_hist_slice            =None,
    V_fibers_slice          =None,
    V_fibers_masked_slice   =None,
    axisHandles             =None,
    reassignVoxels          =True,
    verbose                 =False,
    fiberRadius             =5,
    addDisksAroundCenterPnts=True,
    xOffset                 =0,
    yOffset                 =0,
    textLabels              =True,
    textFontSize            =15
    ):
    # V_hist is only used in validation plots (makePlots=True)

    print("assigning Voxels to Fibers on slice {}/{} on {}".format(imSlice,slicesRange,multiprocessing.current_process().name))

    paddingWidth=0

    SE_disk=morphology.disk(fiberRadius, dtype=np.uint8)*255

    circle_x,circle_y=np.where(SE_disk==255)

    #markers from watershed segmentation (centroid objects have already been matched at extractCenterPoints() )
    fiberMap_slice=V_voxels.copy()
    rejectedMask=np.zeros(fiberMap_slice.shape,np.bool)

    #make list that are easier to iterate over
    watershedCentroids  = [ [centroid.getPnt()[0],centroid.getPnt()[1]] for centroid in watershedData ]
    watershedCentroids  = np.array(watershedCentroids)

    watershedMarkers    = [ centroid.getMarker() for centroid in watershedData ]

    if axisHandles and axisHandles[1]:
        #three channel image of slice for display only
        grayImg = np.array(V_fibers_slice*255,np.uint8)
        grayImg_hist = np.array(V_hist_slice,np.uint8)

        x,y=np.where(grayImg==255)
        flaggedPixels=[]
        for i in range(len(x)):
            flaggedPixels.append((x[i]+xOffset,y[i]+yOffset))

        # attenuate grayImg_hist to make it more readable
        
        oldRange=[0,255]
        newRange=[0,120]
        
        grayImg_hist=np.array(np.round(np.interp(grayImg_hist,oldRange,newRange)),np.uint8)

        imgComp = np.stack([grayImg_hist,grayImg_hist,grayImg_hist],axis=2)

        addFlaggedPixelsToImg(imgComp,flaggedPixels,color=[0,38,230],alpha=0.7)

        #different range for axisHandle[2]
        newRange=[100,230]
        grayImg_hist=np.array(np.round(np.interp(grayImg_hist,oldRange,newRange)),np.uint8)


    dataFib={}
    for keysFib in allCenterPoints.keys():
        dataFib[keysFib]=allCenterPoints[keysFib].copy() # deep copy of dict representing fibers, will be modified with padding distance

    colorsDict=fiberData["colors"]
    colorLabels_legend={}

    fibTrackingCentroids=[] # coordinates only, need to be np.array for knn() to work. 

    for i,iFib in enumerate(dataFib["fiberID"]):
        #padding required for watershed transform
        dataFib["x"][i]+=paddingWidth
        dataFib["y"][i]+=paddingWidth

        fibTrackingCentroids.append([dataFib["x"][i],dataFib["y"][i]])


    
    if axisHandles:
        flaggedPixelsMatched=[]
        flaggedPixelsUnmatched=[]
        watershedTrackedBool=[False]*len(watershedCentroids) # for validation only, not stored
    
    fibTrackingCentroids=np.array(fibTrackingCentroids)

    
    distLateral_knn=10 #TODO: put this in parameters
    if len(fibTrackingCentroids)>0 and len(watershedCentroids)>0: #if not empty
        id_watershed_th,id_tracking_th = knn(fibTrackingCentroids,watershedCentroids,distLateral_knn)
    else:
        id_watershed_th=[]
        id_tracking_th =[]

    allCenterPoints["markerToFiberID_lookUpTable"]={}
    allCenterPoints["fiberIDToMarker_lookUpTable"]={}

    for iMarker in watershedMarkers:
        allCenterPoints["markerToFiberID_lookUpTable"][iMarker]=None # tag all fibers with default marker

    # tag matched pairs of fiber-watershedCentroids
    for iCt in range(len(id_watershed_th)):
        xTrack=fibTrackingCentroids[id_tracking_th[iCt],0]
        yTrack=fibTrackingCentroids[id_tracking_th[iCt],1] 

        xWater=watershedCentroids[id_watershed_th[iCt],0]
        yWater=watershedCentroids[id_watershed_th[iCt],1]  

        allCenterPoints["markerToFiberID_lookUpTable"][watershedMarkers[id_watershed_th[iCt]]]=\
            allCenterPoints["fiberID"][id_tracking_th[iCt]]

        allCenterPoints["fiberIDToMarker_lookUpTable"][ allCenterPoints["fiberID"][id_tracking_th[iCt]] ]=\
            watershedMarkers[id_watershed_th[iCt]]
            
        if axisHandles and axisHandles[1]:
            axisHandles[1].plot([xWater, xTrack],[yWater, yTrack],color="y",linewidth=3)

            watershedTrackedBool[id_watershed_th[iCt]]=True # untracked watershed centroids, for validation only, not stored



    # indicate regions that were not successfully matched between watershed segt and fiber tracking
    unmatchedMarker=-999999
    backgroundMarker=-1

    listUnmatched=allCenterPoints["fiberID"].copy()
    listMatched=[]

    # look up table, number in list trackedCenterPoints
    fiberID_toNum_lut={}

    for fiberNum,fiberID in enumerate(allCenterPoints["fiberID"]):
        fiberID_toNum_lut[fiberID]=fiberNum

    for iX in range(fiberMap_slice.shape[0]):
        for iY in range(fiberMap_slice.shape[1]):
            # every pixel is checked, then assigned to either background or matched fiber. 

            if fiberMap_slice[iX,iY] in [-1,0,2]: 
                #-1 was contour, 0 was intersection of contours, 2 was backgroud, but we need 0,1,2,... for fiberObjs
                fiberMap_slice[iX,iY] = backgroundMarker
            elif allCenterPoints["markerToFiberID_lookUpTable"][fiberMap_slice[iX,iY]] is not None:
                matchedFiberID=int(allCenterPoints["markerToFiberID_lookUpTable"][fiberMap_slice[iX,iY]])
                if matchedFiberID not in listMatched:
                    listMatched.append(matchedFiberID)
                    listUnmatched.remove(matchedFiberID)

                if fiberData["rejected"][matchedFiberID]:
                    rejectedMask[iX,iY]=True

                fiberMap_slice[iX,iY]= matchedFiberID

                if axisHandles:
                    color=colorsDict[fiberData["LUT_fiberID_to_color"][matchedFiberID]]
                    colorLabels_legend[fiberData["LUT_fiberID_to_color"][matchedFiberID]]=color
                    flaggedPixelsMatched.append((iX+xOffset,iY+yOffset,color))
            else:
                fiberMap_slice[iX,iY]=unmatchedMarker # failed to match
                if axisHandles:
                    flaggedPixelsUnmatched.append((iX+xOffset,iY+yOffset))

    ########################################################################################

    ### reassignment: check if unmatched centre points are in a region segmented as fiber:

    ########################################################################################

    markerReassignmentList={}# keys are the markers (fiberID_matched) for regions that need to be split (reassigned)

    addLegendEntry=False

    if reassignVoxels:
        for iCt in listUnmatched:
            unmatchedfiberID=allCenterPoints["fiberID"][fiberID_toNum_lut[iCt]]
            

            xx=int(allCenterPoints["x"]      [fiberID_toNum_lut[iCt]])-xOffset
            yy=int(allCenterPoints["y"]      [fiberID_toNum_lut[iCt]])-yOffset
            
            #attempt to match on the basis of centroid coordinate
            matchedFiberID=fiberMap_slice[xx,yy] #will be negative for rejected fiberObj
            
            if verbose:
                print("\n\n")
                print("\tunmatched fiber:")
                print("\t\tfiber number: ",fiberID_toNum_lut[iCt])

                print("\t\tfiberID       ",unmatchedfiberID)
                print("\t\tx             ",xx)
                print("\t\ty             ",yy)

                print("\tfiberID_matched in fiberMap",matchedFiberID)
            
            if matchedFiberID in [-1,-999999]: # -1==background marker, -999999==unMatched marker
                if verbose:
                    print("\t\tnot in shared region, pass") #will be filled in by post-processing
            else:
                xxTracked=int(allCenterPoints["x"][fiberID_toNum_lut[matchedFiberID]])-xOffset
                yyTracked=int(allCenterPoints["y"][fiberID_toNum_lut[matchedFiberID]])-yOffset
                if verbose:
                    print("\t\tfiberNum:",fiberID_toNum_lut[matchedFiberID])
                    print("\t\txTracked             ",xxTracked)
                    print("\t\tyTracked             ",yyTracked)

            if matchedFiberID not in [-1,-999999] and matchedFiberID not in markerReassignmentList.keys():
                #first untracked fiber to be assigned to this region (marker)
                markerReassignmentList[matchedFiberID]=[{
                    "fiberID_unmatched" :unmatchedfiberID,
                    "x"                 :xx,
                    "y"                 :yy,
                    "matched to:"       :matchedFiberID,
                    "xMatched"          :xxTracked,
                    "yMatched"          :yyTracked
                    }]
            elif matchedFiberID not in [-1,-999999]:
                #following untracked fibers in same region
                markerReassignmentList[matchedFiberID].append({
                    "fiberID_unmatched" :unmatchedfiberID,
                    "x"                 :xx,
                    "y"                 :yy,
                    "matched to:"       :matchedFiberID,
                    "xMatched"          :xxTracked,
                    "yMatched"          :yyTracked
                    })
                
        if markerReassignmentList:

            for matchedFiberID in markerReassignmentList:
                
                mask=np.zeros(fiberMap_slice.shape,np.uint8)

                mask[fiberMap_slice==matchedFiberID]   =255

                seeds=np.zeros(fiberMap_slice.shape,np.int32)# temporary background value
                seeds[mask==255]            =255             # will be set to zero afterwards

                seeds[seeds==0  ]=-1 # backgroundValue
                seeds[seeds==255]= 0 # unknown value, to be attributed by watershed function 

                # adding offset serves to avoid ambiguity with fiber numbers that begin at 0, while for :
                # -1: background
                #  0: outline
                #  1: hack at [1,1]
                offset=2
                seeds=seeds-offset

                fiberID_unmatchedList=[]

                #create a seed map with all the fibers contained in this closed contour
                for fiberInfo in markerReassignmentList[matchedFiberID]:
                    unmatchedfiberID   =fiberInfo["fiberID_unmatched"]
                    x                   =fiberInfo["x"]
                    y                   =fiberInfo["y"]
                    xMatched            =fiberInfo["xMatched"]
                    yMatched            =fiberInfo["yMatched"]

                    seeds[x,  y]                  =unmatchedfiberID     # seed value
                    seeds[xMatched,  yMatched]    =matchedFiberID       # seed value

                    fiberID_unmatchedList.append(unmatchedfiberID)

                seeds=seeds+offset # so fiberID start at 2

                seeds[1,1]=1 #hack, deals with case where perimeter goes all around image

                mask3=np.stack([mask,mask,mask],axis=2)

                voxelMap = cv.watershed(mask3,seeds)

                voxelMap-=offset # return to original fiberID numbers

                voxelMap[voxelMap==-3]=-1 #backgroud value
                voxelMap[voxelMap==-2]=-1 #edge case when two boundaries ovelap

                voxelMap[mask3[:,:,0]==0]=-1 #prevent spilling out of masked region

                # at this point voxelMap contains regions tagged with reassigned regions

                if axisHandles and axisHandles[1]:
                    x,y=np.where(voxelMap>=0)
                    flaggedPixels=[]
                    for i in range(len(x)):
                        flaggedPixels.append((x[i]+xOffset,y[i]+yOffset))

                    # attenuate grayImg_hist to make it more readable
                    
                    oldRange=[0,255]
                    newRange=[0,80]
                    
                    addFlaggedPixelsToImg(imgComp,flaggedPixels,color=[77,166,255],alpha=1.)
                    addLegendEntry=True


                # overwrite the assignment of markers according to split regions
                if fiberData["rejected"][matchedFiberID]: 
                    #if matchedFiberID is rejected, the new region may include fibers that are not, need to update
                    rejectedMask[fiberMap_slice==matchedFiberID]=False

                fiberMap_slice[fiberMap_slice==matchedFiberID]=voxelMap[fiberMap_slice==matchedFiberID]

                
                # update rejectedMask according to fibers in newly made regions
                checkMarkers=list(np.unique(voxelMap))
                #boundaries between regions take values -1, sometimes -2
                checkMarkers=[val for val in checkMarkers if val>0]
                for cMarker in checkMarkers:
                    if fiberData["rejected"][cMarker]:
                        rejectedMask[fiberMap_slice==cMarker]=True

                for newMatch in fiberID_unmatchedList:
                    if newMatch in voxelMap:
                        if newMatch in listMatched:
                            raise RuntimeError("listMatched souldn't contain this marker already")

                        listMatched.append(newMatch)
                        listUnmatched.remove(newMatch)
                        
                        color=[77/255,166/255,1.0]
                        colorLabels_legend["Split region"]=(77/255,166/255,1.0)

                        if axisHandles:
                            x,y=np.where(voxelMap==newMatch)
                            for iX,iY in zip(x,y):
                                flaggedPixelsMatched.append((iX+xOffset,iY+yOffset,color))

    allCenterPoints["listUnmatched"]=listUnmatched

    fiberID_All=np.array(allCenterPoints["fiberID"])
    x_All_interp=np.array(allCenterPoints["x_interp"])-xOffset
    y_All_interp=np.array(allCenterPoints["y_interp"])-yOffset

    x_All=np.array(allCenterPoints["x"])-xOffset
    y_All=np.array(allCenterPoints["y"])-yOffset

    if addDisksAroundCenterPnts:
        for matchedFiberID in listMatched:
            index=np.where(fiberID_All==matchedFiberID)[0][0]

            if fiberData["addDisks_usingInterpolated"][matchedFiberID]:
                fiberCenterX=int(x_All_interp[index])
                fiberCenterY=int(y_All_interp[index])
            else:
                fiberCenterX=int(x_All[index])
                fiberCenterY=int(y_All[index])

            #draw circle where the detected fiber centroid is located
            for cx,cy in zip(circle_x,circle_y):
                if fiberCenterX+cx-fiberRadius>=0 and fiberCenterX+cx-fiberRadius<fiberMap_slice.shape[0]:
                    if fiberCenterY+cy-fiberRadius>=0 and fiberCenterY+cy-fiberRadius<fiberMap_slice.shape[1]:
                        
                        overwrite=False
                        
                        #check if fiber is not already present from other permutation, if so, do not overwrite
                        if V_fibers_masked_slice is None or\
                            V_fibers_masked_slice[fiberCenterX+cx-fiberRadius,fiberCenterY+cy-fiberRadius]==0:

                            # matchedfiberID includes both tracked and rejected fibers, check if tracked fiber present before over
                            if fiberData["rejected"][matchedFiberID]:

                                presentMarker=fiberMap_slice[fiberCenterX+cx-fiberRadius,fiberCenterY+cy-fiberRadius]
                                if presentMarker in [-1,-999999]:
                                    overwrite=True
                                elif fiberData["rejected"][presentMarker]:
                                    overwrite=True

                            else:
                                overwrite=True #always overwrite if fiberID is "tracked"

                            if overwrite:
                                fiberMap_slice[fiberCenterX+cx-fiberRadius,fiberCenterY+cy-fiberRadius]=matchedFiberID
                                rejectedMask[fiberCenterX+cx-fiberRadius,fiberCenterY+cy-fiberRadius]=fiberData["rejected"][matchedFiberID]

    # catch bugs where a background marker could be flipped to fiberID=-1
    backgroundMask=fiberMap_slice==-1
    if np.logical_and(backgroundMask,rejectedMask).any():
        print("Warning: voxels marker both as rejected and background. correcting" )
        rejectedMask[fiberMap_slice==-1]=False

    rejectedFiberIDs=set([fiberID for fiberID,boolVal in fiberData["rejected"].items() if boolVal])

    #flip sign of fiberID to indicate rejection (too short, too steep)
    fiberMap_slice[rejectedMask]=-fiberMap_slice[rejectedMask]

    testSlice=fiberMap_slice.copy()
    testSlice[testSlice<-1]=-1
    markersPresent=set(np.unique(testSlice))
    if -1 in markersPresent: # if no fiber is rejected, wont be present
        markersPresent.remove(-1)

    if markersPresent.intersection(rejectedFiberIDs):
        print("rejected fiberID missed, correcting")
        for m in markersPresent.intersection(rejectedFiberIDs):
            fiberMap_slice[fiberMap_slice==m]=-m


    if axisHandles and axisHandles[0]:

        # voxelsImg=overlayVoxelMapToHist(V_voxels[:,:,imSlice],grayImg_hist)
        # axisHandles[0].imshow(voxelsImg,cmap="ocean")#,cmap="gist_stern")#, cmap='binary')

        fiberImg=fiberMap_slice.copy()
        fiberImg[fiberImg==-999999]=0
        fiberImg[fiberImg<-1]=-fiberImg[fiberImg<-1]

        voxelsImg=overlayVoxelMapToHist(fiberImg,grayImg_hist,xOffset,yOffset)
        axisHandles[0].imshow(voxelsImg,cmap="gist_stern_r")#,cmap="gist_stern")#, cmap='binary')

        axisHandles[0].title.set_text("fiberMap markers mapping, imSlice={}".format(imSlice))

        axisHandles[0].set_xlim([0,voxelsImg.shape[1]])
        axisHandles[0].set_ylim([0,voxelsImg.shape[0]])

    if axisHandles and axisHandles[2]:
        if axisHandles[0] and axisHandles[1]: # 1st step, successfully tracked fibers

            imgHist = np.stack([grayImg_hist,grayImg_hist,grayImg_hist],axis=2) #for axisHandle[2]

            addFlaggedPixelsToImg(imgHist,flaggedPixelsMatched,0.9)

            addFlaggedPixelsToImg(imgHist,flaggedPixelsUnmatched,[255,0,0],0.9)

            axisHandles[2].imshow(imgHist)
            axisHandles[2].title.set_text("Tracked/untracked regions, imSlice={}".format(imSlice))

            axisHandles[2].set_xlim([0,imgHist.shape[1]])
            axisHandles[2].set_ylim([0,imgHist.shape[0]])

            for colorLabel,color in colorLabels_legend.items():
                colorRGB=[val for val in color]
                axisHandles[2].scatter(0.,0.,s=70,marker='.',c=np.array([colorRGB]),label=colorLabel)

            if flaggedPixelsUnmatched:
                axisHandles[2].scatter(0.,0.,s=70,marker='.',c=np.array([[1.,0.,0.]]),label="unmatched")

            axisHandles[2].legend(prop={'size': 16})

    if axisHandles and axisHandles[1]:

        axisHandles[1].imshow(imgComp)
        axisHandles[1].title.set_text("centroid matching with reassigned regions, imSlice={}".format(imSlice))

        singleLabel     =True
        singleLabelRed  =True

        for i,iFib in enumerate(dataFib["fiberID"]):
            if iFib in listUnmatched:
                colorText='r'
                colorScatter="red"
            else:
                colorText='c'
                colorScatter="cyan"

            if textLabels:
                axisHandles[1].text(dataFib["y"][i],dataFib["x"][i]-2.,"t{}".format(iFib),color=colorText,fontsize=textFontSize)
            if colorScatter == "red":
                if singleLabelRed:
                    axisHandles[1].scatter(dataFib["y"][i],dataFib["x"][i],s=70,c=colorScatter,label="unmatched fiber")
                    singleLabelRed=False
                else:
                    axisHandles[1].scatter(dataFib["y"][i],dataFib["x"][i],s=70,c=colorScatter)
            else:
                if singleLabel:
                    axisHandles[1].scatter(dataFib["y"][i],dataFib["x"][i],s=70,c=colorScatter,label="tracked fiber")
                    singleLabel=False
                else:
                    axisHandles[1].scatter(dataFib["y"][i],dataFib["x"][i],s=70,c=colorScatter) 

        singleLabel_watershed=True
        singleLabel_watershedMagenta=True

        for i,iCentroid in enumerate(watershedCentroids):

            if watershedTrackedBool[i]:
                colorText='y'
                colorScatter="yellow"
            else:
                colorText='m'
                colorScatter="magenta"

            if textLabels:
                axisHandles[1].text(iCentroid[1], iCentroid[0]+2.,"w{}".format(i),color=colorText,fontsize=textFontSize) 
            if colorScatter == "magenta":
                if singleLabel_watershedMagenta:
                    axisHandles[1].scatter(iCentroid[1],iCentroid[0],s=30,c=colorScatter,label="unmatched centroid")
                    singleLabel_watershedMagenta=False
                else:
                    axisHandles[1].scatter(iCentroid[1],iCentroid[0],s=30,c=colorScatter)
            else:
                if singleLabel_watershed:
                    axisHandles[1].scatter(iCentroid[1],iCentroid[0],s=30,c=colorScatter,label="centroid")
                    singleLabel_watershed=False
                else:
                    axisHandles[1].scatter(iCentroid[1],iCentroid[0],s=30,c=colorScatter)

        axisHandles[1].set_xlim([0,imgComp.shape[1]])
        axisHandles[1].set_ylim([0,imgComp.shape[0]])
        legend=axisHandles[1].legend()
        if addLegendEntry:
            addPatchToLegend(legend,color=[77/255,166/255,1.0],label="Marker region to split")

    return allCenterPoints,fiberMap_slice
