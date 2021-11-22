# by Facundo Sosa-Rey, 2021. MIT license

from cv2 import fillConvexPoly
import numpy as np

# find most frequent element in a list 
def most_frequent(List): 
    if not List:
        return None
    return max(set(List), key = List.count) 

class centroidObj:

    def __init__(self,x=None,y=None,contour=None,dictObj=None):
        if dictObj is None:
            self.pnt=(  
                int(round(x)),
                int(round(y))
                )
            
            self.marker =   None

            if type(contour)!=type([]):
                if len(contour.shape)==3:
                    self.contour=   [contour]
                else:
                    raise TypeError("contour has wrong shape/type")
            else:
                self.contour=   contour

        else: 
            self.dictToObj(dictObj)

    def getPnt(self):
        return (self.pnt[0],self.pnt[1])

    def getMarker(self,voxelMap=None,exclusionList=None,partialMap=False):

        if self.marker is None or partialMap==True: 
            # only make the search once, else return stored attribute
            marker=voxelMap[self.pnt[0],self.pnt[1]]

            if marker in exclusionList:
                # this step is required when contour has an odd shape, probably concave, and centroid is outside enclosed region
                mask=np.zeros(voxelMap.shape,np.uint8)
                #mark pixels inside closed contour as True
                mask=fillConvexPoly(mask, self.contour[0], color=255)

                xPnts,yPnts=np.where(mask==255)

                markerCandidate=[]
                for i in range(len(xPnts)):
                    markerCandidate.append(voxelMap[xPnts[i],yPnts[i]])

                marker=most_frequent([val for val in markerCandidate if val not in exclusionList])

                if marker is None:
                    # print("\t\tcan't find marker")
                    return None

            if partialMap==False:
                # if partialMap==True, this is in the iterative waterLevel raising, 
                # marker is not the one in general voxelMap, do not store
                self.marker=marker
            
            return marker

        return self.marker

    def setMarker(self,marker):
        # used when regions have already been checked for convexity, and marker is obtained directly from watershed transform
        self.marker=marker
        
    def addFilledContourToImg(self,img,color):
        return fillConvexPoly(img, self.contour[0], color=color)

    def objToDict(self):
        """convert to python dict used to save custom type to disk using JSON format"""
        dictCentroid={}
        dictCentroid["x"]           =self.pnt[0]
        dictCentroid["y"]           =self.pnt[1]

        #convert cv2.contour to list of tuples
        contour= []
        for i,iPnt in enumerate(self.contour[0]):
            if iPnt.shape==(1,2):
                contour.append((int(iPnt[0][0]),int(iPnt[0][1])))
            else:
                raise RuntimeError("not implemented")

        dictCentroid["contour"]     =contour

        dictCentroid["marker"]      =self.marker

        return dictCentroid

    def dictToObj(self,dictObj):
        """construct centroid obj from dict loaded from JSON file"""
        self.pnt=(  
            dictObj["x"],
            dictObj["y"] 
            )

        self.contour = [np.array( np.array([ np.array([np.array(pnTuple)]) for pnTuple in dictObj["contour"]    ]   )  )]
        self.marker  = dictObj["marker"]
