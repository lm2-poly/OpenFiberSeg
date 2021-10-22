# by Facundo Sosa-Rey, 2021. MIT license

import numpy as np
import scipy
from trackingFunctions import knn,knn3D
import time

import multiprocessing

class trackedCenterPointsObj:

    def __init__(self,nSlices,offset):
        #points are indexed by imSlice
        self.points={}

        for iSlice in range(offset,nSlices+offset):
            self.points[iSlice]={}
            self.points[iSlice]["x"]=[]
            self.points[iSlice]["y"]=[]
            self.points[iSlice]["fiberID"]=[]

    def append(self,iSlice,x,y,fiberID):
        iSlice=int(iSlice)
        if fiberID not in self.points[iSlice]["fiberID"]:
            self.points[iSlice]["x"].append(float(x))
            self.points[iSlice]["y"].append(float(y))
            self.points[iSlice]["fiberID"].append(fiberID)
            if len(self.points[iSlice]["x"])!=len(self.points[iSlice]["fiberID"]):
                raise ValueError("inconsistent number of elements in points vs fiberIDs")

    def reject(self,targetPointsObj,iSlice,x,y,fiberID):
        iSlice=int(iSlice)
        if fiberID in self.points[iSlice]["fiberID"]:
            if self.points[iSlice]["fiberID"].count(fiberID)>1:
                raise ValueError("same fiberID occurs more that once in a single imSlice, should never happen")

            indexToPop=self.points[iSlice]["fiberID"].index(fiberID)

            self.points[iSlice]["x"].pop(indexToPop)
            self.points[iSlice]["y"].pop(indexToPop)
            self.points[iSlice]["fiberID"].pop(indexToPop)

            targetPointsObj.append(iSlice,x,y,fiberID)



    def transferID(self,iSlice,oldID,newID):
        self.points[int(iSlice)]["fiberID"]=\
            [newID if fibID==oldID else fibID for fibID in self.points[iSlice]["fiberID"]]
        if self.points[int(iSlice)]["fiberID"].count(newID)>2:
            raise RuntimeError("same fiberID occurs more than once, should never happen")

    def getPoints(self):
        return self.points

    def getPointsInSlice(self,imSlice):
        #needs to return a list of numpy array [x,y]
        pointsObj=self.points[imSlice]
        points=[]
        fiberIDs=[]
        for iCt in range(len(pointsObj["x"])):
            points.append(
                np.array(
                    [   pointsObj["x"][iCt],pointsObj["y"][iCt] ]))
            fiberIDs.append(
                pointsObj["fiberID"][iCt]
            )
        return np.array(points),fiberIDs


def centerPoint_to_tuple():
    pass

class LUT_fiberID_to_centerPointObj:
    # this LUT needs a custom class so as to make the same call to append()
    #   if the fiberObj is new, makes a new list of centerPoints
    #   if the fiberObj exist, adds centerpoints to list
    def __init__(self):
        self.LUT={}

    def append(self,fiberID,centerPointTuple):
        if fiberID in self.LUT.keys():
            self.LUT[fiberID].append(centerPointTuple)

        else:
            self.LUT[fiberID]=[centerPointTuple]

class fiberObj:

    @classmethod
    def initializeClassAttributes(cls,savedAttributes=None):
        # Putting all class attributes in a dict makes for a lighter printout of fiberObj in IDE.
        # For debugging, class attributes are often not needed)
        if savedAttributes is None:
            cls.classAttributes={
                "exclusiveZone"               :[],
                "legendLabels"                :{"basic"},
                "interpolatedCenters"               :{},
                "backTracking"                :{},
                "colors":{
                    # color by type of fiber
                    "basic"                                 :(0.20  ,0.20   ,1.0    ),  # dark blue
                    "stitched_blind(extended)"              :(1.    ,0.1    ,1.     ),  # pink
                    "stitched_blind(added)"                 :(0.14  ,0.9    ,0.14   ),  # green
                    "stitched_smart(extended)"              :(0.47  ,0.04   ,0.14   ),  # burgundy
                    "stitched_smart(added)"                 :(0.55  ,0.76   ,1.00   ),  # cyan
                    "stitched(initial)"                     :(1.    ,1.     ,0.     ),  # yellow
                    "too short"                             :(0.8   ,0.8    ,0.8    ),  # grey       #if string key is changed, modify in getLegendLabel as well
                    "too steep"                             :(0.9   ,0.4    ,0.4    ),  # orange
                    "backTracking"                          :(0.65  ,0.04   ,1.     ),  # violet

                    # colors for all permutations combined
                    "permuted123"           :(0.    ,0.46   ,0.69   ),  # cerulean Blue
                    "permuted132"           :(0.    ,0.50   ,0.     ),  # India Green
                    "permuted321"           :(1.    ,0.50   ,0.     )   # orange
                    },
                "LUT_fiberID_to_color"        :{}, 
                "listFiberIDs_tracked"        :set([])
                }
        else:
            # load from file
            cls.classAttributes=savedAttributes

    @classmethod
    def setExclusiveZone(cls,bounds):
        cls.classAttributes["exclusiveZone"]=bounds

    @classmethod
    def getExclusiveZone(cls):
        return cls.classAttributes["exclusiveZone"]

    @classmethod
    def initLUTs(cls):
        cls.classAttributes["LUT_centerPoint_to_fiberID"]   ={}
        cls.classAttributes["LUT_fiberID_to_centerPoint"]   =LUT_fiberID_to_centerPointObj()
        cls.classAttributes["LUT_fiberID_startCenterPoint"] ={}
        cls.classAttributes["LUT_fiberID_endCenterPoint"]   ={}
        
    @classmethod
    def clearLUTs(cls):
        del cls.classAttributes["LUT_centerPoint_to_fiberID"]
        del cls.classAttributes["LUT_fiberID_to_centerPoint"]
        del cls.classAttributes["LUT_fiberID_startCenterPoint"]
        del cls.classAttributes["LUT_fiberID_endCenterPoint"]

    @classmethod
    def initializeFromLUT(cls,centerPoints,listSlicesLUT,LUT_id_bottom,LUT_id_top,offset):
        if len(LUT_id_bottom)!=len(LUT_id_top):
            raise ValueError("look up tables of incompatible size")

        cls.initLUTs()

        # initialize the trackedCenterPoints object for all slices in volume
        cls.initTrackedCenterPoints(nSlices=len(LUT_id_bottom)+1,offset=offset)

        fiberStruct={}

        for i,imSlice in enumerate(listSlicesLUT):
            if i==0:
                for iCt in range(len(LUT_id_bottom[0])):
                    #first pass of creating fiberObjs, hashed by fiberID, which here correspond to the index of startPnt.
                    fiberStruct[iCt]=fiberObj(
                        iCt,                                               # fiberID
                        centerPoints[imSlice][LUT_id_bottom[0][iCt]][0],          # xcoord
                        centerPoints[imSlice][LUT_id_bottom[0][iCt]][1],      imSlice) # ycoord ,zcoord+offset for exclusize zone with zMin>0

                    # imSlice==0 for bottom, imSlice==1 for top
                    centerPointBottomTuple= (imSlice,     LUT_id_bottom[0][iCt])
                    centerPointTopTuple   = (imSlice+1,   LUT_id_top   [0][iCt])

                    cls.classAttributes["LUT_centerPoint_to_fiberID"][centerPointTopTuple]=iCt
                    cls.classAttributes["LUT_fiberID_to_centerPoint"].append(iCt,centerPointTopTuple)

                    cls.classAttributes["LUT_fiberID_startCenterPoint"][iCt]=centerPointBottomTuple
                    cls.classAttributes["LUT_fiberID_endCenterPoint"][iCt]  =centerPointTopTuple

            else:
                for iCt in range(len(LUT_id_bottom[i])):
                    centerPointBottomTuple = (imSlice  ,LUT_id_bottom [i][iCt])
                    centerPointTopTuple    = (imSlice+1,LUT_id_top    [i][iCt])

                    temp=centerPoints[imSlice][LUT_id_bottom[i][iCt]]

                    xFloat=float(temp[0])
                    yFloat=float(temp[1])

                    if centerPointBottomTuple in cls.classAttributes["LUT_centerPoint_to_fiberID"].keys():
                        # add to existant fiberObj:
                        iFib=cls.classAttributes["LUT_centerPoint_to_fiberID"][centerPointBottomTuple]

                        cls.classAttributes["LUT_centerPoint_to_fiberID"][centerPointTopTuple]=iFib
                        cls.classAttributes["LUT_fiberID_to_centerPoint"].append(iFib,centerPointTopTuple)

                        # point to new endCenterPoint
                        cls.classAttributes["LUT_fiberID_endCenterPoint"][iFib]  =centerPointTopTuple


                        fiberStruct[iFib].append(xFloat,yFloat,imSlice)
                    else:
                        #create new fiberObj
                        fiberIDnew=len(fiberStruct)
                        #                              fNum           xcoord  ycoord       zcoord
                        fiberStruct[fiberIDnew]=fiberObj(fiberIDnew, xFloat,  yFloat, imSlice)

                        cls.classAttributes["LUT_centerPoint_to_fiberID"][centerPointTopTuple]=fiberIDnew
                        cls.classAttributes["LUT_fiberID_to_centerPoint"].append(fiberIDnew,centerPointTopTuple)

                        cls.classAttributes["LUT_fiberID_startCenterPoint"][fiberIDnew]=centerPointBottomTuple
                        cls.classAttributes["LUT_fiberID_endCenterPoint"]  [fiberIDnew]  =centerPointTopTuple


        listFibers,startPnts,endPnts=cls.getAllEndPoints(centerPoints,fiberStruct)

        # add endPnts to fiber objects (fibers were created by adding bottom centerPoints only, need to add last point on top )
        for i,fib in fiberStruct.items():
            fib.append(*endPnts[i])

        #LUT are no longer needed, delete
        cls.clearLUTs()

        return fiberStruct

    @classmethod
    def getAllEndPoints(cls,centerPoints,fiberStruct):
        startPnts  =[]
        endPnts    =[]
        cls.classAttributes["listFiberID"]=np.zeros(len(fiberStruct),np.int32)

        for i,iFibObj in fiberStruct.items():
            fibID=iFibObj.fiberID

            cls.classAttributes["listFiberID"][i]=fibID

            startCenterPointTuple=cls.classAttributes["LUT_fiberID_startCenterPoint"][fibID]
            endCenterPointTuple  =cls.classAttributes["LUT_fiberID_endCenterPoint"]  [fibID]

            z=int(startCenterPointTuple[0])
            xy=centerPoints[z][startCenterPointTuple[1]]
            x=int(xy[0])
            y=int(xy[1])

            startPnts.append(np.array([x,y,z]))

            z=int(endCenterPointTuple[0])
            xy=centerPoints[z][endCenterPointTuple[1]]
            x=int(xy[0])
            y=int(xy[1])

            endPnts.append(np.array([x,y,z]))

        cls.classAttributes["startPnts"]=np.array(startPnts)
        cls.classAttributes["endPnts"]  =np.array(endPnts)

        return cls.classAttributes["listFiberID"],cls.classAttributes["startPnts"],cls.classAttributes["endPnts"]

    @classmethod
    def blindStitching(cls,fiberStruct,blindStitchingMaxDistance,
        blindStitchingMaxLateralDist,verboseHandle):

        from trackingFunctions import knn3D
        ### 3D Knn on endpoints only

        print("\t\n##############################\n")
        print("\t3D simultaneous matching\n\n")

        # create KDTree with all top centerPoints

        knnObj=knn3D(cls.classAttributes["startPnts"])

        id_bottom,id_top,repDict,rejectPositions=knnObj.query(
            cls.classAttributes["endPnts"],blindStitchingMaxDistance)

        # the points that are matched correctly (not a fiber to itself)
        # are checked for backtracking and lateral distance
        testSelfMatch=id_bottom==id_top
        mask_GoodMatches3D=list(np.where(~testSelfMatch)[0])

        # if any of the non self-matched indices have repetitions, delete them
        rejectPositions.sort(reverse=True)
        for i in rejectPositions:
            if i in mask_GoodMatches3D:
                delPos=mask_GoodMatches3D.index(i)
                del mask_GoodMatches3D[delPos]

        #test to see if more than one match in top
        l_top=list(id_top[mask_GoodMatches3D])
        repetitions=[(index,val) for index,val in enumerate(id_top[mask_GoodMatches3D]) if l_top.count(val)>1]

        if len(repetitions)>0:
            raise RuntimeError("there are multiple matches left after deletions, should never happen")

        stitchingChains={}
        stitchingChains_endFib={}
        intermediarySegments=set([])
        # to deal with multiple stitching (chains of fiberObj), a stitchingChain object is
        # created, indexed by the fiber_ID of the tip of the chain (fiberObj with highest z)
        # if a new stitch is required from this chain tip, the chain is poped, and then added
        # to a new chain with the new tip. This way, if a chain is D(top)->C->B->A(bottom),
        # the object A receives the points from the combined object (DCB), not B alone,
        # when extend() is performed. A is the only one considered "real".
        # the other ones are kept for plotting (diagnostic) purposes

        if len(id_bottom)>0:
            #match found
            for index,iBottom in enumerate(id_bottom[mask_GoodMatches3D]):
                iTop=id_top[mask_GoodMatches3D[index]]
                singleEndPnt=cls.classAttributes["endPnts"][iBottom]
                singleStartPnt=cls.classAttributes["startPnts"][iTop]
                #check if backtracking occurs:
                if singleEndPnt[2]<singleStartPnt[2]:
                    #check if lateral distance is below threshold:

                    lateralDist=np.sqrt(
                        (singleEndPnt[0]-singleStartPnt[0])**2+\
                        (singleEndPnt[1]-singleStartPnt[1])**2)
                    if  lateralDist<blindStitchingMaxLateralDist:
                        #create stitchingChains for fiberObj passing both tests

                        fiberID_end  =cls.classAttributes["listFiberID"][iBottom]
                        fiberID_start=cls.classAttributes["listFiberID"][iTop]

                        if fiberID_end in stitchingChains.keys():
                            tempList=stitchingChains.pop(fiberID_end)

                            intermediarySegments.add(fiberID_end)

                            tempList=[fiberID_end]+tempList

                            stitchingChains[fiberID_start]=tempList
                            stitchingChains_endFib[tempList[-1]]=fiberID_start
                        else:
                            stitchingChains[fiberID_start]=[fiberID_end]
                            stitchingChains_endFib[fiberID_end]=fiberID_start

                        if verboseHandle:
                            print("endPnt (bottom):", singleEndPnt)
                            print("startPnt (top) :", singleStartPnt)

                            print("lateralDist",lateralDist)
                            print("####################")

        endPnts          =list(cls.classAttributes["endPnts"])
        listFiberID_end  =list(cls.classAttributes["listFiberID"])

        startPnts        =list(cls.classAttributes["startPnts"])
        listFiberID_start=list(cls.classAttributes["listFiberID"])

        print("\t3D simultaneous matching resulted in {} successful matches out of {} fiberObjs\n".\
            format(len(stitchingChains),len(fiberStruct)))

        #delete point that were correctly matched (not self-matched) on first pass of knn3D()
        #deleting in decreasing order of indices wont affect positions below deletion
        for ele in sorted(id_bottom[mask_GoodMatches3D], reverse = True):
            del endPnts[ele]
            del listFiberID_end[ele]

        for ele in sorted(id_top[mask_GoodMatches3D], reverse = True):
            del startPnts[ele]
            del listFiberID_start[ele]

        endPnts  =np.array(endPnts)
        startPnts=np.array(startPnts)

        listFiberID_end  =np.array(listFiberID_end)
        listFiberID_start=np.array(listFiberID_start)

        print("\n\t##############################")
        print("\n\tsequential matching started")

        # create KDTree only with centerPoints that were self-matched
        knnObjSelfMatched=knn3D(startPnts)

        # Query by specifying k=5 nearest neighbors, which signals to check for self matching, 
        # and returns closest match that is not self-matching 
        id_bottom,id_top,matches=knnObjSelfMatched.query(
            endPnts,
            blindStitchingMaxDistance,
            blindStitchingMaxLateralDist,
            k=5)


        if len(id_bottom)>0:
            for i in range(len(id_bottom)):
                fiberID_start=listFiberID_start[id_top   [i]]
                fiberID_end  =listFiberID_end  [id_bottom[i]]

                if fiberID_start in intermediarySegments or fiberID_end in intermediarySegments:
                    print("Skipping this match: branching of stitching chains. incoherent matching of neighbors: {}->{}".\
                        format(fiberID_start,fiberID_end))
                else:
                    if fiberID_start in stitchingChains_endFib.keys():
                        if fiberID_end in stitchingChains.keys():
                            #case of new link being between two existing stitchingChains
                            existingStart=stitchingChains_endFib[fiberID_start]
                            tempListTop=stitchingChains.pop(existingStart)
                            tempListBottom=stitchingChains.pop(fiberID_end)
                            tempList=tempListTop+[fiberID_end]+tempListBottom
                            stitchingChains[existingStart]=tempList
                            stitchingChains_endFib.pop(fiberID_start)
                            stitchingChains_endFib[tempList[-1]]=existingStart
                            for fibID in tempList[:-1]:
                                intermediarySegments.add(fibID)
                        else:
                            # new link at bottom of existing stitchingChain
                            existingStart=stitchingChains_endFib[fiberID_start]
                            tempList=stitchingChains.pop(existingStart)

                            for fibID in tempList:
                                intermediarySegments.add(fibID)

                            tempList=tempList+[fiberID_end]
                            stitchingChains[existingStart]=tempList
                            stitchingChains_endFib.pop(fiberID_start)
                            stitchingChains_endFib[fiberID_end]=existingStart

                    elif fiberID_end in stitchingChains.keys():
                        # new link at top of existing stitchingChain
                        tempList=stitchingChains.pop(fiberID_end)
                        tempList=[fiberID_end]+tempList
                        stitchingChains[fiberID_start]=tempList
                        stitchingChains_endFib[tempList[-1]]=fiberID_start
                    else:
                        # start new stitchingChain
                        stitchingChains[fiberID_start]=[fiberID_end]
                        stitchingChains_endFib[fiberID_end]=fiberID_start

                    if verboseHandle:
                        lateralDist=matches[id_bottom[i]][1][1]
                        distanceTotal=  matches[id_bottom[i]][1][0]
                        print("endPnt (bottom):", endPnts       [id_bottom[i]], "no: ", id_bottom[i])
                        print("startPnt (top) :", startPnts     [id_top   [i]], "no: ", id_top   [i])
                        print( "endID:",fiberID_end,"startID:", fiberID_start)
                        print("lateralDist",lateralDist)
                        print("distanceTotal",distanceTotal)
                        print("####################")

        stitchedListCache_fiberID =set([])


        # combine fiberObj in stitchingChains
        if len(stitchingChains)>0:

            # check total number of possible stitches, output to console
            numStitchesTotal=0
            for chainLinks in stitchingChains.values():
                numStitchesTotal+=len(chainLinks)

            numStitchesAttempted=0

            for chainEnd in stitchingChains.keys():

                extension_fiberID=chainEnd

                #extend by starting at end of chain, working all the way to the front.
                chainLinks=stitchingChains[chainEnd]
                while chainLinks:
                    segmentToExtend_fiberID=chainLinks.pop(0)
                    numStitchesAttempted+=1

                    extendSuccessful=\
                        fiberStruct[segmentToExtend_fiberID].extendFiber(
                            fiberStruct[extension_fiberID],
                            (segmentToExtend_fiberID,
                            extension_fiberID),
                            stitchingType="blind",
                            checkIfInSegt=False
                            )

                    if extendSuccessful:

                        #keep indices in list so smart stitching is not attempted on fictitious segments
                        # (the "extensions" fiberObj are kept in fiberStruc for plotting purposes)
                        stitchedListCache_fiberID .add(extension_fiberID)


                    extension_fiberID=segmentToExtend_fiberID

                print("BlindStitching: attempted/total: {}/{}".format(numStitchesAttempted,numStitchesTotal))

            print("\ttotal number of successful stitches in blindStitching(): {} out of {} fiberObjs\n".\
                format(len(stitchedListCache_fiberID),len(fiberStruct)))

        return stitchedListCache_fiberID

    @staticmethod
    def smartStitching(fiberStructMain,
        smartStitchingMinFibLength,
        smartStitchingMaxDistance,
        smartStitchingMaxLateralDist,
        smartStitchingAlignAngle,
        smartStitchingBackTrackingLimit,
        processingMinFiberLength,
        tagAngleTooSteep,
        maxSteepnessAngle,
        verboseHandle=False,
        checkIfInSegt=True,
        createNewPoints=True,
        stitchingType="smart",
        preventSelfStitch=False
        ):

        tic=time.perf_counter()

        endPnts={}
        startPnts={}
        orientationVec={}
        coordinateTuplesEnd=[]
        coordinateTuplesStart=[]
        fiberID_list=[]
        lengths=[]

        if preventSelfStitch:
            # in "smart_transposed" or "smart_lastPass":
            # only fibers from different permutations are allowed to be stitched at this stage 
            in123_bottom=[]
            in123_top=[]
            in132_bottom=[]
            in132_top=[]
            in321_bottom=[]
            in321_top=[]
            suffixCheck=(
                in123_bottom,
                in123_top,
                in132_bottom,
                in132_top,
                in321_bottom,
                in321_top
            )
        else:
            in123_bottom=in123_top=in132_bottom=in132_top=in321_bottom=in321_top=None
            suffixCheck=None

        for fibID,fObj in fiberStructMain.items():

            #gather data for ranking matches in smartStitching()
            if fObj.totalLength>smartStitchingMinFibLength:
                endPnts       [fibID]=fObj.endPnt
                startPnts     [fibID]=fObj.startPnt
                orientationVec[fibID]=fObj.orientationVec.copy()/np.linalg.norm(fObj.orientationVec)

                lengths.append(fObj.totalLength)

                coordinateTuplesEnd  .append((*endPnts  [fibID], *orientationVec[fibID]))
                coordinateTuplesStart.append((*startPnts[fibID], *orientationVec[fibID]))

                fiberID_list.append(fibID)

                if preventSelfStitch:
                    in123_bottom.append(fObj.suffix==0.123)
                    in123_top   .append(fObj.suffix==0.123)
                    in132_bottom.append(fObj.suffix==0.132)
                    in132_top   .append(fObj.suffix==0.132)
                    in321_bottom.append(fObj.suffix==0.321)
                    in321_top   .append(fObj.suffix==0.321)


        fiberStructExtended={}

        # create kdTree on terminal points of main fiberObjs only

        if not coordinateTuplesStart:
            return fiberStructExtended,{"6-D knn search only":None }

        knnObj6D=knn3D(coordinateTuplesStart)

        id_bottom,id_top,matches=knnObj6D.query(
            coordinateTuplesEnd,
            smartStitchingMaxDistance,
            smartStitchingMaxLateralDist,
            smartStitchingAlignAngle,
            smartStitchingBackTrackingLimit,
            lengths=lengths,
            k=20,#number of considered neighbors
            suffixCheck=suffixCheck
            )

        times_tracking={"6-D knn search only":time.strftime("%Hh%Mm%Ss", time.gmtime(time.perf_counter()-tic)) }

        print("6-D knn search performed in: {: >0.4f}s".format(time.perf_counter()-tic))

        for i in range(len(id_bottom)):
            fibID_end  =fiberID_list[id_bottom[i]]
            fibID_start=fiberID_list[   id_top[i]]

            if verboseHandle:
                lateralDistTest=matches[id_bottom[i]][1][1]
                distanceTotal=  matches[id_bottom[i]][1][0]
                angle=np.degrees(np.arccos(matches[id_bottom[i]][1][3]))

                print("bottom: \t{: >4} top: \t{: >4}, \tdistanceTotal: {: >5.4f}, \tdistLateral: {: >5.4f}, \tangle: {: >5.4f}".format(
                    fiberStructMain[fibID_end  ].fiberID,
                    fiberStructMain[fibID_start].fiberID,
                    distanceTotal,
                    lateralDistTest,
                    angle)
                    )

            # attempt to combine fiber objects into the initial segment

            if fiberStructMain[fibID_end].addedTo:
                # if this fiber already has been stitched to another -> point to actual initial segment
                fibID_temp=fibID_end
                fibIDcache={fibID_temp}
                while fiberStructMain[fibID_temp].addedTo:
                    fibID_temp=fiberStructMain[fibID_temp].addedTo[0]
                    if fibID_temp in fibIDcache:
                        raise RuntimeError("circular \"addedTo\" loop")
                    fibIDcache.add(fibID_temp)

                mainFiberID=fibID_temp
                addInitial=False
            else:
                # this is the first extension of this fiber:
                # create a new fiber for the purpose of plotting
                # the starting fiberStruc before extension
                fiberStructTemp=fiberStructMain[fibID_end].copy()
                addInitial=True
                mainFiberID=fibID_end

            if stitchingType =="smart":
                suffix=None
                colorInitial    ="stitched(initial)"
                tagInitial      ="initial_stitched_segment"
                tagExtended     ="stitched_smart(extended)"
                doTrimming      =True

            elif stitchingType  =="smart_transposed":
                suffix=mainFiberID%1
                colorInitial    ="stitched(initial)_transposed"
                tagInitial      ="initial_stitched_segment_transposed"
                tagExtended     ="stitched_smart(extended)_transposed"
                doTrimming      =False
            elif stitchingType  =="smart_lastPass":
                suffix=mainFiberID%1
                colorInitial    ="stitched(initial)_lastPass"
                tagInitial      ="initial_stitched_segment_lastPass"
                tagExtended     ="stitched_smart(extended)_lastPass"
                doTrimming      =False
            else:
                raise ValueError("not implemented for stitchingType: {}".format(stitchingType))

            if fiberStructMain[fibID_start].addedTo:
                raise RuntimeError("attempting to add to a fiber when already part of one stitching chain")

            extendSuccessful=fiberStructMain[mainFiberID].extendFiber(
                fiberStructMain[fibID_start],
                (mainFiberID,fibID_start),
                stitchingType,
                checkIfInSegt=checkIfInSegt,
                createNewPoints=createNewPoints,
                suffix=suffix,
                fibersAll=fiberStructMain,
                doTrimming=doTrimming
                )

            if extendSuccessful:#keep extension if majority of new centerPoints are in region segmented as fiber

                if mainFiberID==fibID_end:
                    print("\tSmart stitching of fiberID:{} (bottom) extended by {} (top)".\
                        format(mainFiberID,fibID_start) )
                else:
                    print("\tSmart stitching of fiberID:{} (bottom) extended by {} (top), via intermediary fiber {}".\
                        format(mainFiberID,fibID_start,fibID_end) )

                if addInitial:#if this is the first extension of this fiber
                    #create a new fiber for the sole purpose of seeing the starting fiberStruc before extension

                    if fiberStructExtended:
                        nextFiberID=max(fiberStructExtended.keys())+1
                    else:
                        nextFiberID=max(fiberStructMain.keys())+1

                    fiberStructExtended[nextFiberID]=fiberStructTemp
                    fiberStructExtended[nextFiberID].setColor(colorInitial)
                    fiberStructExtended[nextFiberID].tags.add(tagInitial)
                    fiberStructExtended[nextFiberID].originalFiberID=fibID_end
                    if tagExtended in fiberStructExtended[nextFiberID].tags:
                        raise ValueError("fiber branching")
                    fiberObj.addLegendLabel(colorInitial)

                    #pointer towards initial segment, for plotting purposes
                    fiberStructMain[mainFiberID].initialObj={nextFiberID:fiberStructExtended[nextFiberID]}

                fiberStructMain[mainFiberID].processPointCloudToFiberObj(
                    processingMinFiberLength,
                    tagAngleTooSteep,
                    maxSteepnessAngle,
                    doTrimming=doTrimming
                    )


            if len(fiberStructMain[fibID_end].addedTo)>1:
                #print("Segment added to more than one fiber (causes branching)")
                raise RuntimeError("Segment added to more than one fiber (branching in reverse, converging)")

        return fiberStructExtended,times_tracking



    @classmethod
    def getPointsInSlice(cls,iSlice):
        return cls.classAttributes["trackedCenterPoints"].getPointsInSlice(iSlice)

    @classmethod
    def initTrackedCenterPoints(cls,nSlices,offset):
        cls.classAttributes["trackedCenterPoints" ]=trackedCenterPointsObj(nSlices,offset)
        cls.classAttributes["rejectedCenterPoints"]=trackedCenterPointsObj(nSlices,offset)
        cls.classAttributes["trimmedCenterPoints" ]=trackedCenterPointsObj(nSlices,offset)

    @classmethod
    def getTrackedCenterPoints(cls):
        return cls.classAttributes["trackedCenterPoints"].getPoints()

    @classmethod
    def getRejectedCenterPoints(cls):
        return cls.classAttributes["rejectedCenterPoints"].getPoints()

    @classmethod
    def appendTrackedCenterPoints(cls,iSlice,x,y,fiberID,rejected):
        if rejected:
            cls.classAttributes["rejectedCenterPoints"].append(iSlice,x,y,fiberID)
        else:
            cls.classAttributes["trackedCenterPoints"] .append(iSlice,x,y,fiberID)

    @classmethod
    def rejectTrackedCenterPoints(cls,iSlice,x,y,fiberID,rejected,trimming):
        if trimming:
            if rejected:
                                    # source                                            # target
                cls.classAttributes["rejectedCenterPoints"].reject(cls.classAttributes["trimmedCenterPoints"],iSlice,x,y,fiberID)
            else:                   # source                                            # target
                cls.classAttributes["trackedCenterPoints" ].reject(cls.classAttributes["trimmedCenterPoints"],iSlice,x,y,fiberID)
        else:                       # source                                            # target
            cls.classAttributes["trackedCenterPoints"].reject(cls.classAttributes["rejectedCenterPoints"],iSlice,x,y,fiberID)

    @classmethod
    def restoreRejectedPoints(cls,iSlice,x,y,fiberID):
        cls.classAttributes["rejectedCenterPoints"].reject(cls.classAttributes["trackedCenterPoints"],iSlice,x,y,fiberID)

    @classmethod
    def transferID(cls,iSlice,oldID,newID,rejected):
        if rejected:
            cls.classAttributes["rejectedCenterPoints"].transferID(iSlice,oldID,newID)
        else:
            cls.classAttributes["trackedCenterPoints"] .transferID(iSlice,oldID,newID)


    @classmethod
    def getColor(cls,label):
        return cls.classAttributes["colors"][label]

    @classmethod
    def addLegendLabel(cls,label):
        cls.classAttributes["legendLabels"].add(label)

    @classmethod
    def getLegendLabels(cls,plotRejectedFibers):
        legendLabels=cls.classAttributes["legendLabels"].copy()
        if not plotRejectedFibers:
            if "too steep" in legendLabels:
                legendLabels.remove("too steep")
            if "too short" in legendLabels:
                legendLabels.remove("too short")
        return legendLabels

    @classmethod
    def removeLegendLabels(cls,*labels):
        for label in labels:
            if label in cls.classAttributes["legendLabels"]:
                cls.classAttributes["legendLabels"].remove(label)

    @classmethod
    def loadSegmentationMask(cls,V_fibers):
        cls.classAttributes["V_fibers"]=V_fibers

    @classmethod
    def checkIfPointIsInSegt(cls,x,y,z):
        return cls.classAttributes["V_fibers"][int(z),int(x),int(y)]==255

    @classmethod
    def setTrackingParameters(cls,distance,fraction,fillingNumberAlwaysAllowed,maxTrimPoints):
        cls.classAttributes["collisionDistance"]=distance
        cls.classAttributes["fillingFraction"]  =fraction
        cls.classAttributes["fillingNumberAlwaysAllowed"]  =fillingNumberAlwaysAllowed
        cls.classAttributes["maxTrimPoints"]    =maxTrimPoints

    def __init__(self,fiberID,x=None,y=None,z=None,color="basic",appendTrackedCenterPoint=True):
        #appendTrackedCenterPoint==False for fiberObjs kept only for plotting purposes
        if x is not None:
            self.x=np.array([x],float)
            self.y=np.array([y],float)
            self.z=np.array([z],float)
        else:
            self.x=[]
            self.y=[]
            self.z=[]
        self.extendedBy=[]
        self.extendedByObj={} #pointers to actual fiberObj, serves to update tags and colors
        self.addedTo=[]
        self.tags=set([])
        self.fiberID=fiberID
        self.setColor(color)
        self.rejected=False
        if x is not None and appendTrackedCenterPoint :
            self.appendTrackedCenterPoints(z,float(x),float(y),fiberID,rejected=False)
            self.classAttributes["listFiberIDs_tracked"].add(fiberID)



    def copy(self):
        newFiber=fiberObj(
            self.fiberID,
            color=self.colorLabel,
            appendTrackedCenterPoint=False
            )
        newFiber.x              =self.x.copy()
        newFiber.y              =self.y.copy()
        newFiber.z              =self.z.copy()
        newFiber.setColor(self.colorLabel)
        newFiber.extendedBy     =self.extendedBy
        newFiber.extendedByObj  =self.extendedByObj
        newFiber.addedTo        =self.addedTo
        newFiber.tags           =self.tags.copy()
        newFiber.meanPntCloud   =self.meanPntCloud
        newFiber.startPnt       =self.startPnt
        newFiber.endPnt         =self.endPnt
        newFiber.totalLength    =self.totalLength
        newFiber.orientationVec =self.orientationVec
        if "originalFiberID" in self.__dir__():
            newFiber.originalFiberID=self.originalFiberID
        if "suffix" in self.__dir__():
            newFiber.suffix=self.suffix
        if "trimmedPoints" in self.__dir__():
            newFiber.trimmedPoints=self.trimmedPoints


        return newFiber

    def append(self,x,y,z):

        if len([z])>1:
            raise ValueError("Use extend to append more than one point")
        if (z-self.z[-1])>1.: #(case of dumb stitching)
            # here the endpoint should not be appended right away, as it will be added at the end of this append.
            # the separate append at the end is necessary for all other appends that are not dumb stitiching

            # temporary fiberObj, with interpolated points
            temp=fiberObj(self.fiberID,x,y,z,appendTrackedCenterPoint=False)

            #fill-in with centerPoints between self and temp fiberObj so the watershed transform can be done
            #only the interpolated points are appended, not the endpoints

            fillingSuccessful,zStart=self.filling(temp)

            if fillingSuccessful:
                print("\t\tknn stitching \tfiberID: {: 4.0f} \tfrom slices: {: 4.0f} \tto {: 4.0f}".\
                    format(self.fiberID,int(zStart),z))
                self.x=np.append(self.x,x)
                self.y=np.append(self.y,y)
                self.z=np.append(self.z,z)
                self.appendTrackedCenterPoints(z,x,y,self.fiberID,self.rejected)
                return

        if (z-self.z[-1])==1.:
            self.x=np.append(self.x,x)
            self.y=np.append(self.y,y)
            self.z=np.append(self.z,z)
            self.appendTrackedCenterPoints(z,x,y,self.fiberID,self.rejected)
            return

    def filling(self,otherFiberObj,checkIfInSegt=True,createNewPoints=True,fibersAll=None,doTrimming=True):
        zStart  =self.z[-1]
        zEnd    =otherFiberObj.z[0]

        if zStart>=zEnd:
            if zEnd-self.z[0]<2:
                print("self.fiberID={},other.fiberID={},less than 2 pnts would be left after trimming, do not append".\
                    format(self.fiberID,otherFiberObj.fiberID))
                return False,None

            if doTrimming:
                #trimming wont work when combining fibers that were transposed, as the z value wont be monotonous
                self.trimEndPoints(endPntOther=zEnd)
                zStart  =self.z[-1]
            else:
                print("\ncant do trimming() operation between fibers {: >8.3f} and {: >8.3f}, extend without filling\n".\
                    format(self.fiberID,otherFiberObj.fiberID))
                return True,None

        zFill   =np.linspace(zStart,zEnd,num=int(round(zEnd-zStart+1)),endpoint=True)[1:-1]

        if not createNewPoints:
            # stitchingType: smart_transposed or smart_lastPass:
            # if any point are needed outside of V_fibers==True,
            # tag this fiberObj for post-processing, need to add voxels as well in gap filled with
            self.tags.add("fill_interpolation_secondPass")

            if self.fiberID in self.classAttributes["interpolatedCenters"].keys():
                # if there is more than one gap, postProcessing will select largest
                self.classAttributes["interpolatedCenters"][self.fiberID].append(len(zFill))
            else:
                self.classAttributes["interpolatedCenters"][self.fiberID]=[len(zFill)]

            return True,None

        if len(zFill)>0 and createNewPoints:
            xStart  =self.x[-1]
            xEnd    =otherFiberObj.x[0]

            yStart  =self.y[-1]
            yEnd    =otherFiberObj.y[0]
            #treating x and y coordinates as linear functions of z, i.e. x=f(z)=a*z+b
            xFill   =np.interp(zFill,[zStart,zEnd], [xStart, xEnd])
            yFill   =np.interp(zFill,[zStart,zEnd], [yStart, yEnd])

            #check if the the majority of the added segment is in region segmented as fiber
            testIfInSegt=[]

            for i in range(len(zFill)):
                points,fiberIDs=fiberObj.getPointsInSlice(int(zFill[i]))
                if len(points)>0:
                    # insertion test to check nearest neighbor
                    queryPnt=np.array( [np.array([ xFill[i],yFill[i] ]) ]  )

                    # TODO (would be nice) Here a new kdTree is created at each query, but this can be necessary if
                    # a previous filling() has added points in this slice
                    # Could be optimized to create new kdTree only if there is a change,
                    # but for probably slim performance gain
                    id_bottom_th,id_top_th,dist = knn(
                        queryPnt,
                        points,
                        self.classAttributes["collisionDistance"],
                        returnDist=True
                        )

                    if len(id_bottom_th)>0:
                        for iCollision,iPnt in enumerate(id_bottom_th):
                            print("\t\tcollision at imSlice={: >5.0f},\t(x,y)=({: >6.1f},{: >6.1f}), \tdistance={: >6.1f},\t between fibers:{} and {}".\
                                format(int(zFill[i]),xFill[i],yFill[i],dist[iCollision],fiberIDs[iPnt],self.fiberID))

                            return False,None #collision present, will not stitch fiberObj


                if checkIfInSegt and len(zFill)>self.classAttributes["fillingNumberAlwaysAllowed"]:

                    testIfInSegt.append(fiberObj.checkIfPointIsInSegt(xFill[i],yFill[i],zFill[i]))
                else:
                    # if gap shorter than fillingNumberAlwaysAllowed, always do filling
                    testIfInSegt.append(True)

            # check if majority of potential filling points are in regions segmented as fiber

            doFilling=False
            if testIfInSegt.count(True)/len(testIfInSegt)>self.classAttributes["fillingFraction"]:
                # if any point are needed outside of V_fibers==True,
                # tag this fiberObj for post-processing, need to add voxels as well in gap filled with
                self.tags.add("fill_interpolation")

                if self.fiberID in self.classAttributes["interpolatedCenters"].keys():
                    self.classAttributes["interpolatedCenters"][self.fiberID].append(len(zFill))
                else:
                    self.classAttributes["interpolatedCenters"][self.fiberID]=[len(zFill)]

                testIfInSegt=[True]*len(testIfInSegt)

                doFilling=True

            if doFilling:
                self.x=np.append(self.x,xFill)
                self.y=np.append(self.y,yFill)
                self.z=np.append(self.z,zFill)

                for iCt in range(len(zFill)):
                    #only the new interpolated points are appended, not the endpoints
                    self.appendTrackedCenterPoints(zFill[iCt],xFill[iCt],yFill[iCt],self.fiberID,self.rejected)
                
                return True,zStart

            else:
                for i in range(len(zFill)):
                    if not testIfInSegt[i]:
                        print("\t\tfilling rejected between fibers: {},{}, imSlice={} because not in V_fibers==True regions,\t(x,y)=({: >6.1f},{: >6.1f})".\
                            format(self.fiberID,otherFiberObj.fiberID ,int(zFill[i]),xFill[i],yFill[i]))
                return False,None

        else:
            return True, None # no need to add points, means start and endPnts are on adjacent slices. do blindstitch


    def extendFiber(self,otherFiberObj,linkOrder,stitchingType,checkIfInSegt=True,createNewPoints=True,suffix=None,fibersAll=None,doTrimming=True):
        if len(otherFiberObj.x) != len(otherFiberObj.y) or len(otherFiberObj.x) != len(otherFiberObj.z):
            raise ValueError("inconsistent sizes for x, y, z")

        iFib=linkOrder[0]
        tFib=linkOrder[1]

        #fill-in with centerPoints between self and otherFibObj so the watershed transform can be done
        #collision test results in testIfInSegt
        fillingSuccessful,zStart=self.filling(
            otherFiberObj,
            checkIfInSegt=checkIfInSegt,
            createNewPoints=createNewPoints,
            fibersAll=fibersAll,
            doTrimming=doTrimming
            )

        if fillingSuccessful:
            self.x=np.append(self.x,otherFiberObj.x)
            self.y=np.append(self.y,otherFiberObj.y)
            self.z=np.append(self.z,otherFiberObj.z)
            if stitchingType=="blind":
                labelStr_extended   ="stitched_blind(extended)"
                labelStr_added      ="stitched_blind(added)"

            elif stitchingType=="smart":
                if self.fiberID in self.classAttributes["backTracking"].keys():
                    labelStr_extended="backTracking"
                else:
                    labelStr_extended="stitched_smart(extended)"

                labelStr_added="stitched_smart(added)"

            elif stitchingType =="smart_transposed":
                if self.fiberID in self.classAttributes["backTracking"].keys():
                    labelStr_extended="backTracking_transposed"
                else:
                    labelStr_extended="stitched_smart(extended)_transposed"

                labelStr_added="stitched_smart(added)_transposed"

            elif stitchingType =="smart_lastPass":
                if self.fiberID in self.classAttributes["backTracking"].keys():
                    labelStr_extended="backTracking_lastPass"
                else:
                    labelStr_extended="stitched_smart(extended)_lastPass"

                labelStr_added="stitched_smart(added)_lastPass"

            self.setColor(labelStr_extended)

            self.extendedBy.append(int(tFib)) # can be numpy.int32, from knn() implementation
            self.extendedByObj[int(tFib)]=otherFiberObj
            self.tags.add(labelStr_extended)
            self.addLegendLabel(labelStr_extended)

            otherFiberObj.setColor(labelStr_added)

            if suffix is None:
                otherFiberObj.addedTo.append(int(iFib))
            else:
                otherFiberObj.addedTo.append(int(iFib)+suffix)

            if len(otherFiberObj.addedTo)>1:
                raise RuntimeError("attempting to add segment to more than one fiber (branching in reverse, converging)")

            otherFiberObj.tags.add(labelStr_added)
            otherFiberObj.zOffset=True
            self.addLegendLabel(labelStr_added)
            if len(otherFiberObj.extendedBy)>0:
                # preserve stitching chains
                for extensionID in otherFiberObj.extendedBy:
                    if extensionID not in self.extendedBy:
                        self.extendedBy.append(extensionID)
                        self.extendedByObj[extensionID]=otherFiberObj.extendedByObj[extensionID]

            if otherFiberObj.rejected and not self.rejected:
                # move points from rejectedCenterPoints to trackedCenterPoints, as they are now part of a tracked fiber
                # (fiberID will be removed from listFiberIDs_tracked on subsequent step)
                otherFiberObj.restorePoints()
                #change fiberID on trackedCenterPoints
                for iSlice in otherFiberObj.z:
                    self.transferID(int(iSlice),otherFiberObj.fiberID,self.fiberID,False) # otherFiberObj.rejected=True, but centerPoints are in "tracked" object

            # elif self.rejected and not otherFiberObj.rejected:
                # the points will be handled correctly at processPointCloudToFiberObj

            else:

                #change fiberID on trackedCenterPoints
                for iSlice in otherFiberObj.z:
                    self.transferID(int(iSlice),otherFiberObj.fiberID,self.fiberID,otherFiberObj.rejected)

            if otherFiberObj.fiberID in self.classAttributes["listFiberIDs_tracked"]:
                self.classAttributes["listFiberIDs_tracked"].remove(otherFiberObj.fiberID)

        return fillingSuccessful

    def setColor(self,colorLabel):
        if type(colorLabel) is not str or colorLabel not in fiberObj.classAttributes["colors"].keys():
            raise TypeError("colorLabel must be a string, corresponding to a key in fiberObj.classAttributes[\"colors\"]")

        self.color=fiberObj.classAttributes["colors"][colorLabel]
        self.colorLabel=colorLabel
        self.classAttributes["LUT_fiberID_to_color"][self.fiberID]=colorLabel

    def rejectPoints(self,pos=None):
        if pos is None: #reject entire fiberObj (too short, too steep)
            trimming=False
            rejectRange = range(len(self.z))
            if self.fiberID in self.classAttributes["listFiberIDs_tracked"]:
                self.classAttributes["listFiberIDs_tracked"].remove(self.fiberID)

        else:           #reject only some points, from trimming (trimEndPnts)
            trimming=True
            rejectRange = [pos]

        for iCt in rejectRange:
            self.rejectTrackedCenterPoints(self.z[iCt],self.x[iCt],self.y[iCt],self.fiberID,rejected=self.rejected,trimming=trimming)

    def restorePoints(self):
        self.classAttributes["listFiberIDs_tracked"].add(self.fiberID)
        for iCt in range(len(self.z)):
            self.restoreRejectedPoints(self.z[iCt],self.x[iCt],self.y[iCt],self.fiberID)

    def trimEndPoints(self,endPntOther=None):

        if endPntOther is None:
            startPnt=round(self.startPnt[2])
            endPnt  =round(self.endPnt[2])
            trimStart=True
            trimEnd  =True
        else:
            # this is for the edgecase where smartStitching is done on
            # fibers that have a common z end and startPnt
            endPnt=endPntOther-0.1 # to not allow stric equality in this case
            trimStart=False
            trimEnd  =True
            self.classAttributes["backTracking"][self.fiberID]=endPntOther

            self.setColor("backTracking")

        pos=0
        while trimStart:
            #remove points from trackedCenterPoints object
            if self.z[pos] < startPnt:
                self.rejectPoints(pos)
                pos+=1
            elif pos>self.classAttributes["maxTrimPoints"]:
                print("trimming {} points on fiberID {}. something is wrong".format(pos,self.fiberID))
                self.tags.add("trimmed_by_{}_points".format(pos))
                trimStart=False
            else:
                trimStart=False


        deleteIndicesStart=[i for i in range (pos)]

        originalNumPnt=len(self.z)
        pos=originalNumPnt-1
        while trimEnd:
            if self.z[pos] > endPnt:
                self.rejectPoints(pos)
                pos-=1
                if pos<0:
                    raise RuntimeError("backtracking causes the trimming of entire fiberObj, should not happen")
            elif originalNumPnt-pos>self.classAttributes["maxTrimPoints"]:
                print("trimming {} points on fiberID {}. something is wrong".format(originalNumPnt-pos,self.fiberID))
                self.tags.add("trimmed_by_{}_points".format(originalNumPnt-pos))
                trimEnd=False
            else:
                trimEnd=False

        deleteIndices=deleteIndicesStart+[i for i in range (pos+1,originalNumPnt)]
        if len(deleteIndices)>0:
            self.tags.add("trimmed")
            print("\ttrimming endPoints on fiberObj: {} at positions: {}".format(self.fiberID,deleteIndices))

            if "trimmedPoints" not in self.__dir__():
                self.trimmedPoints={
                    "x":list(self.x[deleteIndices]),
                    "y":list(self.y[deleteIndices]),
                    "z":list(self.z[deleteIndices])
                }

            else:
                self.trimmedPoints["x"].extend(list(self.x[deleteIndices]))
                self.trimmedPoints["y"].extend(list(self.y[deleteIndices]))
                self.trimmedPoints["z"].extend(list(self.z[deleteIndices]))
                
            self.x=np.delete(self.x,deleteIndices)
            self.y=np.delete(self.y,deleteIndices)
            self.z=np.delete(self.z,deleteIndices)

    @staticmethod
    def findFarthestPnts(data,vv,sort=False):
        """ find fartest points along principal direction"""
        dataMean = data.mean(axis=0)

        dist=np.zeros(len(data))

        for iPnt in range(len(data)):
            vecToPnt=data[iPnt,:]-dataMean
            dist[iPnt]=np.dot(vv,vecToPnt)

        #projection of dataPoints farthest from the mean onto the principal vector
        endPntMin=dataMean+vv*np.dot(vv,data[dist.argmin(),:]-dataMean)
        endPntMax=dataMean+vv*np.dot(vv,data[dist.argmax(),:]-dataMean)

        if sort:
            sorting_indices=np.argsort(dist)
            return dataMean,endPntMin,endPntMax,sorting_indices
        else:
            return dataMean,endPntMin,endPntMax,None

    def checkSpread(self,maxSpread,verboseHandle=False):

        if verboseHandle:
            print("checkSpread for fiberID={} on {}".format(self.fiberID,multiprocessing.current_process().name))

        distStart=10000.
        distEnd  =10000.

        data = np.concatenate(
            (
                self.x[:, np.newaxis],
                self.y[:, np.newaxis],
                self.z[:, np.newaxis]
            ),
            axis=1
        )

        trimList=[]

        while distStart>maxSpread or distEnd>maxSpread:

            # Calculate the mean of the points, i.e. the 'center' of the cloud
            datamean = data.mean(axis=0)

            # Do an SVD on the mean-centered data.
            # uu, dd, vv = np.linalg.svd(data - datamean) #for some mysterious reason, the numpy implementation sometime wont converge
            uu, dd, vv = scipy.linalg.svd(data - datamean)

            self.meanPntCloud,startPnt,endPnt,sorting_indices=\
                fiberObj.findFarthestPnts(data,vv[0])

            distStart=np.linalg.norm(data[ 0,:]-startPnt)
            distEnd  =np.linalg.norm(data[-1,:]-endPnt)

            if distStart>maxSpread or distEnd>maxSpread:
                if self.rejected:
                    raise ValueError("checkSpread method is not implemented for rejected fibers")
                if distStart>distEnd:
                    # remove first point

                    # keep list to remove from trackedCenterPoints. since this is a global class attribute, cannot be done in a parallel process due to GIL
                    trimList.append((self.z[0],self.x[0],self.y[0],self.fiberID))

                    self.x=self.x[1:]
                    self.y=self.y[1:]
                    self.z=self.z[1:]
                    data  =data  [1:]

                else:
                    #remove last point
                    trimList.append((self.z[-1],self.x[-1],self.y[-1],self.fiberID))

                    self.x=self.x[:-1]
                    self.y=self.y[:-1]
                    self.z=self.z[:-1]
                    data  =data  [:-1]


        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot3D(self.x,self.y,self.z,c="red")
        # ax.plot3D([startPnt[0],endPnt[0]],[startPnt[1],endPnt[1]],[startPnt[2],endPnt[2]],c="blue")
        # plt.show() 

        return trimList

    def processPointCloudToFiberObj(self,minFiberLength,tagAngleTooSteep,maxSteepnessAngle,sort=False,doTrimming=True):

        data = np.concatenate(
            (
                self.x[:, np.newaxis],
                self.y[:, np.newaxis],
                self.z[:, np.newaxis]
            ),
            axis=1
        )

        # Calculate the mean of the points, i.e. the 'center' of the cloud
        datamean = data.mean(axis=0)

        # Do an SVD on the mean-centered data.
        uu, dd, vv = np.linalg.svd(data - datamean)


        self.meanPntCloud,startPnt,endPnt,sorting_indices=\
            fiberObj.findFarthestPnts(data,vv[0],sort=sort)

        if sort: #used in fiberObj.combine(), because the points will be non-monotonical
            self.x=self.x[sorting_indices]
            self.y=self.y[sorting_indices]
            self.z=self.z[sorting_indices]


        # #TODO: would be nice: unstitching for spread too large
        # #if spread is too large, stitching should be reversed
        # lineEndPnts_lateralSpread=\
        #     findFarthestPnts(data,vv[1])[1]
        # spreadLength=np.linalg.norm(lineEndPnts_lateralSpread[:,1]-lineEndPnts_lateralSpread[:,0])
        # if spreadLength>20:
        #     print("object should be unstitched")

        #     import matplotlib.pyplot as plt
        #     from mpl_toolkits.mplot3d import Axes3D
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.plot3D(self.x,self.y,self.z,c="red")
        #     # ax.plot3D(lineEndPnts[0,:],lineEndPnts[1,:],lineEndPnts[2,:],c="blue")
        #     plt.show()

        # these points are not in the original datacloud:
        # they are projections of the points furthest from mean onto principal vector

        if startPnt[2]>endPnt[2]:
            #sometimes the endpoints are flipped
            buffer=startPnt
            startPnt=endPnt
            endPnt=buffer


        self.startPnt=startPnt
        self.endPnt=endPnt
        orientationVec=endPnt-startPnt
        self.orientationVec=orientationVec
        totalLength=np.linalg.norm(orientationVec)
        self.totalLength=totalLength

        self.oriVec_normalized=orientationVec/totalLength


        rejectedCache=self.rejected

        tooShort=False
        # Mark fibers that are suspiciously short
        if totalLength < minFiberLength:
            self.setColor("too short")

            self.tags.add("length.LT.{: >6.4f}".format(minFiberLength))
            self.addLegendLabel("too short")
            self.rejected=True
            self.rejectPoints()
            tooShort=True
            for extendedFibObj in self.extendedByObj.values():
                extendedFibObj.rejected=True
                if extendedFibObj.fiberID in self.classAttributes["listFiberIDs_tracked"]:
                    self.classAttributes["listFiberIDs_tracked"].remove(extendedFibObj.fiberID)
            if "initialObj" in self.__dir__():
                if len(self.initialObj)>1:
                    raise ValueError("can't be more than one initialObj")
                for fibObj in self.initialObj.values():
                    fibObj.rejected=True

        else:
            self.rejected=False


        if tagAngleTooSteep:
            dotProd=np.dot(orientationVec/totalLength,[0.,0.,1.])
            if dotProd<np.cos(maxSteepnessAngle):
                self.setColor("too steep")
                self.tags.add(
                    "tooSteep_angle={: >6.4f}".format(
                        np.degrees(np.arccos(dotProd))))
                self.addLegendLabel("too steep")
                self.rejected=True
                self.rejectPoints()
                for extendedFibObj in self.extendedByObj.values():
                    extendedFibObj.rejected=True
                    if extendedFibObj.fiberID in self.classAttributes["listFiberIDs_tracked"]:
                        self.classAttributes["listFiberIDs_tracked"].remove(extendedFibObj.fiberID)
                if "initialObj" in self.__dir__():
                    if len(self.initialObj)>1:
                        raise ValueError("can't be more than one initialObj")
                    for fibObj in self.initialObj.values():
                        fibObj.rejected=True


            elif not tooShort:
                self.rejected=False

        if rejectedCache and not self.rejected:

            #used to be rejected but now is long enough
            self.restorePoints()

            for extendedFibObj in self.extendedByObj.values():
                extendedFibObj.rejected=False
            if "initialObj" in self.__dir__():
                if len(self.initialObj)>1:
                    raise ValueError("can't be more than one initialObj")

                for fibObj in self.initialObj.values():
                    fibObj.rejected=False

        # reject points that are beyond startPnt and endPnt in the z direction
        # this is sometimes a consequence of knn sometimes stitching points that
        # are along different real fibers (will be at an angle)
        # this is an edge case that can sometimes cause problems at smartStitching
        # IMPORTANT keep this step last, or trimmed point may be in wrong ("tracked"/"rejected") set 
        # if restorePoints() or rejectPoints() is applied afterwards, wont get removed, and cause collisions
        # on following stitching
        if doTrimming:
            #doTrimming will be False when smartStitching_transpose, as self.z will not increase monotonically
            self.trimEndPoints()

    def transpose(self,permutationVec):
        if permutationVec=="123":
            #add suffix to differentiate between origin referentials
            self.fiberID       =self.fiberID+0.123   
            self.suffix        =0.123           

        if permutationVec=="132":
            temp=self.y
            self.y=self.z
            self.z=temp

            self.endPnt        =self.endPnt         [[0,2,1]]
            self.startPnt      =self.startPnt       [[0,2,1]]
            self.meanPntCloud  =self.meanPntCloud   [[0,2,1]]
            self.orientationVec=self.orientationVec [[0,2,1]]

            # add suffix to differentiate between origin referentials
            self.fiberID       =self.fiberID+0.132
            self.suffix        =0.132

        if permutationVec=="321":
            temp=self.x
            self.x=self.z
            self.z=temp

            self.endPnt        =self.endPnt         [[2,1,0]]
            self.startPnt      =self.startPnt       [[2,1,0]]
            self.meanPntCloud  =self.meanPntCloud   [[2,1,0]]
            self.orientationVec=self.orientationVec [[2,1,0]]

            # add suffix to differentiate between origin referentials
            self.fiberID       =self.fiberID+0.321
            self.suffix        =0.321


        self.tags.add("transposed_from:{}".format(permutationVec))

        # so stitches at combine() stage are stored separately than those from tracking() stage
        self.extendedByFirstPass=self.extendedBy
        self.extendedByObjFirstPass=self.extendedByObj
        self.extendedBy=[]
        self.extendedByObj={}

    def combine(self,otherFiberObj):

        self.x=np.append(self.x,otherFiberObj.x)
        self.y=np.append(self.y,otherFiberObj.y)
        self.z=np.append(self.z,otherFiberObj.z)

        self.processPointCloudToFiberObj(10.,False,None,sort=True,doTrimming=False)

        if "combinedWith" in self.__dir__():
            self.combinedWith   .append(otherFiberObj.fiberID)
            self.combinedWithObj.append(otherFiberObj)
        else:
            self.combinedWith   =[otherFiberObj.fiberID]
            self.combinedWithObj=[otherFiberObj]

        if "combinedWith" in otherFiberObj.__dir__():
            otherFiberObj.combinedWith   .append(self.fiberID)
            otherFiberObj.combinedWithObj.append(self)
        else:
            otherFiberObj.combinedWith   =[self.fiberID]
            otherFiberObj.combinedWithObj=[self]

