# by Facundo Sosa-Rey, 2021. MIT license
import numpy as np

def volumeFractionFromWeightFraction(PEEK_weight_conc,PEI_weight_conc,fiber_weight_conc,porosity_volume_conc):

    #positing total weight is 1 gram

    matrix_weight_conc=1-fiber_weight_conc

    matrix_weight   =matrix_weight_conc*1. #1 gram total
    fiber_weight    =fiber_weight_conc*1.   #1 gram total

    PEEK_dens   =1.32 #g/cm^3
    PEEK_volume =PEEK_weight_conc/PEEK_dens*matrix_weight

    PEI_dens    =1.27 #g/cm^3
    PEI_volume  =PEI_weight_conc/PEI_dens*matrix_weight

    # fiber_dens  =1.81        #g/cm^3 ZOLTEC PX35
    fiber_dens  =1.76        #g/cm^3 Toray T300
    fiber_volume=fiber_weight_conc/fiber_dens*1 #total weight 1 gram

    matrix_volume   =PEEK_volume+PEI_volume
    matrix_dens     =matrix_weight/matrix_volume
    # print("matrix density= \t{: >10.3f} g/cm^3".format(matrix_dens))

    porosity_volume=porosity_volume_conc*(fiber_volume+matrix_volume)/(1-porosity_volume_conc)

    total_volume=fiber_volume+matrix_volume+porosity_volume

    # print("\ntotal volume: \t\t{: >10.3f} cm^3".format(total_volume))
    # print("volume porosity: \t{: >10.3f} cm^3".format(porosity_volume))

    fiber_volume_conc=fiber_volume/(total_volume)
    matrix_volume_conc=matrix_volume/(total_volume)

    # print("volume fibers: \t\t{: >10.3f} cm^3".format(fiber_volume))
    # print("volume matrix: \t\t{: >10.3f} cm^3".format(matrix_volume))
    # print("volume concentration of fibers: {: >10.3f}".format(fiber_volume_conc))
    # print("volume concentration of matrix: {: >10.3f}\n".format(matrix_volume_conc))

    total_weight=fiber_weight+matrix_weight

    total_density=total_weight/total_volume

    return fiber_volume_conc,total_density


def weightFractionFromDensity(PEEK_weight_conc,PEI_weight_conc,density,porosity_volume_conc):

    if density==0.0:
        print("no reading")
        return {"fiber_volume_conc":0.,"fiber_weight_fraction":0.0}

    #supposing total weight is 1 gram

    total_mass=1. 
    total_volume=total_mass/density

    PEEK_dens=1.32 #g/cm^3

    PEI_dens=1.27 #g/cm^3

    if PEEK_weight_conc==1.:
        label="PEEK"
        matrix_dens=PEEK_dens
    elif PEI_weight_conc==1.:
        label="PEI"
        matrix_dens=PEI_dens
    else:
        raise RuntimeError("not implemented for PEEK PEI mixture")

    fiber_dens=1.81        #g/cm^3 ZOLTEC PX35

    matrix_volume= ((1-porosity_volume_conc)*fiber_dens*total_volume-total_mass)/(fiber_dens-matrix_dens)
    
    fiber_volume=(total_mass-matrix_volume*matrix_dens)/fiber_dens

    fiber_weight=fiber_volume*fiber_dens
    matrix_weight=matrix_volume*matrix_dens

    print("material:",label, " total_mass: ",fiber_weight+matrix_weight, ", should be 1.")

    fiber_volume_conc=fiber_volume/total_volume

    fiber_weight_fraction=fiber_weight/total_mass

    return {"fiber_volume_conc":fiber_volume_conc,"fiber_weight_fraction":fiber_weight_fraction}

def calculateFractionsFromVolumes(V_matrix,V_fibers,V_perim,V_pores):
    matrixVolumeFraction=[]
    fibersVolumeFractionTotal=[]
    fibersMatrixVolumeFraction=[]
    poresVolumeFraction=[]

    countMatrixTotal=0
    countPoresTotal =0
    countFibersTotal=0
    countPerimTotal =0

    test=np.logical_and(V_perim==255,V_fibers>-1)
    if np.any(test):
        raise IOError("inconsistent segmentation")
    test=np.logical_and(V_pores==255,V_fibers>-1)
    if np.any(test):
        raise IOError("inconsistent segmentation")
    test=np.logical_and(V_perim==255,V_pores>-1)
    if np.any(test):
        raise IOError("inconsistent segmentation")


    for imSlice in range(V_matrix.shape[0]):
        im_matrix=V_matrix[imSlice]
        im_fibers=V_fibers[imSlice]
        im_perim=V_perim[imSlice]
        im_pores=V_pores[imSlice]

        im_matrix[im_perim==255]=0
        im_matrix[im_pores==255]=0
        im_matrix[im_fibers>-1]=0

        countMatrix=sum(sum(im_matrix==1))
        countPores =sum(sum(im_pores==255))
        countFibers=sum(sum(im_fibers>-1))
        countPerim =sum(sum(im_perim==255))

        countMatrixTotal+=countMatrix
        countPoresTotal+=countPores
        countFibersTotal+=countFibers
        countPerimTotal+=countPerim

        matrixVolumeFraction.append(countMatrix/(countFibers+countMatrix+countPores))
        fibersVolumeFractionTotal.append(countFibers/(countFibers+countMatrix+countPores))
        fibersMatrixVolumeFraction.append(countFibers/(countFibers+countMatrix))
        poresVolumeFraction .append(countPores/(countFibers+countMatrix+countPores))

    totalCount=countPerimTotal+countPoresTotal+countMatrixTotal+countFibersTotal
    voxelCount=V_fibers.shape[0]*V_fibers.shape[1]*V_fibers.shape[2]

    print("total count           = ",totalCount)
    print("total number of voxels= ",voxelCount)

    if totalCount!=voxelCount:
        raise RuntimeError("incompatible count")

    meanFiberFrac=np.mean(fibersVolumeFractionTotal)
    stdFiberFrac=np.std(fibersVolumeFractionTotal)
    errorFiberFrac=1.96*stdFiberFrac/np.sqrt(len(fibersVolumeFractionTotal))

    meanFiberMatrixFrac=np.mean(fibersMatrixVolumeFraction)
    stdFiberMatrixFrac=np.std(fibersMatrixVolumeFraction)
    errorFiberMatrixFrac=1.96*stdFiberFrac/np.sqrt(len(fibersMatrixVolumeFraction))
    
    meanPoresFrac=np.mean(poresVolumeFraction)
    stdPoresFrac=np.std(poresVolumeFraction)
    errorPoresFrac=1.96*stdPoresFrac/np.sqrt(len(poresVolumeFraction))

    print("volume fraction of fibers= {: 8.4f}".format(meanFiberFrac))
    print("volume fraction of fibers/matrix= {: 8.4f}".format(meanFiberMatrixFrac))

    return fibersVolumeFractionTotal,meanFiberFrac,stdFiberFrac,errorFiberFrac,\
        fibersMatrixVolumeFraction,meanFiberMatrixFrac,stdFiberMatrixFrac,errorFiberMatrixFrac,\
        poresVolumeFraction,meanPoresFrac,stdPoresFrac,errorPoresFrac
