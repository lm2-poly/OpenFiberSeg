# by Facundo Sosa-Rey, 2021. MIT license

import os
import pickle

from numpy.core.numeric import True_

import numpy as np

import matplotlib.pyplot as plt 
from matplotlib.ticker import PercentFormatter
from matplotlib import cm

from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import uniform_filter1d


from tifffile import TiffFile

from extractCenterPoints import getTiffProperties
from fiberStatistics_plottingTools import numericalIntegration

plt.rcParams.update({'font.size':26})
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams["font.family"] = "Times New Roman"


path="./TomographicData/PEEK05/processed_x1-901_y1-871_z1-980/2020-01-01_12h00m00/"

filename="fiberStruct_final.pickle"

saveData=False

saveFigures=True

if saveFigures:
    outputPathFigures=os.path.join(path,"FiguresFiberStats")

    if not os.path.exists(outputPathFigures):
        os.mkdir(outputPathFigures)


with open(os.path.join(path,filename),'rb') as f:
    fiberStruct=pickle.load(f)

fibers=fiberStruct["fiberStruct"]

Vtiff_filename="V_fiberMapCombined_postProcessed.tiff"

with TiffFile(os.path.join(path,Vtiff_filename)) as tif:
    xRes,unitTiff=getTiffProperties(tif) 

    if unitTiff=="INCH":
        pixelSize_micron=xRes[1]/xRes[0]*0.0254*1e6
    elif  unitTiff=="CENTIMETER":
        pixelSize_micron=xRes[1]/xRes[0]*0.01*1e6
    else:
        raise ValueError("other units values not implemented in getTiffProperties")

length_inPixels=[]
length_inMicrons=[]
theta=[]
phi=[]

for fib in fibers.values():
    if 'stitched_blind(added)' not in fib.tags and\
        'stitched_smart(added)' not in fib.tags and\
        'initial_stitched_segment' not in fib.tags:
        fib.length=np.linalg.norm(fib.endPnt-fib.startPnt)
        if not fib.rejected:
            length_inPixels.append(fib.length)
            
            length_inMicrons.append(fib.length*pixelSize_micron)


            fib.orientationVecNormalized=fib.orientationVec/fib.length

            fib.theta=np.degrees(np.arccos(np.dot(fib.orientationVecNormalized,np.array([0.,0.,1.]))))

            fib.phi=np.degrees(np.arctan2(fib.orientationVecNormalized[1],fib.orientationVecNormalized[0] ))

            if fib.theta>90.:
                raise ValueError("Should never have a theta>90° because the tracking ensures fibers are upright.")

            phi.append(fib.phi)

            theta.append(fib.theta)


print("average length: {:> 8.4f} microns".format(np.mean(length_inMicrons)))

fig=plt.figure(figsize=[8,5],num="HistogramLengthAngle")

axBottom_theta = fig.add_subplot(111)

axTop_Length = axBottom_theta.twiny()

histLength=axTop_Length.hist(np.array(length_inMicrons),bins=400,density=True,alpha=0.6,color="C0", label='Length')
histTheta =axBottom_theta.hist(theta,bins=400,density=True,alpha=0.6,color="C3", label='Theta')

smoothLength = uniform_filter1d(histLength[0], size=2)
smoothTheta  = uniform_filter1d(histTheta[0] , size=2)

binWidths_length = [histLength[1][i+1]-histLength[1][i]
         for i in range(len(smoothLength)-1)]

binWidths_theta = [histTheta[1][i+1]-histTheta[1][i]
         for i in range(len(smoothTheta)-1)]       

cdf_length = [numericalIntegration(
    binWidths_length[:i-1], smoothLength[:i]) for i in range(1, len(smoothLength))]
cdf_theta = [numericalIntegration(
    binWidths_theta[:i-1], smoothTheta[:i])  for i in range(1, len(smoothTheta))]

cumsum_length = numericalIntegration(binWidths_length, smoothLength)
cumsum_theta  = numericalIntegration(binWidths_theta, smoothTheta)

axBottom_theta.set_xlim([0.,90.])
axTop_Length  .set_xlim([0.,150.])

axBottom_theta.set_xlabel("Deviation angle (degrees)")
axTop_Length.set_xlabel(r"Length ($\mu m$)")
axBottom_theta.set_ylabel("Frequency (%)")
axBottom_theta.yaxis.set_major_formatter(PercentFormatter(1))

# added these three lines
lns = [histLength[2][0],histTheta[2][0]]
labs = ["Length","Deviation"]
axBottom_theta.legend(lns, labs, loc=0)


plt.tight_layout()

if saveFigures:
    plt.savefig(os.path.join(outputPathFigures,"HistogramLengthAngle.png"))

plt.figure(figsize=[12, 8], num="cumulative distribution function: Length")

plt.plot(histLength[1][:-2],  # x values are not evenly spaced
         cdf_length,
         label="Length"
         )

plt.title("cumulative distribution function: Length")
plt.xlabel("Lengths (microns)")
plt.xticks([val for val in range(0,401,50)])
plt.grid("minor")

if saveFigures:
    plt.savefig(os.path.join(outputPathFigures,"CDF_Length.png"))

plt.figure(figsize=[12, 8], num="cumulative distribution function: angle Theta")

plt.plot(histTheta[1][:-2],  # x values are not evenly spaced
         cdf_theta,
         label="Theta"
         )

plt.title("cumulative distribution function: Theta")
plt.xlabel("Deviation angle (degrees)")
plt.xticks([val for val in range(0,91,15)])
plt.grid("minor")

if saveFigures:
    plt.savefig(os.path.join(outputPathFigures,"CDF_Theta.png"))

plt.figure(figsize=[8,8],num="length vs deviation")

plt.scatter(length_inMicrons, theta,s=1)
plt.xlabel(r"Length ($\mu m$)")
plt.ylabel("Deviation angle (degrees)")
plt.tight_layout()

if saveFigures:
    plt.savefig(os.path.join(outputPathFigures,"lengthVSdeviation.png"))

plt.figure(figsize=[8,8],num="Length vs azimuthal angle")

plt.scatter(length_inMicrons, phi,s=1)
plt.xlabel(r"Length ($\mu m$)")
plt.ylabel("Azimuthal angle (degrees)")
plt.tight_layout()

if saveFigures:
    plt.savefig(os.path.join(outputPathFigures,"LengthVSazimuthalAngle.png"))

plt.figure(figsize=[8,8],num="Deviation vs azimuthal angle")

plt.scatter(theta, phi,s=1)
plt.xlabel("Deviation angle (degrees)")
plt.ylabel("Azimuthal angle (degrees)")
plt.tight_layout()

if saveFigures:
    plt.savefig(os.path.join(outputPathFigures,"DeviationVSazimuthalAngle.png"))

fig=plt.figure(figsize=[8,8], num="throwaway histogram")

nBins=256
alpha = 0.3
nLevels = 16

normalizeHist=False
nBins = 256

xMinLength=5.
xMaxLength=500.
cbarScale=3.5
cbar_width_single=0.2

if normalizeHist:
    levels=[val/nLevels/650 for val in range(nLevels)]
else:

    levels=[(np.exp(val/cbarScale)-1)/nBins/nBins for val in range(0,nLevels)]


h,x_edges,y_edges,image=plt.hist2d(length_inMicrons, theta,bins=nBins,density=True,cmap=plt.cm.inferno)

plt.close(fig)

data={
    "hist_length_theta" :h,
    "x_edges"           :x_edges,
    "y_edges"           :y_edges,
    "nFibers"           :len(fibers),
    "nBins"             :nBins,
    "length_inMicrons"  :length_inMicrons,
    "theta"             :theta,
    "phi"               :phi,
    "pixelSize_micron"  :pixelSize_micron
}


if saveData:
    print(f"\tSaving histogram data to disk at: \n{path}")
    with open(os.path.join(path,"histogramFiberData.pickle"),"wb") as f:
        pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)


X,Y=np.meshgrid(x_edges[:-1],y_edges[:-1])

h_smooth=np.transpose(gaussian_filter(h,sigma=1) )

figSingle_1D_Hist=plt.figure(figsize=[8,8],num="2dHist_singleVariableHistogram")

ax2D_hist=figSingle_1D_Hist.add_axes([0.25,0.35,0.5,0.6])
axLengths=figSingle_1D_Hist.add_axes([0.25,0.15,0.5,0.2])
axTheta  =figSingle_1D_Hist.add_axes([0.15,0.35,0.1,0.6])

normalizeHist=True

vals = np.linspace(0., 1., int(nLevels))
vals_rev = [vals[i] for i in range(len(vals)-1, -1, -1)]

colors = [cm.RdYlBu(val) for val in vals_rev]
colors[0] = (0.1, 0.1, 0.1, 1.)

if normalizeHist:
    contourPlot=ax2D_hist.contourf(
        X,
        Y,
        h_smooth,
        levels=levels,
        colors=colors
    )
else:
    contourPlot=ax2D_hist.contourf(
        X,
        Y,
        h_smooth*data["nFibers"]/nBins/nBins,
        levels=levels,
        colors=colors
    )

# recreate 1d histograms
axLengths.hist(data["length_inMicrons"], bins=1024,density=True, color=(0.45,0.45,0.45))
axTheta.hist(data["theta"], bins=256,orientation="horizontal",color="C0")

ax2D_hist.  set_xlim([xMinLength,xMaxLength])
axLengths.set_xlim([xMinLength,xMaxLength])
axTheta.  set_ylim([ 0., 90.])

axTheta.set_ylabel("Deviation (degrees)")
axTheta.set_yticks([0,15,30,45,60,75,90])

axLengths.set_xlabel(r"Fiber length ($\mu m$)")

axLengths.set_ylim([0.,0.04])# HACK
axTheta.invert_xaxis()
axLengths.invert_yaxis()

axTheta.  set_xticklabels("")
axLengths.set_yticklabels("")

ax2D_hist.set_xticklabels("")
ax2D_hist.set_yticklabels("")


# Adding the colorbar axes
cbar_axes1D_hist=figSingle_1D_Hist.add_axes([1.-cbar_width_single, 0.15, 0.02, 0.8])  

# # position for the colorbar

cbar1 = plt.colorbar(
    contourPlot, 
    cax = cbar_axes1D_hist, 
    ticks=levels,
    label="Fiber density (amount/bin)"
    )
cbar1.set_ticks(levels)
cbar1.ax.set_yticklabels(["{:1.2f}".format(val*data["nFibers"]) for val in levels])

if saveFigures:
    plt.savefig(os.path.join(outputPathFigures,"2dHist_singleVariableHistogram.png"))

plt.show()
