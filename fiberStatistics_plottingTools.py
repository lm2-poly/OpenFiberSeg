# by Facundo Sosa-Rey, 2021. MIT license

import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
from matplotlib.lines   import Line2D as Line

import numpy as np


def numericalIntegration(deltaTheta,f_theta):
    if (len(f_theta)-len(deltaTheta))!=1:
        raise("vectors are the wrong length")

    result=0.
    for i in range(len(deltaTheta)):
        result+=deltaTheta[i]*f_theta[i]

    return result

def drawQuantiles(ax,x,cdf,height,color,thickness=0.04):
    q0,q25,q50,q75,q100=np.interp([0.05,0.25,0.5,0.75,.95],cdf,x[:-1])

    ax.add_patch(Rectangle((q25, height-thickness/2,), q75-q25, thickness,alpha=1.,edgecolor=color,facecolor=None,fill=False))
    ax.add_patch(Rectangle((q25, height-thickness/2.), q75-q25, thickness,alpha=1.,color=color,fill=False,linewidth=1.5))

    ax.add_line(Line([q0,   q25], [height,      height],     color=color))
    ax.add_line(Line([q75,  q100],[height,      height],     color=color))
    ax.add_line(Line([q0,   q0],  [height-thickness/4., height+thickness/4.],color=color))
    ax.add_line(Line([q50,  q50], [height-thickness/2., height+thickness/2.],color=color))
    ax.add_line(Line([q100, q100],[height-thickness/4., height+thickness/4.],color=color))


def whiskersLegend(ax,x,height,color,fontsize=16,thickness=0.015):

    q5,q25,q50,q75,q95=x

    ax.add_patch(Rectangle((q25, height-thickness), q75-q25, thickness*2,alpha=1.,edgecolor=color,facecolor=None,fill=False))
    ax.add_patch(Rectangle((q25, height-thickness), q75-q25, thickness*2,alpha=1.,color=color,fill=False,linewidth=1.5))

    ax.add_line(Line([q5,   q25], [height,      height],     color=color))
    ax.add_line(Line([q75,  q95], [height,      height],     color=color))
    ax.add_line(Line([q5,   q5],  [height-thickness/2   , height+thickness/2],color=color))
    ax.add_line(Line([q50,  q50], [height-thickness     , height+thickness  ],color=color))
    ax.add_line(Line([q95,  q95], [height-thickness/2   , height+thickness/2],color=color))

    ax.text(q5 -thickness*0.66, height+thickness*2,r"$P_{5}$" ,color=color,fontsize=fontsize)
    ax.text(q25-thickness*0.66, height+thickness*2,r"$P_{25}$",color=color,fontsize=fontsize)
    ax.text(q50-thickness*0.66, height+thickness*2,r"$P_{50}$",color=color,fontsize=fontsize)
    ax.text(q75-thickness*0.66, height+thickness*2,r"$P_{75}$",color=color,fontsize=fontsize)
    ax.text(q95-thickness*0.66, height+thickness*2,r"$P_{95}$",color=color,fontsize=fontsize)

def mean_std_Legend(ax,mean,std,height,color,fontsize=18,textOffset=0.3):

    ax.add_patch(Rectangle((mean-std, height-0.02), 2*std, 0.04,alpha=1.,edgecolor=color,facecolor=None,fill=False))
    ax.add_patch(Rectangle((mean-std, height-0.02), 2*std, 0.04,alpha=.25,color=color,fill=True,linewidth=2))

    ax.add_line(Line([mean,   mean], [height-0.02,height+0.02],     color=color))

    ax.text(mean-std-textOffset,    height+0.03,r"$\bar{x}-\sigma$" , color=color,fontsize=fontsize)
    ax.text(mean,                   height+0.03,r"$\bar{x}$",         color=color,fontsize=fontsize)
    ax.text(mean+std-textOffset,    height+0.03,r"$\bar{x}+\sigma$",  color=color,fontsize=fontsize)


def drawMean_STD(ax,mean,std,height,color):

    ax.add_patch(Rectangle((mean-std, height-0.02), 2*std, 0.04,alpha=1.,edgecolor=color,facecolor=None,fill=False))
    ax.add_patch(Rectangle((mean-std, height-0.02), 2*std, 0.04,alpha=.25,color=color,fill=True,linewidth=2))

    ax.add_line(Line([mean,   mean], [height-0.02,height+0.02],     color=color))
