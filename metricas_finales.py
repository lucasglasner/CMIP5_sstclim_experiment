#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:07:45 2021

@author: lucas
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
from glob import glob
from cartopy.util import add_cyclic_point
from scipy.stats import percentileofscore
import scipy.stats as st
import pandas as pd
import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    """
    Class useful for normalize divergent colour pallets around 0, it can 
    be used with the following extra argument:
    "norm=MidpointNormalize(midpoint=0)"
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))     

def percentileofscore_3d(data,score,kind="rank"):
    sh = data.shape
    result = np.empty((sh[1],sh[2]))
    for i in range(sh[1]):
        for j in range(sh[2]):
            result[i,j] = percentileofscore(data[:,i,j],score[i,j],kind=kind)
    return result

#%%
#Load data
models = ["CSIRO-Mk3-6-0","INMCM4","IPSL-CM5A-LR","MIROC5","MPI-ESM-LR","MPI-ESM-MR","MRI-CGCM3"]
paths_sstclim   = glob("cmip5/sstclim/pr/regrid/*_anual.nc")   #Path to SSTCLIM files
paths_picontrol = glob("cmip5/piControl/pr/regrid/*_anual.nc") #Path to PICONTROL files

#Create data bag
data = {key:{model:None for model in models} for key in ["sstclim","picontrol"]}
for mod in range(len(models)):
    if models[mod] != "MULTI_MODEL_AVG":
        data["sstclim"][models[mod]]   = xr.open_dataset(paths_sstclim[mod])["pr"].squeeze().dropna(dim="year")
        data["picontrol"][models[mod]] = xr.open_dataset(paths_picontrol[mod])["pr"].squeeze().dropna(dim="year")
#%%
#Attempt with different metrics
#Metric1: Ratio of sstclim_std/picontrol_std
#Metric2: Ratio of sstclim_whiskersrange/picontrol_whiskersrange
#Metric3: F_sstclim(F-1_picontrol(95%))
#Metric4: F_sstclim(F-1_picontrol(5%))
#Compute percentile of picontrol and sstclim
target_freq = 95
picontrol_percentile = {model:None for model in models}
metric1              = {model:None for model in models+["M.MODEL-AVG"]}
metric2              = {model:None for model in models+["M.MODEL-AVG"]}
metric3              = {model:None for model in models+["M.MODEL-AVG"]}
metric4              = {model:None for model in models+["M.MODEL-AVG"]}

for mod in models:
    print(mod)
    picontrol_percentile[mod] = np.percentile(data["picontrol"][mod],target_freq,axis=0)
    metric1[mod]  = data["sstclim"][mod].values.std(axis=0)/data["picontrol"][mod].values.std(axis=0)
    Q1_sc,Q3_sc   = np.percentile(data["sstclim"][mod],25,axis=0),np.percentile(data["sstclim"][mod],75,axis=0)
    Q1_pi,Q3_pi   = np.percentile(data["picontrol"][mod],25,axis=0),np.percentile(data["picontrol"][mod],75,axis=0)
    metric2[mod]  = (Q3_sc-Q1_sc)/(Q3_pi-Q1_pi)
    metric3[mod]  = percentileofscore_3d(data["sstclim"][mod],Q3_pi)
    metric4[mod]  = percentileofscore_3d(data["sstclim"][mod],Q1_pi)
    

#%%
#Multi model average of metrics
for metric in ["metric1","metric2","metric3","metric4"]:
    eval(metric)["M.MODEL-AVG"] = np.stack(list(eval(metric).values())[:-1],axis=0).mean(axis=0) 

#%%
var = "metric1"
plt.boxplot([m.ravel() for m in list(eval(var).values())],sym="")

#%%
var = "metric1"
#Map of metrics###
fig,ax = plt.subplots(2,4,subplot_kw={"projection":ccrs.Robinson()},num=1,figsize=(16,4))
ax = ax.ravel()
# norm = mpl.colors.Normalize(90,100)
# norm=MidpointNormalize(midpoint=95,vmin=70,vmax=100)
norm=colors.TwoSlopeNorm(1,0.6,1.4)
# im   = mpl.cm.ScalarMappable(norm=norm,cmap="BrBG")

for i in range(len(ax)):
    if i<len(ax)-1:
        ax[i].coastlines()
        ax[i].set_global()
        mapa = eval(var)[models[i]]
        lat,lon = data["sstclim"][models[0]].lat.values,data["sstclim"][models[0]].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap="BrBG",norm=norm)
        ax[i].set_title(models[i])
    else:
        ax[i].coastlines()
        ax[i].set_global()
        mapa = eval(var)["M.MODEL-AVG"]
        lat,lon = data["sstclim"][models[0]].lat.values,data["sstclim"][models[0]].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap="BrBG",norm=norm)
        ax[i].set_title("M.MODEL-AVG")
        pass


cax = fig.add_axes([0.93,0.1,0.01,0.8])
cb = fig.colorbar(cf,cax=cax)

# fig.show()
plt.savefig("plots/metricas_finales/"+var+".pdf",dpi=150,bbox_inches="tight")

#%%
#Statistical significance of std differences

    












