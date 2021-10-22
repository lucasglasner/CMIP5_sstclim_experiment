#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:48:10 2021

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
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.colors as colors

def percentileofscore_3d(data,score,kind="rank",method="scipy"):
    sh = data.shape
    result = np.empty((sh[1],sh[2]))
    for i in range(sh[1]):
        for j in range(sh[2]):
            if method=="scipy":
                result[i,j] = percentileofscore(data[:,i,j],score[i,j],kind=kind)
            elif method=="statsmodels":
                ecdf = ECDF(data[:,i,j].data.flatten())
                result[i,j] = ecdf(score[i,j])*100
    return result



def test_extremes(ts_sstclim,ts_picontrol,p=90,N_montecarlo=1999,bootstrap_hand=30,kind="greater"):
    tspan = ts_picontrol.values.flatten().shape[0]
    metric = np.empty(N_montecarlo)
    ecdf = ECDF(ts_sstclim.values.flatten())
    for i in range(N_montecarlo):
        random_positions = np.random.randint(0,tspan-1,bootstrap_hand)
        sample = ts_picontrol.values.flatten()[random_positions]
        metric[i] = ecdf(np.percentile(sample,p))*100
    if kind=="greater":
        count = (metric>p).sum()
    elif kind=="less":
        count = (metric<p).sum()
    return (count+1)/(N_montecarlo+1)
        
#%%
#Load data
models = ["CSIRO-Mk3-6-0","INMCM4","IPSL-CM5A-LR","MIROC5","MPI-ESM-LR","MPI-ESM-MR","MRI-CGCM3"]
paths_sstclim   = glob("cmip5/sstclim/pr/regrid/*_anual.nc")   #Path to SSTCLIM files
paths_picontrol = glob("cmip5/piControl/pr/regrid/*_anual.nc") #Path to PICONTROL files

#Create data bag
data = {key:{model:None for model in models} for key in ["sstclim","picontrol"]}
for mod in range(len(models)):
    print(models[mod])
    if models[mod] != "MULTI_MODEL_AVG":
        data["sstclim"][models[mod]]   = xr.open_dataset(paths_sstclim[mod],chunks={"time":30})["pr"].squeeze().dropna(dim="year")
        data["picontrol"][models[mod]] = xr.open_dataset(paths_picontrol[mod],chunks={"time":500})["pr"].squeeze().dropna(dim="year")

#%%
#metric1: cdf_sstclim(percentile90_picontrol)
#metric2: cdf_sstclim(percentile10_picontrol)

metric1 = {model:None for model in models+["M.MODEL-AVG"]} #Wet extremes
metric2 = {model:None for model in models+["M.MODEL-AVG"]} #Dry extremes
# picontrol_percentile = {model:None for model in models+["M.MODEL-AVG"]}
for mod in models:
    print(mod)
    picontrol_p90 = np.percentile(data["picontrol"][mod], 90, axis=0)
    picontrol_p10 = np.percentile(data["picontrol"][mod], 10, axis=0)
    metric1[mod]  = percentileofscore_3d(data["sstclim"][mod],picontrol_p90, method="statsmodels")
    metric2[mod]  = percentileofscore_3d(data["sstclim"][mod],picontrol_p10, method="statsmodels")
metric1["M.MODEL-AVG"] = np.stack(list(metric1.values())[:-1],axis=0).mean(axis=0)    
metric2["M.MODEL-AVG"] = np.stack(list(metric2.values())[:-1],axis=0).mean(axis=0)     

#%%
var = "metric1"
#Map of metrics###
fig,ax = plt.subplots(2,4,subplot_kw={"projection":ccrs.Robinson(central_longitude=180)},num=1,figsize=(16,4))
ax = ax.ravel()
norm=colors.TwoSlopeNorm(90,0,100)
# norm = colors.Normalize(vmin=0,vmax=100)
cmap = "RdBu"
mapas = []
for i in range(len(ax)):
    if i<len(ax)-1:
        ax[i].coastlines()
        ax[i].set_global()
        mapa = eval(var)[models[i]]
        lat,lon = data["sstclim"][models[0]].lat.values,data["sstclim"][models[0]].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap=cmap,norm=norm,
                              rasterized=True,antialiased=False)
        ax[i].set_title(models[i])
    else:
        ax[i].coastlines()
        ax[i].set_global()
        mapa = eval(var)["M.MODEL-AVG"]
        lat,lon = data["sstclim"][models[0]].lat.values,data["sstclim"][models[0]].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        # masks["M.MODEL-AVG"] = np.logical_and.reduce(list(masks.values())[:-1])
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap=cmap,norm=norm, 
                              rasterized=True,antialiased=False)
        # cf = ax[i].pcolormesh(lon,lat,pvalues["M.MODEL-AVG"],#levels=np.arange(-1,1.1,0.1),
        #                       transform=ccrs.PlateCarree(), cmap=cmap,norm=norm)
        ax[i].set_title("M.MODEL-AVG")
        pass


cax = fig.add_axes([0.93,0.1,0.01,0.8])
cb = fig.colorbar(cf,cax=cax)

# fig.show()
plt.savefig("plots/metricas_finales/extremos/extremos_lluviosos_statsmodels.pdf",dpi=150,bbox_inches="tight")

#%%
from tqdm import trange
#compute pvalue

pvalues_wet = {model:np.ones(metric1[model].shape) for model in models+["M.MODEL-AVG"]}
pvalues_dry = {model:np.ones(metric1[model].shape) for model in models+["M.MODEL-AVG"]}

dry_samples = {model:[] for model in models}
wet_samples = {model:[] for model in models}
N_montecarlo = 1999
for mod in trange(len(models)):
    mod = models[mod]
    print(mod)
    count = 0
    tspan = data["picontrol"][mod].shape[0]
    while count<N_montecarlo:
        print(count)
        random_positions = np.random.randint(0,tspan-1,30)
        picontrol_p90 = np.percentile(data["picontrol"][mod][random_positions,:,:], 90, axis=0)
        picontrol_p10 = np.percentile(data["picontrol"][mod][random_positions,:,:], 10, axis=0)
        sample_dry = percentileofscore_3d(data["sstclim"][mod],picontrol_p10, method="statsmodels")       
        sample_wet = percentileofscore_3d(data["sstclim"][mod],picontrol_p90, method="statsmodels")
        dry_samples[mod].append(sample_dry)
        wet_samples[mod].append(sample_wet)
        count+=1
    print("MC end.")
    dry_samples[mod] = np.stack(dry_samples[mod],axis=0)
    wet_samples[mod] = np.stack(wet_samples[mod],axis=0)
    for i in range(data["picontrol"][mod].shape[1]):
        for j in range(data["picontrol"][mod].shape[2]):
            pvalues_dry[mod][i,j] = ((dry_samples[mod][:,i,j]<=10).sum()+1)/(N_montecarlo+1)
            pvalues_wet[mod][i,j] = ((dry_samples[mod][:,i,j]<=90).sum()+1)/(N_montecarlo+1)
    dry_samples.pop(mod)
    wet_samples.pop(mod)    

# SE DEMORA MUCHOOOOOOOOOO!!!!
# for mod in models:
#     print(mod)
#     for i in trange(metric1[mod].shape[0]):
#         for j in range(metric1[mod].shape[1]):
#             pvalues_wet[mod][i,j] = test_extremes(data["sstclim"][mod][:,i,j],
#                                                   data["picontrol"][mod][:,i,j],
#                                                   p=90, N_montecarlo=1999, bootstrap_hand=30,
#                                                   kind="greater")
#             pvalues_dry[mod][i,j] = test_extremes(data["sstclim"][mod][:,i,j],
#                                                   data["picontrol"][mod][:,i,j],
#                                                   p=10, N_montecarlo=1999, bootstrap_hand=30,
#                                                   kind="greater")
# pvalues_wet["M.MODEL-AVG"] = np.stack(list(pvalues_wet.values())[:-1],axis=0).mean(axis=0)    
# pvalues_dry["M.MODEL-AVG"] = np.stack(list(pvalues_dry.values())[:-1],axis=0).mean(axis=0)    

#%%