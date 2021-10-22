#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 15:43:27 2021

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
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import trange


def test_std(ts_sstclim,ts_picontrol,N_montecarlo=999,confidence_level=0.05,
             bootstrap_sample=30):
    std_samples = np.empty(N_montecarlo)
    tspan = ts_picontrol.shape[0]
    for i in range(N_montecarlo):
        random_positions = np.random.randint(0,tspan-1,bootstrap_sample)
        std_samples[i] = np.nanstd(ts_picontrol.values[random_positions])
    std_sstclim = np.nanstd(ts_sstclim.values)
    pvalue = (np.count_nonzero(std_samples>std_sstclim)+1)/(N_montecarlo+1)
    # ecdf = ECDF(std_samples,side="left")
    # pvalue = ecdf(std_sstclim)
    return pvalue
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
metric1 = {model:None for model in models+["M.MODEL-AVG"]}  
for mod in models:
    print(mod)
    metric1[mod]  = data["sstclim"][mod].values.std(axis=0)/data["picontrol"][mod].values.std(axis=0)
metric1["M.MODEL-AVG"] = np.stack(list(metric1.values())[:-1],axis=0).mean(axis=0) 

#%%
#Compute statistical significance of differences
# pvalues     = {model:np.empty(metric1[models[0]].shape) for model in models+["M.MODEL-AVG"]}
# N_montecarlo = 999
# for mod in models:
#     print(mod)
#     for i in range(data["picontrol"][mod].shape[1]):
#         for j in trange(data["picontrol"][mod].shape[2]):
#             pvalues[mod][i,j] = test_std(data["sstclim"][mod][:,i,j],
#                                          data["picontrol"][mod][:,i,j],
#                                          N_montecarlo=N_montecarlo)
            

#%%
from scipy.stats import ttest_1samp
#Compute statistical significance of differences
pvalues     = {model:np.empty(metric1[models[0]].shape) for model in models+["M.MODEL-AVG"]}
std_samples = {model:[] for model in models}
N_montecarlo = 1999
for mod in models:
    print(mod)
    count = 0
    tspan = data["picontrol"][mod].shape[0]
    while count<N_montecarlo:
        random_positions = np.random.randint(0,tspan-1,30)
        std_samples[mod].append(data["picontrol"][mod][random_positions,:,:].values.std(axis=0))
        count+=1
    std_samples[mod] = np.stack(std_samples[mod],axis=0)
    std_sstclim = data["sstclim"][mod].values.std(axis=0)
    for i in range(data["picontrol"][mod].shape[1]):
        for j in range(data["picontrol"][mod].shape[2]):
            # crit1 = np.quantile(std_samples[mod][:,i,j],0.05)
            # crit2 = np.quantile(std_samples[mod][:,i,j],1-0.05)
            # interval = np.min((crit1,crit2))
            # if std_sstclim[i,j] <= interval:
            #     pvalues[mod][i,j] = False
            # else:
            #     pvalues[mod][i,j] = True
            # ecdf = ECDF(std_samples[mod][:,i,j])
            # pvalues[mod][i,j] = ecdf(std_sstclim[i,j])
            pvalues[mod][i,j] = ((std_samples[mod][:,i,j]<=std_sstclim[i,j]).sum()+1)/(N_montecarlo+1)
            # pvalues[mod][i,j] = ttest_1samp(std_samples[mod][:,i,j],std_sstclim[i,j]).pvalue
    std_samples.pop(mod)
#%%
pvalues["M.MODEL-AVG"] = np.stack(list(pvalues.values())[:-1],axis=0).mean(axis=0) 

#%%
#significance plot
var = "pvalues"
#Map of metrics###
fig,ax = plt.subplots(2,4,subplot_kw={"projection":ccrs.Robinson(central_longitude=180)},num=1,figsize=(16,4))
ax = ax.ravel()
# norm=colors.TwoSlopeNorm(1,0,1))
norm = colors.Normalize(vmin=0,vmax=1)
cmap = "Blues"
mapas = []
masks = {model:None for model in models+["M.MODEL-AVG"]}
for i in range(len(ax)):
    if i<len(ax)-1:
        ax[i].coastlines()
        ax[i].set_global()
        mapa = eval(var)[models[i]]
        lat,lon = data["sstclim"][models[0]].lat.values,data["sstclim"][models[0]].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        masks[models[i]] = mapa < 0.05
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap=cmap,norm=norm,rasterized=True)
        ax[i].set_title(models[i])
    else:
        ax[i].coastlines()
        ax[i].set_global()
        mapa = eval(var)["M.MODEL-AVG"]
        lat,lon = data["sstclim"][models[0]].lat.values,data["sstclim"][models[0]].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        # masks["M.MODEL-AVG"] = np.logical_and.reduce(list(masks.values())[:-1])
        masks["M.MODEL-AVG"] = mapa < 0.05
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap=cmap,norm=norm,rasterized=True)
        # cf = ax[i].pcolormesh(lon,lat,pvalues["M.MODEL-AVG"],#levels=np.arange(-1,1.1,0.1),
        #                       transform=ccrs.PlateCarree(), cmap=cmap,norm=norm)
        ax[i].set_title("M.MODEL-AVG")
        pass


cax = fig.add_axes([0.93,0.1,0.01,0.8])
cb = fig.colorbar(cf,cax=cax)

# fig.show()
plt.savefig("plots/metricas_finales/pvalue_std.pdf",dpi=150,bbox_inches="tight")


#%%
from scipy.ndimage import uniform_filter
var = "metric1"
#Map of metrics###
fig,ax = plt.subplots(2,4,subplot_kw={"projection":ccrs.Robinson(central_longitude=180)},num=1,figsize=(16,4))
ax = ax.ravel()
# norm = mpl.colors.Normalize(90,100)
# norm=MidpointNormalize(midpoint=95,vmin=70,vmax=100)
norm=colors.TwoSlopeNorm(1,0.6,1.4)
# im   = mpl.cm.ScalarMappable(norm=norm,cmap="BrBG")
cmap = "BrBG_r"
for i in range(len(ax)):
    if i<len(ax)-1:
        ax[i].coastlines()
        ax[i].set_global()
        mapa = eval(var)[models[i]]
        lat,lon = data["sstclim"][models[0]].lat.values,data["sstclim"][models[0]].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        # mapa = np.where(~masks[models[i]],np.nan,mapa)
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap=cmap,norm=norm, rasterized=True)
        mapa3 = pvalues[models[i]]
        pv = ax[i].contour(lon[:-1],lat,uniform_filter(mapa3,9),transform=ccrs.PlateCarree(),levels=[0.05],
                            colors="r",zorder=2, linewidths=1)
        ax[i].set_title(models[i])
    else:
        ax[i].coastlines()
        ax[i].set_global()
        mapa = eval(var)["M.MODEL-AVG"]
        lat,lon = data["sstclim"][models[0]].lat.values,data["sstclim"][models[0]].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        # mapa = np.where(~masks["M.MODEL-AVG"],np.nan,mapa)
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap=cmap,norm=norm,rasterized=True)
        # ax[i].pcolormesh(lon,lat,np.where(~masks["M.MODEL-AVG"],masks["M.MODEL-AVG"],np.nan),transform=ccrs.PlateCarree(),
        #                  cmap="Blues_r",norm=mpl.colors.Normalize(0,1))
        mapa3 = pvalues["M.MODEL-AVG"]
        pv = ax[i].contour(lon[:-1],lat,uniform_filter(mapa3,9),transform=ccrs.PlateCarree(),levels=[0.05],
                            colors="r",zorder=2, linewidths=1)
        ax[i].set_title("M.MODEL-AVG")
        pass


cax = fig.add_axes([0.93,0.1,0.01,0.8])
cb = fig.colorbar(cf,cax=cax)

# fig.show()
plt.savefig("plots/metricas_finales/"+var+"_masked.pdf",dpi=150,bbox_inches="tight")