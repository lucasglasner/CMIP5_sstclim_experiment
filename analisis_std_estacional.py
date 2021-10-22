#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:14:23 2021

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

#%%
#Load data
models = ["CSIRO-Mk3-6-0","INMCM4","IPSL-CM5A-LR","MIROC5","MPI-ESM-LR","MPI-ESM-MR","MRI-CGCM3"]
paths_sstclim   = glob("cmip5/sstclim/pr/regrid/*_season.nc")   #Path to SSTCLIM files
paths_picontrol = glob("cmip5/piControl/pr/regrid/*_season.nc") #Path to PICONTROL files

#Create data bag
data = {key:{model:None for model in models} for key in ["sstclim","picontrol"]}
for mod in range(len(models)):
    print(models[mod])
    if models[mod] != "MULTI_MODEL_AVG":
        data["sstclim"][models[mod]]   = xr.open_dataset(paths_sstclim[mod], chunks={"time":30})[["pr_winter","pr_summer"]].squeeze().dropna(dim="time")
        data["picontrol"][models[mod]] = xr.open_dataset(paths_picontrol[mod], chunks={"time":500})[["pr_winter","pr_summer"]].squeeze().dropna(dim="time")
        
        
#%%
metric_all    = {model:None for model in models+["M.MODEL-AVG"]}  
metric_winter = {model:None for model in models+["M.MODEL-AVG"]}  
metric_summer = {model:None for model in models+["M.MODEL-AVG"]}  
for mod in models:
    print(mod)
    sstclim_winter = data["sstclim"][mod]["pr_winter"]*86400*365/2
    sstclim_summer = data["sstclim"][mod]["pr_summer"]*86400*365/2
    sstclim_all    = np.concatenate((sstclim_winter,sstclim_summer),axis=0)
    
    picontrol_winter = data["picontrol"][mod]["pr_winter"]*86400*365/2
    picontrol_summer = data["picontrol"][mod]["pr_summer"]*86400*365/2
    picontrol_all    = np.concatenate((picontrol_winter,picontrol_summer),axis=0)
    
    
    metric_summer[mod] = sstclim_summer.values.std(axis=0)/picontrol_summer.values.std(axis=0)
    metric_winter[mod] = sstclim_winter.values.std(axis=0)/picontrol_winter.values.std(axis=0) 
    metric_all[mod]    = sstclim_all.std(axis=0)/picontrol_all.std(axis=0)
    
    del sstclim_winter,sstclim_summer,sstclim_all,picontrol_winter,picontrol_summer,picontrol_all
#%%
metric_all["M.MODEL-AVG"] = np.stack(list(metric_all.values())[:-1],axis=0).mean(axis=0) 
metric_winter["M.MODEL-AVG"] = np.stack(list(metric_winter.values())[:-1],axis=0).mean(axis=0) 
metric_summer["M.MODEL-AVG"] = np.stack(list(metric_summer.values())[:-1],axis=0).mean(axis=0) 

#%%
#Compute statistical significance of differences
pvalues_all     = {model:np.empty(metric_all[models[0]].shape) for model in models+["M.MODEL-AVG"]}
pvalues_winter  = {model:np.empty(metric_all[models[0]].shape) for model in models+["M.MODEL-AVG"]}
pvalues_summer  = {model:np.empty(metric_all[models[0]].shape) for model in models+["M.MODEL-AVG"]}

std_samples_all     = {model:[] for model in models}
std_samples_winter  = {model:[] for model in models}
std_samples_summer  = {model:[] for model in models}
N_montecarlo = 1999
for mod in models:
    print(mod)
    for x in ["winter","summer","all"]:
        print(x)
        std_samples = eval("std_samples_"+x)
        pvalues     = eval("pvalues_"+x)
        if x!="all":
            picontrol   = data["picontrol"][mod]["pr_"+x].values*86400*365/2
            sstclim     = data["sstclim"][mod]["pr_"+x].values*86400*365/2
        else:
            picontrol   = np.concatenate((data["picontrol"][mod]["pr_summer"].values*86400*365/2,
                                          data["picontrol"][mod]["pr_winter"].values*86400*365/2), axis=0)
            
            sstclim     = np.concatenate((data["sstclim"][mod]["pr_summer"].values*86400*365/2,
                                          data["sstclim"][mod]["pr_winter"].values*86400*365/2), axis=0)
        count = 0
        tspan = picontrol.shape[0]
        while count<N_montecarlo:
            random_positions = np.random.randint(0,tspan-1,30)
            std_samples[mod].append(picontrol[random_positions,:,:].std(axis=0))
            count+=1
        std_samples[mod] = np.stack(std_samples[mod],axis=0)
        std_sstclim = sstclim.std(axis=0)
        for i in range(pvalues[mod].shape[0]):
            for j in range(pvalues[mod].shape[1]):
                pvalues[mod][i,j] = ((std_samples[mod][:,i,j]<=std_sstclim[i,j]).sum()+1)/(N_montecarlo+1)
        std_samples.pop(mod)
#%%
for pvalues in [pvalues_all, pvalues_winter, pvalues_summer]:
    pvalues["M.MODEL-AVG"] = np.stack(list(pvalues.values())[:-1],axis=0).mean(axis=0)
    
# del pvalues, picontrol, sstclim, std_sstclim, std_samples_all, std_samples_winter,

#%%
#significance plot
var = "pvalues_all"
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
        lat,lon = data["sstclim"][models[0]]["pr_winter"].lat.values,data["sstclim"][models[0]]["pr_winter"].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        masks[models[i]] = mapa < 0.05
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap=cmap,norm=norm,
                              rasterized=True)
        ax[i].set_title(models[i])
    else:
        ax[i].coastlines()
        ax[i].set_global()
        mapa = eval(var)["M.MODEL-AVG"]
        lat,lon = data["sstclim"][models[0]]["pr_winter"].lat.values,data["sstclim"][models[0]]["pr_winter"].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        # masks["M.MODEL-AVG"] = np.logical_and.reduce(list(masks.values())[:-1])
        masks["M.MODEL-AVG"] = mapa < 0.05
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap=cmap,norm=norm,
                              rasterized=True)
        # cf = ax[i].pcolormesh(lon,lat,pvalues["M.MODEL-AVG"],#levels=np.arange(-1,1.1,0.1),
        #                       transform=ccrs.PlateCarree(), cmap=cmap,norm=norm)
        ax[i].set_title("M.MODEL-AVG")
        pass


cax = fig.add_axes([0.93,0.1,0.01,0.8])
cb = fig.colorbar(cf,cax=cax)

# fig.show()
plt.savefig("plots/season/pvalue_std_all.pdf",dpi=150,bbox_inches="tight")

#%%
from matplotlib.colors import ListedColormap
from scipy.ndimage import uniform_filter, gaussian_filter
var = "metric_all"
#Map of metrics###
fig,ax = plt.subplots(2,4,subplot_kw={"projection":ccrs.Robinson(central_longitude=180)},num=1,figsize=(16,4))
ax = ax.ravel()
# norm = mpl.colors.Normalize(90,100)
# norm=MidpointNormalize(midpoint=95,vmin=70,vmax=100)
norm=colors.TwoSlopeNorm(1,0.4,1.6)
# im   = mpl.cm.ScalarMappable(norm=norm,cmap="BrBG")
cmap = "BrBG_r"
for i in range(len(ax)):
    if i<len(ax)-1:
        ax[i].coastlines(lw=0.8, color="k")
        ax[i].set_global()
        mapa = eval(var)[models[i]]
        lat,lon = data["sstclim"][models[0]]["pr_winter"].lat.values,data["sstclim"][models[0]]["pr_winter"].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        mapa2 = np.where(~masks[models[i]],np.nan,mapa)
        mapa3 = eval("pvalues"+var.replace("metric",""))[models[i]]
        cf = ax[i].pcolormesh(lon,lat,mapa,
                              transform=ccrs.PlateCarree(), cmap=cmap,norm=norm,
                              rasterized=True)
        pv = ax[i].contour(lon[:-1],lat,uniform_filter(mapa3,9),transform=ccrs.PlateCarree(),levels=[0.05],
                            colors="r",zorder=2, linewidths=1)
        # pv = ax[i].pcolor(lon,lat,mapa2, hatch="xxx", cmap=ListedColormap(['none']),
        #                   edgecolor="silver",
        #                   rasterized=True, transform=ccrs.PlateCarree(), lw=0, zorder=2)
        ax[i].set_title(models[i])
    else:
        ax[i].coastlines()
        ax[i].set_global()
        mapa = eval(var)["M.MODEL-AVG"]
        lat,lon = data["sstclim"][models[0]]["pr_winter"].lat.values,data["sstclim"][models[0]]["pr_winter"].lon.values
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        mapa3 = eval("pvalues"+var.replace("metric",""))["M.MODEL-AVG"]
        # mapa = np.where(~masks["M.MODEL-AVG"],np.nan,mapa)
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap=cmap,norm=norm,
                              rasterized=True)
        pv = ax[i].contour(lon[:-1],lat,uniform_filter(mapa3,9),transform=ccrs.PlateCarree(),levels=[0.05],
                            colors="r",zorder=2, linewidths=1)
        # ax[i].pcolormesh(lon,lat,np.where(~masks["M.MODEL-AVG"],masks["M.MODEL-AVG"],np.nan),transform=ccrs.PlateCarree(),
        #                  cmap="Blues_r",norm=mpl.colors.Normalize(0,1))
        ax[i].set_title("M.MODEL-AVG")
        pass


cax = fig.add_axes([0.93,0.1,0.01,0.8])
cb = fig.colorbar(cf,cax=cax)

# fig.show()
plt.savefig("plots/season/std_"+var.replace("metric","")+".pdf",dpi=150,bbox_inches="tight")