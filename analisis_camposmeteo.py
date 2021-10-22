#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 12:27:36 2021

@author: lucas
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
from glob import glob
from cartopy.util import add_cyclic_point
import pandas as pd
import matplotlib.colors as colors
import os


#%%
#Load data
models = ["CSIRO-Mk3-6-0","IPSL-CM5A-LR","MIROC5","MPI-ESM-LR","MPI-ESM-MR","MRI-CGCM3","INMCM4"]
paths_sstclim_pr   = sorted(glob("cmip5/sstclim/pr/regrid/anomaly/*_manomaly.nc"))   #Path to sstclim pr anomalies
paths_sstclim_ts   = sorted(glob("cmip5/sstclim/ts/regrid/anomaly/*_manomaly.nc"))   #Path to sstclim ts anomalies
paths_sstclim_psl  = sorted(glob("cmip5/sstclim/psl/regrid/anomaly/*_manomaly.nc"))  #Path to sstclim psl anomalies
exp = "sstclim"
season="summer"
#Create data bag
data = {model:{var:None for var in ["pr","ts","psl"]} for model in models}
for n in range(len(models)):
    print(models[n])
    for var in ["pr","ts","psl"]:
        paths = eval("paths_sstclim_"+var.replace("",""))[n]
        print(paths)
        if exp=="sstclim":
            ck = 30
        elif exp=="picontrol":
            ck=500
        else:
            assert False
        if "yanomaly" in paths:
            data[models[n]][var] = xr.open_dataset(paths,chunks={"time":ck})[var].squeeze().dropna(dim="time")
        else:
            if season=="summer":
                lis = [1,2,3,10,11,12]
            elif season=="winter":
                lis = [4,5,6,7,8,9]
            else:
                assert False
            dic = xr.open_dataset(paths,chunks={"time":ck})[var].squeeze().dropna(dim="time")
            dic2 = xr.merge([dic.groupby("time.month")[i] for i in lis])[var].squeeze()
            data[models[n]][var] = dic2.groupby("time.year").mean()
        
#%%
case   = "dry"
#Quinta normal precipitation
qn_coords  = (-33.44,360-70.68)
pr_qn  = {mod:None for mod in models}
for mod in models:
    if season=="summer":
        pr_qn[mod] = pd.read_csv("estaciones/quintanormal_"+mod+"_"+exp+"_season.csv",index_col=0)
        pr_qn[mod] = pr_qn[mod]["pr"][~pr_qn[mod].iloc[:,0]]
    elif season=="winter":
        pr_qn[mod] = pd.read_csv("estaciones/quintanormal_"+mod+"_"+exp+"_season.csv",index_col=0)
        pr_qn[mod] = pr_qn[mod]["pr"][pr_qn[mod].iloc[:,0]]
    else:
        pr_qn[mod] = pd.read_csv("estaciones/quintanormal_"+mod+"_"+exp+".csv",index_col=0)
    # pr_qn[mod] = pd.read_csv("estaciones/quintanormal_"+mod+"_"+exp+".csv",index_col=0)
    # pr_qn[mod] = data[mod]["pr"].sel(lat=qn_coords[0],lon=qn_coords[1],method="nearest").to_series()

# Quinta normal dry/wet years
dry_times = {mod:None for mod in models}
wet_times = {mod:None for mod in models}
for mod in models:
    dry_times[mod] = pr_qn[mod][pr_qn[mod]<np.quantile(pr_qn[mod],0.1)].dropna().index.values
    wet_times[mod] = pr_qn[mod][pr_qn[mod]>np.quantile(pr_qn[mod],0.9)].dropna().index.values

#%%
#Mask data to only extremes in Quinta normal. Save in new variables

dry_extremes = {model:{var:None for var in ["pr","ts","psl"]} for model in models}
wet_extremes = {model:{var:None for var in ["pr","ts","psl"]} for model in models}
for mod in models:
    print(mod)
    for var in ["pr","ts","psl"]:
        print(var)
        if season == "annual":
            if var != "pr":
                try:
                    time = list(map(lambda x: x.year,data[mod][var].time.values))
                except:
                    time = list(map(lambda x: pd.to_datetime(x).year,data[mod][var].time.values))
                dry_extremes[mod][var] = data[mod][var].assign_coords({"time":time}).sel(time=dry_times[mod])
                wet_extremes[mod][var] = data[mod][var].assign_coords({"time":time}).sel(time=wet_times[mod])
            else:
                dry_extremes[mod][var] = data[mod][var].sel(time=dry_times[mod])
                wet_extremes[mod][var] = data[mod][var].sel(time=wet_times[mod])
        else:
            dry_extremes[mod][var] = data[mod][var].sel(year=dry_times[mod])
            wet_extremes[mod][var] = data[mod][var].sel(year=wet_times[mod])

#%%
#plots dry/wet extremes maps
lon,lat = data[models[0]]["pr"].lon.values,data[models[0]]["pr"].lat.values
lon2d,lat2d = np.meshgrid(lon,lat)
fig,ax = plt.subplots(3,3,sharex=True,sharey=True,subplot_kw={"projection":ccrs.Robinson(180)},
                      figsize=(14,10))
for axis in ax.ravel():
    axis.coastlines()
# norm_pr  = colors.TwoSlopeNorm(0,-250,250)
# norm_pr  = colors.TwoSlopeNorm(0,-150,150)
norm_pr  = colors.TwoSlopeNorm(0,-50,50)
norm_ts  = colors.TwoSlopeNorm(0,-.5,.5)
norm_psl = colors.TwoSlopeNorm(0,-100,100)


cax_pr  = fig.add_axes([ax[2,0].get_position().xmin-0.02,ax[2,0].get_position().ymin-0.02,0.2,0.015])
cax_ts  = fig.add_axes([ax[2,1].get_position().xmin-0.02,ax[2,1].get_position().ymin-0.02,0.2,0.015])
cax_psl = fig.add_axes([ax[2,2].get_position().xmin-0.02,ax[2,2].get_position().ymin-0.02,0.2,0.015])


# mod = models[0]
for mod in models:
    ann = []
    for i in range(3):
            pr  = ax[i,0].pcolormesh(lon2d,lat2d,eval(case+"_extremes")[mod]["pr"][i,:,:]*86400*365/6,
                                     transform=ccrs.PlateCarree(),cmap="BrBG",rasterized=True,
                                     norm=norm_pr)
            ts  = ax[i,1].pcolormesh(lon2d,lat2d,eval(case+"_extremes")[mod]["ts"][i,:,:],
                                     transform=ccrs.PlateCarree(),cmap="coolwarm",rasterized=True,
                                     norm=norm_ts)
            psl = ax[i,2].pcolormesh(lon2d,lat2d,eval(case+"_extremes")[mod]["psl"][i,:,:],
                                     transform=ccrs.PlateCarree(),cmap="RdBu_r",rasterized=True,
                                     norm=norm_psl)
            if season == "annual":
                an=ax[i,0].annotate(xy=(0,1.0),xytext=(0.0,1.0),text=eval(case+"_extremes")[mod]["pr"].time[i].item(),
                                  fontsize=20,xycoords="axes fraction")
            else:
                an=ax[i,0].annotate(xy=(0,1.0),xytext=(0.0,1.0),text=eval(case+"_extremes")[mod]["pr"].year[i].item(),
                                  fontsize=20,xycoords="axes fraction")
            ann.append(an)
            # ax[i,0].set_ylabel(eval(extremes+"_extremes")[mod]["pr"].year[i].item())
    
    fig.colorbar(pr,cax=cax_pr,orientation="horizontal",label="pr")
    fig.colorbar(ts,cax=cax_ts,orientation="horizontal",label="ts")
    fig.colorbar(psl,cax=cax_psl,orientation="horizontal",label="psl")
    ax[0,1].set_title(mod,fontsize=22)
    
    plt.savefig("plots/campos/"+case+"_extremes_"+mod+"_"+season+"_anomaly.pdf",dpi=150,bbox_inches="tight")
    for an in ann:
        an.remove()
        
        
        
#%%
# cluster based upon pressure anomalies
matrix = {var:[] for var in ["pr","ts","psl"]}
for var in matrix.keys():
    for mod in models:
        print(mod)
        model_extremes = eval(case+"_extremes")[mod][var].values*np.sqrt(np.cos(np.deg2rad(lat2d)))
        dim = model_extremes.shape
        matrix[var].append(model_extremes.reshape((dim[0],dim[1]*dim[2])))
    matrix[var] = np.vstack(matrix[var])

from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten
print("running kmeans")
kmeans = KMeans(n_clusters=3,init="k-means++",n_init=199)
kmeans = kmeans.fit((matrix["psl"]))

#%%

print("plot")


fig,ax = plt.subplots(3,3,figsize=(14,10), subplot_kw={"projection":ccrs.Robinson(central_longitude=180)})
for axis in ax.flatten():
    axis.coastlines()

cax_pr  = fig.add_axes([ax[2,0].get_position().xmin-0.03,ax[2,0].get_position().ymin-0.02,0.2,0.015])
cax_ts  = fig.add_axes([ax[2,1].get_position().xmin-0.03,ax[2,1].get_position().ymin-0.02,0.2,0.015])
cax_psl = fig.add_axes([ax[2,2].get_position().xmin-0.03,ax[2,2].get_position().ymin-0.02,0.2,0.015])

# cax_pr  = fig.add_axes([ax[3,0].get_position().xmin-0.04,ax[3,0].get_position().ymin-0.02,0.2,0.015])
# cax_ts  = fig.add_axes([ax[3,1].get_position().xmin-0.04,ax[3,1].get_position().ymin-0.02,0.2,0.015])
# cax_psl = fig.add_axes([ax[3,2].get_position().xmin-0.04,ax[3,2].get_position().ymin-0.02,0.2,0.015])


# norm_pr  = colors.TwoSlopeNorm(0,-300,300)
norm_pr  = colors.TwoSlopeNorm(0,-20,20)
# norm_pr  = colors.TwoSlopeNorm(0,-50,50)
norm_ts  = colors.TwoSlopeNorm(0,-.5,.5)
norm_psl = colors.TwoSlopeNorm(0,-50,50)

method="kmeans"
for i in range(np.shape(ax)[0]):
    # if method=="pca":
    #     pr  = ax[i,0].pcolormesh(lon2d,lat2d,pca["pr"].components_[i,:].reshape(180,360),
    #                               transform=ccrs.PlateCarree(),cmap="BrBG",rasterized=True,
    #                               norm=norm_pr)
    #     ts  = ax[i,1].pcolormesh(lon2d,lat2d,pca["ts"].components_[i,:].reshape(180,360),
    #                               transform=ccrs.PlateCarree(),cmap="RdBu_r",rasterized=True,
    #                               norm=norm_ts)
    #     psl = ax[i,2].pcolormesh(lon2d,lat2d,pca["psl"].components_[i,:].reshape(180,360),
    #                               transform=ccrs.PlateCarree(),cmap="Spectral_r",rasterized=True,
    #                               norm=norm_psl)
    if method=="kmeans":
        pr  = ax[i,0].pcolormesh(lon2d,lat2d,matrix["pr"][kmeans.labels_==i,:].mean(axis=0).reshape(180,360)*86400*365/6,
                                  transform=ccrs.PlateCarree(),cmap="BrBG",rasterized=True,
                                  norm=norm_pr)
        ts  = ax[i,1].pcolormesh(lon2d,lat2d,matrix["ts"][kmeans.labels_==i,:].mean(axis=0).reshape(180,360),
                                  transform=ccrs.PlateCarree(),cmap="coolwarm",rasterized=True,
                                  norm=norm_ts)
        psl = ax[i,2].pcolormesh(lon2d,lat2d,matrix["psl"][kmeans.labels_==i,:].mean(axis=0).reshape(180,360),
                                  transform=ccrs.PlateCarree(),cmap="RdBu_r",rasterized=True,
                                  norm=norm_psl)
        ax[i,0].annotate(xy=(0,1.0),xytext=(0,1.0),text=(kmeans.labels_==i).sum(),fontsize=20,xycoords="axes fraction")
        
fig.colorbar(pr,cax=cax_pr,orientation="horizontal",label="pr")
fig.colorbar(ts,cax=cax_ts,orientation="horizontal",ticks=np.arange(-1,1+0.5,0.5),label="ts")
fig.colorbar(psl,cax=cax_psl,orientation="horizontal",label="psl")


plt.savefig("plots/clusters/clusters_sstclim_"+case+"_"+method+"_"+season+"_SLP.pdf",dpi=150,bbox_inches="tight")
# plt.savefig("plots/clusters/clusters_sstclim_"+case+"_"+method+"_season.pdf",dpi=150,bbox_inches="tight")
#%%
#Build matrix for PCA, reduce time dimensions tacking into consideration all models
# case   = "dry"
# matrix = {var:[] for var in ["psl"]}
# for var in ["psl"]:
#     print(var)
#     for mod in models:
#         print(mod)
#         model_data = eval(case+"_extremes")[mod][var].values.reshape(len(eval(case+"_times")[mod]),180*360)
#         matrix[var].append(model_data)
#     matrix[var] = np.vstack(matrix[var])
    

# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from scipy.cluster.vq import whiten
# #Cluster with PCA and KMeans
# # pca     = {var:None for var in ["pr","ts","psl"]}
# # kmeans  = {var:None for var in ["pr","ts","psl"]}
# # for var in ["pr","ts","psl"]:
# #     # pca[var]    = PCA(n_components=10).fit(matrix[var])
# #     kmeans[var] = KMeans(n_clusters=3,max_iter=300).fit(matrix[var])
# print("running kmeans")
# kmeans   = KMeans(n_clusters=3,init="k-means++",n_init=199).fit(whiten(np.hstack([matrix[var] for var in ["psl"]])))

#%%
# print("plot")


# fig,ax = plt.subplots(3,3,figsize=(14,10), subplot_kw={"projection":ccrs.Robinson(central_longitude=180)})
# for axis in ax.flatten():
#     axis.coastlines()

# cax_pr  = fig.add_axes([ax[2,0].get_position().xmin-0.03,ax[2,0].get_position().ymin-0.02,0.2,0.015])
# cax_ts  = fig.add_axes([ax[2,1].get_position().xmin-0.03,ax[2,1].get_position().ymin-0.02,0.2,0.015])
# cax_psl = fig.add_axes([ax[2,2].get_position().xmin-0.03,ax[2,2].get_position().ymin-0.02,0.2,0.015])

# # cax_pr  = fig.add_axes([ax[3,0].get_position().xmin-0.04,ax[3,0].get_position().ymin-0.02,0.2,0.015])
# # cax_ts  = fig.add_axes([ax[3,1].get_position().xmin-0.04,ax[3,1].get_position().ymin-0.02,0.2,0.015])
# # cax_psl = fig.add_axes([ax[3,2].get_position().xmin-0.04,ax[3,2].get_position().ymin-0.02,0.2,0.015])


# # norm_pr  = colors.TwoSlopeNorm(0,-300,300)
# norm_pr  = colors.TwoSlopeNorm(0,-20,20)
# # norm_pr  = colors.TwoSlopeNorm(0,-50,50)
# norm_ts  = colors.TwoSlopeNorm(0,-.5,.5)
# norm_psl = colors.TwoSlopeNorm(0,-50,50)

# method="kmeans"
# for i in range(np.shape(ax)[0]):
#     # if method=="pca":
#     #     pr  = ax[i,0].pcolormesh(lon2d,lat2d,pca["pr"].components_[i,:].reshape(180,360),
#     #                               transform=ccrs.PlateCarree(),cmap="BrBG",rasterized=True,
#     #                               norm=norm_pr)
#     #     ts  = ax[i,1].pcolormesh(lon2d,lat2d,pca["ts"].components_[i,:].reshape(180,360),
#     #                               transform=ccrs.PlateCarree(),cmap="RdBu_r",rasterized=True,
#     #                               norm=norm_ts)
#     #     psl = ax[i,2].pcolormesh(lon2d,lat2d,pca["psl"].components_[i,:].reshape(180,360),
#     #                               transform=ccrs.PlateCarree(),cmap="Spectral_r",rasterized=True,
#     #                               norm=norm_psl)
#     if method=="kmeans":
#         # pr  = ax[i,0].pcolormesh(lon2d,lat2d,matrix["pr"][kmeans.labels_==i,:].mean(axis=0).reshape(180,360)*86400*365/12,
#         #                           transform=ccrs.PlateCarree(),cmap="BrBG",rasterized=True,
#         #                           norm=norm_pr)
#         # ts  = ax[i,1].pcolormesh(lon2d,lat2d,matrix["ts"][kmeans.labels_==i,:].mean(axis=0).reshape(180,360),
#         #                           transform=ccrs.PlateCarree(),cmap="coolwarm",rasterized=True,
#         #                           norm=norm_ts)
#         psl = ax[i,2].pcolormesh(lon2d,lat2d,matrix["psl"][kmeans.labels_==i,:].mean(axis=0).reshape(180,360),
#                                   transform=ccrs.PlateCarree(),cmap="RdBu_r",rasterized=True,
#                                   norm=norm_psl)
#         ax[i,0].annotate(xy=(0,1.0),xytext=(0,1.0),text=(kmeans.labels_==i).sum(),fontsize=20,xycoords="axes fraction")
        
# fig.colorbar(pr,cax=cax_pr,orientation="horizontal",label="pr")
# fig.colorbar(ts,cax=cax_ts,orientation="horizontal",ticks=np.arange(-1,1+0.5,0.5),label="ts")
# fig.colorbar(psl,cax=cax_psl,orientation="horizontal",label="psl")

# plt.savefig("plots/clusters/clusters_sstclim_"+case+"_"+method+".pdf",dpi=150,bbox_inches="tight")
# plt.savefig("plots/clusters/clusters_sstclim_"+case+"_"+method+"_season.pdf",dpi=150,bbox_inches="tight")





