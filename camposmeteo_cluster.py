#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:55:27 2021

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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.vq import whiten


# %%
# Load data
models = ["CSIRO-Mk3-6-0", "IPSL-CM5A-LR", "MIROC5",
          "MPI-ESM-LR", "MPI-ESM-MR", "MRI-CGCM3", "INMCM4"]
# Path to sstclim pr anomalies
paths_picontrol_pr = sorted(
    glob("cmip5/piControl/pr/regrid/anomaly/*_yanomaly.nc"))
# Path to sstclim ts anomalies
paths_picontrol_ts = sorted(
    glob("cmip5/piControl/ts/regrid/anomaly/*_yanomaly.nc"))
# Path to sstclim psl anomalies
paths_picontrol_psl = sorted(
    glob("cmip5/piControl/psl/regrid/anomaly/*_yanomaly.nc"))
exp = "picontrol"
season = "annual"
# Create data bag
data = {model: {var: None for var in ["pr", "ts", "psl"]} for model in models}
for n in range(len(models)):
    print(models[n])
    for var in ["pr", "ts", "psl"]:
        paths = eval("paths_picontrol_"+var.replace("", ""))[n]
        print(paths)
        if exp == "sstclim":
            ck = 30
        elif exp == "picontrol":
            ck = 500
        else:
            assert False
        if "yanomaly" in paths:
            data[models[n]][var] = xr.open_dataset(paths, chunks={"time": ck})[
                var].squeeze().dropna(dim="time")
        else:
            dic = xr.open_dataset(paths, chunks={"time": ck})[
                var].squeeze().dropna(dim="time")
            dic2 = xr.merge([dic.groupby("time.month")[i]
                            for i in [4, 5, 6, 7, 8, 9]])[var].squeeze()
            data[models[n]][var] = dic2.groupby("time.year").mean()

# %% Quinta normal precipitation
qn_coords = (-33.44, 360-70.68)
pr_qn = {mod: None for mod in models}
for mod in models:
    if season == "summer":
        pr_qn[mod] = pd.read_csv(
            "estaciones/quintanormal_"+mod+"_"+exp+"_season.csv", index_col=0)
        pr_qn[mod] = pr_qn[mod]["pr"][~pr_qn[mod].iloc[:, 0]]
    elif season == "winter":
        pr_qn[mod] = pd.read_csv(
            "estaciones/quintanormal_"+mod+"_"+exp+"_season.csv", index_col=0)
        pr_qn[mod] = pr_qn[mod]["pr"][pr_qn[mod].iloc[:, 0]]
    else:
        pr_qn[mod] = pd.read_csv(
            "estaciones/quintanormal_"+mod+"_"+exp+".csv", index_col=0)
    # pr_qn[mod] = pd.read_csv("estaciones/quintanormal_"+mod+"_"+exp+".csv",index_col=0)
    # pr_qn[mod] = data[mod]["pr"].sel(lat=qn_coords[0],lon=qn_coords[1],method="nearest").to_series()

# Quinta normal dry/wet years position
dry_times = {mod: None for mod in models}
wet_times = {mod: None for mod in models}
for mod in models:
    dry_times[mod] = np.where(list(map(lambda x: float(
        x) < np.quantile(pr_qn[mod], 0.1), pr_qn[mod].squeeze())))[0]
    wet_times[mod] = np.where(list(map(lambda x: float(
        x) > np.quantile(pr_qn[mod], 0.9), pr_qn[mod].squeeze())))[0]

# %%
# Mask data to only extremes in Quinta normal. Save in new variables

dry_extremes = {model: {var: None for var in [
    "pr", "ts", "psl"]} for model in models}
wet_extremes = {model: {var: None for var in [
    "pr", "ts", "psl"]} for model in models}
for mod in models:
    print(mod)
    for var in ["pr", "ts", "psl"]:
        print(var)
        dry_extremes[mod][var] = data[mod][var][list(dry_times[mod]), :, :]
        wet_extremes[mod][var] = data[mod][var][list(wet_times[mod]), :, :]

# %%
# Build matrix for cluster
case = "dry"
matrix = {model: {None} for model in models}
var = ["ts", "psl"]
for mod in models:
    model_data = [eval(case+"_extremes")[mod][v].values.reshape(len(eval(case+"_times")[mod]), 180*360)
                  for v in var]
    shapes = [i.shape for i in model_data]
    matrix[mod] = whiten(np.hstack(model_data))

# %%
# pca = {mod:None for mod in models}
# kmeans = {mod:None for mod in models}
# for mod in models:
#     pca[mod] = PCA(n_components=5).fit(matrix[mod])

# %% compute correlation between pc and variables
# from scipy.stats import pearsonr
# corr = {mod:{v:[] for v in var} for mod in models}
# for mod in models:
#     print(mod)
#     serie = pca[mod].transform(matrix[mod])
#     for v in var:
#         print(v)
#         for pc in range(5):
#             corr[mod][v].append(np.empty((180,360)))
#             for i in range(180):
#                 for j in range(360):
#                     corr[mod][v][pc][i,j] = pearsonr(serie[:,pc],eval(case+"_extremes")[mod][v][:,i,j].values)[0]

    # kmeans[mod] = KMeans(n_clusters=3,max_iter=300).fit(matrix[mod])
case = "dry"
matrix = {var: [] for var in ["pr", "ts", "psl"]}
for var in ["pr", "ts", "psl"]:
    print(var)
    for mod in models:
        print(mod)
        model_data = eval(
            case+"_extremes")[mod][var].values.reshape(len(eval(case+"_times")[mod]), 180*360)
        matrix[var].append(model_data)
    matrix[var] = np.vstack(matrix[var])


# Cluster with PCA and KMeans
# # pca     = {var:None for var in ["pr","ts","psl"]}
kmeans = {var: None for var in ["pr", "ts", "psl"]}
for var in ["pr", "ts", "psl"]:
    # #     # pca[var]    = PCA(n_components=10).fit(matrix[var])
    kmeans[var] = KMeans(n_clusters=3, max_iter=300).fit(matrix[var])
# print("running kmeans")
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=99).fit(
    whiten(np.hstack([matrix[var] for var in ["pr", "ts", "psl"]])))

# %%
# lon,lat = data[models[0]]["pr"].lon.values,data[models[0]]["pr"].lat.values
# lon2d,lat2d = np.meshgrid(lon,lat)
# fig,ax = plt.subplots(5,3,subplot_kw={"projection":ccrs.Robinson(180)},figsize=(8,8))

# for axis in ax.ravel():
#     axis.coastlines()

# fig.text(0.5,.9,mod,ha="center",va="center",fontsize=20)
# for i in range(5):
#     # ax[i,0].pcolormesh(lon2d,lat2d,pca[mod]["pr"].components_[i,:].reshape((180,360)),
#     #                    transform=ccrs.PlateCarree(),cmap="BrBG")
#     # ax[i,1].pcolormesh(lon2d,lat2d,pca[mod]["ts"].components_[i,:].reshape((180,360)),
#     #                    transform=ccrs.PlateCarree(),cmap="coolwarm")
#     # ax[i,2].pcolormesh(lon2d,lat2d,pca[mod]["psl"].components_[i,:].reshape((180,360)),
#     #                    transform=ccrs.PlateCarree(),cmap="RdBu")
#     # ax[i,0].set_title("{:.1%}".format(pca[mod]["pr"].explained_variance_ratio_[i]))

#     ax[i,0].annotate(xy=(-0.55,0.5),xytext=(-0.55,0.5),text="{:.1%}".format(pca[mod].explained_variance_ratio_[i]),
#                       fontsize=15,xycoords="axes fraction")

#     loadings = (pca[mod].components_.T * np.sqrt(pca[mod].explained_variance_)).T
#     pr=ax[i,0].pcolormesh(lon2d,lat2d,loadings[i,:64800].reshape((180,360)),
#                         transform=ccrs.PlateCarree(),cmap="BrBG",vmin=-.5,vmax=.5)
#     ts=ax[i,1].pcolormesh(lon2d,lat2d,loadings[i,64800:129600].reshape((180,360)),
#                         transform=ccrs.PlateCarree(),cmap="coolwarm_r",vmin=-.5,vmax=.5)
#     # psl=ax[i,2].pcolormesh(lon2d,lat2d,loadings[i,129600:].reshape((180,360)),
#     #                     transform=ccrs.PlateCarree(),cmap="RdBu",vmin=-.5,vmax=.5)


#     # ax[i,0].pcolormesh(lon2d,lat2d,eval(case+"_extremes")[mod]["pr"][kmeans[mod].labels_==i,:,:].mean(axis=0),
#     #                    transform=ccrs.PlateCarree(),cmap="RdBu")
#     # ax[i,1].pcolormesh(lon2d,lat2d,eval(case+"_extremes")[mod]["ts"][kmeans[mod].labels_==i,:,:].mean(axis=0),
#     #                    transform=ccrs.PlateCarree(),cmap="RdBu")
#     # ax[i,2].pcolormesh(lon2d,lat2d,eval(case+"_extremes")[mod]["psl"][kmeans[mod].labels_==i,:,:].mean(axis=0),
#     #                    transform=ccrs.PlateCarree(),cmap="RdBu")


# cax_pr  = fig.add_axes([ax[i,0].get_position().xmin+0.01,ax[i,0].get_position().ymin-0.025,0.2,0.015])
# cax_ts  = fig.add_axes([ax[i,1].get_position().xmin+0.01,ax[i,1].get_position().ymin-0.025,0.2,0.015])
# cax_psl = fig.add_axes([ax[i,2].get_position().xmin+0.01,ax[i,2].get_position().ymin-0.025,0.2,0.015])

# fig.colorbar(pr,cax=cax_pr,orientation="horizontal",label="pr")
# fig.colorbar(ts,cax=cax_ts,orientation="horizontal",ticks=np.arange(-1,1+0.5,0.5),label="ts")
# # fig.colorbar(psl,cax=cax_psl,orientation="horizontal",label="psl")
#     #
