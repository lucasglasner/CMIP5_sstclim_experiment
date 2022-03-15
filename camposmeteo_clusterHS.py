#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 10:08:37 2021

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
import matplotlib.path as mpath
# %%
# Load data
exp = "sstclim"
models = ["CSIRO-Mk3-6-0", "IPSL-CM5A-LR", "MIROC5",
          "MPI-ESM-LR", "MPI-ESM-MR", "MRI-CGCM3", "INMCM4"]
# Path to sstclim pr anomalies
paths_picontrol_pr = sorted(
    glob("cmip5/"+exp+"/pr/regrid/anomaly/*_manomaly.nc"))
# Path to sstclim ts anomalies
paths_picontrol_ts = sorted(
    glob("cmip5/"+exp+"/ts/regrid/anomaly/*_manomaly.nc"))
# Path to sstclim psl anomalies
paths_picontrol_psl = sorted(
    glob("cmip5/"+exp+"/psl/regrid/anomaly/*_manomaly.nc"))
season = "summer"
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
        if season == "annual":
            dic = xr.open_dataset(paths, chunks={"time": ck})[
                var].squeeze().dropna(dim="time")
            dic = dic.sel(lat=slice(*[-90, 0]))
            data[models[n]][var] = dic
        else:
            dic = xr.open_dataset(paths, chunks={"time": ck})[
                var].squeeze().dropna(dim="time")
            dic = dic.sel(lat=slice(*[-90, 0]))
            if season == "winter":
                s = [4, 5, 6, 7, 8, 9]
            else:
                s = [1, 2, 3, 10, 11, 12]
            dic2 = xr.merge([dic.groupby("time.month")[i]
                             for i in s])[var].squeeze()
            data[models[n]][var] = dic2.groupby("time.year").mean()
# %%Quinta normal precipitation
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
# build matrix for cluster
case = 'wet'
varss = ["psl", 'pr', 'ts']
matrix = {var: [] for var in varss}
for var in varss:
    print(var)
    for mod in models:
        print(mod)
        model_data = eval(
            case+"_extremes")[mod][var].values.reshape(len(eval(case+"_times")[mod]), 90*360)
        matrix[var].append(model_data)
    matrix[var] = np.vstack(matrix[var])

# %%
print('running kmeans')
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=99).fit(
    whiten(np.hstack([matrix[var] for var in ["psl"]])))

# %%

fig, ax = plt.subplots(3, 3, figsize=(10, 10),
                       subplot_kw={'projection': ccrs.SouthPolarStereo()})
lon, lat = data[models[0]]["pr"].lon.values, data[models[0]]["pr"].lat.values
lon2d, lat2d = np.meshgrid(lon, lat)
for axis in ax.ravel():
    axis.coastlines()
    axis.gridlines(linestyle=":")
    axis.set_extent([-180, 180, -90, 0], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    axis.set_boundary(circle, transform=axis.transAxes)

for i in range(3):
    pr = ax[i, 0].pcolormesh(lon2d, lat2d, 86400*365/6*matrix['pr'][kmeans.labels_ == i, :].mean(axis=0).reshape(90, 360),
                             transform=ccrs.PlateCarree(), rasterized=True, cmap="BrBG",
                             norm=colors.Normalize(-75, 75))
    ts = ax[i, 1].pcolormesh(lon2d, lat2d, matrix['ts'][kmeans.labels_ == i, :].mean(axis=0).reshape(90, 360),
                             transform=ccrs.PlateCarree(), rasterized=True, cmap="coolwarm",
                             norm=colors.Normalize(-.5, .5))
    psl = ax[i, 2].pcolormesh(lon2d, lat2d, matrix['psl'][kmeans.labels_ == i, :].mean(axis=0).reshape(90, 360),
                              transform=ccrs.PlateCarree(), rasterized=True, cmap="RdBu_r",
                              norm=colors.Normalize(-50, 50))
    ax[i, 0].set_title(np.count_nonzero(
        kmeans.labels_ == i), loc='left', fontsize=18)

cax_pr = fig.add_axes([ax[-1, 0].get_position().xmin+0.01,
                      ax[-1, 0].get_position().ymin-0.025, 0.2, 0.015])
cax_ts = fig.add_axes([ax[-1, 1].get_position().xmin+0.01,
                      ax[-1, 1].get_position().ymin-0.025, 0.2, 0.015])
cax_psl = fig.add_axes([ax[-1, 2].get_position().xmin +
                       0.01, ax[-1, 2].get_position().ymin-0.025, 0.2, 0.015])

fig.colorbar(pr, cax=cax_pr, orientation="horizontal", label="pr")
fig.colorbar(ts, cax=cax_ts, orientation="horizontal",
             ticks=np.arange(-1, 1+0.5, 0.5), label="ts")
fig.colorbar(psl, cax=cax_psl, orientation="horizontal", label="psl")

plt.savefig('plots/clusters/HS/clusters_'+exp+'_'+case+'_kmeans_'+season+'_SLP.pdf',
            dpi=150, bbox_inches='tight')
