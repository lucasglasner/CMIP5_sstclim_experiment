#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:14:13 2021

@author: lucas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt
import glob
import os
from seaborn import kdeplot,ecdfplot
from statsmodels.nonparametric.kde import KDEUnivariate
import matplotlib as mpl
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import trange
from scipy.stats import ttest_1samp, percentileofscore

def corregir_ECR2(df):
    """
    Función para transformar las series de tiempo del CR2 a series
    de tiempo trabajables facilmente con los algoritmos de pandas
    para series de tiempo.

    Parameters
    ----------
    df : Data Frame de pandas con una serie de tiempo descargada
         del explorador climático (directo de pandas.read_csv()).

    Returns
    -------
    df : Serie de Pandas con los indices de la serie objetos de 
         tiempo.

    """
    index=[]
    for i in range(len(df)):
        index.append(dt.datetime(df["agno"][i],df[" mes"][i],df[" dia"][i]))
    df.index = index
    df.drop(["agno"," mes"," dia"],axis=1,inplace=True)
    df = df.resample("d").asfreq()[" valor"].rename("serie")
    return df

#%%
#Datos Quinta Normal
dataQN = pd.read_csv("estaciones/quintanormal_estacion.csv")
dataQN = corregir_ECR2(dataQN)
dataQN = dataQN.resample("M").sum().to_period()
dataQN = dataQN.groupby([dataQN.index.year,
                         [i in [4,5,6,7,8,9] for i in dataQN.index.month]]
                        ).sum()

#miscelaneo
modelos  = ["CSIRO-Mk3.6.0","INM-CM4","IPSL-CM5A-LR","MIROC5","MPI-ESM-LR","MPI-ESM-MR","MRI-CGCM3"]
qn_coords  = (-33.44,360-70.68)
dias_mes = [31,28,31,30,31,30,31,31,30,31,30,31]

#%%
#Datos sstclim
pr_sstclim = []
paths_sstclim = np.array(glob.glob("cmip5/sstclim/pr/*[!_anual].nc"))
for i in range(len(paths_sstclim)):
    path = "estaciones/quintanormal_"+modelos[i]+"_sstclim_season.csv"
    if os.path.isfile(path) == False:
        file = paths_sstclim[i]
        datos = xr.open_dataset(file)["pr"].squeeze().sel(lat=qn_coords[0],lon=qn_coords[1],
                                                          method="nearest").to_series()
        datos = datos.groupby([datos.index.year,datos.index.month]).sum()
        mult  = np.array(list(map(lambda x: dias_mes[x-1],datos.index.get_level_values(1))))
        # datos = datos.resample("M").sum().to_period()
        datos = datos*mult*3600*24
        datos = datos.groupby([datos.index.get_level_values(0),
                               [i in [4,5,6,7,8,9] for i in datos.index.get_level_values(1)]]
                              ).sum()
        datos.to_csv(path)
        pr_sstclim.append(datos)
        
        # datos = datos*mult*3600*24
    #     pr_sstclim.append(datos.groupby(datos.index.year).sum())
    else:
        pr_sstclim.append(pd.read_csv(path, squeeze=True, index_col=[0,1]))
    
#%%
#datos picontrol
pr_picontrol = []
paths_picontrol = np.array(glob.glob("cmip5/piControl/pr/*[!_anual].nc"))
for i in range(len(paths_picontrol)):
    path = "estaciones/quintanormal_"+modelos[i]+"_picontrol_season.csv"
    if os.path.isfile(path) == False:
        file = paths_picontrol[i]
        datos = xr.open_dataset(file)["pr"].squeeze().sel(lat=qn_coords[0],lon=qn_coords[1],
                                                          method="nearest").to_series()
        datos = datos.groupby([datos.index.year,datos.index.month]).sum()
        mult  = np.array(list(map(lambda x: dias_mes[x-1],datos.index.get_level_values(1))))
        # datos = datos.resample("M").sum().to_period()
        datos = datos*mult*3600*24
        datos = datos.groupby([datos.index.get_level_values(0),
                               [i in [4,5,6,7,8,9] for i in datos.index.get_level_values(1)]]
                              ).sum()
        datos.to_csv(path)
        pr_picontrol.append(datos)
        
        # datos = datos*mult*3600*24
    #     pr_sstclim.append(datos.groupby(datos.index.year).sum())
    else:
        pr_picontrol.append(pd.read_csv(path, squeeze=True, index_col=[0,1]))
        
#%%

sstclim = pr_sstclim
picontrol = pr_picontrol
#%%

#plot
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color="k")

    
def bp_colors(bp, edge_color, fill_color):
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)   

plt.figure(figsize=(10,5))

bp_QN        = plt.boxplot(dataQN,positions=np.array([0]),sym='',widths=0.3,patch_artist=True)
# bp_amip      = plt.boxplot(pr_amip,positions=np.arange(1,1+len(modelos))*2-0.35, sym='',widths=0.3,patch_artist=True)
bp_sstclim   = plt.boxplot(pr_sstclim,positions=np.arange(1,1+len(modelos))*2, sym='',widths=0.3,patch_artist=True)
bp_picontrol = plt.boxplot(pr_picontrol,positions=np.arange(1,1+len(modelos))*2+0.35, sym='',widths=0.3,patch_artist=True)

bp_colors(bp_QN,"k","royalblue")
# bp_colors(bp_amip,"k","firebrick")
bp_colors(bp_sstclim,"k","limegreen")
bp_colors(bp_picontrol,"k","mediumorchid")


plt.plot([], c='royalblue', label='AWS Quinta_Normal')
# plt.plot([], c='firebrick', label='AMIP')
plt.plot([], c='limegreen', label='sstClim')
plt.plot([], c='mediumorchid', label='piControl')
plt.legend()

plt.xticks(np.arange(0,1+len(modelos))*2,["Quinta\nNormal"]+modelos,rotation=45)
# plt.xlim(-2, len(["Quinta\nNormal"]+modelos)*2)
plt.ylabel("Precipitation (mm/yr)")
plt.savefig("plots/season/pr_season_QN.pdf",dpi=150,bbox_inches="tight")
plt.show()

#%%
#plot
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color="k")

    
def bp_colors(bp, edge_color, fill_color):
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)   

plt.figure(figsize=(10,5))

bp_QN        = plt.boxplot(dataQN.unstack().iloc[:,1],positions=np.array([0]),sym='',widths=0.3,patch_artist=True)
# bp_amip      = plt.boxplot(pr_amip,positions=np.arange(1,1+len(modelos))*2-0.35, sym='',widths=0.3,patch_artist=True)
bp_sstclim   = plt.boxplot([sstclim.unstack().iloc[:,1] for sstclim in pr_sstclim],
                           positions=np.arange(1,1+len(modelos))*2, sym='',
                           widths=0.3,patch_artist=True)
bp_picontrol = plt.boxplot([picontrol.unstack().iloc[:,1] for picontrol in pr_picontrol],
                           positions=np.arange(1,1+len(modelos))*2+0.35, sym='',
                           widths=0.3,patch_artist=True)

bp_colors(bp_QN,"k","royalblue")
# bp_colors(bp_amip,"k","firebrick")
bp_colors(bp_sstclim,"k","limegreen")
bp_colors(bp_picontrol,"k","mediumorchid")


plt.plot([], c='royalblue', label='AWS Quinta_Normal')
# plt.plot([], c='firebrick', label='AMIP')
plt.plot([], c='limegreen', label='sstClim')
plt.plot([], c='mediumorchid', label='piControl')
plt.legend()

plt.xticks(np.arange(0,1+len(modelos))*2,["Quinta\nNormal"]+modelos,rotation=45)
# plt.xlim(-2, len(["Quinta\nNormal"]+modelos)*2)
plt.ylabel("Precipitation (mm/yr)")
plt.savefig("plots/season/pr_winter_QN.pdf",dpi=150,bbox_inches="tight")
plt.show()

#%%
fig,ax = plt.subplots(2,4,figsize=(12,5),sharey=True)
fig.tight_layout(pad=3.5)
fig.text(0,0.5,"Probability Density",ha="center",va="center",rotation=90)
fig.text(0.5,0,"$\sigma$_picontrol\nQuinta_Normal",ha="center",va="center")
ax = ax.ravel()
colores = mpl.cm.tab10(np.linspace(0,1,10))
alpha = 0.05
N_montecarlo = 1999
for j in range(len(modelos)):   
    std_samples = np.empty(N_montecarlo)
    tspan = pr_picontrol[j].shape[0]
    for i in range(N_montecarlo):
        random_positions = np.random.randint(0,tspan-1,30)
        std_samples[i] = np.nanstd(pr_picontrol[j].values[random_positions])
    std_sstclim = np.nanstd(pr_sstclim[j].values)
    pvalue = (np.count_nonzero(std_samples<=std_sstclim)+1)/(N_montecarlo+1)
    ax[j].hist(std_samples,bins="auto",density=True,color=colores[j],alpha=0.8)
    kdeplot(std_samples,color=colores[j],ax=ax[j],lw=2)
    label="pvalue: "+"{:.3f}".format(pvalue)
    # ax[j].axvline(std_sstclim,label=label,color="k")
    # ax[j].legend()
    if j==6:
        ax[j].axvline(np.quantile(std_samples,1-alpha/2),color="k",ls=":",label="Critical Values\n"+r"$\alpha$ = "+str(alpha))
        ax[j].axvline(np.quantile(std_samples,alpha/2),color="k",ls=":")
        ax[j].axvline(std_sstclim,label="$\sigma$_sstclim",color="k",lw=1.5)
        ax[j].set_title(modelos[j]+"\n"+label)
        ax[j].legend(loc=(1.5,0.5))
    else:
        ax[j].axvline(np.quantile(std_samples,1-alpha/2),color="k",ls=":")
        ax[j].axvline(np.quantile(std_samples,alpha/2),color="k",ls=":")
        ax[j].axvline(std_sstclim,color="k",lw=1.5)    
        ax[j].set_title(modelos[j]+"\n"+label)
    ax[j].set_ylabel("")
    ax[j].set_xlabel("")
    
ax[7].axis("off")
plt.savefig("plots/season/null_distributions.pdf",dpi=150,bbox_inches="tight")

#%%
fig,ax = plt.subplots(2,4,figsize=(12,5),sharey=True)
fig.tight_layout(pad=3.5)
fig.text(0,0.5,"Probability Density",ha="center",va="center",rotation=90)
fig.text(0.5,0,"$\sigma$_picontrol\nQuinta_Normal",ha="center",va="center")
ax = ax.ravel()
colores = mpl.cm.tab10(np.linspace(0,1,10))
alpha = 0.05
N_montecarlo = 1999
for j in range(len(modelos)):   
    std_samples = np.empty(N_montecarlo)
    tspan = [picontrol.unstack().iloc[:,0] for picontrol in pr_picontrol][j].shape[0]
    for i in range(N_montecarlo):
        random_positions = np.random.randint(0,tspan-1,30)
        std_samples[i] = np.nanstd([picontrol.unstack().iloc[:,0] for picontrol in pr_picontrol][j].values[random_positions])
    std_sstclim = np.nanstd([sstclim.unstack().iloc[:,0] for sstclim in pr_sstclim][j].values)
    pvalue = (np.count_nonzero(std_samples<=std_sstclim)+1)/(N_montecarlo+1)
    ax[j].hist(std_samples,bins="auto",density=True,color=colores[j],alpha=0.8)
    kdeplot(std_samples,color=colores[j],ax=ax[j],lw=2)
    label="pvalue: "+"{:.3f}".format(pvalue)
    # ax[j].axvline(std_sstclim,label=label,color="k")
    # ax[j].legend()
    if j==6:
        ax[j].axvline(np.quantile(std_samples,1-alpha/2),color="k",ls=":",label="Critical Values\n"+r"$\alpha$ = "+str(alpha))
        ax[j].axvline(np.quantile(std_samples,alpha/2),color="k",ls=":")
        ax[j].axvline(std_sstclim,label="$\sigma$_sstclim",color="k",lw=1.5)
        ax[j].set_title(modelos[j]+"\n"+label)
        ax[j].legend(loc=(1.5,0.5))
    else:
        ax[j].axvline(np.quantile(std_samples,1-alpha/2),color="k",ls=":")
        ax[j].axvline(np.quantile(std_samples,alpha/2),color="k",ls=":")
        ax[j].axvline(std_sstclim,color="k",lw=1.5)    
        ax[j].set_title(modelos[j]+"\n"+label)
    ax[j].set_ylabel("")
    ax[j].set_xlabel("")
    
ax[7].axis("off")
plt.savefig("plots/season/null_distributions_summer.pdf",dpi=150,bbox_inches="tight")

#%%
pr_sstclim = [s.unstack().iloc[:,0] for s in sstclim]
pr_picontrol = [s.unstack().iloc[:,0] for s in picontrol]
ecdfs = [[] for mod in modelos]
ecdfs_sstclim = [ECDF(pr_sstclim[mod]) for mod in range(len(modelos))]
wet_ex = [np.empty(N_montecarlo) for mod in modelos]
pvalue_wet = np.empty(len(modelos)) 
dry_ex = [np.empty(N_montecarlo) for mod in modelos]
pvalue_dry = np.empty(len(modelos))
for j in range(len(modelos)):
    print(modelos[j])
    tspan = pr_picontrol[j].shape[0]
    for i in range(N_montecarlo):
        random_positions = np.random.randint(0,tspan-1,30)
        sample = pr_picontrol[j].values[random_positions]
        ecdfs[j].append(ECDF(sample))
        wet_ex[j][i] = ecdfs_sstclim[j](np.quantile(sample,0.9))*100
        dry_ex[j][i] = ecdfs_sstclim[j](np.quantile(sample,0.1))*100
    pvalue_wet[j]  = ((wet_ex[j]>90).sum()+1)/(N_montecarlo+1)
    pvalue_dry[j]  = ((dry_ex[j]<10).sum()+1)/(N_montecarlo+1)


#%%
fig,ax = plt.subplots(2,4,figsize=(12,5),sharey=True)
fig.tight_layout(pad=3.5)
fig.text(0,0.5,"Probability Density",ha="center",va="center",rotation=90)
fig.text(0.5,0,"Precipitation (mm/yr)\nQuinta_Normal",ha="center",va="center")
ax = ax.ravel()
for j in range(len(modelos)):
    if j!=6:
        for i in range(N_montecarlo):
            ax[j].plot(ecdfs[j][i].x,ecdfs[j][i].y,color="grey",alpha=0.1)
        # ax[j].plot(ECDF(pr_sstclim[j]).x,ECDF(pr_sstclim[j]).y,color="limegreen",lw=2)
        kdeplot(pr_sstclim[j], cumulative=True, color="limegreen",ax=ax[j], lw=2,bw_adjust=0.8)
        kdeplot(pr_picontrol[j], cumulative=True, color="mediumorchid",ax=ax[j], lw=2,bw_adjust=0.8)
        # ax[j].plot(ECDF(pr_picontrol[j]).x,ECDF(pr_picontrol[j]).y,color="mediumorchid",lw=2)
        ax[j].set_title(modelos[j])
        ax[j].axvline(np.quantile(pr_picontrol[j],0.1),color="k",ls=":")
        ax[j].axvline(np.quantile(pr_picontrol[j],0.9),color="k",ls=":")
        ax[j].set_xlabel("")
        ax[j].set_ylabel("")
    else:
        for i in range(N_montecarlo):
            ax[j].plot(ecdfs[j][i].x,ecdfs[j][i].y,color="grey",alpha=0.1)
        # ax[j].plot(ECDF(pr_sstclim[j]).x,ECDF(pr_sstclim[j]).y,color="limegreen",lw=2,label="sstClim")
        kdeplot(pr_sstclim[j], cumulative=True, color="limegreen",ax=ax[j], label="sstClim",bw_adjust=0.8, lw=2)
        kdeplot(pr_picontrol[j], cumulative=True, color="mediumorchid",ax=ax[j], lw=2, label="picontrol",bw_adjust=0.8)
        # ax[j].plot(ECDF(pr_picontrol[j]).x,ECDF(pr_picontrol[j]).y,color="mediumorchid",lw=2,label="piControl")
        ax[j].set_title(modelos[j])
        ax[j].axvline(np.quantile(pr_picontrol[j],0.1),color="k",ls=":",label="Percentiles picontrol: P10, P90")
        ax[j].axvline(np.quantile(pr_picontrol[j],0.9),color="k",ls=":")
        ax[j].legend(loc=(1.3,0.5))
        ax[j].set_xlabel("")
        ax[j].set_ylabel("")
ax[7].axis("off")

plt.savefig("plots/season/cdf_quintanormal_summer.pdf",dpi=150,bbox_inches="tight")