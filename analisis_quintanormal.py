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
dataQN = dataQN.resample("y").sum()

#miscelaneo
modelos  = ["CSIRO-Mk3.6.0","INM-CM4","IPSL-CM5A-LR","MIROC5","MPI-ESM-LR","MPI-ESM-MR","MRI-CGCM3"]
qn_coords  = (-33.44,360-70.68)
dias_mes = [31,28,31,30,31,30,31,31,30,31,30,31]
#%%
#Datos sstclim
pr_sstclim = []
paths_sstclim = np.array(glob.glob("cmip5/sstclim/pr/*[!_anual].nc"))
for i in range(len(paths_sstclim)):
    path = "estaciones/quintanormal_"+modelos[i]+"_sstclim.csv"
    if os.path.isfile(path) == False:
        file = paths_sstclim[i]
        datos = xr.open_dataset(file)["pr"].squeeze().sel(lat=qn_coords[0],lon=qn_coords[1],method="nearest").to_series()
        mult  = np.array(list(map(lambda x: dias_mes[x-1],datos.index.month)))
        datos = datos*mult*3600*24
        pr_sstclim.append(datos.groupby(datos.index.year).sum())
    else:
        pr_sstclim.append(pd.read_csv(path,index_col=0, squeeze=True))
    
#%%
#datos picontrol
pr_picontrol = []
paths_picontrol = np.array(glob.glob("cmip5/piControl/pr/*[!_anual].nc"))
for i in range(len(paths_picontrol)):
    path = "estaciones/quintanormal_"+modelos[i]+"_picontrol.csv"
    if os.path.isfile(path) == False:
        datos = xr.open_dataset(file)["pr"].squeeze().sel(lat=qn_coords[0],lon=qn_coords[1],method="nearest").to_series()
        mult  = np.array(list(map(lambda x: dias_mes[x-1],datos.index.month)))
        datos = datos*mult*3600*24
        pr_picontrol.append(datos.groupby(datos.index.year).sum())
    else:
        pr_picontrol.append(pd.read_csv(path,index_col=0, squeeze=True))
#%%
# datos amip
# pr_amip = []
# for file in np.array(glob.glob("cmip5/amip/pr/*[!_anual].nc")):
#     path = "estaciones/quintanormal_"+modelos[i]+"_amip.csv"
#     if os.path.isfile(path) == False:
#         datos = xr.open_dataset(file)["pr"].squeeze().sel(lat=qn_coords[0],lon=qn_coords[1],method="nearest").to_series()
#         mult  = np.array(list(map(lambda x: dias_mes[x-1],datos.index.month)))
#         datos = datos*mult*3600*24
#         pr_amip.append(datos.groupby(datos.index.year).sum())
#     else:
#         pr_amip.append(pd.read_csv(path,index_col=0,squeeze=True))



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
# plt.savefig("plots/pr_anual_QN.pdf",dpi=150,bbox_inches="tight")
plt.show()

#%%
#COMPUTE STATISTICAL SIGNIFICANCE OF THE NULL HYPOTHESIS H0: STD_SSTCLIM >= STD_PICONTROL
# np.random.seed(1999)
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
plt.savefig("plots/metricas_finales/null_distributions.pdf",dpi=150,bbox_inches="tight")
#%%
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

plt.savefig("plots/metricas_finales/extremos/cdf_quintanormal.pdf",dpi=150,bbox_inches="tight")


#%%
fig,ax = plt.subplots(2,4,figsize=(12,5),sharey=True)
fig.tight_layout(pad=3.5)
fig.text(0,0.5,"Probability Density",ha="center",va="center",rotation=90)
fig.text(0.5,0,"$F_{sstclim}$(P90$_{picontrol}$) (mm/yr)\nQuinta_Normal",ha="center",va="center")
ax = ax.ravel()
for j in range(len(modelos)):
    label = modelos[j]+"\npvalue: "+str(pvalue_wet[j])
    ax[j].set_title(label)
    # ax[j].hist(wet_ex[j],bins=5,density=True,color=colores[j],alpha=0.8)
    kdeplot(wet_ex[j],ax=ax[j],color=colores[j],lw=2, bw_adjust=2)
    if j!=6:
        ax[j].axvline(np.quantile(wet_ex[j],1-alpha/2),color="k",ls=":")
        ax[j].axvline(np.quantile(wet_ex[j],alpha/2),color="k",ls=":")
        ax[j].axvline(90,color="k",lw=1.5)
        
    else:
        ax[j].axvline(np.quantile(wet_ex[j],1-alpha/2),color="k",ls=":",label="Critical Values\n"+r"$\alpha$ = "+str(alpha))
        ax[j].axvline(np.quantile(wet_ex[j],alpha/2),color="k",ls=":")
        ax[j].axvline(90,color="k",lw=1.5,label="90%")
        ax[j].legend(loc=(1.3,0.5))
    ax[j].set_xlabel("")
    ax[j].set_ylabel("")
ax[7].axis("off")

plt.savefig("plots/metricas_finales/extremos/null_drydistributions.pdf",dpi=150,bbox_inches="tight")
#%%

#%%
def test_std(ts_sstclim,ts_picontrol,N_montecarlo=999,confidence_level=0.05,
             bootstrap_sample=30):
    std_samples = np.empty(N_montecarlo)
    tspan = ts_picontrol.shape[0]
    for i in range(N_montecarlo):
        random_positions = np.random.randint(0,tspan-1,bootstrap_sample)
        std_samples[i] = np.nanstd(ts_picontrol.values[random_positions])
    std_sstclim = np.nanstd(ts_sstclim.values)
    pvalue = (np.count_nonzero(std_samples>std_sstclim)+1)/(N_montecarlo+1)
    # pvalue = ttest_1samp(std_samples,std_sstclim).pvalue
    # ecdf = ECDF(std_samples,side="left")
    # pvalue = ecdf(std_sstclim)
    return pvalue

#%%
# check pvalue sensibility to Number of montecarlo simulations
N_montecarlo = np.arange(1,5000,5)
pvalues = [np.empty(N_montecarlo.shape) for mod in modelos] 
for i in trange(len(modelos)):
    for j in trange(len(N_montecarlo)):
        pvalues[i][j] = test_std(pr_sstclim[i],pr_picontrol[i],j+1)
        
fig = plt.figure(figsize=(12,5))
ax  = fig.add_subplot()
pd.DataFrame(pvalues,index=modelos,columns=N_montecarlo).T.plot(ax=ax)
plt.legend(ncol=2)
plt.xlabel("Number of bootstrap samples",fontsize=12)
plt.ylabel("pvalue",fontsize=12)
# plt.savefig("plots/metricas_finales/montecarlo_optimumsize.pdf",dpi=150,bbox_inches="tight")

#%%
# check pvalue sensibility to bootstrap sample size
bootstrap_samples = np.arange(1,500,1)
pvalues = [np.empty(bootstrap_samples.shape) for mod in modelos] 
for i in trange(len(modelos)):
    for j in range(len(bootstrap_samples)):
        pvalues[i][j] = test_std(pr_sstclim[i],pr_picontrol[i],N_montecarlo=1999,bootstrap_sample=bootstrap_samples[j])
#%%        
fig = plt.figure(figsize=(12,5))
ax  = fig.add_subplot()
(1-pd.DataFrame(pvalues,index=modelos,columns=bootstrap_samples)).T.plot(ax=ax)
plt.legend(ncol=2)
plt.xlabel("Bootstrap sample size",fontsize=12)
plt.ylabel("pvalue",fontsize=12)
plt.axvline(30)
# plt.savefig("plots/metricas_finales/bootstrap_optimumsamplesize.pdf",dpi=150,bbox_inches="tight")

#%%
# fig,ax = plt.subplots(2,4,figsize=(12,5),sharey=True)
# fig.tight_layout(pad=2)
# fig.text(0.5,0,"Precipitacion (mm/año)",ha="center",va="center",fontsize=13)
# ax = ax.ravel()

# x = kdeplot(dataQN,ax=ax[0], cumulative=True)
# # ecdfplot(dataQN,ax=ax[0])
# ax[0].set_xlabel("")
# ax[0].set_title("Quinta Normal")
# ax[0].set_ylabel("")
# ax[0].set_xlim([0,dataQN.max()*1.1])
# # hist,bins,patches=ax[0].hist(dataQN, cumulative=True, histtype="step", bins=50, density=True)
# # kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(dataQN.values.reshape(1,-1))
# for i in range(0,7):
#     # kdeplot(pr_amip[i], cumulative=True, color="firebrick", ax=ax[i+1], label="AMIP",bw_adjust=0.8)
#     kdeplot(pr_sstclim[i], cumulative=True, color="limegreen",ax=ax[i+1], label="sstClim",bw_adjust=0.8)
#     kdeplot(pr_picontrol[i], cumulative=True, color="mediumorchid",ax=ax[i+1], label="piControl",bw_adjust=0.8)
#     # ecdfplot(pr_picontrol[i],ax=ax[i+1],color="mediumorchid",label="piControl")
#     # ecdfplot(pr_sstclim[i],ax=ax[i+1],color="limegreen",label="sstClim")
#     # ecdfplot(pr_amip[i],ax=ax[i+1],color="firebrick",label="AMIP")
#     ax[i+1].set_xlabel("")
#     ax[i+1].set_ylabel("")
#     ax[i+1].set_title(modelos[i])
#     # ax[i+1].set_xlim([0,np.max((pr_amip[i].max(),pr_sstclim[i].max(),pr_picontrol[i].max()))*1.1])
#     # ax[i+1].hist(pr_picontrol[i], cumulative=True, histtype="step", bins=50, density=True, color="mediumorchid")
# ax[1].legend()
# # # plt.savefig("plots/distribuciones_ecdf_pranualQN.pdf",dpi=150,bbox_inches="tight")

#%%
# from seaborn import kdeplot,ecdfplot
# fig,ax = plt.subplots(2,4,figsize=(12,5),sharey=True)
# fig.tight_layout(pad=2)
# fig.text(0.5,0,"Precipitacion (mm/año)",ha="center",va="center",fontsize=13)
# ax = ax.ravel()

# kdeplot(dataQN,ax=ax[0], cumulative=False,lw=2,bw_adjust=1)
# # ax[0].hist(dataQN,density=True,color="tab:blue",bins=10,alpha=0.5)
# # ecdfplot(dataQN,ax=ax[0])
# ax[0].set_xlabel("")
# ax[0].set_title("Quinta Normal")
# ax[0].set_ylabel("")
# ax[0].set_xlim([0,dataQN.max()*1.1])
# # hist,bins,patches=ax[0].hist(dataQN, cumulative=True, histtype="step", bins=50, density=True)
# # kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(dataQN.values.reshape(1,-1))
# for i in range(0,7):
#     # kdeplot(pr_amip[i], cumulative=False, color="firebrick", ax=ax[i+1], label="AMIP",bw_adjust=1)
#     kdeplot(pr_sstclim[i], cumulative=False, color="limegreen",ax=ax[i+1], label="sstClim",bw_adjust=1)
#     kdeplot(pr_picontrol[i], cumulative=False, color="mediumorchid",ax=ax[i+1], label="piControl",bw_adjust=1)
#     # ax[i+1].hist(pr_amip[i],color="firebrick",density=True,alpha=0.5,bins=5)
#     # ax[i+1].hist(pr_sstclim[i],color="limegreen",density=True,alpha=0.5,bins=5)
#     # ax[i+1].hist(pr_picontrol[i],color="mediumorchid",density=True,alpha=0.5,bins=5)
#     # ecdfplot(pr_picontrol[i],ax=ax[i+1],color="mediumorchid",label="piControl")
#     # ecdfplot(pr_sstclim[i],ax=ax[i+1],color="limegreen",label="sstClim")
#     # ecdfplot(pr_amip[i],ax=ax[i+1],color="firebrick",label="AMIP")
#     ax[i+1].set_xlabel("")
#     ax[i+1].set_ylabel("")
#     ax[i+1].set_title(modelos[i])
#     # ax[i+1].set_xlim([0,np.max((pr_amip[i].max(),pr_sstclim[i].max(),pr_picontrol[i].max()))*1.1])
#     # ax[i+1].hist(pr_picontrol[i], cumulative=True, histtype="step", bins=50, density=True, color="mediumorchid")
# ax[1].legend()
# plt.savefig("plots/distribuciones_histpdf_pranualQN.pdf",dpi=150,bbox_inches="tight")
