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

def invnorm_3d(data,score):
    sh = data.shape
    result = np.empty((sh[1],sh[2]))
    for i in range(sh[1]):
        for j in range(sh[2]):
            result[i,j] = st.norm.cdf(score[i,j],*st.norm.fit(data[:,i,j]))
    return result

def testnorm_3d(data,method="ks"):
    sh = data.shape
    result = np.empty((sh[1],sh[2]))
    for i in range(sh[1]):
        for j in range(sh[2]):
            if method=="ks":
                result[i,j] = st.kstest(data[:,i,j],"norm")[1]
            if method=="shapiro":
                result[i,j] = st.shapiro(data[:,i,j])[1]    
            if method=="anderson":
                result[i,j] = st.anderson(data[:,i,j],dist="norm")
    return result

#%%
#Load data
models = ["CSIRO-Mk3-6-0","INMCM4","IPSL-CM5A-LR","MIROC5","MPI-ESM-LR","MPI-ESM-MR","MRI-CGCM3"]
paths_sstclim   = glob("cmip5/sstclim/pr/*_anual.nc")   #Path to SSTCLIM files
paths_picontrol = glob("cmip5/piControl/pr/*_anual.nc") #Path to PICONTROL files

#Create data bag
data = {key:{model:None for model in models} for key in ["sstclim","picontrol"]}
for mod in range(len(models)):
    data["sstclim"][models[mod]]   = xr.open_dataset(paths_sstclim[mod])["pr"].squeeze().dropna(dim="year")
    data["picontrol"][models[mod]] = xr.open_dataset(paths_picontrol[mod])["pr"].squeeze().dropna(dim="year")

#%%
#Get back boxplots
# qn = (-33,360-71)
# qn_sstclim   = [data["sstclim"][mod].sel(lat=qn[0],lon=qn[1],method="nearest").to_series() for mod in models]
# qn_picontrol = [data["picontrol"][mod].sel(lat=qn[0],lon=qn[1],method="nearest").to_series() for mod in models]

# plt.figure(figsize=(10,3))
# plt.boxplot(qn_sstclim,positions=np.arange(0,len(models))*2-0.3,widths=0.3,sym='')
# plt.boxplot(qn_picontrol,positions=np.arange(0,len(models))*2+0.3,widths=0.3,sym='')

# plt.xticks(np.arange(0,len(models))*2,models)
# plt.show()

#%%
#Attempt with different metrics
#Metric1: Normalized difference of sstclim-picontrol percentiles at X%
#Metric2: Ratio of sstclim/percentiles percentiles at X%
#Metric3: Ratio of sstclim/percentiles standard deviations
#Metric5: Ratio of empirical cdf-1_sstclim(PERCENILE_PICONTROL(X%))/X%
#Metric4: Normalized difference of empirical cdf-1_sstclim(PERCENILE_PICONTROL(X%)) and X%
#Metric6: Ratio of normal cdf-1_sstclim(PERCENILE_PICONTROL(X%))/X%
#Metric7: Normalized difference of normal cdf-1_sstclim(PERCENILE_PICONTROL(X%)) and X%
#Metric8: Normalized difference of standard deviations
#Metric9: Ratio of empirical cdf-1_picontrol(PERCENILE_sstclim(X%))/X%
#Metric10: Normalized difference of empirical cdf-1_picontrol(PERCENILE_sstclim(X%))/X%
#Compute percentile of picontrol and sstclim
target_freq = 95
picontrol_percentile = {model:None for model in models}
sstclim_percentile   = {model:None for model in models}
metric1              = {model:None for model in models}
metric2              = {model:None for model in models}
metric3              = {model:None for model in models}
metric4              = {model:None for model in models}
metric5              = {model:None for model in models}
metric6              = {model:None for model in models}
metric7              = {model:None for model in models}
metric8              = {model:None for model in models}
metric9              = {model:None for model in models}
metric10             = {model:None for model in models}
for mod in models:
    print(mod)
    picontrol_percentile[mod] = np.percentile(data["picontrol"][mod],target_freq,axis=0)
    sstclim_percentile[mod]   = np.percentile(data["sstclim"][mod],target_freq,axis=0)
    metric1[mod] = (-picontrol_percentile[mod]+sstclim_percentile[mod])/(picontrol_percentile[mod]+sstclim_percentile[mod])
    metric2[mod] = sstclim_percentile[mod]/picontrol_percentile[mod]
    metric3[mod] = data["sstclim"][mod].values.std(axis=0)/data["picontrol"][mod].values.std(axis=0)
    metric4[mod] = percentileofscore_3d(data["sstclim"][mod],picontrol_percentile[mod])
    metric5[mod] = metric4[mod]/target_freq
    metric4[mod] = (-target_freq+metric4[mod])/(metric4[mod]+target_freq)
    metric6[mod] = invnorm_3d(data["sstclim"][mod],picontrol_percentile[mod])*100
    metric7[mod] = (metric6[mod]-target_freq)/(metric6[mod]+target_freq)
    metric6[mod] = metric6[mod]/target_freq
    metric8[mod] = (data["sstclim"][mod].values.std(axis=0)-data["picontrol"][mod].values.std(axis=0))/(data["sstclim"][mod].values.std(axis=0)+data["picontrol"][mod].values.std(axis=0))
    metric9[mod] = percentileofscore_3d(data["picontrol"][mod],sstclim_percentile[mod])
    metric10[mod] = (metric9[mod]-target_freq)/(metric9[mod]+target_freq)
    metric9[mod] = metric9[mod]/target_freq

#%%
metricnames = {"metric2":"metric1","metric3":"metric3","metric5":"metric5","metric6":"metric7","metric1":"metric2","metric8":"metric4","metric4":"metric6","metric7":"metric8","metric9":"metric9","metric10":"metric10"}
metricas = {model:[] for model in models}
for mod in models:
    for m in ["metric"+str(i) for i in range(1,11)]:
        metric = eval(m)[mod]
        metric = xr.DataArray(data=metric,
                              coords=dict(lon=(["lon"],data["sstclim"][mod].lon.values),
                                          lat=(["lat"],data["sstclim"][mod].lat.values)),
                              dims=["lat","lon"],
                              name=metricnames[m])
        metricas[mod].append(metric)
        metric.to_netcdf("outputs/metricas_pr/"+metricnames[m]+"_"+mod+"_P"+str(target_freq)+".nc")
#%%
valores = []
for mod in models:
    m=metricas[mod]
    for x in m:
        y=x.sel(lat=-33.44,lon=289.32,method="nearest").item(),x.name,mod
        valores.append(y)
valores = pd.DataFrame(np.stack(valores)).pivot(values=0,index=2,columns=1).applymap(lambda x: np.round(float(x),4))
valores.to_excel("outputs/metricas_quintanormal_P"+str(target_freq)+".xlsx")
#%%
metrics = ["metric"+str(i) for i in range(1,11)]
#Distribution of index by model
fig,ax = plt.subplots(2,1,figsize=(10,6),num=0)
ax = ax.ravel()
colors = plt.cm.tab10(np.linspace(0,1,7))
for i in range(len(models)):
    mod = models[i]
    box_data = [eval(mx)[mod].ravel()[~np.isnan(eval(mx)[mod].ravel())] for mx in ["metric2","metric3","metric5","metric6","metric9"]]
    bp = ax[0].boxplot(box_data,sym="",positions=np.arange(1,6)*10-3+i,widths=0.6,patch_artist=True)
    plt.setp(bp["medians"],color="k",linewidth=1.3)
    for patch in bp['boxes']:
        patch.set(facecolor=colors[i])   
    ax[0].plot([],[],color=colors[i],label=models[i])
        
    box_data = [eval(mx)[mod].ravel()[~np.isnan(eval(mx)[mod].ravel())] for mx in ["metric1","metric8","metric4","metric7","metric10"]]
    bp = ax[1].boxplot(box_data,sym="",positions=np.arange(1,6)*10-3+i,widths=0.6,patch_artist=True)
    plt.setp(bp["medians"],color="k",linewidth=1.3)
    for patch in bp['boxes']:
        patch.set(facecolor=colors[i])   
ax[0].legend(ncol=3)
ax[0].set_xticks([10,20,30,40,50])
ax[0].set_xticklabels(["Metric1:\nRatioPercentiles","Metric3:\nRatioStd","Metric5:\nRatioEmpCDF","Metric7:\nRatioNormCDF","Metric9:\nRatioEmpCDF_2"])

ax[1].set_xticks([10,20,30,40,50])
ax[1].set_xticklabels(["Metric2:\nNDPercentiles","Metric4:\nNDStd","Metric6:\nNDEmpCDF","Metric8:\nNDNormCDF","Metric10:\nNDEmpCDF_2"])
fig.text(0.5,.9,"Target Frequency: "+str(target_freq)+"%",fontsize=15,ha="center",va="center")
# plt.savefig("plots/distribution_P"+str(target_freq)+".pdf",dpi=200,bbox_inches="tight")

#%%
var = "metric10"
#Map of metrics###
fig,ax = plt.subplots(2,4,subplot_kw={"projection":ccrs.Robinson()},num=1,figsize=(16,4))
ax = ax.ravel()
norm = mpl.colors.Normalize(-0.1,0.1)
# norm=MidpointNormalize(midpoint=1,vmin=0,vmax=2)
# im   = mpl.cm.ScalarMappable(norm=norm,cmap="BrBG")
for i in range(len(ax)):
    if i<len(ax)-1:
        ax[i].coastlines()
        ax[i].set_global()
        lat,lon = data["sstclim"][models[i]].lat.values,data["sstclim"][models[i]].lon.values
        mapa = eval(var)[models[i]]
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap="BrBG",norm=norm)

        # ax2 = fig.add_axes([ax[i].get_position().xmin-0.035,ax[i].get_position().ymax,0.05,0.1])
        # ax2.hist(mapa.ravel(),density=True,bins=20,edgecolor="k")
        # ax2.set_yticklabels([])
        # ax2.set_xticks([-0.3,0,0.3])
        # ax2.set_xlim([-0.5,0.5])
        # ax2.spines["top"].set_visible(False)
        # ax2.spines["right"].set_visible(False)
        # # ax2.spines["bottom"].set_visible(False)
        # ax2.spines["left"].set_visible(False)
        # ax2.axes.get_yaxis().set_visible(False)
        # ax2.tick_params(axis='both', which='both', labelsize=6)

        ax[i].set_title(models[i])
    else:
        ax[i].axis("off")
        #multi model average
        pass


cax = fig.add_axes([0.93,0.1,0.01,0.8])
cb = fig.colorbar(cf,cax=cax)

# fig.show()
# plt.savefig("plots/"+var+"_P"+str(target_freq)+".pdf",dpi=150,bbox_inches="tight")

#%%

# kstest       = {mod: None for mod in models}
shapirotest  = {mod: None for mod in models}


for mod in models:
    print(mod)
    # kstest[mod]      = testnorm_3d(data["picontrol"][mod],method="ks")
    shapirotest[mod] = testnorm_3d(data["picontrol"][mod],method="shapiro")

#%%

fig,ax = plt.subplots(2,4,subplot_kw={"projection":ccrs.Robinson()},num=1,figsize=(16,4))
ax = ax.ravel()
norm = mpl.colors.Normalize(0,1)
# norm=MidpointNormalize(midpoint=1,vmin=0,vmax=2)
# im   = mpl.cm.ScalarMappable(norm=norm,cmap="BrBG")
for i in range(len(ax)):
    if i<len(ax)-1:
        ax[i].coastlines()
        ax[i].set_global()
        lat,lon = data["sstclim"][models[i]].lat.values,data["sstclim"][models[i]].lon.values
        mapa = kstest[models[i]]>0.05
        mapa,lon = add_cyclic_point(mapa,lon,axis=1)
        cf = ax[i].pcolormesh(lon,lat,mapa,#levels=np.arange(-1,1.1,0.1),
                              transform=ccrs.PlateCarree(), cmap="Blues_r",norm=norm)

        # cf = ax[i].contourf(lon,lat,mapa,levels=np.arange(0,1.1,0.1),
        #                     transform=ccrs.PlateCarree(), cmap="Blues_r",norm=norm)
                
        ax[i].set_title(models[i])
    else:
        ax[i].axis("off")
        #multi model average
        pass


cax = fig.add_axes([0.93,0.1,0.01,0.8])
cb = fig.colorbar(cf,cax=cax)

# plt.savefig("plots/normality_ks_picontrol.pdf",dpi=150,bbox_inches="tight")


















