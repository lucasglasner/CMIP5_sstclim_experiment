import xarray as xr
import numpy as np
import glob
import os

#Misc
paths_sstclim   = glob.glob("cmip5/sstclim/pr/*[!_anual].nc")
paths_picontrol = glob.glob("cmip5/piControl/pr/*[!_anual].nc")
paths_amip      = glob.glob("cmip5/amip/pr/*[!_anual].nc")
#Loop over models
dias_mes = [31,28,31,30,31,30,31,31,30,31,30,31]
for i in range(len(paths_sstclim)):
		for exp in ["sstclim","amip","picontrol"]:
			print("Model: "+paths_sstclim[i].split("_")[2]+" ; Experiment: "+exp.upper())
			os.system("rm -rf "+eval("paths_"+exp)[i][:-3]+"_anual.nc")
			precip = xr.open_dataset(eval("paths_"+exp)[i],chunks={"time":12})
			attrs   = precip.attrs
			attrs["units"] = "mm/yr"
			attrs["standard_name"] = "precipitation flux"
			attrs["long_name"] = "Precipitation"
			precip = precip.pr.squeeze()
			mult2   = np.array(list(map(lambda x: dias_mes[x-1],precip.time.to_series().index.month)))
			mult    = np.ones(precip.shape)
			for j in range(mult.shape[0]):
				mult[j,:,:] = mult[j,:,:]*mult2[j]
			precip = precip*mult*3600*24
			# precip = precip.resample(time="Y").sum()
			precip = precip.groupby("time.year").sum().load()
			precip.attrs = attrs
			precip.to_netcdf(eval("paths_"+exp)[i][:-3]+"_anual.nc")
			del mult2,mult,precip,attrs
