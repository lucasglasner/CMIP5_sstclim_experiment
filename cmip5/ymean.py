#!bin/python


import xarray as xr
import sys

file_path = sys.argv[2]
variable  = sys.argv[1]
out_path  = sys.argv[3]
print("acaaaaaaaaaaaaaaaaa     "+file_path)
print("opening data")
data = xr.open_dataset(file_path)[variable].squeeze()
print("saving data")
data.mean(dim="time").squeeze().to_netcdf(out_path)
