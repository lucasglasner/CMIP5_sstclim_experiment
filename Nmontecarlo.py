#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 19:31:28 2021

@author: lucas
"""

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
 

#%%
#Load data
models = ["CSIRO-Mk3-6-0","INMCM4","IPSL-CM5A-LR","MIROC5","MPI-ESM-LR","MPI-ESM-MR","MRI-CGCM3"]
paths_picontrol = glob("cmip5/piControl/pr/regrid/*_anual.nc") #Path to PICONTROL files

#Create data bag
data = {model:None for model in models}
for mod in range(len(models)):
    data[models[mod]] = xr.open_dataset(paths_picontrol[mod])["pr"].squeeze().dropna(dim="year")
        
#%%