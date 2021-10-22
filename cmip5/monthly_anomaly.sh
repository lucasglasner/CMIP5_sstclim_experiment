#!bin/bash
#

folders="sstclim piControl"
var="pr psl ts"
for folder in $folders;do
    for v in $var; do
	    for file in $folder/$v/regrid/*[!_anual][!_winter][!_summer][!_wmean][!_smean][!_season][!_mean][!_mmean][!_manomaly][!_ymean][!_yanomaly].nc; do
	        file_name=${file%.nc}
	        echo Folder: $folder
	        echo Variable: $v
	        echo File: $file_name
	        rm -rf ${file_name}_mmean.nc
	        rm -rf ${file_name}_manomaly.nc
	        cdo ymonmean ${file_name}.nc ${file_name}_mmean.nc
	        cdo ymonsub ${file_name}.nc ${file_name}_mmean.nc ${file_name}_manomaly.nc
        done
	done
done
