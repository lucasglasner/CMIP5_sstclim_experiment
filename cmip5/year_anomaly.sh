#!bin/bash
#

folders="sstclim"
var="ts psl"
for folder in $folders;do
    for v in $var; do
	    for file in $folder/$v/regrid/*[!_anual][!_winter][!_summer][!_season][!_mmean].nc; do
	        file_name=${file%.nc}
	        echo Folder: $folder
	        echo Variable: $v
	        echo File: $file_name
	        rm -rf ${file_name}_mean.nc
	        rm -rf ${file_name}_yanomaly.nc

            #cdo timmean -chname,year,time ${file_name}_anual.nc ${file_name}_mean.nc
            python ymean.py $v ${file_name}_anual.nc ${file_name}_ymean.nc
            cdo sub -chname,year,time ${file_name}_anual.nc ${file_name}_ymean.nc ${file_name}_yanomaly.nc

        done
	done
done

exit
