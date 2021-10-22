#!bin/bash
#Merge winter and summer netcdfs into a single one with 2 variables

folders="sstclim piControl"
var="pr psl ts"
for folder in $folders;do
    for v in $var; do
	    for file in $folder/$v/regrid/*[!_anual][!_winter][!_summer][!_season][!_mmean][!_manomaly][!_mean].nc; do
	        file_name=${file%.nc}
	        echo Folder: $folder
	        echo Variable: $v
	        echo File: $file_name
	        cdo select,name=${v}_winter ${file_name}_season.nc ${file_name}_winter.nc
	        cdo select,name=${v}_summer ${file_name}_season.nc ${file_name}_summer.nc
        done
	done
done

exit
