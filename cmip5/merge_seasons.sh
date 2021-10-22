#!bin/bash
#Merge winter and summer netcdfs into a single one with 2 variables

folders="sstclim piControl"
var="pr psl ts"
for folder in $folders;do
    for v in $var; do
	    for file in $folder/$v/regrid/*[!_anual][!_winter][!_summer][!_wmean][!_smean][!_season][!_mean][!_mmean][!_manomaly][!_ymean][!_yanomaly].nc; do
	        file_name=${file%.nc}
	        echo Folder: $folder
	        echo Variable: $v
	        echo File: $file_name
            rm -rf ${file_name}_season.nc
            rm -rf ${file_name}_season_winter.nc
            rm -rf ${file_name}_season_summer.nc
            cdo merge -chname,$v,${v}_summer ${file_name}_summer.nc -chname,$v,${v}_winter ${file_name}_winter.nc ${file_name}_season.nc
        done
	done
done

exit
