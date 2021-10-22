#!/bin/bash
#Loop over each monthly netcdf and join them in a new netcdf with two seasons

folders="sstclim piControl"
for folder in $folders;do
	for file in $folder/pr/regrid/*[!_anual][!_winter].nc; do
		echo var: pr
		echo Folder: $folder
		echo File: $file
		cdo selmon,4/9 $file $folder/pr/regrid/tmp.nc
		cdo selmon,10,11,12,1,2,3 $file $folder/pr/regrid/tmp_2.nc
		cdo yearsum $folder/pr/regrid/tmp.nc ${file%.nc}_winter.nc
		cdo yearsum $folder/pr/regrid/tmp_2.nc ${file%.nc}_summer.nc
		rm -rf $folder/pr/regrid/tmp.nc $folder/pr/regrid/tmp_2.nc
	done

	for file in $folder/ts/regrid/*[!_anual][!_winter].nc; do
		echo var: ts
		echo Folder: $folder
		echo File: $file
		cdo selmon,4/9 $file $folder/ts/regrid/tmp.nc
		cdo selmon,10,11,12,1,2,3 $file $folder/ts/regrid/tmp_2.nc
		cdo yearmean $folder/ts/regrid/tmp.nc ${file%.nc}_winter.nc
		cdo yearmean $folder/ts/regrid/tmp_2.nc ${file%.nc}_summer.nc
		rm -rf $folder/ts/regrid/tmp.nc $folder/ts/regrid/tmp_2.nc
	done
	
	for file in $folder/psl/regrid/*[!_anual][!_winter].nc; do
	    echo var: psl
		echo Folder: $folder
		echo File: $file
		cdo selmon,4/9 $file $folder/psl/regrid/tmp.nc
		cdo selmon,10,11,12,1,2,3 $file $folder/psl/regrid/tmp_2.nc
		cdo yearmean $folder/psl/regrid/tmp.nc ${file%.nc}_winter.nc
		cdo yearmean $folder/psl/regrid/tmp_2.nc ${file%.nc}_summer.nc
		rm -rf $folder/psl/regrid/tmp.nc $folder/psl/regrid/tmp_2.nc
	done
done

exit

