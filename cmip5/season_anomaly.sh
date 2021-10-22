folders="sstclim"
var="pr ts psl"

for folder in $folders;do
    for v in $var; do
	    for file in $folder/$v/regrid/*[!_anual][!_winter][!_summer][!_season][!_mean][!_mmean][!_manomaly][!_yanomaly][!_ymean][!_smean][!_wmean].nc; do
	        file_name=${file%.nc}
	        echo Folder: $folder
	        echo Variable: $v
	        echo File: $file_name
	        rm -rf ${file_name}_mmean.nc
	        rm -rf ${file_name}_manomaly.nc
	        rm -rf ${file_name}_wmean.nc
	        rm -rf {$file_name}_wmean.nc
	        python ymean.py ${v}_winter ${file_name}_winter.nc ${file_name}_wmean.nc
            cdo sub -chname,year,time ${file_name}_winter.nc ${file_name}_wmean.nc ${file_name}_wanomaly.nc
        
            python ymean.py ${v}_summer ${file_name}_summer.nc ${file_name}_smean.nc
            cdo sub -chname,year,time ${file_name}_summer.nc ${file_name}_smean.nc ${file_name}_sanomaly.nc
        done
	done
done

for folder in $folders;do
    for v in $var; do
        path=~/Documents/proyectos/sstclim/cmip5/$folder/$v/regrid/
        echo $path
        cd $path
        mv *_wanomaly.nc anomaly/
        mv *_sanomaly.nc anomaly/
    done
done
