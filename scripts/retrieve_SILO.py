import glob
import os
import argparse

from netCDF4 import Dataset
import rasterio
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

import requests


def getCmdargs():
    p = argparse.ArgumentParser()
    p.add_argument("-wd", "--work_dir", type=str, default="/scratch/rsc8/yongjingm/SILO",
                   help="Work directory")
    cmdargs = p.parse_args()
    return cmdargs

def drr_to_tif(in_dir, varname, dst_dir):
    """
    Convert drr files to tif file
    """
    if varname == 'temperature':
        fileList = glob.glob(os.path.join(in_dir, '*.drr'))
    else:
        fileList = glob.glob(os.path.join(in_dir, '**', '*.drr'))

    for file in fileList:
        basename = os.path.basename(file)
        if varname == 'temperature':
            year_month = int(basename.split('.')[0].split('_')[-1][:6])
        else:
            year_month = int(basename.split('.')[0])
        out_name = os.path.join(dst_dir, '{}_{}.tif'.format(varname, year_month))
            
        if (year_month) > 198608 and not (os.path.exists(out_name)):
            cmd = 'gdal_translate -of GTiff {} {}'.format(file, out_name)
            os.system(cmd)

def merge_tif(in_dir, varname, out_dir):
    """
    Merge the montly data into the same tif file
    """
    #Read montly data
    fileList = sorted(glob.glob(os.path.join(in_dir, '{}_*.tif'.format(varname))))

    count = 0
    dates = []
    for filename in fileList:
        count += 1
        date = datetime.datetime.strptime(os.path.basename(filename).split('_')[1][0:6], '%Y%m')
        dates.append(date)

        with rasterio.open(filename) as src:
            if varname == 'temperature':
                out_image = np.expand_dims(src.read(3), 0)
            else:
                out_image = np.expand_dims(src.read(1), 0)

            # Concat images
            if count == 1:
                out_images = out_image
            else:
                out_images = np.concatenate((out_images, out_image), axis=0)
        os.remove(filename)
        
    out_meta = src.meta 
    profile = out_meta.copy()
    profile['count'] = out_images.shape[0]
    if varname == 'temperature':
        profile['nodata'] = 0
    else:
        profile['nodata'] = -1
    # Save image data
    print('Save images')
    with rasterio.Env():    
        with rasterio.open(os.path.join(out_dir, '{}.tif'.format(varname)), 'w', **profile) as dst:
            dst.write(out_images)

    # Save metadata
    print('Save metadata')
    out_df = pd.DataFrame(
        {'Date':dates,
        'Temperature':np.nanmean(np.where(out_images==profile['nodata'], np.nan, out_images), (1,2))}
    )
    out_df.to_csv(os.path.join(out_dir, '{}.csv'.format(varname)), index=False)
        
        
def mainRoutine():
    
    cmdargs = getCmdargs()
    
    # Work directory where data will be saved
    work_dir = cmdargs.work_dir
    in_dirs = {'temperature': '/apollo/qccce/dres/ag/dres/graspOutputs/climate/src3/',
               'rainfall': '/sdata/metfiles/monthlyRainfall/'}
    
    for var in in_dirs.keys():       
        # Retrieve data and convert to tif
        print('Converting {}'.format(var))
        drr_to_tif(in_dirs[var], var, work_dir)
        print('Merging {}'.format(var))
        merge_tif(work_dir, var, work_dir)
        
if __name__ == "__main__":
    mainRoutine()
        