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
    p.add_argument("-wd", "--work_dir", type=str, default="/scratch/rsc8/yongjingm/WaterOutlook",
                   help="Work directory")
    cmdargs = p.parse_args()
    return cmdargs


def save_file_from_url(url, save_path):
    """
    Download data from url
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print("File saved successfully at {}".format(save_path))
    else:
        print("Failed to download the file")
        
def netCDF_to_tif(netCDF_dir, varname, meta_dir, tif_dir):
    """
    Save AWO data from netcdf to tif
    """
    # Filter range of dates
    epoch_date = datetime.datetime.strptime('1900-01-01', "%Y-%m-%d")
    s_date = datetime.datetime.strptime('1987-01-01', "%Y-%m-%d")
    s_day = (s_date - epoch_date).days
    
    data = Dataset(netCDF_dir, "r", format="NETCDF4")
    time = data.variables['time'][:]
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    
    # Retrieve netCDF data
    time_idx = np.where(time>s_day)
    var_data = data.variables[varname][:][time_idx, :, :].data
    var_data = np.squeeze(var_data, 0)
    
    # Save metadata
    Dates = pd.date_range(start=s_date, periods=var_data.shape[0], freq='MS')
    Result = np.nanmean(np.where(var_data==-999, np.nan, var_data), (1,2))
    df = pd.DataFrame({'Date':Dates, 'Soil Moisture': Result})
    df.to_csv(meta_dir, index=False)
    
    # Save tif data
    transform = rasterio.transform.from_bounds(
        min(lon)-0.025, min(lat)-0.025, max(lon)+0.025, max(lat)+0.025, len(lon), len(lat)
    )
    
    profile = {'driver':'GTiff',
           'width':var_data.shape[2],
           'height':var_data.shape[1],
           'count':var_data.shape[0],
           'dtype':'float64',
           'crs':'epsg:4326', 
           'transform':transform,
           'nodata':-999}
    
    with rasterio.Env():
        with rasterio.open(tif_dir, 'w', **profile) as dst:
            dst.write(var_data)
        
        
def mainRoutine():
    
    cmdargs = getCmdargs()
    
    # Work directory where data will be saved
    work_dir = cmdargs.work_dir
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    
    # Download link for runoff data
    runoff_link = ("https://dapds00.nci.org.au/"
                   "thredds/fileServer/iu04/australian-water-outlook/"
                   "historical/v1/AWRALv7/processed/values/month/qtot.nc")
    
    # Download link for soil moisture data
    sm_link = ("https://dapds00.nci.org.au/"
               "thredds/fileServer/iu04/australian-water-outlook/"
               "historical/v1/AWRALv7/processed/values/month/sm_pct.nc")    

    varnames = {'soilmoisture': 'sm_pct',
                'runoff': 'qtot'}
    links = {'soilmoisture': sm_link,
             'runoff': runoff_link}
    
    for var in varnames.keys():
        
        # Target saving dir
        netcdf_path = os.path.join(work_dir, "{}.nc".format(var))
        
        # Save data
        print('Download {}'.format(var))
        save_file_from_url(links[var], netcdf_path)

        # Convert netCDF data to tif data
        print('Convert {}'.format(var))
        meta_dir = os.path.join(work_dir, "{}.csv".format(var))
        tif_dir = os.path.join(work_dir, "{}.tif".format(var))  
        netCDF_to_tif(netcdf_path, varnames[var], meta_dir, tif_dir)

if __name__ == "__main__":
    mainRoutine()
        