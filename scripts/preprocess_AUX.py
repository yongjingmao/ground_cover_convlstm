import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from fiona.crs import from_epsg
from skimage.transform import resize_local_mean
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import cascaded_union

import os
import argparse
import sys
from datetime import datetime

def getCmdargs():
    p = argparse.ArgumentParser()
    p.add_argument("-wd", "--work_dir", type=str, default="/scratch/rsc8/yongjingm/Auxiliary data",
                   help="Work directory")
    p.add_argument("-SILO", "--SILO_dir", type=str, default="/scratch/rsc8/yongjingm/SILO",
               help="SILO directory")
    p.add_argument("-AWO", "--AWO_dir", type=str, default="/scratch/rsc8/yongjingm/WaterOutlook",
               help="AWO directory")
    p.add_argument("-ROI", "--ROI_dir", type=str,
                   default="/scratch/rsc8/yongjingm/Github/ground_cover_convlstm/Data/Shapefiles/GBRCA_grazing.zip",
                   help="ROI directory")
    cmdargs = p.parse_args()
    return cmdargs

def img_resample(img_path, resolution, aoi, crs, buffer=1, save=False, dst_path=None, 
                 method=Resampling.bilinear):
    """
    Image reproject and resampling
    """
    with rasterio.open(img_path) as src:

        # Crop to the AOI
        aoi = cascaded_union(aoi.to_crs(crs).envelope.to_crs(src.crs)).buffer(buffer)
        #bounds  = box(*aoi.bounds)
        out_image, out_transform = mask(src, [aoi], crop=True)
        out_meta = src.meta.copy()

    # Update the metadata with the new shape, transform and CRS
    out_meta.update({
        'driver': 'GTiff',
        'height': out_image.shape[1],
        'width': out_image.shape[2],
        'transform': out_transform,
        'nodata': src.nodata
    })

    # Calculate the bounds of the cropped image
    left, bottom, right, top = rasterio.transform.array_bounds(
        out_meta['height'], out_meta['width'], out_transform
    )

    # Resample the cropped image to the desired resolution
    # Calculate the new shape and transform
    transform, width, height = calculate_default_transform(
        out_meta['crs'], crs, out_meta['width'], out_meta['height'],
        left, bottom, right, top,
        resolution=resolution)

    # Create a new numpy array for the resampled data
    resampled_image = np.empty((out_meta['count'], height, width))

    reproject(
        source=out_image,
        destination=resampled_image,
        src_transform=out_transform,
        src_crs=out_meta['crs'],
        src_nodata=out_meta['nodata'],
        dst_transform=transform,
        dst_crs=crs,
        dst_nodata=out_meta['nodata'],
        resampling=Resampling.bilinear
    )

    # Update the metadata
    out_meta.update({
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'transform': transform,
        'crs': crs
    })

    if save and (dst_path is not None):
        with rasterio.open(dst_path, 'w', **out_meta) as dst:
            dst.write(resampled_image.astype(out_meta['dtype']))
 
    return resampled_image, transform, out_meta

def mainRoutine():
    
    cmdargs = getCmdargs()
    
    # Work directory where data will be saved
    work_dir = cmdargs.work_dir
    SILO_dir = cmdargs.SILO_dir
    AWO_dir = cmdargs.AWO_dir
    ROI_dir = cmdargs.ROI_dir
    
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    
    aux_paths = {'rainfall': SILO_dir,
                 'temperature': SILO_dir,
                 'soilmoisture': AWO_dir,
                 'runoff': AWO_dir}
    
    aux_files = {'rainfall':'rainfall',
                 'temperature':'temperature',
                 'soilmoisture': 'soilmoisture',
                 'runoff': 'runoff'}

    # Train_end and Train_start define the window of training
    # They were used to constrain the max and min values in a fixed range
    # and keep the consistency of the normalization
    train_ends = {'rainfall':'2022-10-01',
                  'temperature':'2022-10-01',
                  'runoff':'2022-09-01',
                  'soilmoisture':'2022-09-01'}

    train_starts = {'rainfall':'1987-01-01',
                    'temperature':'1987-01-01',
                    'runoff':'1987-01-01',
                    'soilmoisture':'1987-01-01'}
    
    ROI = gpd.read_file(ROI_dir)
    crs = from_epsg(3577)
    
    """
    Normalize data
    """
    for aux_var in ['rainfall', 'temperature', 'soilmoisture', 'runoff']:
        filename = aux_files[aux_var]
        aux_path = aux_paths[aux_var]
        input_df = pd.read_csv(os.path.join(aux_path, '{}.csv'.format(filename)))
        
        with rasterio.open(os.path.join(aux_path, '{}.tif'.format(filename))) as src:
            out_image = src.read()

            if aux_var in ['rainfall', 'runoff']:
                out_image = np.where(out_image!=src.nodata,
                                     np.log10(out_image+1),
                                     src.nodata)
            train_idx = input_df.index[(input_df['Date']>=train_starts[aux_var]
                                       )&(input_df['Date']<=train_ends[aux_var])]
            train_image = out_image[train_idx]
            image_max = np.max(train_image[train_image!=src.nodata])
            image_min = np.min(train_image[train_image!=src.nodata])

            out_image = np.where(out_image!=src.nodata, 
                                 (out_image-image_min)/(image_max-image_min), 
                                 src.nodata)

            out_meta = src.meta
            out_meta.update(
                    {'dtype': 'float64'}
                )

        print('Save {} data'.format(aux_var))
        with rasterio.Env():    
            with rasterio.open(os.path.join(work_dir, '{}_norm.tif'.format(filename)), 'w', **out_meta) as dst:
                dst.write(out_image)

        """
        Clip, reproject and resampling image to AOI
        """
        img_path = os.path.join(work_dir, '{}_norm.tif'.format(filename))
        out_path = os.path.join(work_dir, '{}_GBRCA_norm.tif'.format(filename))
        print('Reproj {} data'.format(aux_var))
        resampled_image, transform, out_meta = img_resample(
            img_path, 3000, ROI, crs, buffer=1, save=True, dst_path=out_path, 
            method=Resampling.bilinear)
        
        averages = np.nanmean(np.where(resampled_image==out_meta['nodata'], np.nan, resampled_image), (1, 2))
        input_df.iloc[:, 1] = averages
        input_df.to_csv(os.path.join(work_dir, '{}.csv'.format(filename)), index=False)
                
if __name__ == "__main__":
    mainRoutine()
        