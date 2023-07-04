import os
import argparse
import shutil
import glob

import rasterio
from shapely.geometry import box
import numpy as np
from datetime import datetime
from rasterio.enums import Resampling
import rasterio.mask
from rasterio.plot import show

from sklearn.model_selection import train_test_split
import pandas as pd
import geopandas as gpd


def getCmdargs():
    p = argparse.ArgumentParser()
    p.add_argument("-wd", "--work_dir", type=str, default="/scratch/rsc8/yongjingm/ConvLSTM_GBRCA",
                   help="Work directory")
    p.add_argument("-ad", "--aux_dir", type=str, default="/scratch/rsc8/yongjingm/Auxiliary data",
                   help="Auxiliary data directory")
    p.add_argument("-gd", "--gc_dir", type=str, default="/scratch/rsc3/fractionalcover3_cache/ground_cover_seasonal/qld",
                   help="Work directory")
    p.add_argument("-AOI", "--AOI_dir", type=str, 
                   default="/scratch/rsc8/yongjingm/Github/ground_cover_convlstm/Data/Shapefiles/AOIs.shp.zip",
                   help="Work directory")
    p.add_argument("--Site", type=int, 
                   help="Index of Site")
    p.add_argument("--window_length", '-l', type=int, 
                   help="Length of window")
    p.add_argument("--img_width", '-iw', type=int, default=128,
                   help="Width of image")
    p.add_argument("--img_height", '-ih', type=int, default=128,
                   help="Height of image")
    p.add_argument("--train_ratio", '-tr', type=float, default=0.5,
                   help="Ratio to split training data")
    p.add_argument("--val_test_ratio", '-vtr', type=float, default=0.5,
                   help="Ratio to split validation and testing data")
    p.add_argument("--rainfall", "-rf", action="store_true",
                    help="Whether include rainfall in the auxiliary data")
    p.add_argument("--temperature", "-tp", action="store_true",
                    help="Whether include temperature in the auxiliary data") 
    p.add_argument("--soilmoisture", "-sm", action="store_true",
                help="Whether include soil temperature in the auxiliary data") 
    p.add_argument("--runoff", "-ro", action="store_true",
            help="Whether include runoff in the auxiliary data") 
    
    cmdargs = p.parse_args()
    return cmdargs

def mainRoutine():
    
    cmdargs = getCmdargs()
    
    work_dir = cmdargs.work_dir
    aux_dir = cmdargs.aux_dir
    gc_dir = cmdargs.gc_dir
    
    AOI_dir = cmdargs.AOI_dir
    AOI_idx = cmdargs.Site
    window_size = cmdargs.window_length # Temporal length
    width = cmdargs.img_width # Width of tile
    height = cmdargs.img_height # Height of tile
    train_ratio = cmdargs.train_ratio # Split ratio for training and others
    val_test_ratio = cmdargs.val_test_ratio # Split ratio for testing and validation
    
    AOIs = gpd.read_file(AOI_dir)
    
    target_dir = os.path.join(work_dir, 'Sites/Site{}'.format(AOI_idx))
    
    if not os.path.exists(os.path.join(work_dir, 'Sites')):
        os.mkdir(os.path.join(work_dir, 'Sites'))
    
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    
    
    aux_vars = []
    if cmdargs.rainfall:
        aux_vars.append('rainfall')
    if cmdargs.temperature:
        aux_vars.append('temperature')
    if cmdargs.soilmoisture:
        aux_vars.append('soilmoisture')
    if cmdargs.runoff:
        aux_vars.append('runoff')
    
    
    aux_files = {'rainfall':'rainfall',
             'temperature':'temperature',
             'soilmoisture': 'soilmoisture',
             'runoff': 'runoff'}
    
    # Clip auxiliary images
    for aux_var in aux_vars:       
        if not os.path.exists(os.path.join(target_dir, "{}.tif".format(aux_var))):
            filename = aux_files[aux_var]
            input_df = pd.read_csv(os.path.join(aux_dir, '{}.csv'.format(filename)))

            with rasterio.open(os.path.join(aux_dir, '{}_GBRCA_norm.tif'.format(filename))) as src:
                AOIs = AOIs.to_crs(src.crs)
                AOI = AOIs.loc[AOI_idx, 'geometry']
                out_image, out_transform = rasterio.mask.mask(src, [box(*AOI.bounds)], crop=True, all_touched=True)
                out_meta = src.meta
                out_meta.update(
                        {'width': out_image.shape[2],
                        'height': out_image.shape[1],
                        'count': out_image.shape[0],
                        'transform':out_transform,
                        'dtype':'float64'}
                    )

            dates = input_df['Date']
            values = np.nanmean(np.where(out_image==src.nodata, np.nan, out_image), (1,2))

            output_df = pd.DataFrame({'Date':dates, aux_var:values})
            output_df.to_csv(os.path.join(target_dir, "{}.csv".format(aux_var)), index=False)
            print('Save {} data'.format(aux_var))
            with rasterio.Env():    
                with rasterio.open(os.path.join(target_dir, "{}.tif".format(aux_var)), 
                                   'w', **out_meta) as dst:
                    dst.write(out_image)
    
    #Clip ground cover image data
    if not os.path.exists(os.path.join(target_dir, "seasonal_masks.tif")):
        data_dir = gc_dir
        filenames = os.listdir(data_dir)
        dates = []
        cloudcover = []

        count = 0
        for filename in filenames:
            count += 1
            #print(filename)
            date = datetime.strptime(filename.split('_')[2][1:7], '%Y%m')
            dates.append(date)

            with rasterio.open(os.path.join(data_dir, filename)) as src:    
                AOIs = AOIs.to_crs(src.crs)
                AOI = AOIs.loc[AOI_idx, 'geometry']
                out_image, out_transform = rasterio.mask.mask(src, [box(*AOI.bounds)], crop=True, all_touched=True)
                out_image = out_image[0].reshape(1, out_image.shape[1], out_image.shape[2])
                out_mask = np.where(out_image==src.nodata, 1, 0).reshape(1, out_image.shape[1], out_image.shape[2])
                out_meta = src.meta

                # Calculate cloud cover
                cloudcover.append(
                    np.sum(out_image[0]==src.nodata)/(out_image.shape[1]*out_image.shape[2])*100
                )


                # Concat images
                if count == 1:
                    out_images = out_image
                    out_masks = out_mask
                else:
                    out_images = np.concatenate((out_images, out_image), axis=0)
                    out_masks = np.concatenate((out_masks, out_mask), axis=0)

                out_meta.update(
                    {'width': out_images.shape[2],
                    'height': out_images.shape[1],
                    'count': out_images.shape[0],
                    'dtype': 'uint8',
                    'transform':out_transform}
                )


        meta_resample = out_meta.copy()
        meta_resample['count'] = out_images.shape[0]
        meta_mask = meta_resample.copy()
        meta_mask['count'] = out_masks.shape[0]
        
        # Save metadata
        print('Save metadata')
        out_df = pd.DataFrame(
            {'Name':filenames,
            'Date':dates,
            'CloudCover':cloudcover}
        )
        out_df.to_csv(os.path.join(target_dir, "seasonal_meta.csv"), index=False)  
        
        # Save image data
        print('Save images')
        with rasterio.Env():    
            profile = meta_resample    
            with rasterio.open(os.path.join(target_dir, "seasonal_groundcover.tif"), 
                               'w', **profile) as dst:
                dst.write(out_images.astype(rasterio.uint8))

        # Save mask data
        print('Save masks')
        with rasterio.Env():    
            profile = meta_mask    
            with rasterio.open(os.path.join(target_dir, "seasonal_masks.tif"), 
                                            'w', **profile) as dst:
                dst.write(out_masks.astype(rasterio.uint8))

  
    
    # Read Ground cover metadata (seasonal)
    df_gc = pd.read_csv(os.path.join(target_dir, "seasonal_meta.csv"))
    df_gc['Date'] = pd.to_datetime(df_gc['Date'])
    df_gc['gc_idx'] = range(len(df_gc))
    df_merge = df_gc.set_index('Date').copy()
    
    
    # Read auxiliary metadata (monthly)
    for aux_var in aux_vars:
        df_aux = pd.read_csv(os.path.join(target_dir, "{}.csv".format(aux_var)))
        df_aux['Date'] = pd.to_datetime(df_aux['Date'])
        df_aux['{}_idx'.format(aux_var)] = range(len(df_aux))
        df_aux.set_index('Date', inplace=True)
        df_merge = df_merge.join(df_aux, on='Date').dropna().sort_index()
        
    df_merge = df_merge[~df_merge.index.duplicated(keep='last')]

    # Resample data to input tile sizes
    def image_resample(img, width, height):
        with rasterio.open(img, 'r') as src:
            profile = src.profile
            # resample data to target shape
            data = src.read(
                out_shape=(
                    src.count,
                    height,
                    width
                ),
                resampling=Resampling.bilinear
            )

            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]),
                (src.height / data.shape[-2])
            )
        return data, src.nodata    
    

    
    # Prepare image data to (w, h, c, t)
    print('Process image')
    ground_cover, _ = image_resample(
        os.path.join(target_dir, "seasonal_groundcover.tif"), width, height)
    mask, _ = image_resample(
        os.path.join(target_dir,"seasonal_masks.tif"), width, height)
        
    image_data = np.concatenate((np.expand_dims(ground_cover, axis=-1), np.expand_dims(mask, axis=-1)), axis=-1)
    image_data = np.moveaxis(image_data, 0, -1)
    image_data = image_data[:, :, :, df_merge['gc_idx'].astype(int).values].astype(float)
    
    # Scaled data
    image_data = np.where(image_data==255, 0, image_data)
    image_data[:, :, 0, :] = image_data[:, :, 0, :].astype(float)/100.0 
    
    # Prepare aux data to (w, h, c, t)
    count = 0
    for aux_var in aux_vars:
        print('Process {}'.format(aux_var))
        aux_month, nodata = image_resample(
            os.path.join(target_dir, "{}.tif".format(aux_var)), width, height)
        aux_month = np.where(aux_month==nodata, np.nan, aux_month)
        
        # fill nan with mean of each time step
        aux_means = np.nanmean(aux_month, (1, 2))
        aux_means = np.expand_dims(aux_means, (1,2)).repeat(aux_month.shape[1], 1).repeat(aux_month.shape[2], 2)
        aux_month = np.where(np.isnan(aux_month), aux_means, aux_month)
        
        idx = df_merge['{}_idx'.format(aux_var)]
        max_id = max(idx)
        if len(aux_month) == max_id + 1:
            aux_month = np.concatenate((aux_month,  aux_month[[max_id], :, :], aux_month[[max_id], :, :]), axis=0)
        elif len(aux_month) == max_id + 2:
            aux_month = np.concatenate((aux_month,  aux_month[[max_id], :, :]), axis=0)
        aux_season = (aux_month[idx, :, :]+aux_month[idx+1, :, :]+aux_month[idx+2, :, :])/3
        
        aux_season = np.expand_dims(aux_season, axis=-1)
                      
        if count == 0:
            aux_data = aux_season
        else:
            aux_data = np.concatenate((aux_data, aux_season), axis=-1)
        count+=1
    aux_data = np.moveaxis(aux_data, 0, -1)

    # Slice images into small sequence according to input window length
    def slice_images(data, window_size):
        dataslices = []
        idx = []
        s_time = []
        e_time = []
        for i in range(data.shape[-1]-window_size+1):
            images = data[:, :, :, i:i+window_size]
            dataslices.append(images)
            idx.append(i)
            s_time.append(df_merge.index[i])
            e_time.append(df_merge.index[i+window_size-1])
        return dataslices, idx, s_time, e_time
    
    
    # Split to training, testing and validation
    image_slices, idx, s_time, e_time = slice_images(image_data, window_size)
    aux_slices, idx, s_time, e_time = slice_images(aux_data, window_size)
    
    if train_ratio == 0:
        idx_train = []; idx_val_test = idx
    elif train_ratio == 1:
        idx_train = idx; idx_val_test = []
    else:
        idx_train, idx_val_test = train_test_split(idx, test_size=1-train_ratio, shuffle=False)
    if val_test_ratio == 0:
        idx_val = idx; idx_test = []
    elif val_test_ratio == 1:
        idx_val = []; idx_test = idx
    else:
        idx_val, idx_test = train_test_split(idx_val_test, test_size=val_test_ratio, shuffle=False)

    # Collect data to folders
    datafolder = os.path.join(target_dir, 'data')
    if not os.path.exists(datafolder):
        os.mkdir(datafolder)

    for foldername in ['train', 'test', 'val']:
        outfolder = os.path.join(datafolder, foldername)
        if not os.path.exists(outfolder):
            os.mkdir(outfolder)
        else:
            shutil.rmtree(outfolder, ignore_errors=True)
            os.mkdir(outfolder)   

    # Save data
    for i in idx:
        if i in idx_train:
            group = 'train'
        elif i in idx_val:
            group = 'val'
        else:
            group = 'test' 

        folder = os.path.join(datafolder, group)            
        filename = os.path.join(folder, '{}_{}'.format(s_time[i].strftime("%Y-%m"), e_time[i].strftime("%Y-%m")))
        image_slice = image_slices[i]
        aux_slice = aux_slices[i]
        np.savez(filename, image=image_slice, auxiliary=aux_slice)
        
if __name__ == "__main__":
    mainRoutine()
        
        