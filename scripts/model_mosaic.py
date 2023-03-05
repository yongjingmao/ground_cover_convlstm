import os
import sys
import glob
import shutil
import argparse

import math
import numpy as np
import scipy
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import skimage.transform as st
import rasterio
import rasterio.mask
import rasterio.features
from rasterio.plot import show
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform
from pyproj import Proj, transform
from shapely.geometry import box
from shapely.ops import cascaded_union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
import torch
from load_model_data import load_data_point, load_model
from core.models.ModelWrapperSampling import model_wrapper
from Data.data_preparation import DataModule, Dataset

def getCmdargs():
    p = argparse.ArgumentParser()
    p.add_argument("-wd", "--work_dir", type=str, default=None,
                   help="Work directory")
    p.add_argument("-rd", "--ROI_dir", type=str, default="/scratch/rsc8/yongjingm/ConvLSTM_Burdekin/Burdekin_grazing.zip",
                   help="ROI directory")
    p.add_argument("-gd", "--gc_dir", type=str, 
                   default="/scratch/rsc3/fractionalcover3_cache/ground_cover_seasonal/qld",
                   help="Ground cover directory")
    p.add_argument("-ad", "--aux_dir", type=str, 
                   default="/scratch/rsc8/yongjingm/ConvLSTM_Burdekin/Auxiliary data",
                   help="Auxiliary data directory")
    p.add_argument("-md", "--model_dir", type=str,
                   default="/scratch/rsc8/yongjingm/Github/ground_cover_convlstm/trained_models/PredRNN_rs/predrnn.ckpt",
                   help="Model directory")
    p.add_argument("-tsd", "--train_start_date", type=str, default="1991-12-01",
                   help="Starting date of training dataset")
    p.add_argument("-ted", "--train_end_date", type=str, default="2022-06-01",
                   help="End date of training dataset")
    p.add_argument("-tw", "--tile_width", type=int, default=128,
                   help="Width of each tile")
    p.add_argument("-th", "--tile_height", type=int, default=128,
                   help="Height of each tile")    
    p.add_argument("-ce", "--clear_cache", action="store_true",
                   help="Whether clear cache")
    cmdargs = p.parse_args()
    return cmdargs

def create_grid(gc_dir, ROI_gdf, tile_width, tile_height):
    '''
    This function tiles the ROI into grids with 50% overlapping,
    saves the results polygons
    '''
    # Read image data
    filenames = os.listdir(gc_dir)
    with rasterio.open(os.path.join(gc_dir, filenames[0])) as src:
        row_count = math.ceil(src.height/tile_height)
        col_count = math.ceil(src.width/tile_width)

        rows_start = np.arange(0, tile_height*row_count, tile_height/2)
        cols_start = np.arange(0, tile_width*col_count, tile_width/2)

        rows_end = rows_start + tile_height
        cols_end = cols_start + tile_width

        rows_start_grid, cols_start_grid = np.meshgrid(rows_start, cols_start)
        rows_end_grid, cols_end_grid = np.meshgrid(rows_end, cols_end)

        xs_start, ys_start = rasterio.transform.xy(src.transform, rows_start_grid.reshape(-1), cols_start_grid.reshape(-1))
        xs_end, ys_end = rasterio.transform.xy(src.transform, rows_end_grid.reshape(-1), cols_end_grid.reshape(-1))
    
    geometry = [box(x1, y1, x2, y2) for x1,y1,x2,y2 in zip(
    np.array(xs_start), np.array(ys_end), np.array(xs_end)-15, np.array(ys_start)-15)]
    grids_all = gpd.GeoDataFrame(geometry=geometry, crs=src.crs) 
    grids_intersect = grids_all[grids_all.intersects(cascaded_union(ROI_gdf.to_crs(grids_all.crs).geometry))]
    return grids_intersect.reset_index(drop=True)

def get_gc(grid_id, grids_gdf, pred_df, gc_dir, context_length, img_width=128, img_height=128):
    """
    Read and process ground cover data for each tile
    """
    count = 1
    for filename in pred_df['Name']:
        with rasterio.open(os.path.join(gc_dir, filename)) as src:    
            AOIs = grids_gdf.to_crs(src.crs)
            AOI = AOIs.loc[grid_id, 'geometry']
            out_image, transform = rasterio.mask.mask(src, [box(*AOI.bounds)], crop=True, all_touched=True)
            new_transform = rasterio.Affine(transform.a*out_image.shape[1]/img_width, transform.b, transform.c, 
                                transform.d, transform.e*out_image.shape[2]/img_height, transform.f)
            out_image = out_image[0].reshape(1, out_image.shape[1], out_image.shape[2])
            #out_mask = np.where(out_image==src.nodata, 1, 0).reshape(1, out_image.shape[1], out_image.shape[2])
            out_image = np.where(out_image==255, np.nan, out_image)
            out_image = st.resize(out_image, (1, img_width, img_height))
            out_mask = np.where(np.isnan(out_image), 1, 0)

            
            out_meta = src.meta

            # Concat images
            if count == 1:
                out_images = out_image
                out_masks = out_mask
            else:
                out_images = np.concatenate((out_images, out_image), axis=0)
                out_masks = np.concatenate((out_masks, out_mask), axis=0)

            count += 1

    image_data = np.concatenate((np.expand_dims(out_images, axis=-1), np.expand_dims(out_masks, axis=-1)), axis=-1)
    image_data = np.moveaxis(image_data, 0, -1)
    image_data = np.nan_to_num(image_data.astype(float), nan = 0.0)
    image_data[:, :, 0, :] = image_data[:, :, 0, :]/100  
    
    out_meta['transform'] = new_transform
    out_meta['width'] = img_width
    out_meta['height'] = img_height
    out_meta['count'] = 1
    return image_data, new_transform, out_meta


def get_aux(grid_id, grids_gdf, pred_df, train_df, aux_dir, aux_vars, aux_filenames, img_width=128, img_height=128):
    """
    Read and process auxiliary data for a tile
    """
    count = 0
    for aux_var in aux_vars:       
        filename = aux_filenames[aux_var]
        with rasterio.open(os.path.join(aux_dir, '{}.tif'.format(filename))) as src:
            AOIs = grids_gdf.to_crs(src.crs)
            AOI = AOIs.loc[grid_id, 'geometry']
            aux_month, aux_transform = rasterio.mask.mask(src, [box(*AOI.bounds)], crop=True, all_touched=True)
            aux_month = np.where(aux_month==src.nodata, np.nan, aux_month)
            aux_month = st.resize(aux_month, (aux_month.shape[0], img_width, img_height), order=0)

            aux_means = np.nanmean(aux_month, (1, 2))
            aux_means = np.expand_dims(aux_means, (1,2)).repeat(aux_month.shape[1], 1).repeat(aux_month.shape[2], 2)
            aux_month = np.where(np.isnan(aux_month), aux_means, aux_month)

            idx_pred = pred_df['{}_idx'.format(aux_var)]
            idx_train = train_df['{}_idx'.format(aux_var)]
            max_id = max(max(idx_pred), max(idx_train))
            
            if len(aux_month) == max_id + 1:
                aux_month = np.concatenate((aux_month,  aux_month[[max_id], :, :], aux_month[[max_id], :, :]), axis=0)
            elif len(aux_month) == max_id + 2:
                aux_month = np.concatenate((aux_month,  aux_month[[max_id], :, :]), axis=0)
            
            aux_season_pred = (aux_month[idx_pred, :, :]+aux_month[idx_pred+1, :, :]+aux_month[idx_pred+2, :, :])/3
            aux_season_train = (aux_month[idx_train, :, :]+aux_month[idx_train+1, :, :]+aux_month[idx_train+2, :, :])/3
            aux_season = (aux_season_pred - np.nanmin(aux_season_train))/(np.nanmax(aux_season_train) - np.nanmin(aux_season_train))

            aux_season = np.expand_dims(aux_season, axis=-1)

            if count == 0:
                aux_data = aux_season
            else:
                aux_data = np.concatenate((aux_data, aux_season), axis=-1)
            count+=1
    aux_data = np.moveaxis(aux_data, 0, -1)
    aux_data = np.nan_to_num(aux_data, nan = 0.0)
    return aux_data

def spline_window(window_size, power=2, channel=1):
    """
    Squared spline (power=2) window function:
    a filter with large weights in center and low weights around edges
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    
    wind = np.expand_dims(wind, 1)
    wind_2D = wind*wind.transpose()/4
    wind_final = np.expand_dims(wind_2D, 0)
    wind_final = np.repeat(wind_final, channel, 0)
    return wind_final


def merge_smooth_edge(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
    """
    Function for mosaic
    """
    old_data[:] = old_data + new_data*wind_weights


def pred_mosaic(grid_tile_fps, ROI_gdf, n_channel):
    global wind_weights
    wind_weights = spline_window(128, 2, n_channel)
    src_files_to_mosaic = []
    for fp in grid_tile_fps:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic, resampling=rasterio.enums.Resampling.bilinear,
                             method=merge_smooth_edge)
    
    ROI_gdf_proj = ROI_gdf.to_crs(src.crs)
    AOI = ROI_gdf_proj.loc[0, 'geometry']
    AOI_raster = rasterio.features.rasterize([AOI], out_shape=mosaic.shape[1:],
                                            transform=out_trans)
    mosaic = np.where(AOI_raster==1, mosaic, 255)
    
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                      "width": mosaic.shape[2],
                      "transform": out_trans,
                      }
                     )
    
    return mosaic, out_trans, out_meta

def mainRoutine():
    
    # Get command arguments
    cmdargs = getCmdargs()
    if not cmdargs.work_dir is None:
        work_dir = cmdargs.work_dir
    else:
        work_dir = os.getcwd()
    ROI_dir = cmdargs.ROI_dir
    gc_dir = cmdargs.gc_dir
    aux_dir = cmdargs.aux_dir
    model_dir = cmdargs.model_dir
    train_start_date = cmdargs.train_start_date
    train_end_date = cmdargs.train_end_date
    tile_height = cmdargs.tile_height
    tile_width = cmdargs.tile_width
    clear_cache = cmdargs.clear_cache
    
    
    # Define variables
    aux_vars = ['rainfall', 'temperature', 'soilmoisture', 'runoff']
    aux_filenames = {'rainfall':'rainfall',
             'temperature':'temperature',
             'soilmoisture': 'soil_moisture',
             'runoff': 'runoff'}
    context_length = 16
    img_width = 128
    img_height = 128
    
    cache_path = work_dir+'/image_cache'
    
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        
    
    # Generate grids for ROI
    ROI_gdf = gpd.read_file(ROI_dir)
    if not os.path.exists(work_dir+'/grids.shp'):
        grids_gdf = create_grid(gc_dir, ROI_gdf, tile_width, tile_height)
        grids_gdf.to_file(work_dir+'/grids.shp')
    else:
        grids_gdf = gpd.read_file(work_dir+'/grids.shp')

    # Process ground cover meta data
    gc_filenames = os.listdir(gc_dir)
    dates = []

    for filename in gc_filenames:
        #print(filename)
        date = datetime.strptime(filename.split('_')[2][1:7], '%Y%m')
        dates.append(date)

    gc_df = pd.DataFrame(
        {'Name':gc_filenames,
        'Date':dates}
    ).sort_values('Date')
    gc_df['Date'] = pd.to_datetime(gc_df['Date'])
    gc_df['gc_idx'] = range(len(gc_df))
    merge_df = gc_df.set_index('Date').copy()

    # Process aux metadata data
    for aux_var in aux_vars:
        aux_df = pd.read_csv(aux_dir+"/{}.csv".format(aux_filenames[aux_var]))
        aux_df['Date'] = pd.to_datetime(aux_df['Date'])
        aux_df['{}_idx'.format(aux_var)] = range(len(aux_df))
        aux_df.set_index('Date', inplace=True)
        merge_df = merge_df.join(aux_df, on='Date').dropna().sort_index()

    merge_df = merge_df[~merge_df.index.duplicated(keep='last')]
    pred_df = merge_df.iloc[-context_length:, :]
    train_df = merge_df.loc[(merge_df.index>=train_start_date)&(merge_df.index<=train_end_date), :]


    # Load trained model
    model = load_model(model_dir).to('cuda')

    # Predict each grid
    for grid_id in sorted(grids_gdf.index):
    
        pred_fp = cache_path + '/pred_grid{:03d}.tif'.format(grid_id)
        obs_fp = cache_path + '/obs_grid{:03d}.tif'.format(grid_id)
        
        if not os.path.exists(obs_fp):        
            print('Predicting grid {}/{}'.format(grid_id, len(grids_gdf)))

            # Prepare image and auxiliary data
            image_data, transform_image, meta = get_gc(grid_id, grids_gdf, pred_df, gc_dir, context_length, img_width, img_height)
            aux_data = get_aux(grid_id, grids_gdf, pred_df, train_df, aux_dir, aux_vars, aux_filenames, img_width, img_height)

            all_data = np.append(image_data, aux_data, axis=-2)
            all_data = torch.Tensor(all_data).permute(2, 0, 1, 3)
            truth = all_data.unsqueeze(dim=0).to('cuda')

            # Make prediction
            preds = model(truth, 16, 0, sampling=None).to('cpu')
            preds = preds[0, 0, :, :, :].detach().squeeze()*100
            preds = np.where(np.isnan(preds), 255, preds).astype(int)
            print(preds.shape)
            obs = image_data[:, :, 0, :].squeeze()*100
            obs_mask = image_data[:, :, 1, :].squeeze()
            obs = np.where(obs_mask==1, 255, obs).astype(int)
            
            meta['count'] = obs.shape[-1]
            # Save prediction
            with rasterio.Env():
                profile = meta
                with rasterio.open(pred_fp, 'w', **profile) as dst:
                    dst.write(preds.transpose(2, 0, 1))
            
            # Save prediction
            with rasterio.Env():
                profile = meta
                with rasterio.open(obs_fp, 'w', **profile) as dst:
                    dst.write(obs.transpose(2, 0, 1))
                    
        
    pred_fps = glob.glob(cache_path+'/pred_grid*.tif')
    obs_fps = glob.glob(cache_path+'/obs_grid*.tif')
    mosaic_pred, transform_pred, meta_pred = pred_mosaic(pred_fps, ROI_gdf, len(pred_df))
    mosaic_obs, _, _ = pred_mosaic(obs_fps, ROI_gdf, len(pred_df))
    
    date_next_season = pred_df.index[-1] + relativedelta(months=+3)
    str_last_season = datetime.strftime(pred_df.index[-1], "%Y-%m-%d")
    str_next_season = datetime.strftime(date_next_season, "%Y-%m-%d")
    
    mosaic_pred_fp = work_dir+'/Pred_mosaic_{}.tif'.format(str_next_season)
    with rasterio.open(mosaic_pred_fp, "w", **meta_pred) as dest:
         dest.write(mosaic_pred)
            
    mosaic_obs_fp = work_dir+'/Obs_mosaic_{}.tif'.format(str_last_season)
    with rasterio.open(mosaic_obs_fp, "w", **meta_pred) as dest:
         dest.write(mosaic_obs)
            
    gc_obs = np.where(mosaic_obs==255, np.nan, 100-mosaic_obs)
    gc_pred = np.where(mosaic_pred==255, np.nan, 100-mosaic_pred)
    
    last_obs = gc_obs[-1, :, :]
    last_pred = gc_pred[-2, :, :]
    new_pred = gc_pred[-1, :, :]
    diff = new_pred - last_pred

    fig, axes = plt.subplots(1, 4, figsize = [16, 4])
    ax1 = axes[0]
    im1 = show(last_obs.squeeze(), transform=transform_pred, interpolation='nearest',
               vmin=50, vmax=100,  cmap='RdYlGn', ax=ax1)
    ax1.set_title('Ground cover obs\n {}'.format(str_last_season))
    x_ticks = ax1.get_xticks()
    y_ticks = ax1.get_yticks()
    x_coords, y_coords = np.meshgrid(x_ticks, y_ticks)
    lats, lons = transform(meta_pred['crs'], 'epsg:4326',x_coords,y_coords)
    lat_labels = list(lats[:, 0])
    lon_labels = list(lons[0, :])
    lat_labels = ['{:.0f}'.format(label) for label in lat_labels]
    lon_labels = ['{:.0f}'.format(label) for label in lon_labels]
    lat_labels[1] = '-25'
    ax1.set_xticklabels(lon_labels)
    ax1.set_yticklabels(lat_labels)
    plt.colorbar(im1.get_images()[0], ax=ax1)
    ax2 = axes[1]
    im2 = show(last_pred.squeeze(), transform=transform_pred, interpolation='nearest',
               vmin=50, vmax=100,  cmap='RdYlGn', ax=ax2)
    plt.colorbar(im2.get_images()[0], ax=ax2)
    ax2.set_title('Ground cover pred\n {}'.format(str_last_season))
    ax2.set_xticklabels(lon_labels)
    ax2.set_yticklabels(lat_labels)
    ax3 = axes[2]
    im3 = show(new_pred.squeeze(), transform=transform_pred, interpolation='nearest',
               vmin=50, vmax=100,  cmap='RdYlGn', ax=ax3)
    plt.colorbar(im3.get_images()[0], ax=ax3)
    ax3.set_title('Ground cover pred\n {}'.format(str_next_season))
    ax3.set_xticklabels(lon_labels)
    ax3.set_yticklabels(lat_labels)
    ax4 = axes[3]
    im4 = show(diff.squeeze(), transform=transform_pred, interpolation='nearest',
               vmin=-20, vmax=20,  cmap='bwr_r', ax=ax4)
    ax4.set_title('Ground cover change')
    ax4.set_xticklabels(lon_labels)
    ax4.set_yticklabels(lat_labels)
    plt.colorbar(im4.get_images()[0], ax=ax4)
    plt.savefig(work_dir+'/Pred.jpg', dpi=300)

    if clear_cache:
        shutil.rmtree(cache_path, ignore_errors=True)

if __name__ == "__main__":
    mainRoutine()