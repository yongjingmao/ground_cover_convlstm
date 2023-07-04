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
from matplotlib.dates import DateFormatter
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import skimage.transform as st
from skimage.metrics import structural_similarity as ssim
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
    p.add_argument("-rd", "--ROI_dir", type=str, 
                   default="/scratch/rsc8/yongjingm/Github/ground_cover_convlstm/Data/Shapefiles/GBRCA_grazing.zip",
                   help="ROI directory")
    p.add_argument("-gd", "--gc_dir", type=str, 
                   default="/scratch/rsc3/fractionalcover3_cache/ground_cover_seasonal/qld",
                   help="Ground cover directory")
    p.add_argument("-ad", "--aux_dir", type=str, 
                   default="/scratch/rsc8/yongjingm/Auxiliary data",
                   help="Auxiliary data directory")
    p.add_argument("-md", "--model_dir", type=str,
                   default="/scratch/rsc8/yongjingm/Github/ground_cover_convlstm/trained_models/PredRNN_rs/predrnn.ckpt",
                   help="Model directory")
    p.add_argument("--config_dir", type=str,
                   default="/scratch/rsc8/yongjingm/Github/ground_cover_convlstm/config",
                   help="Config directory")
    p.add_argument("-tw", "--tile_width", type=int, default=1280,
                   help="Width of each tile")
    p.add_argument("-th", "--tile_height", type=int, default=1280,
                   help="Height of each tile")    
    p.add_argument("-ce", "--clear_cache", action="store_true",
                   help="Whether clear cache")
    cmdargs = p.parse_args()
    return cmdargs

def create_grid(gc_dir, ROI_gdf, tile_width, tile_height):
    '''
    This function tiles the ROI into grids with 50% overlapping,
    and saves the resulted polygons
    
    param gc_dir: Directory for ground cover data
    param ROI_gdf: GeoDataFrame for Region of Interest (ROI)
    param tile_width: Width of each individual tile (in terms of ground cover pixels)
    param tile_height: Height of each individual tile (in terms of ground cover pixels)
    
    return grids_intersect: Tiles within AOI
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

def get_gc(grid_id, grids_gdf, pred_df, gc_dir, img_width=128, img_height=128):
    """
    Read and process ground cover data for each tile
    
    param grid_id: the id of grid for processing
    param grids_gdf: the GeoDataFrame of grids
    param pred_df: the DataFrame of metadata for context images
    param gc_dir: Directory for ground cover data
    param img_width: Width of images (after transformation and resampling)
    param img_height: Height of imags (after transformation and resampling)
    
    return image_data: the ground cover data in the tile after transformation
    return new_transformation: the transformation metrix
    return out_meta: the metadata for the output image
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


def get_aux(grid_id, grids_gdf, pred_df, aux_dir, aux_vars, aux_filenames, img_width=128, img_height=128):
    """
    Read and process auxiliary data for a tile
    
    param grid_id: the id of grid for processing
    param grids_gdf: the GeoDataFrame of grids
    param pred_df: the DataFrame of metadata for context images
    param aux_dir: Directory for auxiliary data
    param aux_vars: List of aux variables
    param aux_filenames: List of names for aux variables    
    param img_width: Width of images (after transformation and resampling)
    param img_height: Height of imags (after transformation and resampling)
    
    return aux_data: the ground cover data in the tile after transformation
    """
    count = 0
    for aux_var in aux_vars:       
        filename = aux_filenames[aux_var]
        with rasterio.open(os.path.join(aux_dir, '{}_GBRCA_norm.tif'.format(filename))) as src:
            AOIs = grids_gdf.to_crs(src.crs)
            AOI = AOIs.loc[grid_id, 'geometry']
            aux_month, aux_transform = rasterio.mask.mask(src, [box(*AOI.bounds)], crop=True, all_touched=True)
            aux_month = np.where(aux_month==src.nodata, np.nan, aux_month)
            aux_month = st.resize(aux_month, (aux_month.shape[0], img_width, img_height), order=0)

            aux_means = np.nanmean(aux_month, (1, 2))
            aux_means = np.expand_dims(aux_means, (1,2)).repeat(aux_month.shape[1], 1).repeat(aux_month.shape[2], 2)
            aux_month = np.where(np.isnan(aux_month), aux_means, aux_month)

            idx_pred = pred_df['{}_idx'.format(aux_var)].astype(int)
            max_id = int(max(idx_pred))
            
            if len(aux_month) == max_id + 1:
                aux_month = np.concatenate((aux_month,  aux_month[[max_id], :, :], aux_month[[max_id], :, :]), axis=0)
            elif len(aux_month) == max_id + 2:
                aux_month = np.concatenate((aux_month,  aux_month[[max_id], :, :]), axis=0)
            
            aux_season = (aux_month[idx_pred, :, :]+aux_month[idx_pred+1, :, :]+aux_month[idx_pred+2, :, :])/3
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


def pred_mosaic(grid_tile_fps, ROI_gdf, img_size, n_channel):
    """
    Mosaic predicted tiles
    """
    global wind_weights
    wind_weights = spline_window(img_size, 2, n_channel)
    src_files_to_mosaic = []
    for fp in grid_tile_fps:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    mosaic, out_trans = merge(src_files_to_mosaic, resampling=rasterio.enums.Resampling.bilinear,
                             method=merge_smooth_edge)
    
    ROI_gdf_proj = ROI_gdf.to_crs(src.crs)
    AOI = cascaded_union(ROI_gdf_proj['geometry'])
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
    config_dir = cmdargs.config_dir
    tile_height = cmdargs.tile_height
    tile_width = cmdargs.tile_width
    clear_cache = cmdargs.clear_cache
    
    
    # Define variables
    aux_vars = ['rainfall', 'temperature', 'soilmoisture', 'runoff']
    aux_filenames = {'rainfall':'rainfall',
             'temperature':'temperature',
             'soilmoisture': 'soilmoisture',
             'runoff': 'runoff'}
    
    cfg_training = json.load(open(os.path.join(config_dir, "Training.json"), 'r'))
    model_type = cfg_training['project_name'].split('_')[0]
    cfg_model= json.load(open(os.paht.join(config_dir, model_type + ".json"), 'r'))
    
    context_length = cfg_training['context_training'] + cfg_training['future_training']
    img_width = cfg_model['img_width']
    img_height = cfg_model['img_height']
    
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
    gc_filenames = glob.glob(os.path.join(gc_dir, '*.tif'))
    dates = []
    
    for filename in gc_filenames:
        #print(filename)
        basename = os.path.basename(filename)
        date = datetime.strptime(basename.split('_')[2][1:7], '%Y%m')
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


    # Load trained model
    model = load_model(model_dir).to('cuda')

    # Predict each grid
    for grid_id in sorted(grids_gdf.index):
    
        pred_fp = cache_path + '/pred_grid{:03d}.tif'.format(grid_id)
        obs_fp = cache_path + '/obs_grid{:03d}.tif'.format(grid_id)
        
        if not os.path.exists(obs_fp):        
            print('Predicting grid {}/{}'.format(grid_id, len(grids_gdf)))

            # Prepare image and auxiliary data
            image_data, transform_image, meta = get_gc(grid_id, grids_gdf, pred_df, gc_dir, img_width, img_height)
            aux_data = get_aux(grid_id, grids_gdf, pred_df, aux_dir, aux_vars, aux_filenames, img_width, img_height)

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
            
            # Save observation
            with rasterio.Env():
                profile = meta
                with rasterio.open(obs_fp, 'w', **profile) as dst:
                    dst.write(obs.transpose(2, 0, 1))
                    
    # Glob output predictions and observations
    pred_fps = glob.glob(cache_path+'/pred_grid*.tif')
    obs_fps = glob.glob(cache_path+'/obs_grid*.tif')
    mosaic_pred, transform_pred, meta_pred = pred_mosaic(pred_fps, ROI_gdf, img_width, len(pred_df))
    mosaic_obs, _, _ = pred_mosaic(obs_fps, ROI_gdf, img_width, len(pred_df))
    
    # Retrieve the dates for predictions
    date_last_end = pred_df.index[-1] + relativedelta(months=+2)
    date_next_start = pred_df.index[-1] + relativedelta(months=+3)
    date_next_end = date_next_start + relativedelta(months=+2)
    str_last_start = datetime.strftime(pred_df.index[-1], "%Y-%m-%d")
    str_last_end = datetime.strftime(date_last_end, "%Y-%m-%d")
    str_next_start = datetime.strftime(date_next_start, "%Y-%m-%d")
    str_next_end = datetime.strftime(date_next_end, "%Y-%m-%d")
    all_seasons = list(pred_df.index)
    all_seasons.append(date_next_start)
    
    # Mosaice the observation and predictions
    mosaic_pred_fp = work_dir+'/Pred_mosaic_{}.tif'.format(str_next_start)
    with rasterio.open(mosaic_pred_fp, "w", **meta_pred) as dest:
         dest.write(mosaic_pred)
            
    mosaic_obs_fp = work_dir+'/Obs_mosaic_{}.tif'.format(str_last_start)
    with rasterio.open(mosaic_obs_fp, "w", **meta_pred) as dest:
         dest.write(mosaic_obs)
    
    # Mask the invalid data and calculate the frequency of cloud at each pixel
    gc_obs = np.where(mosaic_obs==255, np.nan, 100-mosaic_obs)
    gc_pred = np.where(mosaic_pred==255, np.nan, 100-mosaic_pred)
    cloud_feq = np.sum(np.isnan(gc_obs), (0)).astype(float)/gc_obs.shape[0]
    
    # Retreive the last season observation and prediciton as well as next season prediction
    # Calculate the predicted difference
    last_obs = gc_obs[-1, :, :]
    last_pred = np.where(cloud_feq<0.9, gc_pred[-2, :, :], np.nan)
    new_pred = np.where(cloud_feq<0.9, gc_pred[-1, :, :], np.nan)
    diff = new_pred - last_pred
    
    # Plot the comparison of prediction and observations
    fig, axes = plt.subplots(2, 2, figsize = [6, 8])
    ax1 = axes[0, 0]
    ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes)
    im1 = show(last_obs.squeeze(), transform=transform_pred, interpolation='nearest',
               vmin=50, vmax=100,  cmap='RdYlGn', ax=ax1)
    ax1.set_title('Ground cover obs\n {}~{}'.format(str_last_start, str_last_end), fontsize=10)
    t = ax1.yaxis.get_offset_text()
    t.set_x(-0.2)
    plt.colorbar(im1.get_images()[0], ax=ax1)
    ax2 = axes[0, 1]
    ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes)
    im2 = show(last_pred.squeeze(), transform=transform_pred, interpolation='nearest',
               vmin=50, vmax=100,  cmap='RdYlGn', ax=ax2)
    plt.colorbar(im2.get_images()[0], ax=ax2)
    ax2.set_title('Ground cover pred\n {}~{}'.format(str_last_start, str_last_end), fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3 = axes[1, 0]
    ax3.text(0.05, 0.95, '(c)', transform=ax3.transAxes)
    im3 = show(new_pred.squeeze(), transform=transform_pred, interpolation='nearest',
               vmin=50, vmax=100,  cmap='RdYlGn', ax=ax3)
    plt.colorbar(im3.get_images()[0], ax=ax3)
    ax3.set_title('Ground cover pred\n {}~{}'.format(str_next_start, str_next_end), fontsize=10)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4 = axes[1, 1]
    ax4.text(0.05, 0.95, '(d)', transform=ax4.transAxes)
    im4 = show(diff.squeeze(), transform=transform_pred, interpolation='nearest',
               vmin=-20, vmax=20,  cmap='bwr_r', ax=ax4)
    ax4.set_title('Ground cover\n predicted change', fontsize=10)
    ax4.set_xticks([])
    ax4.set_yticks([])

    ax5 = ax4.inset_axes([0.45, 0.7, 0.45, 0.25])
    sns.kdeplot(last_obs.squeeze().reshape(-1), ax=ax5, 
                linestyle='-',color='#d96f6f',label='(a)', gridsize=100, bw_adjust=3)
    sns.kdeplot(last_pred.squeeze().reshape(-1).reshape(-1), ax=ax5, 
                linestyle='-', color='#729c6b',label='(b)', gridsize=100, bw_adjust=3)
    sns.kdeplot(new_pred.squeeze().reshape(-1).reshape(-1), ax=ax5, 
                linestyle='--', color='#729c6b',label='(c)', gridsize=100, bw_adjust=3)
    ax5.set_yticks([])
    ax5.set_ylabel('')
    ax5.set_xlim([50, 105])
    ax5.set_xticks([50, 100])
    ax5.legend(loc=3, bbox_to_anchor=(-1, -2.8, 0.5, 0.5),
              fontsize='x-small', markerscale=0.5)


    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.colorbar(im4.get_images()[0], ax=ax4)
    plt.savefig(work_dir+'/Pred.jpg', dpi=300, bbox_inches='tight')
    
    # Plot time series of predicted sequence
    all_seasons = np.array(all_seasons)
    Obs_ts = np.nanmean(gc_obs, (1,2))
    Obs_inc = Obs_ts[1:] - Obs_ts[:-1]
    Pred_ts = np.nanmean(gc_pred, (1,2))
    Pred_inc = Pred_ts[1:] - Pred_ts[:-1]
    MAE_ts = np.nanmean(np.abs(gc_obs[1:16, :, :]-gc_pred[0:15, :, :]), (1, 2))
    MAE_std = np.nanstd(np.abs(gc_obs[1:16, :, :]-gc_pred[0:15, :, :]), (1, 2))
    SSIM_ts = [ssim(obs, pred) for (obs, pred) in zip(mosaic_obs[1:16, :, :], mosaic_pred[0:15, :, :])]
    f, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1.2]})
    ax0 = axes[0]
    ax1 = ax0.twinx()
    ax0.set_zorder(1) 
    ax0.set_frame_on(False)
    ax2 = axes[1]
    ax3 = ax2.twinx()
    ax0.plot(all_seasons[:16], Obs_ts, color='#e34f4f', linestyle='-',  label='Obs', zorder=1)
    ax0.plot(all_seasons[1:17], Pred_ts, color='#e34f4f', linestyle='--',  label='Pred', zorder=2)
    ax0.legend(loc=2)
    ax0.set_ylabel('Ground cover (spatial average)')
    bar_width = (all_seasons[1]-all_seasons[0])/3
    ax1.bar(all_seasons[2:-1]-bar_width, Obs_inc[1:], width=bar_width, edgecolor='#b37b7b', facecolor='#b37b7b',
           label='Obs Inc', zorder=0)
    ax1.bar(all_seasons[2:-1], Pred_inc[:-1], width=bar_width, edgecolor='#b37b7b', facecolor='None',
           label='Pred Inc', zorder=-1)
    ax1.set_xticks(all_seasons[0:17:2]+relativedelta(days=1))
    ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax1.set_ylabel('Ground cover increment')
    ax1.legend(loc=1)
    ax1.set_title('Spatial average of ground cover')


    ax2.errorbar(all_seasons[1:16], MAE_ts, yerr=MAE_std, color='#e34f4f', linestyle='-',  label='MAE')
    ax2.legend(loc=2)
    ax2.set_ylabel('MAE')
    ax3.plot(all_seasons[1:16], SSIM_ts, color='#e34f4f', linestyle='--',  label='SSIM')
    ax3.legend(loc=0)
    ax3.set_ylabel('SSIM')
    ax3.set_xticks(ax1.get_xticks())
    ax3.set_xticklabels(ax1.get_xticklabels())
    ax3.set_xlim(ax1.get_xlim())
    ax3.set_title('Scoring metrics')
    plt.savefig(work_dir+'/Timeseries.jpg', dpi=300,
               bbox_inches='tight')

    if clear_cache:
        shutil.rmtree(cache_path, ignore_errors=True)

if __name__ == "__main__":
    mainRoutine()