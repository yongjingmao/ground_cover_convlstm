import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import datetime as dt
import glob
from datetime import datetime
import json

import torch
from skimage.metrics import structural_similarity as ssim

sys.path.append('..')
from Data import data_preparation
from Data.data_preparation import DataModule
import load_model_data
from load_model_data import load_data_point, load_model

#sys.path.append(os.getcwd()+'/demos')
import gc
torch.cuda.empty_cache()
gc.collect()

def getCmdargs():
    p = argparse.ArgumentParser()
    p.add_argument("-wd", "--work_dir", type=str, 
                   help="Work directory")
    p.add_argument("-md", "--model_dir", type=str, 
                   help="Model directory")
    cmdargs = p.parse_args()
    return cmdargs

def mainRoutine():
    cmdargs = getCmdargs()
    work_dir = cmdargs.work_dir
    # Load data 
    cfg_training = json.load(open(work_dir + "/config/Training.json", 'r'))
    project_name = cfg_training['project_name']
    pred_count = cfg_training['future_training']
    train_count = cfg_training['context_training']
    total_count = pred_count + train_count
    if cmdargs.model_dir is not None:
        model_dir = cmdargs.model_dir
    else:
        model_dir = glob.glob(work_dir + '/checkpoints/{}/*.ckpt'.format(project_name))[-1]
    model = load_model(model_dir).to('cuda')
    dataset = DataModule(data_dir=cfg_training["pickle_dir"], 
                         train_batch_size=cfg_training["train_batch_size"],
                         val_batch_size=cfg_training["val_batch_size"], 
                         test_batch_size=cfg_training["test_batch_size"], 
                         include_non_pred = cfg_training["include_non_pred"])


    # Calculate time series
    model = model.to('cuda')
    path_lists = {
        'train': dataset.training_path_list,
        'val': dataset.validation_path_list,
        'test': dataset.testing_path_list
    }

    columns = ['Date', 'Label']
    for i in range(pred_count):
        columns.append('Obs{}'.format(i+1))
        columns.append('Pred{}'.format(i+1))
        columns.append('MAE{}'.format(i+1))
        columns.append('SSIM{}'.format(i+1))
        columns.append('Cloud{}'.format(i+1))

    record_count = len(dataset.training_path_list) +\
    len(dataset.validation_path_list) + \
    len(dataset.testing_path_list)

    df_ts = pd.DataFrame(columns=columns, index=range(record_count)) # Dataframe to save results

    count = 0
    for group in path_lists.keys():    
        path_list = path_lists[group]
        for i in range(len(path_list)):
            truth, context, target, npf, name = load_data_point(cfg_training["pickle_dir"], group, cfg_training["include_non_pred"],
                                                         context_ratio=pred_count/(pred_count+train_count), index=i)
            pred_all = model(truth.to('cuda'), train_count, pred_count, sampling=None).to('cpu')
            date = datetime.strptime(name.split('_')[-1], '%Y-%m')

            df_ts.loc[count, 'Date'] = date
            df_ts.loc[count, 'Label'] = group

            for j in range(pred_count):            
                mask = truth[:, 1, :, :, train_count+j].detach().squeeze()
                obs = 100-100*truth[:, 0, :, :, train_count+j].detach().squeeze()
                pred = 100-100*pred_all[:, :, :, :, train_count+j-1].detach().squeeze()

                obs = np.where(mask, np.nan, obs)
                pred = np.where(mask, np.nan, pred)
                SSIM = ssim(np.where(np.isnan(obs), 0, obs), np.where(np.isnan(obs), 0, pred))
                MAE = np.nanmean(np.abs(obs-pred))

                df_ts.loc[count, 'Obs{}'.format(j+1)] = np.nanmean(obs)
                df_ts.loc[count, 'Pred{}'.format(j+1)] = np.nanmean(pred)
                df_ts.loc[count, 'SSIM{}'.format(j+1)] = SSIM
                df_ts.loc[count, 'MAE{}'.format(j+1)] = MAE
                df_ts.loc[count, 'Cloud{}'.format(j+1)] = np.float(mask.sum()/(mask.shape[0]*mask.shape[1]))

            print(date)
            count+=1

    df_ts.sort_values('Date', inplace=True)

    # Visualize the result for the best predicted data record
    data_index = (df_ts.loc[df_ts['Label']=='test', 'MAE1']).astype('float').argmin()
    truth, context, target, npf, name = load_data_point(cfg_training["pickle_dir"], 'test', cfg_training["include_non_pred"],
                                                 context_ratio=train_count/total_count, index=data_index)
    preds = model(truth.to('cuda'), train_count, pred_count, sampling=None).to('cpu')


    f, axes = plt.subplots(nrows=4, ncols=pred_count, figsize=(16, 8))
    f.subplots_adjust(right=0.95)
    cbar_ax1 = f.add_axes([0.96, 0.35, 0.02, 0.45])
    cbar_ax2 = f.add_axes([0.96, 0.13, 0.02, 0.15])

    for i in range(total_count):
        mask = truth[:, 1, :, :, i].detach().squeeze()
        obs = truth[:, 0, :, :, i].detach().squeeze()
        obs = np.where(mask, np.nan, obs)

        if i<train_count:
            ax1 = axes[0, i]
            im1 = ax1.imshow(100-100*obs, vmin=0, vmax=100,  cmap='RdYlGn')
            ax1.set_title('Context season{}'.format(i+1))
            ax1.set_xticks([])
            ax1.set_yticks([])

        else:
            pred = preds[:, :, :, :, i-1].detach().squeeze()
            ax2 = axes[1, i-train_count]
            ax2.imshow(100-100*obs, vmin=0, vmax=100,  cmap='RdYlGn')
            ax2.set_title('Target season{}'.format(i-train_count+1))
            ax2.set_xticks([])
            ax2.set_yticks([])

            ax3 = axes[2, i-train_count]
            ax3.imshow(100-100*pred, vmin=0, vmax=100, cmap='RdYlGn')
            ax3.set_title('Pred season{}'.format(i-train_count+1))
            ax3.set_xticks([])
            ax3.set_yticks([])

            ax4 = axes[3, i-train_count]
            im2 = ax4.imshow(100*(np.abs(pred-obs)), vmin=0, vmax=25, cmap='Reds')
            ax4.set_title('|Target{}-Pred{}|'.format(i-train_count+1, i-train_count+1))
            ax4.set_xticks([])
            ax4.set_yticks([])

    f.patch.set_facecolor('white')
    f.colorbar(im1, cax=cbar_ax1)
    cbar_ax1.set_ylabel('Ground cover (%)')
    f.colorbar(im2, cax=cbar_ax2)
    cbar_ax2.set_ylabel('Difference (%)')

    if not os.path.exists("Figures"):
        os.mkdir("Figures")
    if not os.path.exists("Figures/{}".format(project_name)):
        os.mkdir("Figures/{}".format(project_name))

    plt.savefig(os.path.join(work_dir, "Figures/{}/images.jpg".format(project_name)))


    # Visualize the time series of first step prediction
    colors = {'train':'#4d8047', 'val':'#064e78', 'test':'#e34f4f'}
    if 'trained_models' in model_dir:
        df_ts['Label'] = 'test'
    f, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), gridspec_kw={'height_ratios': [2, 1.2]})
    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax3 = axes[1]
    ax4 = ax3.twinx()
    for group in colors.keys():
        idx = (df_ts['Label']==group)
        ax1.plot(df_ts.loc[idx, 'Date'], df_ts.loc[idx, 'Obs1'], 
                 color=colors[group], linestyle='-', label='{}_obs'.format(group))
        ax1.plot(df_ts.loc[idx, 'Date'], df_ts.loc[idx, 'Pred1'],
                 color=colors[group], linestyle='--',  label='{}_pred'.format(group))
        if group == 'test':

            label = 'Cloud cover (%)'
        else:
            label = ''
        ax2.bar(df_ts.loc[idx, 'Date'], df_ts.loc[idx, 'Cloud1']*100,
                width=0.5*(df_ts['Date'][1]-df_ts['Date'][0]), zorder=-1, 
                color='grey', alpha=0.5, label=label)
        ax3.plot(df_ts.loc[idx, 'Date'], df_ts.loc[idx, 'MAE1'],
                 color=colors[group], linestyle='-')
        ax4.plot(df_ts.loc[idx, 'Date'], df_ts.loc[idx, 'SSIM1'],
                 color=colors[group], linestyle='--')

    ax1.legend(ncol=3)
    ax1.set_ylabel('Ground cover (%)')
    ax1.set_zorder(2)
    ax1.set_facecolor("none")
    ax2.set_zorder(1)
    ax2.set_ylabel('Cloud cover (%)')
    ax2.legend()
    #ax2.set_yscale('log')
    ax3.set_ylabel('MAE')
    ax3.plot([], [], linestyle='-', color='k', label='MAE')
    ax3.legend(loc=2, ncol=1)
    ax3.set_xticks([])
    ax4.set_ylabel('SSIM')
    ax4.plot([], [], linestyle='--', color='k', label='SSIM')
    ax4.legend(loc=1, ncol=1)

    test_MAE = np.mean(df_ts.loc[df_ts['Label']=='test', 'MAE1'])
    test_SSIM = np.mean(df_ts.loc[df_ts['Label']=='test', 'SSIM1'])
    print("Test MAE: {:.3f}".format(test_MAE))
    print("Test SSIM: {:.3f}".format(test_SSIM))
    ax4.set_title("Test MAE: {:.3f}, Test SSIM: {:.3f}".format(test_MAE, test_SSIM))
    plt.savefig(os.path.join(work_dir, "Figures/{}/timeseries.jpg".format(project_name)))


if __name__ == "__main__":
    mainRoutine()