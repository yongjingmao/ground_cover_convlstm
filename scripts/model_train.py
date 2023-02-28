import sys
import os
from os import listdir
from pathlib import Path
import shutil
import argparse

import json
#import wandb
import torch
import tensorboard
import random

#import pytorch_lightning as pl
sys.path.append('..')
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from core.models.ModelWrapperSampling import model_wrapper
from Data.data_preparation import DataModule
#from callbacks import WandbTrain_callback
from datetime import datetime

import gc
torch.cuda.empty_cache()
gc.collect()

def getCmdargs():
    p = argparse.ArgumentParser()
    p.add_argument("-wd", "--work_dir", type=str, 
                   help="Work directory")
    p.add_argument("-ce", "--clear_existing", action="store_true",
                   help="Whether clear existing checkpoint and logs")
    cmdargs = p.parse_args()
    return cmdargs

def mainRoutine():
    cmdargs = getCmdargs()
    work_dir = cmdargs.work_dir
    
    # Load data and initialize model
    cfg_training = json.load(open(work_dir + "/config/Training.json", 'r'))
    model_type = cfg_training['project_name'].split('_')[0]
    cfg_model= json.load(open(work_dir + "/config/" + model_type + ".json", 'r'))

    dataset = DataModule(data_dir=cfg_training["pickle_dir"], 
                         train_batch_size=cfg_training["train_batch_size"],
                         val_batch_size=cfg_training["val_batch_size"], 
                         test_batch_size=cfg_training["test_batch_size"], 
                         include_non_pred = cfg_training["include_non_pred"])

    model = model_wrapper(model_type, cfg_model, cfg_training)

    project_name = cfg_training['project_name']
    checkpoint_path = work_dir + '/checkpoints/{}'.format(project_name)
    log_path = work_dir + '/lightning_logs/{}'.format(project_name)
    
    if cmdargs.clear_existing:
        if os.path.exists(checkpoint_path):
            print('Clear checkpoint')
            shutil.rmtree(checkpoint_path, ignore_errors=True)
        if os.path.exists(log_path):
            print('Clear log')
            shutil.rmtree(log_path, ignore_errors=True)

    # Set call backs
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, 
                                          monitor="val_loss",
                                          save_on_train_epoch_end=True, 
                                          save_last=True,
                                          save_top_k=1,
                                          filename='model_{epoch:03d}')

    earlystopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=cfg_training['patience'])

    if cfg_training["early_stopping"]:
        callbacks = [checkpoint_callback, earlystopping_callback]
    else:
        callbacks = [checkpoint_callback]

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_path)
    
    # Set model trainer
    trainer = Trainer(max_epochs=cfg_training["epochs"], 
                      devices=1,
                      accelerator="auto",
                      num_sanity_val_steps=1,
                      check_val_every_n_epoch=1,
                      precision=cfg_training["precision"],
                      logger = tb_logger,
                      log_every_n_steps=5,
                      callbacks=[checkpoint_callback, earlystopping_callback])
    
    if os.path.exists(checkpoint_path):
        checkpoint_lists = sorted(listdir(checkpoint_path))
    else:
        checkpoint_lists = []
        
    if len(checkpoint_lists)==0:
        ckpt_path = None
    else:
        if 'last.ckpt' in checkpoint_lists:
            ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        else:
            ckpt_path = os.path.join(checkpoint_path, checkpoint_lists[-1])
    
    # Train the model
    trainer.fit(model, dataset, ckpt_path=ckpt_path)
    
if __name__ == "__main__":
    mainRoutine()