import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import os
from os.path import join
import pickle5 as pickle
import glob

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths, include_non_pred):
        '''
            paths: list of input paths
            include_non_pred: whether include non-predict feature

            The dataset combines the different components of a earchnet data cube.

            ground cover (seasonal) - leave as it is
            auxiliary                - replicate the values to the entire image & each channel represents a property
                                       | Precipitation
        '''
        self.paths = paths
        self.include_non_pred = include_non_pred
        
    def __setstate__(self, d):
        self.paths = d["paths"]

    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, index):
        # load the item from data
        context = np.load(self.paths[index], allow_pickle=True)
        
        images = np.nan_to_num(context['image'].astype(float), nan = 0.0)
        
        # Scaled data
        images = np.where(images==255, 0, images)
        images[:, :, 0, :] = images[:, :, 0, :]/100  

        auxiliary = np.nan_to_num(context['auxiliary'], nan = 0.0)
        
        if self.include_non_pred:
            all_data = np.append(images, auxiliary, axis=-2)
        else:
            all_data = images
        
        ''' 
            Permute data so that it fits the Pytorch conv2d standard. From (w, h, c, t) to (c, w, h, t)
            w = width
            h = height
            c = channel
            t = time
        '''
        all_data = torch.Tensor(all_data).permute(2, 0, 1, 3)
        
        return all_data

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = "./", 
                 train_batch_size = 4,
                 val_batch_size = 4,
                 test_batch_size = 4, 
                 include_non_pred = True):
        """
        This is wrapper for all of our datasets. It preprocesses the data into a format that can be fed into our model.

        Parameters:
            data_dir: Location of pickle file with paths to the train/validation/test datapoints or a dictionary of file lists
            train_batch_size: Number of data points in a single train batch
            val_batch_size: Number of data points in a single validation batch
            test_batch_size: Number of data points in a single test batch
            train_batch_size: Number of data points in a single train batch
        """
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.include_non_pred = include_non_pred
        
        if os.path.basename(data_dir).split('.')[-1] != 'pickle':
            self.training_path_list = glob.glob(os.path.join(self.data_dir, 'train', '*.npz'))
            self.testing_path_list = glob.glob(os.path.join(self.data_dir, 'test', '*.npz'))
            self.validation_path_list = glob.glob(os.path.join(self.data_dir, 'val', '*.npz'))
        else:
            with open(data_dir, 'rb') as f:
                filelists = pickle.load(f)
            
            self.training_path_list = filelists['train']
            self.testing_path_list = filelists['test']
            self.validation_path_list = filelists['val']
            


    def setup(self, stage):
        # assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            # training
            self.training_data = Dataset(self.training_path_list, self.include_non_pred)
            # validation
            self.validation_data = Dataset(self.validation_path_list, self.include_non_pred)


        # assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            self.testing_data = Dataset(self.testing_path_list, self.include_non_pred)
            
    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.val_batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.testing_data, batch_size=self.test_batch_size, shuffle=False)

