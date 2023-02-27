import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import os
from os.path import join
import pickle
import glob

class Dataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        '''
            paths: list of input paths
            ms_cut: indxs for relevant mesoscale data

            The dataset combines the different components of a earchnet data cube.

            ground cover (seasonal) - leave as it is
            auxliary                - replicate the values to the entire image & each channel represents a property
                                       | Precipitation
        '''
        self.paths = paths
        self.data = np.load(self.paths, allow_pickle=True)
        self.fake_weather = False
        
    def __setstate__(self, d):
        self.paths = d["paths"]

    def __len__(self):
        return self.data.shape[1]
 
    def __getitem__(self, index):
        # load the item from data
        data = self.data
        image = data[:, [index], :, :].transpose(1,2,3,0)
        mask = np.zeros(image.shape)
        image_mask = np.append(image, mask, axis=0)      
        # Scaled data
        image_mask[0, :, :, :] = image_mask[0, :, :, :]/255  
        
        all_data = torch.Tensor(image_mask)
        
        return all_data

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str = "./", 
                 train_batch_size = 4,
                 val_batch_size = 4,
                 test_batch_size = 4):
        """
        This is wrapper for all of our datasets. It preprocesses the data into a format that can be fed into our model.

        Parameters:
            data_dir: Location of pickle file with paths to the train/validation/test datapoints
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
        
        self.training_path = os.path.join(self.data_dir, 'train', 'train.npy')
        self.testing_path = os.path.join(self.data_dir, 'test', 'test.npy')
        self.validation_path = os.path.join(self.data_dir, 'val', 'val.npy')


    def setup(self, stage):
        # assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            # training
            self.training_data = Dataset(self.training_path)
            # validation
            self.validation_data = Dataset(self.validation_path)


        # assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            self.testing_data = Dataset(self.testing_path)
            
    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.val_batch_size)

    def test_dataloader(self):
        return DataLoader(self.testing_data, batch_size=self.test_batch_size)

