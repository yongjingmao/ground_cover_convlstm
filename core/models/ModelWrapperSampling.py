import pytorch_lightning as pl
import numpy as np
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .model_parts.Conv_LSTM_Sampling import Conv_LSTM
from .model_parts.Pred_RNN_Sampling import Pred_RNN
from ..losses import get_loss_from_name
from ..optimizers import get_opt_from_name


class model_wrapper(pl.LightningModule):
    def __init__(self, model_type, model_cfg, training_cfg):
        """
        This is the supermodel that wraps all the possible models to do ground cover forecasting
        with resampling incorporated. 
        It uses the model (specified by 'model_type') that it wraps to predict.

        Parameters:
            cfg (dict) -- model configuration parameters
        """
        super().__init__()
        self.save_hyperparameters()

        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        
        if model_type == "ConvLSTM":
            self.model = Conv_LSTM(input_dim=self.model_cfg["input_channels"],
                                   output_dim=self.model_cfg["output_channels"],
                                   hidden_dims=self.model_cfg["hidden_channels"],
                                   big_mem=self.model_cfg["big_mem"],
                                   num_layers=self.model_cfg["n_layers"],
                                   kernel_size=self.model_cfg["kernel"],
                                   memory_kernel_size=self.model_cfg["memory_kernel"],
                                   dilation_rate=self.model_cfg["dilation_rate"],
                                   baseline=self.training_cfg["baseline"],
                                   layer_norm_flag=self.model_cfg["layer_norm"],
                                   img_width=self.model_cfg["img_width"],
                                   img_height=self.model_cfg["img_height"],
                                   mask_channel = self.model_cfg['mask_channel'],
                                   peephole=self.model_cfg["peephole"])
        elif model_type == "PredRNN":
            self.model = Pred_RNN(input_dim=self.model_cfg["input_channels"],
                                 output_dim=self.model_cfg["output_channels"],
                                 hidden_dims=self.model_cfg["hidden_channels"],
                                 num_layers=self.model_cfg["n_layers"],
                                 kernel_size=self.model_cfg["kernel"],
                                 baseline=self.training_cfg["baseline"],
                                 layer_norm_flag=self.model_cfg["layer_norm"],
                                 img_width=self.model_cfg["img_width"],
                                 img_height=self.model_cfg["img_height"],
                                 mask_channel = self.model_cfg['mask_channel'])        
            
        self.baseline = self.training_cfg["baseline"]
        self.context_training = self.training_cfg["context_training"]
        self.future_training = self.training_cfg["future_training"]
        self.total_training = self.context_training + self.future_training
        self.learning_rate = self.training_cfg["start_learn_rate"]
        self.mask_channel = self.model_cfg["mask_channel"]
        
        self.training_sampling = self.training_cfg["training_sampling"]
        self.r_sampling_step_1 = int(self.training_cfg["epochs"]/4)
        self.r_sampling_step_2 = int(self.training_cfg["epochs"]/2)
        self.r_exp_alpha = self.training_cfg["epochs"]/20
        
        self.sampling_stop_epoch = int(self.training_cfg["epochs"]*0.8)
        self.sampling_start_value = 1
        self.sampling_changing_rate = 1/self.sampling_stop_epoch
        
        self.train_batch_size = self.training_cfg["train_batch_size"]
        
        
        self.train_loss = get_loss_from_name(self.training_cfg["train_loss"])
        self.test_loss = get_loss_from_name(self.training_cfg["test_loss"])
        
        self.input_flag = np.zeros((self.train_batch_size,
                      self.model_cfg["input_channels"],
                      self.model_cfg["img_width"],
                      self.model_cfg["img_height"],
                      self.total_training - 1))


    def forward(self, x, context_count, future_count, sampling=None):
        """
        :param x: All features of the input time steps.
        :param future_count: The amount of time steps that should be predicted all at once.
        :param non_pred_feat: Only need if prediction_count > 1. All features that are not predicted
            by the model for all the future time steps we want to predict.
        :return: preds: Full predicted images.
        """
        epoch = self.current_epoch
        eta = self.sampling_start_value - self.sampling_changing_rate*epoch
        if eta<0:
            eta = 0
        
        if sampling == 'rs':
            input_flag = self.reverse_schedule_sampling(epoch)
        elif sampling == 'ss':
            input_flag = self.schedule_sampling(eta, epoch)
        else:
            input_flag = self.input_flag
            
        preds = self.model(x, input_flag, context_count, future_count, sampling)
        return preds
    
    def reverse_schedule_sampling(self, epoch):
        # Create reverse schedule sampling mask
        if epoch < self.r_sampling_step_1:
            r_eta = 0.5
        elif epoch < self.r_sampling_step_2:
            r_eta = 1.0 - 0.5 * math.exp(-float(epoch - self.r_sampling_step_1) / self.r_exp_alpha)
        else:
            r_eta = 1.0

        if epoch < self.r_sampling_step_1:
            eta = 0.5
        elif epoch < self.r_sampling_step_2:
            eta = 0.5 - (0.5 / (self.r_sampling_step_2 - self.r_sampling_step_1)) * (epoch - self.r_sampling_step_1)
        else:
            eta = 0.0

        r_random_flip = np.random.random_sample(
            (self.train_batch_size, self.context_training - 1))
        r_true_token = (r_random_flip < r_eta)

        random_flip = np.random.random_sample(
            (self.train_batch_size, self.future_training - 1))
        true_token = (random_flip < eta)

        trues = np.ones((self.model_cfg["input_channels"],
                        self.model_cfg["img_width"],
                        self.model_cfg["img_height"]))
        falses = np.zeros((self.model_cfg["input_channels"],
                        self.model_cfg["img_width"],
                        self.model_cfg["img_height"]))

        real_input_flag = np.zeros((self.train_batch_size,
                      self.model_cfg["input_channels"],
                      self.model_cfg["img_width"],
                      self.model_cfg["img_height"],
                      self.total_training - 2))
        
        for i in range(self.train_batch_size):
            for j in range(self.total_training-2):
                if j < self.context_training - 1:
                    if r_true_token[i, j]:
                        real_input_flag[i, :, :, :, j] = trues
                    else:
                        real_input_flag[i, :, :, :, j] = falses
                else:
                    if true_token[i, j - (self.context_training - 1)]:
                        real_input_flag[i, :, :, :, j] = trues
                    else:
                        real_input_flag[i, :, :, :, j] = falses
        return real_input_flag


    def schedule_sampling(self, eta, epoch):
        # Create schedule sampling mask
        random_flip = np.random.random_sample(
            (self.train_batch_size, self.future_training - 1))
        true_token = (random_flip < eta)
        trues = np.ones((self.model_cfg["input_channels"],
                        self.model_cfg["img_width"],
                        self.model_cfg["img_height"]))
        falses = np.zeros((self.model_cfg["input_channels"],
                        self.model_cfg["img_width"],
                        self.model_cfg["img_height"]))
        
        real_input_flag = np.zeros((self.train_batch_size,
                      self.model_cfg["input_channels"],
                      self.model_cfg["img_width"],
                      self.model_cfg["img_height"],
                      self.future_training - 1))
        
        for i in range(self.train_batch_size):
            for j in range(self.future_training - 1):
                if true_token[i, j]:
                    real_input_flag[i, :, :, :, j] = trues
                else:
                    real_input_flag[i, :, :, :, j] = falses
        return real_input_flag

    def batch_loss(self, batch, sampling=None, loss=None):
        # Calculate batch loss
        input_tensor = batch
        target = batch[:, :self.mask_channel+1, :, :, 1:self.total_training]
        
        x_preds = self(input_tensor, context_count=self.context_training, future_count=self.future_training,
                     sampling=sampling)
        
        if loss is None:
            return self.train_loss(labels=target, prediction=x_preds[...,:target.shape[-1]], 
                                   mask_channel=self.mask_channel)
        else:
            return loss(labels=target, prediction=x_preds[...,:target.shape[-1]], 
                        mask_channel = self.mask_channel)

    def configure_optimizers(self):        
        optimizer = get_opt_from_name(self.training_cfg["optimizer"],
                                      params=self.parameters(),
                                      lr=self.training_cfg["start_learn_rate"])
        scheduler = ReduceLROnPlateau(optimizer, 
                                      mode='min', 
                                      factor=self.training_cfg["lr_factor"], 
                                      patience=self.training_cfg["patience"],
                                      threshold=self.training_cfg["lr_threshold"],
                                      verbose=True)
        lr_sc = {
            'scheduler': scheduler,
            'monitor': 'val_loss'
        }
        return [optimizer] , [lr_sc]
    
    def training_step(self, batch, batch_idx):
        '''
        all_data of size (b, w, h, c, t)
            b = batch_size
            c = channels
            w = width
            h = height
            t = time
        '''
        if self.training_cfg["train_loss"] == 'Ensamble':
            l, ssim, mse, huber = self.batch_loss(batch, sampling=self.training_sampling, loss=self.train_loss)
        else:
            l = self.batch_loss(batch, sampling=self.training_sampling, loss=self.train_loss)
        self.log("train_loss", l, on_epoch=True, on_step=False)
        return l
    
    def validation_step(self, batch, batch_idx):
        '''
        all_data of size (b, w, h, c, t)
            b = batch_size
            c = channels
            w = width
            h = height
            t = time
        '''
        if self.training_cfg["test_loss"] == 'Ensamble':
            l, ssim, mse, huber = self.batch_loss(batch, sampling=self.training_sampling, loss=self.test_loss)
            metrics = {
                'val_loss': l,
                'SSIM': ssim,
                'MSE': mse,
                'HUBER': huber}
        else:
            l = self.batch_loss(batch, sampling=self.training_sampling, loss=self.test_loss)
            metrics = {
                'val_loss': l
            }
            
        self.log_dict(metrics, on_epoch=True, on_step=False)
        return l
    
    def test_step(self, batch, batch_idx):
        '''
        all_data of size (b, w, h, c, t)
            b = batch_size
            c = channels
            w = width
            h = height
            t = time
        '''
        # across all test sets 1/3 is context, 2/3 target
        _, l = self.batch_loss(batch, sampling=None, loss=self.test_loss)
        return l
