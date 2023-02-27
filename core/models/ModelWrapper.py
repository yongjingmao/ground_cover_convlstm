import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .model_parts.AutoencLSTM import AutoencLSTM
from .model_parts.Conv_Transformer import Conv_Transformer_npf
from .model_parts.Conv_LSTM import Conv_LSTM
from .model_parts.Pred_RNN import Pred_RNN
from ..losses import get_loss_from_name
from ..optimizers import get_opt_from_name


class model_wrapper(pl.LightningModule):
    def __init__(self, model_type, model_cfg, training_cfg):
        """
        This is the supermodel that wraps all the possible models to do satellite image forecasting. 
        It uses the model (specified by 'model_type') that it wraps to predict a deviation onto the 
        specified baseline (specified in 'self.training_cfg["baseline"]').

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
        elif model_type == "AutoencLSTM":
            self.model = AutoencLSTM(input_dim=self.model_cfg["input_channels"],
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
        elif model_type == "ConvTransformer":
            self.model = Conv_Transformer_npf(num_hidden=self.model_cfg["num_hidden"],
                                              output_dim=self.model_cfg["output_channels"],
                                              depth=self.model_cfg["depth"],
                                              dilation_rate=self.model_cfg["dilation_rate"],
                                              num_conv_layers=self.model_cfg["num_conv_layers"],
                                              kernel_size=self.model_cfg["kernel_size"],
                                              img_width=self.model_cfg["img_width"],
                                              non_pred_channels=self.model_cfg["non_pred_channels"],
                                              in_channels=self.model_cfg["in_channels"],
                                              mask_channel = self.model_cfg["mask_channel"],
                                              baseline=self.training_cfg["baseline"])
            
        self.baseline = self.training_cfg["baseline"]
        self.future_training = self.training_cfg["future_training"]
        self.future_validation = self.training_cfg["future_validation"]
        self.learning_rate = self.training_cfg["start_learn_rate"]
        self.mask_channel = self.model_cfg["mask_channel"]
        self.train_loss = get_loss_from_name(self.training_cfg["train_loss"])
        self.test_loss = get_loss_from_name(self.training_cfg["test_loss"])


    def forward(self, x, prediction_count=1, non_pred_feat=None):
        """
        :param x: All features of the input time steps.
        :param prediction_count: The amount of time steps that should be predicted all at once.
        :param non_pred_feat: Only need if prediction_count > 1. All features that are not predicted
            by the model for all the future time steps we want to predict.
        :return: preds: Full predicted images.
        :return: predicted deltas: Predicted deltas with respect to baselines.
        :return: baselines: All future baselines as computed by the predicted deltas. Note: These are NOT the ground truth baselines.
        Do not use these for computing a loss!
        """

        preds, pred_deltas, baselines = self.model(x, non_pred_feat=non_pred_feat, prediction_count=prediction_count)

        return preds, pred_deltas, baselines

    def batch_loss(self, batch, mask_channel, t_future=20, loss=None):
        cmc = mask_channel # cloud_mask channel
        T = batch.size()[4]
        t0 = T - t_future
        context = batch[:, :, :, :, :t0]       # b, c, h, w, t
        target = batch[:, :cmc + 1, :, :, t0:] # b, c, h, w, t
        npf = batch[:, cmc + 1:, :, :, t0:]

        x_preds, _, _ = self(context, prediction_count=T-t0, non_pred_feat=npf)
        
        if loss is None:
            return self.train_loss(labels=target, prediction=x_preds, mask_channel=cmc)
        else:
            return loss(labels=target, prediction=x_preds, mask_channel = cmc)

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
            l, ssim, mse, huber = self.batch_loss(batch, self.mask_channel, t_future=self.future_training, loss=self.train_loss)
        else:
            l = self.batch_loss(batch, self.mask_channel, t_future=self.future_training, loss=self.train_loss)
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
        if self.training_cfg["test_loss"] == 'ENS':
            v_loss = self.batch_loss(batch, self.mask_channel, t_future=self.future_validation, loss=self.test_loss)
            l = v_loss[0]
            metrics = {
                'val_loss': v_loss[0],
                'MAD': v_loss[1],
                'SSIM': v_loss[2],
                'OLS': v_loss[3],
                'EMD': v_loss[4]}
        elif self.training_cfg["test_loss"] == 'Ensamble':
            l, ssim, mse, huber = self.batch_loss(batch, self.mask_channel, t_future=self.future_validation, loss=self.test_loss)
            metrics = {
                'val_loss': l,
                'SSIM': ssim,
                'MSE': mse,
                'HUBER': huber}
        else:
            l = self.batch_loss(batch, self.mask_channel, t_future=self.future_validation, loss=self.test_loss)
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
        _, l = self.batch_loss(batch, self.mask_channel, t_future=self.future_validation, loss=self.test_loss)
        return l
