import sys
import os
from os.path import join
import pickle
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from core.models.ModelWrapperSampling import model_wrapper 
from Data.data_preparation import Dataset

def load_model(model_path = "trained_models/SGConvLSTM.ckpt"):
    """
        load a model from file for inference
    """
    model = model_wrapper.load_from_checkpoint(model_path, strict=False)
    model.eval()
    return model

def load_data_point(dataset_path, label='test', include_non_pred=True, context_ratio=0.9, index=0, target_count=1):
    list_path =  glob.glob(os.path.join(dataset_path, label, '*.npz'))
    dataset = Dataset(list_path, include_non_pred)
    # get cube
    truth = dataset.__getitem__(index)
    # cube splitting
    truth = truth.unsqueeze(dim=0)
    T = truth.shape[-1]
    t0 = int(T*context_ratio)
    context = truth[:, :, :, :, :t0] # b, c, h, w, t
    target = truth[:, :target_count+1, :, :, t0:] # b, c, h, w, t
    npf = truth[:, target_count+1:, :, :, t0:]
    name = os.path.splitext(os.path.basename(list_path[index]))[0]
    return truth, context, target, npf, name