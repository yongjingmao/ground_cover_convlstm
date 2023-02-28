import argparse
import os
import json
from operator import index
from os.path import join

def train_line_parser():
    # load default args from json

    # parse settable args from terminal
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-wd', '--work_dict', type=str, help='work dictory')
    parser.add_argument('-pd', '--pickle_dir', type=str, default=None, help='directory with the desired pickle files')
    parser.add_argument('-mt', '--model_type', type=str, default='ConvLSTM', choices=['ConvLSTM', 'PredRNN','AutoencLSTM', 'ConvTransformer'], 
                        help='type of model architecture')
    parser.add_argument('-ts', "--training_sampling", type=str, default='None', help='future steps for training')
    
    parser.add_argument('-iw', '--img_width', type=int, default=None, help='number of layers')
    parser.add_argument('-ih', '--img_height', type=int, default=None, help='number of layers')
    parser.add_argument('-nl', '--num_layers', type=int, default=None, help='number of layers')
    parser.add_argument('-hc', '--hidden_channels', type=int, default=None, help='number of hidden channels')
    parser.add_argument('-mc', '--mask_channel', type=int, default=None, help='index of mask_channel')
    parser.add_argument('-npc', '--include_non_pred', type=str, default=None, help='whether include non predicting variable')
    parser.add_argument('-oc', '--output_channels', type=int, default=None, help='number of output variable')
    parser.add_argument('-ic', '--input_channels', type=int, default=None, help='number of input variable')
    
    parser.add_argument('-bs', '--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='starting learning rate')
    parser.add_argument('-e',  '--epochs', type=int, default=None, help='training epochs')
    
    parser.add_argument('-ft', '--future_training', type=int, default=None, help='future steps for training')
    parser.add_argument('-ct', '--context_training', type=int, default=None, help='context steps for training')
    parser.add_argument('-fv', '--future_validation', type=int, default=None, help='future steps for validation')
    
    parser.add_argument('-tl', '--train_loss', type=str, default='l2', choices=['l1','l2','Huber', 'Ensamble'], help='loss function used for training')
    parser.add_argument('-vl', '--validation_loss', type=str, default='Ensamble', choices=['ENS','Ensamble'], help='loss function used for validation/testing')
    
    parser.add_argument('-bm', '--big_memory', type=str, default="N", help='big memory or small: t = ture, f = false')
    parser.add_argument('-ph', '--peephole', type=str, default="Y", help='use peephole: t = ture, f = false')  
    parser.add_argument('-ln', '--layer_norm', type=str, default="Y", help='layer normalization: t = true, f = false')
    parser.add_argument('-k',  '--kernel_size', type=int, default=5, help='convolution kernel size')
    parser.add_argument('-mk', '--mem_kernel_size', type=int, default=5, help='memory kernel size')
    parser.add_argument('-dl', '--dilation_rate', type=int, default=1, help='diliation rate')
        
    parser.add_argument('-lf', '--learning_factor', type=float, default=0.1, help='learning rate factor')
    parser.add_argument('-lrt','--lr_threshold', type=float, default=0.001, help='learning rate threshold')
    parser.add_argument('-p',  '--patience', type=int, default=100, help='patience')
    parser.add_argument('-pr', '--precision', type=int, default=32,choices=[16,32,64], help='bit precision')
    parser.add_argument('-bf', '--baseline_function', type=str, default='zeros', choices=['mean_cube', 'last_frame', 'zeros'], help='baseline function')
    parser.add_argument('-cp', '--checkpoint', type=str, default=None, help='checkpoint to continue from')
    parser.add_argument('-es', '--early_stopping', type=str, default="Y", help='whether implement eartly stopping')

    args = parser.parse_args()
    
    return args
    
if __name__ == '__main__':

    args = train_line_parser()
    cfg_training = {}
    cfg_model = {}
    
    if not os.path.exists(args.work_dict):
        os.mkdir(args.work_dict)    

    if args.batch_size is not None:
        cfg_training["train_batch_size"] = args.batch_size
        cfg_training["val_batch_size"] = args.batch_size
        cfg_training["test_batch_size"] = args.batch_size

    if args.big_memory is not None:
        if args.big_memory == "y" or args.big_memory == "Y" or args.big_memory == "T" or args.big_memory == "t":
            cfg_model["big_mem"] = True
        elif args.big_memory == "n" or args.big_memory == "N" or args.big_memory == "f" or args.big_memory == "F":
            cfg_model["big_mem"] = False
            
    if args.peephole is not None:
        if args.peephole == "y" or args.peephole == "Y" or args.peephole == "T" or args.peephole == "t":
            cfg_model["peephole"] = True
        elif args.peephole == "n" or args.peephole == "N" or args.peephole == "f" or args.peephole == "F":
            cfg_model["peephole"] = False
    
    if args.layer_norm is not None:
        if args.layer_norm == "y" or args.layer_norm == "Y" or args.layer_norm == "T" or args.layer_norm == "t":
            cfg_model["layer_norm"] = True
        elif args.layer_norm == "n" or args.layer_norm == "N" or args.layer_norm == "f" or args.layer_norm == "F":
            cfg_model["layer_norm"] = False
            
    if args.early_stopping is not None:
        if args.early_stopping == "y" or args.early_stopping == "Y" or args.early_stopping == "T" or args.early_stopping == "t":
            cfg_training["early_stopping"] = True
        elif args.early_stopping == "n" or args.early_stopping == "N" or args.early_stopping == "f" or args.early_stopping == "F":
            cfg_training["early_stopping"] = False
    
    if args.include_non_pred is not None:
        if args.include_non_pred == "y" or args.include_non_pred == "Y" or args.include_non_pred == "T" or args.include_non_pred == "t":
            cfg_training["include_non_pred"] = True
        elif args.include_non_pred == "n" or args.include_non_pred == "N" or args.include_non_pred == "f" or args.include_non_pred == "F":
            cfg_training["include_non_pred"] = False

    if args.img_width is not None:
        cfg_model["img_width"] = args.img_width
        
    if args.img_height is not None:
        cfg_model["img_height"] = args.img_height
        
    if args.hidden_channels is not None:
        cfg_model["hidden_channels"] = args.hidden_channels
    
    if args.input_channels is not None:
        if cfg_training["include_non_pred"]:
            cfg_model["input_channels"] = args.input_channels
        else:
            cfg_model["input_channels"] = args.mask_channel+1
        
    if args.output_channels is not None:
        cfg_model["output_channels"] = args.output_channels
    
    if args.mask_channel is not None:
        cfg_model["mask_channel"] = args.mask_channel
    
    if args.kernel_size is not None:
        cfg_model["kernel"] = args.kernel_size

    if args.mem_kernel_size is not None:
        cfg_model["memory_kernel"] = args.mem_kernel_size
    
    if args.dilation_rate is not None:
        cfg_model["dilation_rate"] = args.dilation_rate

    if args.num_layers is not None:
        cfg_model["n_layers"] = args.num_layers

    if args.future_training is not None:
        cfg_training["future_training"] = args.future_training
        
    if args.context_training is not None:
        cfg_training["context_training"] = args.context_training
        
    if args.future_validation is not None:
        cfg_training["future_validation"] = args.future_validation

    if args.learning_rate is not None:
        cfg_training["start_learn_rate"] = args.learning_rate

    if args.learning_factor is not None:
        cfg_training["lr_factor"] = args.learning_factor
    
    if args.lr_threshold is not None:
        cfg_training["lr_threshold"] = args.lr_threshold

    if args.patience is not None:
        cfg_training["patience"] = args.patience

    if args.precision is not None:
        cfg_training["precision"] = args.precision

    if args.epochs is not None:
        cfg_training["epochs"] = args.epochs

    if args.baseline_function is not None:
        cfg_training["baseline"] = args.baseline_function

    if args.train_loss is not None:
        cfg_training["train_loss"] = args.train_loss

    if args.validation_loss is not None:
        cfg_training["test_loss"] = args.validation_loss
        
    if args.pickle_dir is not None:
        cfg_training["pickle_dir"] = args.pickle_dir
        
    
    model_type = args.model_type
    #cfg_training["checkpoint"] = args.checkpoint
    cfg_training["training_sampling"] = args.training_sampling
    cfg_training["project_name"] = "{}_{}".format(model_type, args.training_sampling)
    cfg_training["optimizer"] = 'adamW'
    cfg_training["accelerator"] = 'auto'
    
    
    if not os.path.exists(args.work_dict + "/config"):
        os.mkdir(args.work_dict + "/config")
    
    with open(args.work_dict + "/config/Training.json", 'w') as f:
        json.dump(cfg_training, f, sort_keys=True, indent=4)
        
    with open(args.work_dict + "/config/" + args.model_type + ".json", 'w') as f:
        json.dump(cfg_model, f, sort_keys=True, indent=4)
    

