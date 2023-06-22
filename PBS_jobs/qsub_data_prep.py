import sys, os
import argparse
import pickle
import glob
import datetime
from rsc.utils import metadb
from rsc.utils import batchque


"""
This script submits jobs to run set_train_data.py for different sites parallelly
"""


def submitJob(Site):
    
    """
    Set parameters
    """
    work_path = "/scratch/rsc8/yongjingm/ConvLSTM_GBRCA"
    container_path = "/scratch/containers/denhamr/tfgpu.sif"
    venv_path = "/scratch/rsc8/yongjingm/envs/convlstm/bin/python3"
    script_path = "/scratch/rsc8/yongjingm/Github/ground_cover_convlstm/scripts"
    
    walltime = 10
    
    AOI_index = Site
    target_folder = "{}/Sites/Site{}".format(work_path, AOI_index)
    pickle_dir = "{}/Sites/Site{}/merged_train_path.pickle".format(work_path, AOI_index)


    img_width = 128
    img_height = 128
    window_length = 16
    aux_vars = ['rainfall', 'temperature', 'soilmoisture', 'runoff']
    
    model_name = "PredRNN"
    sampling = "rs"
    input_channel = 2+len(aux_vars)

    context = int(window_length/2) 
    
    """
    Create folder
    """
    if not os.path.exists(os.path.join(work_path, 'Sites')):
        os.mkdir(os.path.join(work_path, 'Sites'))
    
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
        
    jobdir = os.path.join(target_folder, 'Jobscript')
    if not os.path.isdir(jobdir):
        os.mkdir(jobdir)
        
    """
    Set command lines
    """
    
    cmd1 = "cd {}".format(work_path)
    cmd2 = "module load singularity"
    
    if AOI_index < 50:
        train_ratio = 1
        val_test_ratio = 0
    elif AOI_index < 75:
        train_ratio = 0
        val_test_ratio = 0
    else:
        train_ratio = 0
        val_test_ratio = 1

    cmd3 = "singularity exec {} {} {}/set_train_data.py -wd {} --Site {} -l {} -iw {} -ih {} -tr {} -vtr {}".format(
        container_path, venv_path, script_path, work_path, AOI_index, window_length, 
        img_width, img_height, train_ratio, val_test_ratio)
    
    for aux_var in aux_vars:
        cmd3 += ' --{}'.format(aux_var)

    """
    Write and submit jobscript
    """
    
    jobscript = os.path.join(jobdir, '{}_{}.sh'.format(model_name, AOI_index))
    logfile = os.path.join(jobdir, '{}_{}.log'.format(model_name, AOI_index))
    
    jobstring = '\n'.join([cmd1, cmd2, cmd3])
    batchque.makeJobScript(
        jobstring, jobscript, logfile=logfile,jobname="{}_{}".format(
            model_name, AOI_index), 
        walltime=1, ncpus=1, memsize=4)
    
    job_cmd = "qsub {}".format(os.path.abspath(jobscript))
    jobid = batchque.submitJobCmd(job_cmd)
    
    return(jobid)


for site in range(100):
    jobid = submitJob(site)
    print(jobid)