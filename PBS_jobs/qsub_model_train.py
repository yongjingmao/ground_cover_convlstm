import sys, os
import argparse
import pickle
import glob
import datetime
from rsc.utils import metadb
from rsc.utils import batchque



"""
This script sets and submits jobs for long-time (>100 hours) model training 
by submitting multiple jobs with dependency set
"""

def submitJob(dep_job_list):
    
    """
    Set parameters
    """
    work_path = "/scratch/rsc8/yongjingm/ConvLSTM_GBRCA"
    container_path = "/scratch/containers/denhamr/tfgpu.sif"
    venv_path = "/scratch/rsc8/yongjingm/envs/convlstm/bin/python3"
    script_path = "/scratch/rsc8/yongjingm/Github/ground_cover_convlstm/scripts"
    walltime = 96
    
    target_folder = "{}/Sites/SiteMerge".format(work_path)
    pickle_dir = "{}/Sites/SiteMerge/merged_train_path.pickle".format(work_path)

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
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
        
    jobdir = os.path.join(target_folder, 'Jobscript')
    
    if not os.path.isdir(jobdir):
        os.mkdir(jobdir)
    
    """
    Glob file paths
    """
    training_paths = glob.glob('Sites/Site*/data/train/*.npz')
    testing_paths = glob.glob('Sites/Site*/data/test/*.npz')
    validation_paths = glob.glob('Sites/Site*/data/val/*.npz')
    all_paths = {
        'train': training_paths,
        'test': testing_paths,
        'val': validation_paths
    }
    
    with open(pickle_dir, 'wb') as handle:
        pickle.dump(all_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    """
    Set command lines
    """
        
    cmd1 = "source /usr/local/bin/pbs_set_cuda_env.bash"
    cmd2 = "cd {}".format(work_path)
    cmd3 = "module load singularity"
    cmd4 = "singularity exec {} {} \
    {}/config.py -wd {} -pd {} -mt {} -ts {} \
    -iw {} -ih {} -nl 3 -hc 64 -mc 1 -npc Y -oc 1 -ic {} -bs 4 -e 1000 \
    -ft {} -ct {} -fv {}".format(
        container_path, venv_path, script_path,
        target_folder, pickle_dir, model_name, sampling, img_width, img_height,
        input_channel, context, context, context)
    
    cmd5 = "singularity exec --nv {} {} {}/model_train.py -wd {}".format(
        container_path, venv_path, script_path, target_folder)

    
    """
    Write and submit jobscript
    """
    jobscript = os.path.join(jobdir, '{}_Merge.sh'.format(model_name))
    if len(dep_job_list) == 0:
        logfile = os.path.join(jobdir, '{}_Merge.log'.format(model_name))
    else:
        logfile = os.path.join(jobdir, '{}_Merge_after_{}.log'.format(model_name, dep_job_list[-1]))
    

    jobstring = '\n'.join([cmd1, cmd2, cmd3, cmd4, cmd5])
    batchque.makeJobScript(
        jobstring, jobscript, logfile=logfile,jobname="{}_Merge".format(
            model_name), 
        walltime=walltime, ncpus=1, memsize=8, ngpus=1, gpumem=32,
        job_dependency_list = dep_job_list
    )        

    if len(dep_job_list)>0:
        job_cmd = "qsub -W depend=afternotok:{} {}".format(':'.join(dep_job_list), os.path.abspath(jobscript))
    else:
        job_cmd = "qsub {}".format(os.path.abspath(jobscript))
    print(job_cmd)
    jobid = batchque.submitJobCmd(job_cmd)
    
    return(jobid)

jobids = []
num_jobs = 5 # Number of training jobs started sequentially
for i in range(num_jobs):
    jobid = submitJob(jobids)
    jobids.append(jobid.split('.')[0])
    print(jobid)