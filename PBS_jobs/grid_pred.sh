#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=16gb
#PBS -l walltime=20:00:00
#PBS -l ngpus=1
#PBS -l cudadrv=1
#PBS -l gpumem=16gb
#PBS -j eo
#PBS -r n
#PBS -N grid_pred

if [ -r /usr/local/bin/pbs_set_cuda_env.bash ]; then
  . /usr/local/bin/pbs_set_cuda_env.bash
fi


# Run application using the GPU
source /usr/local/bin/pbs_set_cuda_env.bash
cd /scratch/rsc8/yongjingm/Github/ground_cover_convlstm/scripts
module load singularity
singularity exec --nv /scratch/containers/denhamr/tfgpu.sif /scratch/rsc8/yongjingm/envs/convlstm/bin/python3 model_mosaic.py -tw 1280 -th 1280 -wd /scratch/rsc8/yongjingm/Github/ground_cover_convlstm/predicted_maps
    

