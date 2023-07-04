#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=16gb
#PBS -l walltime=10:00:00
#PBS -j eo
#PBS -r n
#PBS -N data_retrieve


cd /scratch/rsc8/yongjingm/Github/ground_cover_convlstm/scripts

module load singularity
SINGULARITY_IMAGE=/scratch/containers/devenv_rsc_latest.sif
singularity exec --bind /sdata $SINGULARITY_IMAGE /scratch/rsc8/yongjingm/envs/data_fusion/bin/python3 retrieve_AWO.py -wd /scratch/rsc8/yongjingm/WaterOutlook
singularity exec --bind /sdata $SINGULARITY_IMAGE /scratch/rsc8/yongjingm/envs/data_fusion/bin/python3 retrieve_SILO.py -wd /scratch/rsc8/yongjingm/SILO
singularity exec --bind /sdata $SINGULARITY_IMAGE /scratch/rsc8/yongjingm/envs/data_fusion/bin/python3 preprocess_AUX.py -wd /scratch/rsc8/yongjingm/Auxiliary\ data
