#!/bin/bash
#SBATCH -A m3898_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=4
#SBATCH --exclusive
#SBATCH --account m3898_g
set +x

# -C : constraints 
#-n : ntasks
#-c : --cpus-per-task
#-G : --gpus-per-task

source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
conda activate 3DCNN


python main.py --mode pretraining --framework simclr --ckpt_dir ./checkpoint_simclr
