#!/bin/bash
#SBATCH -A m3898_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH --nodes 32
#SBATCH --ntasks-per-node=4
#SBATCH -c 10
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J UKBya64c_s15
#SBATCH --chdir=../
#SBATCH -o logs/%j-%x.out
#SBATCH --mail-user=kjb961013@snu.ac.kr
#SBATCH --mail-type=BEGIN
set +x

# -C : constraints 
#-n : ntasks
#-c : --cpus-per-task
#-G : --gpus-per-task

source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
conda activate 3DCNN

env | grep SLURM

srun python main.py --mode pretraining --framework yaware --ckpt_dir ./checkpoint_yaware_only_cutout_steplr_age_s15 --tb_dir ./tb_yaware_only_cutout_steplr_age_s15 --batch_size 64 --tf cutout --label_name age --lr_policy step --lr_decay_iters 20 --gamma 0.5 --sigma 15

