#!/bin/bash
#SBATCH -A m3898_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH --nodes 8
#SBATCH --ntasks-per-node=4
#SBATCH -c 10
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J simc_steplr
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

srun python main.py --mode pretraining --framework simclr --ckpt_dir ./checkpoint_simclr_only_cutout_steplr_220607 --tb_dir ./tb_simclr_only_cutout_steplr_220607 --batch_size 64 --tf cutout --label_name age --lr_policy step

