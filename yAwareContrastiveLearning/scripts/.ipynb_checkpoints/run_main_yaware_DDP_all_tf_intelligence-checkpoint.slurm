#!/bin/bash
#SBATCH -A m4244_g
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 2:00:00
#SBATCH --nodes 8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH --exclusive
#SBATCH --chdir=../
#SBATCH -o logs/%j-%x.out
#SBATCH --mail-user=kjb961013@snu.ac.kr
#SBATCH --mail-type=BEGIN
set +x

# -C : constraints 
#-n : ntasks
#-c : --cpus-per-task
#-G : --gpus-per-task

#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
#conda activate 3DCNN
module load pytorch/1.13.1

env | grep SLURM

srun python main.py --mode pretraining --framework yaware --ckpt_dir ./checkpoint_yaware_all_tf_step_intelligence_rescaled --tb_dir ./tb_yaware_all_tf_cutout_step_intelligence_rescaled --batch_size 64 --tf all_tf --label_name intelligence --lr_policy step --sigma 5 --gamma 0.5


