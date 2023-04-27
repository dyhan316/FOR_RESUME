#!/bin/bash
#SBATCH -t 240:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --gpus-per-node=1 #2 might also work..
#SBATCH -J lr_5e-5 #custom_4
#SBATCH --chdir=../
#SBATCH -o logs/%j-%x.out
#SBATCH -e logs/%j-%x.err

set +x

echo "this is past lab server version.. probs won't work.... "

#will use 16 cpus!! (num of workers)
# -C : constraints 
#-n : ntasks
#-c : --cpus-per-task
#-G : --gpus-per-task

#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
#conda activate 3DCNN

#env | grep SLURM


#try later with 64!! ans 128!!

srun python main_multilabel.py  --mode pretraining --framework yaware --ckpt_dir ./ckpt_different_lr_policy/ckpt_intelligence_lr_5e-5 --batch_size 32 --label_name intelligence --lr 5e-5 #--lr_policy custom_WR_4


