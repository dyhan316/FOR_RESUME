#!/bin/bash
#SBATCH --job-name sex_1lr_baseline  #job name을 다르게 하기 위해서
#SBATCH -p volta
#SBATCH -t 00:40:00 #volta can only do four hours for voltadebug...
#SBATCH --gres=gpu:2 #how many gpus each job array should have 
#SBATCH --gpus-per-node=2 #2 if DDP
#SBATCH --chdir=../
#SBATCH -o sdcc_logs/%j-%x.out
#SBATCH -e sdcc_logs/%j-%x.err
#SBATCH -c 16
#SBATCH --mem-per-cpu=6GB

set +x

export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=29012

#export WORLD_SIZE=2


#will use 16 cpus!! (num of workers)
# -C : constraints 
#-n : ntasks
#-c : --cpus-per-task
#-G : --gpus-per-task

#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
#conda activate 3DCNN

#env | grep SLURM
save_path=/hpcgpfs01/scratch/dyhan316/yAware_results_save/


export PYTHONUNBUFFERED=TRUE #performance penallty may arise  (https://docs.ccv.brown.edu/oscar/software/python-in-batch-jobs) 


srun python main_multilabel_DALI.py --mode pretraining --framework yaware --ckpt_dir ${save_path}/ckpt_TEST_TEST_2 --batch_size 8 --dataset2use ABCD --run_where SDCC --lr 1e-4 # --label_name intelligence #--lr_policy custom_WR_2 #--

#srun python main_multilabel_DALI.py --mode pretraining --framework yaware --ckpt_dir ${save_path}/ckpt_TEST_TEST --batch_size 32 --dataset2use ABCD --run_where SDCC --lr 1e-4 # --label_name intelligence #--lr_policy custom_WR_2 #--lr_policy onecyclelr --
