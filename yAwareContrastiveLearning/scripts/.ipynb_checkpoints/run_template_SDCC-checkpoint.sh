#!/bin/bash
#SBATCH --job-name sex_1lr_baseline  #job name을 다르게 하기 위해서
#SBATCH -p volta
#SBATCH -t 72:00:00 #volta can only do four hours for voltadebug...
#SBATCH --gpus-per-node=2 #2 if DDP
#SBATCH --ntasks-per-node=2
#SBATCH --chdir=../
#SBATCH -o sdcc_logs/%j-%x.out
#SBATCH -e sdcc_logs/%j-%x.err

#SBATCH -c 8
#SBATCH --mem-per-cpu=6GB

set +x

export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=29500

#env | grep SLURM
save_path=/hpcgpfs01/scratch/dyhan316/yAware_results_save/

srun python main_multilabel.py --mode pretraining --framework yaware --ckpt_dir $save_path/ckpt_TEST_DDP_batch_32_with_sync --batch_size 32 --dataset2use test --run_where SDCC --wandb_name test --lr 1e-3 --sigma 0.05