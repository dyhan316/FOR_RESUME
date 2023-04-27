#!/bin/bash
#SBATCH --job-name CHA_yAwa  #job name을 다르게 하기 위해서
#SBATCH -p volta
#SBATCH -t 48:00:00 #volta can only do four hours for voltadebug...
#SBATCH --chdir=../
#SBATCH -o sdcc_logs/%j-%x.out
#SBATCH -e sdcc_logs/%j-%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 #number of tasks (DDP rnaks) to use PER NODE
#SBATCH --gpus-per-node=2 #MUST BE SAME AS ntasks-per-node!! (because task갯수만큼 gpu만들어야하니)
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6GB
set +x

export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=29502


#will use 16 cpus!! (num of workers)
# -C : constraints 
#-n : ntasks
#-c : --cpus-per-task
#-G : --gpus-per-task

#source /global/common/software/nersc/shasta2105/python/3.8-anaconda-2021.05/etc/profile.d/conda.sh
#conda activate 3DCNN

#env | grep SLURM
save_path=/hpcgpfs01/scratch/dyhan316/yAware_results_save/
#srun python main.py --mode pretraining --framework yaware --ckpt_dir $save_path/ckpt_CHA_sex_basline --batch_size 64 --dataset2use CHA_secHC --run_where SDCC --label_name sex

#for age 
srun python main.py --mode pretraining --framework yaware --ckpt_dir ./ckpt_CHA_10sigma_onecyclelr --batch_size 64 --dataset2use CHA_secHC --run_where SDCC --sigma 0.6176507443170335 --lr #--lr_policy onecyclelr --

#srun python main.py --mode pretraining --framework yaware --ckpt_dir ./ckpt_CHA_default --batch_size 64 --data CHA_secHC #--sigma 0.006176507443170335

