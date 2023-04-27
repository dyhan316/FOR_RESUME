#!/bin/bash
#SBATCH --job-name sex_1lr_baseline  #job name을 다르게 하기 위해서
#SBATCH -p volta
#SBATCH -t 72:00:00 #volta can only do four hours for voltadebug...
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
export MASTER_PORT=29135

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

#intelligence
#srun python main_multilabel.py --mode pretraining --framework yaware --ckpt_dir ${save_path}/ckpt_UKB_intel_10lr --batch_size 64 --dataset2use UKB --run_where SDCC --sigma 0.6176507443170335 --lr 1e-3 --lr_policy None --label_name intelligence #--lr_policy onecyclelr --



##multiple
#srun python main_multilabel.py --mode pretraining --framework yaware --ckpt_dir ${save_path}/ckpt_UKB_intel_age --batch_size 64 --dataset2use UKB --run_where SDCC --sigma 0.6176507443170335/0.6176507443170335 --lr 1e-4 --lr_policy None --label_name age/intelligence #--lr_policy onecyclelr --


## trying to change lr and so on to make intelligence go down (set to batch size 32!)
srun python main_multilabel_DALI_optuna.py --mode pretraining --framework yaware --ckpt_dir ${save_path}/ckpt_UKB_USE_THIS_DALI/ckpt_UKB_age_dali_test_64_optuna --batch_size 64 --dataset2use UKB --run_where SDCC  --label_name age #--lr 1e-5 #intelligence #--lr_policy custom_WR_2 #--lr_policy onecyclelr --


##trying 64 batc h for yAwrae intelligence thing
#srun python main_multilabel.py --mode pretraining --framework yaware --ckpt_dir ${save_path}/ckpt_UKB_intel_modify_Batchsize/ckpt_b64_lr_none --batch_size 64 --dataset2use UKB --run_where SDCC --lr 1e-4 --label_name intelligence #--lr_policy custom_WR_2 #--lr_policy onecyclelr --



##trying sex thing (batchsize 32)
#srun python main_multilabel.py --mode pretraining --framework yaware --ckpt_dir ${save_path}/ckpt_UKB_SEX_hyperparam_tuning/ckpt_lr_none --batch_size 32 --dataset2use UKB --run_where SDCC --lr 1e-4 --label_name sex #--lr_policy custom_WR_2 #--lr_policy onecyclelr --


