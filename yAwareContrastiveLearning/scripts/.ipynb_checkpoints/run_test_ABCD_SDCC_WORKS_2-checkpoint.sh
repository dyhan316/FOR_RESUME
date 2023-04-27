#!/bin/bash
#SBATCH --job-name ABCD_TEST  #job name을 다르게 하기 위해서
#SBATCH -p volta
#SBATCH -t 00:10:00 #volta can only do four hours for voltadebug...
#SBATCH --chdir=../
#SBATCH -o sdcc_logs/%j-%x.out
#SBATCH -e sdcc_logs/%j-%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 #number of tasks (DDP rnaks) to use PER NODE
#SBATCH --gres=gpu:2 #MUST BE SAME AS ntasks-per-node! (b/c its' gpu per node이다) #gpu-per-task는 slurm안되더라
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6GB

set +x

export MASTER_ADDR=`/bin/hostname -s`
export MASTER_PORT=29012 


#여기 + 준범쌤 modified : https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904 
#export WORLD_SIZE=2 #지정안해줘도 slurm에서 찾아서 하더라 (modified due to python script)

#env | grep SLURM
save_path=/hpcgpfs01/scratch/dyhan316/yAware_results_save/


export PYTHONUNBUFFERED=TRUE #performance penallty may arise  (https://docs.ccv.brown.edu/oscar/software/python-in-batch-jobs) 


srun python main_multilabel_DALI.py --mode pretraining --framework yaware --ckpt_dir ${save_path}/ckpt_TEST_TEST_2 --batch_size 32 --dataset2use ABCD --run_where SDCC 


###example of doing multinode job (2 nodes, 2 task (gpu) per node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2 #number of tasks (DDP rnaks) to use PER NODE
#SBATCH --gres=gpu:2 #MUST BE SAME AS ntasks-per-node! (gpu per node이다) #gpu-per-task는 slurm안되더라

##note that the --gres optoin was TWO!! (becasue tasks per node was two)