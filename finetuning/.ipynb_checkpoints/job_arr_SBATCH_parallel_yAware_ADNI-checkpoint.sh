#!/bin/bash

#SBATCH --job-name pll_yfreeze  #job name을 다르게 하기 위해서
#SBATCH --nodes=1
#SBATCH --nodelist=node3 #used node4
#SBATCH -t 48:00:00 # Time for running job #길게 10일넘게 잡음
#SBATCH -o ./shell_output/output_%A_%a.output
#SBATCH -e ./shell_output/error_%A_%a.error
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4 #upto (number of tasks) -1  there are

#sometimes not doing stratify =balan  could be important, as 그렇게하면 majority case가 minority case의 갯수만큼 되도록 강제로 sample을 버리니

##trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/)
base_dir=/scratch/connectome/study_group/VAE_ADHD/junbeom_weights
weight_list=( None ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 



task_list=( ADNI_ALZ_ADCN ADNI_ALZ_ADMCI ADNI_ALZ_CNMCI ADNI_sex ADNI_age )


task="$task_list[${SLURM_ARRAY_TASK_ID}]" #select the corresponding jobarray part from the list of tasks to do 




for weight in "${weight_list[@]}"
do 
python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 10 --layer_control freeze --stratify balan --random_seed 0 --task $task --input_option yAware --learning_rate 1e-3 &\
python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 30 --layer_control freeze --stratify balan --random_seed 0 --task $task --input_option yAware --learning_rate 1e-3 &\
python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 50 --layer_control freeze --stratify balan --random_seed 0 --task $task --input_option yAware --learning_rate 1e-3 &\
python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 100 --layer_control freeze --stratify balan --random_seed 0 --task $task --input_option yAware --learning_rate 1e-3 &\
python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 200 --layer_control freeze --stratify balan --random_seed 0 --task $task  --input_option yAware --learning_rate 1e-3 #now do tuneall vary train_num

done 


#remove NEW_ later

#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 10 --layer_control tune_all --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 30 --layer_control tune_all --stratify balan --random_seed 0 --task $task

#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 50 --layer_control tune_all --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 100 --layer_control tune_all --stratify balan --random_seed 0 --task $task

#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 200 --layer_control tune_all --stratify balan --random_seed 0 --task $task 