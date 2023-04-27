#!/bin/bash

#SBATCH --job-name parallel_finetune  #job name을 다르게 하기 위해서
#SBATCH --nodes=1
#SBATCH --nodelist=node3 #used node4
#SBATCH -t 48:00:00 # Time for running job #길게 10일넘게 잡음
#SBATCH -o ./shell_output/output_%J.output
#SBATCH -e ./shell_output/error_%J.error
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16

#sometimes not doing stratify =balan  could be important, as 그렇게하면 majority case가 minority case의 갯수만큼 되도록 강제로 sample을 버리니


#task=test #for testing
##trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/)
base_dir=/scratch/connectome/study_group/VAE_ADHD/junbeom_weights
weight_list=( None ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 

task_list=( ADNI_ALZ_ADCN ADNI_ALZ_ADMCI ADNI_ALZ_CNMCI ADNI_sex ADNI_age )


for task in "${task_list[@]}"
do
for weight in "${weight_list[@]}"
do 
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 10 --layer_control freeze --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 30 --layer_control freeze --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 50 --layer_control freeze --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 100 --layer_control freeze --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 200 --layer_control freeze --stratify balan --random_seed 0 --task $task &\ #now do tuneall vary train_num
python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 10 --layer_control tune_all --stratify balan --random_seed 0 --task $task &\
python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 30 --layer_control tune_all --stratify balan --random_seed 0 --task $task &\
python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 50 --layer_control tune_all --stratify balan --random_seed 0 --task $task &\
python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 100 --layer_control tune_all --stratify balan --random_seed 0 --task $task &\
python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 200 --layer_control tune_all --stratify balan --random_seed 0 --task $task 
done 
done


#task=ADNI_ALZ_ADCN
#
###trying to do it using for loops (from https://www.cyberciti.biz/faq/unix-linux-iterate-over-a-variable-range-of-numbers-in-bash/)
#base_dir=/scratch/connectome/study_group/VAE_ADHD/junbeom_weights
#weight_list=( ${base_dir}/UKBsim32a.pth ${base_dir}/UKBsim64c.pth ${base_dir}/UKByAa32a.pth ${base_dir}/UKByAa64c.pth  ${base_dir}/UKByAg64c.pth ${base_dir}/UKByAs64c.pth ${base_dir}/UKBsim64a.pth  ${base_dir}/UKBsim+yAa64c.pth  ${base_dir}/UKByAa64a.pth  ${base_dir}/UKByAa64cS15.pth  ${base_dir}/UKByAi64c.pth ) 
#
#
#for weight in "${weight_list[@]}"
#do 
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 10 --layer_control freeze --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 20 --layer_control freeze --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 40 --layer_control freeze --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 80 --layer_control freeze --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 100 --layer_control freeze --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 200 --layer_control freeze --stratify balan --random_seed 0 --task $task &\ #now do tuneall vary train_num
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 10 --layer_control tune_all --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 20 --layer_control tune_all --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 40 --layer_control tune_all --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 80 --layer_control tune_all --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 100 --layer_control tune_all --stratify balan --random_seed 0 --task $task &\
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 200 --layer_control tune_all --stratify balan --random_seed 0 --task $task 
#done 

##tuneall vary train_num


#
###freeze vary batchsize
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 100 --layer_control freeze --stratify balan --random_seed 0 --task $task --batch-size 32
#
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 100 --layer_control freeze --stratify balan --random_seed 0 --task $task --batch-size 16
#

###just temporary trash
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 20 --layer_control freeze --stratify balan --random_seed 0 --task $task
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 40 --layer_control freeze --stratify balan --random_seed 0 --task $task
#python3 main_DEBUG.py --pretrained_path $weight --mode finetuning --train_num 80 --layer_control freeze --stratify balan --random_seed 0 --task $task
#

##freeze vary transfomration
##freeze vary learning rate
##freeze vary wegiht decay
##freeze vary task 
##등등 하기 
