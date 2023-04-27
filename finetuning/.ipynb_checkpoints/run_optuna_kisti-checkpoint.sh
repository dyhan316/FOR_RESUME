# #!/bin/bash

# #SBATCH --job-name CHA  #job name을 다르게 하기 위해서
# #SBATCH -t 96:00:00 #volta can only do four hours for voltadebug...
# #SBATCH -N 1
# #SBATCH --gres=gpu:1 #how many gpus each job array should have 
# #SBATCH --ntasks=1 #여기서부터는 내가 추가
# #SBATCH -o ./shell_output/output_%A_%a.output
# #SBATCH -e ./shell_output/error_%A_%a.error
# #SBATCH --array=0-2 #upto (number of tasks) -1  there are 
# #SBATCH --cpus-per-task=16 #같이 하는게 훨씬 빠름(?)(test해보기.. .전에 넣은것이랑 비교해서)
# #SBATCH --mem-per-cpu=4GB


# ##강제로 기다리기
# ##SBATCH --nodelist=node3
# sleep_list=( 1 30 60 90 120 ) #( 1e-3 1e-4 1e-5 ) #default : 1e-4, but found out that that is too small for freeze
# sleep_time="${sleep_list[${SLURM_ARRAY_TASK_ID}]}"

# sleep $sleep_time #sleep for this much time 

resource=kisti
save_path=finetune_RESULTS/CHA_ASDGDD
wandb_name=test1
weight=./weights/UKByAa64a.pth 

dataset=CHA
task=CHA_ASDGDD 
stratify=balan_iter_strat

# train_num="--train_num 237" #maximum for iter_strat of CHA 
n_workers=3
epochs=100
batch=32 
scheduler=cosine_annealing_decay
patience=20
prune=False

# change stratify to balan_iter_strat
python3 main_optuna.py --run_where $resource --save_path $save_path --wandb_name $wandb_name --pretrained_path $weight \
--dataset $dataset --task $task --stratify $stratify $train_num --num_workers $n_workers \
--layer_control tune_all  --random_seed 0 --input_option yAware --batch_size $batch --binary_class True --early_criteria loss \
--lr_schedule $scheduler --lr_range 2e-4/2.5e-4 --wd_range 2e-2/1e0 --AMP True --prune $prune --nb_epochs 100 --patience 20