#!/bin/bash
#SBATCH --job-name=BIDS_conversion_first_batch
#SBATCH --partition=flat-quadrant #normal can only last 24 hrs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=./shell_output/dcm2niix_output%A_%x.out
#SBATCH --error=./shell_output/dcm2niix_error%A_%x.err
#SBATCH --mail-user=dyhan0316@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=TG-IBN180001
#SBATCH --cpus-per-task=30


python REAL_step3_group_by_50.py
#python REAL_step3_BIDS_one_per_sub.py
#REAL_step3_BIDS_one_per_sub.py
