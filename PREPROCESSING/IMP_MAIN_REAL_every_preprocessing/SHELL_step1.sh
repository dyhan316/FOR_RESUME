#!/bin/bash
#SBATCH --job-name=BIDS_conversion_first_batch
#SBATCH --partition=long #normal can only last 24 hrs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=96:00:00
#SBATCH --output=./shell_output/dcm2niix_output%A_%x.out
#SBATCH --error=./shell_output/dcm2niix_error%A_%x.err
#SBATCH --mail-user=dyhan0316@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=TG-IBN180001
#SBATCH --cpus-per-task=30

python REAL_step1_dcm2niix_BIDS.py

