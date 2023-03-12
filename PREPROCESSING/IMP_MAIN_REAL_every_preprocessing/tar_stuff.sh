#!/bin/bash
#SBATCH --job-name=BIDS_conversion_first_batch
#SBATCH --partition=flat-quadrant #normal can only last 24 hrs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=./tardcm2niix_output%A_%x.out
#SBATCH --error=./tardcm2niix_error%A_%x.err
#SBATCH --mail-user=dyhan0316@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=TG-IBN180001
#SBATCH --cpus-per-task=30


base_dir=/scratch/08834/tg881334/T2_new/fmriresults01
tar -cvf $base_dir/NDARINV0-5.tar $base_dir/NDARINV0-5
tar -cvf $base_dir/NDARINV6-9.tar $base_dir/NDARINV6-9
tar -cvf $base_dir/NDARINVA-D.tar $base_dir/NDARINVA-D
tar -cvf $base_dir/NDARINVE-K.tar $base_dir/NDARINVE-K
tar -cvf $base_dir/NDARINVL-O.tar $base_dir/NDARINVL-O
tar -cvf $base_dir/NDARINVP-Z.tar $base_dir/NDARINVP-Z


#plot_loss(ckpt_lambda['lambda_1percent'],label = "batch 64, $\lambda$ 1%")
