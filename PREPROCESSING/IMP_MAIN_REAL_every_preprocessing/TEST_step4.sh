#!/bin/bash


#SBATCH -J run_qsiprep_preproc#name of the job #SBATCH -N 2  #24 sub per node, so for 128 subjects, 5.01 neede,d so doing 6  #how many nodes you need #SBATCH -n $1  #how many jobs 
#SBATCH -p skx-dev #skx-dev for faster allocation#flat-quadrant #the queue on stampede 2 to use
#SBATCH -o ./shell_output/skx_normal_QSIPREP_preproc.o%j  #change according to you job name
#SBATCH -e ./shell_output/skx_normal_QSIPREP_preproc.e%j #change according to you job name
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 96
##SBATCH –-mail-user=dyhan0316@gmail.com

echo "hi"
#data_pth=/scratch/08834/tg881334/CHA_preproc/2.age_9_20 #/scratch/08834/tg881334/CHA_preproc_try/2.one_BIDS
data_pth=/scratch/08834/tg881334/CHA_preproc/2.BIDS_data/age_7_to_9
#sub_save_dir=/scratch/08834/tg881334/CHA_preproc_try/3.qsipreproc_results_2
sub_save_dir=/scratch/08834/tg881334/CHA_preproc/3.qsipreproc_results_2
supp_dir=/work2/08834/tg881334/stampede2/CHA_preproc/CHA_preproc_supplementary_files
sing_img_dir=${supp_dir}/qsiprep-0.14.3.sif


module load tacc-singularity

#infant when 4.5~5 등등
singularity run --cleanenv -B ${data_pth}/:/data:ro,${sub_save_dir}/:/out,${supp_dir}:/freesurfer ${sing_img_dir} /data /out participant -w /out/tmp --output_resolution 1.2 --denoise_after_combining --unringing_method mrdegibbs --b0_to_t1w_transform Affine --intramodal_template_transform SyN --do_reconall --fs-license-file /freesurfer/license.txt --skip_bids_validation --omp-nthreads 4
