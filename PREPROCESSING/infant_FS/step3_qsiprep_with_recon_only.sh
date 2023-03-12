#! /bin/bash

#SBATCH -J qsiprep_icx-normal#name of the job #SBATCH -N 2  #24 sub per node, so for 128 subjects, 5.01 neede,d so doing 6  #how many nodes you need #SBATCH -n $1  #how many jobs 
#SBATCH -p icx-normal #skx-dev for faster allocation#flat-quadrant #the queue on stampede 2 to use
#SBATCH -o ./step3_shell_outputs/icx-normal_QSIPREP_preproc.o%j  #change according to you job name
#SBATCH -e ./step3_shell_outputs/icx-normal_QSIPREP_preproc.e%j #change according to you job name
#SBATCH -t 48:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 160
##SBATCH –-mail-user=dyhan0316@gmail.com

##this is run by itself 

echo "parallelize this later!"

echo "hi"

data_pth=/scratch/08834/tg881334/2019/2.BIDS/

sub_save_dir=/scratch/08834/tg881334/2019/5.qsiprep_results/
supp_dir=/work2/08834/tg881334/stampede2/CHA_preproc/CHA_preproc_supplementary_files
sing_img_dir=${supp_dir}/qsiprep-0.16.1.sif
fs_dir=/scratch/08834/tg881334/2019/4.FS_untar

module load tacc-singularity
#change nthreads per omp to like 4 
#infant when 4.5~5 등등
singularity run --cleanenv -B ${data_pth}/:/data:ro,${sub_save_dir}/:/out,${supp_dir}:/freesurfer,${fs_dir}:/fs_input ${sing_img_dir} /data /out participant -w /out/tmp --output_resolution 1.2 --denoise_after_combining --unringing_method mrdegibbs --b0_to_t1w_transform Affine --intramodal_template_transform SyN --fs-license-file /freesurfer/license.txt --skip_bids_validation --freesurfer-input /fs_input --recon-spec mrtrix_multishell_msmt_ACT-hsvs --omp-nthreads 4 --recon-only  #treid recononly






#singularity run --cleanenv -B ${data_pth}/:/data:ro,${sub_save_dir}/:/out,${supp_dir}:/freesurfer ${sing_img_dir} /data /out participant -w /out/tmp --output_resolution 1.2 --denoise_after_combining --unringing_method mrdegibbs --b0_to_t1w_transform Affine --intramodal_template_transform SyN --fs-license-file /freesurfer/license.txt --skip_bids_validation --freesurfer-input $fs_dir ----recon-spec mrtrix_multishell_msmt_ACT-hsvs --omp-nthreads 32
