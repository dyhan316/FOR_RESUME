#! /bin/bash

#SBATCH --job-name QSIPREP_preproc_WECANDOIT  #job name을 다르게 하기 위해서
#SBATCH --nodes=1
#SBATCH --nodelist=node3 #used node4
#SBATCH -t 360:00:00 # Time for running job #길게 10일넘게 잡음
#SBATCH -o /scratch/connectome/dyhan316/WECANDOIT/output_%J.out #%j : job id 가 들어가는 것
#SBATCH -e /scratch/connectome/dyhan316/WECANDOIT/error_%J.error
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=500MB
#SBATCH --cpus-per-task=30


# uses older versino of qsiprep because new versino of qsiprep doesn't have reconall, and doing reconall seperately results in dwi T1 registration errors.... 

docker run --rm --name WECANDOIT_fs_reconstruction \
-v /storage/bigdata/WECANDOIT/3.BIDS_raw_v2/:/data:ro \
-v /storage/bigdata/WECANDOIT/5.qsiprep_results_REAL_v2:/out \
-v /scratch/connectome/dyhan316/dwMRI:/freesurfer \
pennbbl/qsiprep:0.14.3 \
/data \
/out \
participant \
-w /out/tmp \
--output_resolution 1.2 \
--denoise_after_combining \
--unringing_method mrdegibbs \
--b0_to_t1w_transform Affine \
--intramodal_template_transform SyN \
--do_reconall \
--fs-license-file /freesurfer/license.txt \
--skip_bids_validation \
--nthreads 30 \
