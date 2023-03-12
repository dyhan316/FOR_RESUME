#!/bin/bash

#SBATCH --job-name WECANDOIT_recon_only #job name을 다르게 하기 위해서
#SBATCH --nodes=1
#SBATCH --nodelist=node3
#SBATCH -t 360:00:00 # Time for running job
#SBATCH -o /scratch/connectome/dyhan316/WECANDOIT/try_stuff/output_%J.out #%j : job id 가 들어가는 것
#SBATCH -e /scratch/connectome/dyhan316/WECANDOIT/try_stuff/error_%J.error
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=500MB
#SBATCH --cpus-per-task=40

docker run --rm -v /scratch/connectome/dyhan316/dwMRI/license.txt:/opt/freesurfer/license.txt:ro\
 -v /storage/bigdata/WECANDOIT/3.BIDS_raw_v2:/data:ro\
 -v /storage/bigdata/WECANDOIT/5.qsiprep_results_REAL_v2:/qsiprep-output:ro\
 -v /storage/bigdata/WECANDOIT/5.qsiprep_results_REAL_v2/freesurfer:/sngl/freesurfer-input:ro\
 -v /storage/bigdata/WECANDOIT/5.qsiprep_results_REAL_v2:/out pennbbl/qsiprep:0.15.4\
 /data /out participant --recon-input /qsiprep-output --freesurfer-input /sngl/freesurfer-input\
 --recon-spec mrtrix_multishell_msmt_ACT-hsvs --output_resolution 1.2 --denoise_after_combining\
 --unringing_method mrdegibbs --b0_to_t1w_transform Affine --intramodal_template_transform SyN\
 --nthreads 40 --skip_bids_validation --recon_only
