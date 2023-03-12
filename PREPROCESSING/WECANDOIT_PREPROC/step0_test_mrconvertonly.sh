#!/bin/bash

###################################################
######Extracting the inputs tha will be used#######
###################################################

#dicom_data_input= #/scratch/connectome/dyhan316/WECANDOIT/try_input_two_subjects
dicom_data_input='/storage/bigdata/WECANDOIT/1.raw_test_jan'
extracted_inputs_save_dir='/storage/bigdata/WECANDOIT/2.extracted_inputs_test_jan_2' #output directory

for before in $dicom_data_input/*         #possible because we already made subject folders in the output
do
    sub=${before}/study_1*
    sub_idx="${before:(-3)}" #subject number #extracts the last three (ex : 101 from A101)

    sub_out=${extracted_inputs_save_dir}/A${sub_idx}
    mkdir -p ${sub_out}

    echo $sub
    for files in $sub/* #do DICOM to niix for EACH SUBJECT
    do
###################### anat ######################

  
    # && : if the one in the front is successful, proceed to the next
    # conditional statement [ $a == $b ] requires spaces between all stuff

    if [[ "${files}" == *"T1"* ]] || [[ "${files}" == *"TASK"* ]] || [[ "${files}" == *"gre"* ]] || [[ "${files}" == *"LR_B1000" ]] || [[ "${files}" == *"LR_B2000" ]] || [[ "${files}" == *"LR_B3000" ]] || [[ "${files}" == *"DTI_BLIP_LR" ]] || [[ "${files}" == *"DTI_BLIP_RL" ]] 
    then cp -r  $files $sub_out

    fi
    done

done


####################################################################################################################
##########NOW that the proper inputs (extracted inputs) are made, let's change them into NIfTI and do BIDS###########
#####################################################################################################################


input_dir=$extracted_inputs_save_dir
output_dir='/storage/bigdata/WECANDOIT/3.BIDS_raw_test_jan_2' #output directory


for sub in $input_dir/*
do
    sub_idx="${sub:(-3)}"
    mkdir -p "${output_dir}/sub-${sub_idx}"
done


for sub in $input_dir/*         #possible because we already made subject folders in the output
do
    sub_idx="${sub:(-3)}" #subject number #extracts the last three (ex : 101 from A101)

###################### 
    echo $sub			# this sub folder has the absolute path for each stuff!

    sub_out="${output_dir}/sub-${sub_idx}" #output directory for this specific subject
    mkdir "${sub_out}/anat"
    mkdir "${sub_out}/dwi"
    mkdir "${sub_out}/func"
    mkdir "${sub_out}/fmap"
    mkdir "${sub_out}/tmp"
######################

    #check which B1000 to use
    #if ${sub
    mrconvert -nthreads 20 ${sub}/*DTI_MB3_LR_B1000 ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b1000_dir-LR_dwi.nii.gz -json_export ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b1000_dir-LR_dwi.json -export_grad_fsl ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b1000_dir-LR_dwi.bvecs ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b1000_dir-LR_dwi.bval
    mrconvert -nthreads 20 ${sub}/*DTI_MB3_LR_B2000 ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b2000_dir-LR_dwi.nii.gz -json_export ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b2000_dir-LR_dwi.json -export_grad_fsl ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b2000_dir-LR_dwi.bvecs ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b2000_dir-LR_dwi.bval
    mrconvert -nthreads 20 ${sub}/*DTI_MB3_LR_B3000 ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b3000_dir-LR_dwi.nii.gz -json_export ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b3000_dir-LR_dwi.json -export_grad_fsl ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b3000_dir-LR_dwi.bvecs ${sub_out}/dwi/sub-${sub_idx}_acq-mb3_singleshell_b3000_dir-LR_dwi.bval
    mrcat -nthreads 20 ${sub}/*DTI_MB3_LR_B1000/ ${sub}/*DTI_MB3_LR_B2000/ ${sub}/*_DTI_MB3_LR_B3000/\
    - | mrconvert -nthreads 20 - ${sub_out}/tmp/sub-${sub_idx}_acq-mb3multishell_dir-LR_dwi.nii.gz -json_export ${sub_out}/tmp/multishell.json -export_grad_fsl ${sub_out}/tmp/bvecs ${sub_out}/tmp/bvals
    #dcm2niix -b y -ba n -o ${sub_out}/tmp -z y "${sub}/"
    echo `dcm2niix -b y -ba n -o ${sub_out}/tmp -z y "${sub}/"` #save to the sub_out folder
    
done