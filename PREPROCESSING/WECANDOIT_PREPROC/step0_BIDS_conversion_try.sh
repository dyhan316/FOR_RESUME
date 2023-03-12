#!/bin/bash

###################################################
######Extracting the inputs tha will be used#######
###################################################

#dicom_data_input= #/scratch/connectome/dyhan316/WECANDOIT/try_input_two_subjects
dicom_data_input='/storage/bigdata/WECANDOIT/1.raw_v2'
extracted_inputs_save_dir='/storage/bigdata/WECANDOIT/2.extracted_inputs_v2' #output directory

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

    if [[ "${files}" == *"T1"* ]] || [[ "${files}" == *"TASK"* ]] || [[ "${files}" == *"gre"* ]] || [[ "${files}" == *"DTI_BLIP_LR" ]] || [[ "${files}" == *"DTI_BLIP_RL" ]] || [[ "${files}" == *"LR_B1000_ac-pc" ]] || [[ "${files}" == *"LR_B1000" ]] ||[[ "${files}" == *"LR_B2000" ]] || [[ "${files}" == *"LR_B3000" ]]
    then cp -r  $files $sub_out

    fi
    done

done


####################################################################################################################
##########NOW that the proper inputs (extracted inputs) are made, let's change them into NIfTI and do BIDS###########
#####################################################################################################################


input_dir=$extracted_inputs_save_dir
output_dir='/storage/bigdata/WECANDOIT/3.BIDS_raw_v2' #output directory


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

    mrcat -nthreads 20 ${sub}/*DTI_MB3_LR_B1000*/ ${sub}/*DTI_MB3_LR_B2000/ ${sub}/*_DTI_MB3_LR_B3000/\
 - | mrconvert -nthreads 20 - ${sub_out}/tmp/sub-${sub_idx}_acq-mb3multishell_dir-LR_dwi.nii.gz -json_export ${sub_out}/tmp/multishell.json -export_grad_fsl ${sub_out}/tmp/bvecs ${sub_out}/tmp/bvals
    echo `dcm2niix -b y -ba n -o ${sub_out}/tmp -z y "${sub}/"` #save to the sub_out folder

    for files in $sub_out/tmp/* #do DICOM to niix for EACH SUBJECT
    do
###################### anat ######################

    # && : if the one in the front is successful, proceed to the next
    # conditional statement [ $a == $b ] requires spaces between all stuff

    # t1
    if [[ "${files}" == *"T1"* ]] && [[ "${files}" == *".json"* ]]
    then
        mv $files "${sub_out}/anat/sub-${sub_idx}_T1w.json"

    elif [[ "${files}" == *"T1"* ]] && [[ "${files}" == *".nii.gz"* ]]
    then
        mv $files "${sub_out}/anat/sub-${sub_idx}_T1w.nii.gz"

###################### dwi ######################
    ################multishell####### (move the previous things to the tmp folder thing)
    elif [[ "${files}" == *"multishell.json"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-mb3multishell_dir-LR_dwi.json"

    elif [[ "${files}" == *"mb3multishell"* ]] && [[ "${files}" == *".nii.gz"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-mb3multishell_dir-LR_dwi.nii.gz"

    elif [[ "${files}" == *"bvals"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-mb3multishell_dir-LR_dwi.bval"

    elif [[ "${files}" == *"bvecs"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-mb3multishell_dir-LR_dwi.bvec"


    ################ blip
    elif [[ "${files}" == *"BLIP_RL"* ]] && [[ "${files}" == *".json"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-blip_dir-RL_dwi.json"

    elif [[ "${files}" == *"BLIP_RL"* ]] && [[ "${files}" == *".nii.gz"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-blip_dir-RL_dwi.nii.gz"

    elif [[ "${files}" == *"BLIP_RL"* ]] && [[ "${files}" == *".bval"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-blip_dir-RL_dwi.bval"

    elif [[ "${files}" == *"BLIP_RL"* ]] && [[ "${files}" == *".bvec"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-blip_dir-RL_dwi.bvec"

    elif [[ "${files}" == *"BLIP_LR"* ]] && [[ "${files}" == *".json"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-blip_dir-LR_dwi.json"

    elif [[ "${files}" == *"BLIP_LR"* ]] && [[ "${files}" == *".nii.gz"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-blip_dir-LR_dwi.nii.gz"

    elif [[ "${files}" == *"BLIP_LR"* ]] && [[ "${files}" == *".bval"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-blip_dir-LR_dwi.bval"

    elif [[ "${files}" == *"BLIP_LR"* ]] && [[ "${files}" == *".bvec"* ]]
    then
        mv $files "${sub_out}/dwi/sub-${sub_idx}_acq-blip_dir-LR_dwi.bvec"

###################### func;memory ######################
    echo "========================================================="
    elif [[ "${files}" == *"1TASK"* ]] &&[[ "${files}" == *".nii.gz"* ]]
    then
        mv $files "${sub_out}/func/sub-${sub_idx}_task-memory_run-1_bold.nii.gz"

    elif [[ "${files}" == *"1TASK"* ]] &&[[ "${files}" == *".json"* ]]
    then
        mv $files "${sub_out}/func/sub-${sub_idx}_task-memory_run-1_bold.json"

    elif [[ "${files}" == *"2TASK"* ]] &&[[ "${files}" == *".nii.gz"* ]]
    then
        mv $files "${sub_out}/func/sub-${sub_idx}_task-memory_run-2_bold.nii.gz"

    elif [[ "${files}" == *"2TASK"* ]] &&[[ "${files}" == *".json"* ]]
    then
        mv $files "${sub_out}/func/sub-${sub_idx}_task-memory_run-2_bold.json"

    elif [[ "${files}" == *"3TASK"* ]] &&[[ "${files}" == *".nii.gz"* ]]
    then
        mv $files "${sub_out}/func/sub-${sub_idx}_task-memory_run-3_bold.nii.gz"

    elif [[ "${files}" == *"3TASK"* ]] &&[[ "${files}" == *".json"* ]]
    then
        mv $files "${sub_out}/func/sub-${sub_idx}_task-memory_run-3_bold.json"

    elif [[ "${files}" == *"4TASK"* ]] &&[[ "${files}" == *".nii.gz"* ]]
    then
        mv $files "${sub_out}/func/sub-${sub_idx}_task-memory_run-4_bold.nii.gz"

    elif  [[ "${files}" == *"4TASK"* ]] &&[[ "${files}" == *".json"* ]]
    then
        mv $files "${sub_out}/func/sub-${sub_idx}_task-memory_run-4_bold.json"

###################### func;learning ######################
    elif [[ "${files}" == *"TASK1"* ]] &&[[ "${files}" == *".nii.gz"* ]]
        then
            mv $files "${sub_out}/func/sub-${sub_idx}_task-learning_run-1_bold.nii.gz"

    elif  [[ "${files}" == *"TASK1"* ]] &&[[ "${files}" == *".json"* ]]
        then
            mv $files "${sub_out}/func/sub-${sub_idx}_task-learning_run-1_bold.json"

    elif [[ "${files}" == *"TASK2"* ]] &&[[ "${files}" == *".nii.gz"* ]]
        then
            mv $files "${sub_out}/func/sub-${sub_idx}_task-learning_run-2_bold.nii.gz"

    elif  [[ "${files}" == *"TASK2"* ]] &&[[ "${files}" == *".json"* ]]
        then
            mv $files "${sub_out}/func/sub-${sub_idx}_task-learning_run-2_bold.json"

    elif [[ "${files}" == *"TASK3"* ]] &&[[ "${files}" == *".nii.gz"* ]]
        then
            mv $files "${sub_out}/func/sub-${sub_idx}_task-learning_run-3_bold.nii.gz"

    elif  [[ "${files}" == *"TASK3"* ]] &&[[ "${files}" == *".json"* ]]
        then
            mv $files "${sub_out}/func/sub-${sub_idx}_task-learning_run-3_bold.json"

    elif [[ "${files}" == *"TASK4"* ]] &&[[ "${files}" == *".nii.gz"* ]]
        then
            mv $files "${sub_out}/func/sub-${sub_idx}_task-learning_run-4_bold.nii.gz"

    elif  [[ "${files}" == *"TASK4"* ]] &&[[ "${files}" == *".json"* ]]
        then
            mv $files "${sub_out}/func/sub-${sub_idx}_task-learning_run-4_bold.json"

#######################STUDY HERE!!! DON'T KNOW WHAT TO DO HERE#########

###################### fmap ######################
#fmap these are weird names so gotta be careful
    elif [[ "${files}" == *"_12_ph"* ]] && [[ "${files}" == *"json"* ]]
        then
            mv $files "${sub_out}/fmap/sub-${sub_idx}_dir-AP_epi.json"

    elif [[ "${files}" == *"_12_ph"* ]] && [[ "${files}" == *"nii.gz"* ]]
        then
            mv $files "${sub_out}/fmap/sub-${sub_idx}_dir-AP_epi.nii.gz"

    elif [[ "${files}" == *"_14_ph"* ]] && [[ "${files}" == *"json"* ]]
        then
            mv $files "${sub_out}/fmap/sub-${sub_idx}_dir-PA_epi.json"

    elif [[ "${files}" == *"_14_ph"* ]] && [[ "${files}" == *"nii.gz"* ]]
        then
            mv $files "${sub_out}/fmap/sub-${sub_idx}_dir-PA_epi.nii.gz"
    fi
    done
rm -r ${sub_out}/fmap
rm -r ${sub_out}/func
rm -r ${sub_out}/tmp
done
