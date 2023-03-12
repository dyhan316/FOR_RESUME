#! /bin/bash

input_dir=/work2/08834/tg881334/stampede2/CHA_preproc/2010-2021_acutal_preproc_files/2.dcm2niix/$1
save_dir=/scratch/08834/tg881334/CHA_preproc/2.one_BIDS/$1
json_dir=/work2/08834/tg881334/stampede2/CHA_preproc/CHA_preproc_supplementary_files/dataset_description.json

#IMPLEMENT_doing it over the whole data (not providiing $1 inputs)

echo $save_dir
mkdir $save_dir

for sub in $input_dir/sub-*
do
    echo $sub
    sub_name="${sub:(-10)}"
    one_BIDS=$save_dir/$sub_name
    mkdir $one_BIDS
    cp $json_dir $one_BIDS
    cp -r $sub $one_BIDS


done

