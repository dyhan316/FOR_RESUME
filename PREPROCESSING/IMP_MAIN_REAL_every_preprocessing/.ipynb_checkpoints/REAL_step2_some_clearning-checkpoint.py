import os
import shutil
import subprocess
import sys
print("DO NOT RUN THIS UNLESS ABSOLUTELY NECESSARY (remove exit below if u're gonna run it)")
sys.exit()

##remove all the DICOM folders for ones that were converted to nii, and only leave their subject name folders
#dicom_folder_dir = "/work2/08834/tg881334/stampede2/CHA_preproc/CHA_data_unzip/2016-2021/final_extracted_files/seperated_by_age"
dicom_folder_dir = "/work2/08834/tg881334/stampede2/CHA_preproc/CHA_data_unzip/2010-2015/final_extracted_files/seperated_by_age"
#dcm2niix_path = "/work2/08834/tg881334/stampede2/CHA_preproc/CHA_data_unzip/2016-2021/2.dcm2niix"
dcm2niix_path = "/work2/08834/tg881334/stampede2/CHA_preproc/CHA_data_unzip/2010-2015/2.dcm2niix"

subprocess.run('rm -r {}/*/20*/*'.format(dicom_folder_dir), shell = True)

##rename the subjects into sub-XXX format (forgot to do this in step1)

for age_group in os.listdir(dcm2niix_path):
    print("doing age group : ", age_group)
    for sub in os.listdir(os.path.join(dcm2niix_path, age_group)):
        #print(sub)
        sub_path = os.path.join(dcm2niix_path, age_group, sub)
        #print(os.listdir(sub_path))
        new_name = 'sub-' + sub[2:4] + sub[6:]
        
        new_name_path = os.path.join(dcm2niix_path, age_group, new_name)
        #print(new_name_path)
        shutil.move(sub_path, new_name_path)
    
