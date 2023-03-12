# -*- coding: utf-8 -*- 
"""1. SETUP"""
import os
from REAL_step0_get_infant_adult_df import get_info, get_year_file
import pandas as pd

#get info templates
infant_pd, adult_pd = get_info()()


import shutil
import os
import subprocess


##SETUP THINGS
MRI_path = "/work2/08834/tg881334/stampede2/CHA_preproc/CHA_data_unzip/2010-2015/dicom_2010_2015"
#MRI_path = "/work2/08834/tg881334/stampede2/CHA_preproc/CHA_data_unzip/2016-2021/dicom_2016_2021"
#MRI_path = "/scratch/08834/tg881334/CHA_preproc/CHA_data/test_try_dcm2niix/sample_dataset/test_CHA_data_backup/haha"
#MRI_path = "/scratch/08834/tg881334/CHA_preproc/CHA_data/not_moved_data/not_moved_1/2010_2015_not_moved"
#MRI_path = "/scratch/08834/tg881334/CHA_preproc/CHA_data/not_moved_data/not_moved_2/2016_2021_not_moved"
#MRI_path = "/storage/bigdata/CHA_bigdata/download_CHA_other_years/CHA_age_seperation_testing/new_test_files/" #이미 normal, missing 두개다 되어있는 것 
age_list = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 13, 16, 18,20]

##dataframe concat
infant_df_list = []
adult_df_list = []
for year in infant_pd.keys():
    infant_df_list.append(infant_pd[year].rename(columns = {"Unnamed: 0" : "파일 유무"})) #Unnamed : 0 으로 되어있는 것들을 파일유무라는 것으로 rename을 다시하기
    
for year in adult_pd.keys():
    adult_df_list.append(adult_pd[year].rename(columns = {"Unnamed: 0" : "파일 유무"})) #Unnamed : 0 으로 되어있는 것들을 파일유무라는 것으로 rename을 다시하기

#concat the things
infant_df_total = pd.concat([i for i in infant_df_list])
adult_df_total = pd.concat([i for i in adult_df_list])
total_df = pd.concat([infant_df_total, adult_df_total])
total_set = set()
for year in range(2010, 2021+1):
    total_set = total_set.union(set(get_year_file(year)['영상파일 번호'].values))


    
   
"""2. 이상한 폴더들 해서 분리하는 것 하기"""
def find_wrong_files(path): 
    #path : folder path of the DICOM files
                        ##subject_df : T1, DWI 따로따로 가능하게 하기 (partition missing의 함수의 output들을 하나하나 넣을 수 있도록)
    
    survive_files = [] #survive한 파일들 
    missing_files = []
    empty_DTI_files = []
    empty_FSPGR_files = []
    multiple_sub_files = []
    no_tabular_data = []
    
    ##find weird files 
    for dicom_folders in os.listdir(path):
        abs_dicom_folders = os.path.join(path, dicom_folders)
        #if DTI and FSPGR aren't the only folders, get them out
        if set(os.listdir(abs_dicom_folders)) != set(["DTI", "FSPGR"]):
            print("DTI or FSPGR folder itself missing : ", dicom_folders)
            missing_files.append(abs_dicom_folders)
            continue #stop so that the next stage (checking empty files aren't performed)(skips to the next iteration)

        #if DTI and FSPGR folders are emtpy, get them out
        elif os.listdir(os.path.join(abs_dicom_folders, "DTI")) == []:
            print("DTI empty : ", dicom_folders)
            empty_DTI_files.append(abs_dicom_folders)
            continue
        
        elif os.listdir(os.path.join(abs_dicom_folders, "FSPGR")) == []:
            print("FSPGR empty : ", dicom_folders)
            empty_FSPGR_files.append(abs_dicom_folders)
            continue
        
        #if folder name is -1, -2 or etc, save it to another file
        elif dicom_folders.count('-') ==2:
            print("mulitple subject versions, but normal otherwise : ", dicom_folders)
            multiple_sub_files.append(abs_dicom_folders)
            continue

        #dicom 파일은 있는데 tabular이 없는 경우
        elif dicom_folders not in total_set:
            print("no tabular data exists : ", dicom_folders)
            no_tabular_data.append(abs_dicom_folders)
            continue
        
        #if 위의 관문들을 통과했으면, 남겨도 되는 폴더라고 생각하고 남길 것이다 (아니라면 continue에서 걸러짐)
        elif dicom_folders in set( total_df['영상파일 번호'].values): #weird quirkcs hahafuck
            survive_files.append(abs_dicom_folders)
        else :
            print("weird subject : ", dicom_folders) 

    ##dict 로 return해서 그 dict에다가 mv하는 함수 쓰기 
    return {"survived_files" : survive_files,"DTI_or_FSPGR_folder_missing": missing_files, "DTI_folder_missing":empty_DTI_files, 
            "FSPGR_folder_missing" : empty_FSPGR_files, "mulitple_vers_of_subject" : multiple_sub_files, "no_tabular_data" : no_tabular_data}
            # 이파트는 그냥.... 직접 내가 하자! 사람 많지도 않은데 

    
file_dict = find_wrong_files(MRI_path)

print("missing files")
print([len(i) for i in file_dict.values()])
print(file_dict['survived_files'])

"""3. 폴더들 옮기기!"""
save_path = os.path.join(str(MRI_path), "../final_extracted_files") #위 디렉토리에서 saved files디렉토리 만들어서 하ㅏㅈ하자 

os.mkdir(save_path)
for file_name in file_dict.keys():
    sub_file_dir = os.path.join(save_path, file_name)
    os.mkdir(sub_file_dir)
    for dicom_file in file_dict[file_name]:
        shutil.move(dicom_file, sub_file_dir)

###making folders by age first (which we will use to seperate by age )
####uncomment and use later 



sep_age_path = os.path.join(save_path, "seperated_by_age")  #path where the seperated by age is

os.mkdir(sep_age_path)

lower_age = 0 #deafult 값
for age in age_list:
    upper_age = age

    os.mkdir(os.path.join(sep_age_path,"age_{}_to_{}".format(lower_age, upper_age)))
    
    lower_age = age #다음번에는 이것을 lower age로 하기 

###uncomment and use later
#위에서 만든 age seperated file로 mv시키기
raw_path = os.path.join(save_path, "survived_files")
#for sub in os.listdir(raw_path):
for sub_path in file_dict['survived_files']:
    sub = sub_path[-10:] 
    sub_age = total_df[total_df['영상파일 번호'] == str(sub)]['age'].values[0]
    
    lower_age = 0
    for upper_age in age_list:
        if sub_age > lower_age and sub_age < upper_age:
            print(os.path.join(raw_path,sub))
            print(os.path.join(sep_age_path,"age_{}_to_{}".format(lower_age, upper_age)))
            shutil.move(os.path.join(raw_path,sub),os.path.join(sep_age_path,"age_{}_to_{}".format(lower_age, upper_age)))
            #print(sub, " : this age is it:{}".format(sub_age), upper_age)
        lower_age = upper_age

def get_all_metadata():
    return total_set

"""4. dcm2niix : 위에서 것들을 그대로 가져와서하기 + sub-형태로 이름 바꾸기"""

nii_path = os.path.join(MRI_path, "../2.dcm2niix")
[os.makedirs(os.path.join(nii_path,file_name)) for file_name in os.listdir(sep_age_path)]


for age_group in os.listdir(sep_age_path):
    print("================age range : {} has started================".format(age_group))
    for sub in os.listdir(os.path.join(sep_age_path, age_group)): 
        print(sub)

        nii_path_name = os.path.join(nii_path, age_group,sub)
        os.mkdir(nii_path_name)
        #running dcm2niix 
        subprocess.run('dcm2niix -b y -ba n -o {} -z y {}'.format(nii_path_name, os.path.join(sep_age_path, age_group, sub)), shell=True)
        
        ##now(밑에서부터) : dcm2niix밑의 결과들을 BIDS compliant하게 하는 범
        new_sub_id = 'sub-' + sub[2:4] + sub[6:]
        
        
        file_list = os.listdir(nii_path_name)
        
        if len(file_list) !=6: #i.e. some DTI or FSPGR folders are missing (error일어날때)
            print("this file is missing some files from dcm2niix : ", sub)
            shutil.move(nii_path_name, os.path.join(nii_path, "ERROR_dcm2niix", sub)) #rename the file to ERROR _sub_ddd or sth
            continue   #i.e. skip this iteration
        bval_ex = [x for x in file_list if ".bval" in x][0]
        
        DTI_list = [x for x in file_list if bval_ex[:-5] in x ]
        T1_list = list(set(file_list).difference(set(DTI_list))) #i.e. get the remaining things and save as T1_list

        ##making new names (BIDS-compliant)
        suffix_list = [".json", ".nii.gz", ".bval", ".bvec"]
        DTI_new_names = []
        for DTI_file in DTI_list:
            DTI_new_names.append([new_sub_id+"_dwi"+suffix for suffix in suffix_list if suffix in DTI_file][0])
        
        T1_new_names = []
        for T1_file in T1_list:
            T1_new_names.append([new_sub_id+"_T1w"+suffix for suffix in suffix_list if suffix in T1_file][0])

        
        ##creating folders and acutally moving into the while renaming
        #creaitng folders
        dwi_folder_pth = os.path.join(nii_path_name, "dwi")
        anat_folder_pth = os.path.join(nii_path_name, "anat")
        
        os.mkdir(dwi_folder_pth)
        os.mkdir(anat_folder_pth)
        
        #moving to the folders
        for i,file in enumerate(DTI_new_names):
            shutil.move(os.path.join(nii_path_name,DTI_list[i]), os.path.join(dwi_folder_pth, file)) #[i] 번째가 둘다 동일하기에 안전하다
        
        for i,file in enumerate(T1_new_names):
            shutil.move(os.path.join(nii_path_name,T1_list[i]), os.path.join(anat_folder_pth, file)) #[i] 번째가 둘다 동일하기에 안전하다       
        print("========================")
        #break #몇개만 해보고 싶어서
