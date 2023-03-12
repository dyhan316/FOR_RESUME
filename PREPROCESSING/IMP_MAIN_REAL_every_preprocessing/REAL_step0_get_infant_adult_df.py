# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("ASSUME THAT PREPROC_LATER AND STUFF => 이미 옮겼다고 가정!")

#tabular_path = '/storage/bigdata/CHA_bigdata/scripts_IMP/CHA_tabular_data/' #이거는 ver 1, 밑에 missing_tabular_path가 ver2여서
#밑의 것만 봐도 된다
missing_tabular_path = '../CHA_tabular_data/missing/'

###DO NOT USE TABULAR_PATH!! IT'S THE FIRST VERSION! DON'T WANT 중복
##missing_tabular_list 얻기
missing_tab_file_list = os.listdir(missing_tabular_path)



def get_year_file(year):
    for i, file_name in enumerate(missing_tab_file_list): #file돌리며 해당 연도 파일 찾기 
        if str(year) in file_name:
            file_dir = missing_tabular_path + file_name

    total_df = pd.read_excel(file_dir)
    return total_df


###define function to iterate over for removing NaN and extracting age
def partition_missing(df_file):
    file = df_file
    try : 
        file = file.dropna(subset=['파일 유무'])  #excel에서 원래 hiding함
        
        no_MRI_filter = (file['파일 유무'] != 'X')

    except : 
        print("'파일유무' didn't exist, so removing based on Unnamed: 0 instead")
        file = file.dropna(subset = ['Unnamed: 0'])
        
        no_MRI_filter = (file['Unnamed: 0'] != 'X')
    file = file[no_MRI_filter]  
    ##change to things to numeric ('1',1 두개가 다있더라..)
    file["MRI"] = pd.to_numeric(file["MRI"])
    file["DTI"] = pd.to_numeric(file["DTI"])
    
    MRI_mask, DTI_mask = file["MRI"].values == 1, file["DTI"].values == 1
    
    file_both = file[MRI_mask &  DTI_mask]
    file_MRI =  file[MRI_mask & ~ DTI_mask]
    file_DTI =  file[~MRI_mask & DTI_mask]
    file_neither = file[~MRI_mask & ~DTI_mask]
    
    return file_both, file_MRI, file_DTI, file_neither

def seperate_by_year(file, year= 4.5):
    ##get age
    mri_date = file['MRI(DTI) 시행 날짜']
    birth_date = file['Date of birth']
    
    #change age to pandas dtype : datetime64 (to get change in dates to get age)
    mri_date = pd.to_datetime(mri_date)
    birth_date = pd.to_datetime(birth_date)
    
    #compute ages
    age_days = mri_date - birth_date #age in days
    age_years = age_days/np.timedelta64(1, 'Y')
    
    ##append age_years to the dataset
    file['age'] = age_years
    
    infant_mask = file['age'] <= year
    adult_mask = file['age'] > year
    
    
    return file[infant_mask], file[adult_mask]

infant_pd = {}
adult_pd = {}


class get_info():
    def __init__(self):
        pass
     
    
    def __call__(self):
        for year in range(2010, 2021+1):
            sep_missing = partition_missing(get_year_file(year))
    
            both = sep_missing[0] #file that has both MRI and DTI 
    
            sep_both = seperate_by_year(both) #seperate by year
    
            infant_pd[str(year)] = sep_both[0]
            adult_pd[str(year)] = sep_both[1]
            print("year {} finished".format(year))
        return infant_pd, adult_pd
    #infant_pd

