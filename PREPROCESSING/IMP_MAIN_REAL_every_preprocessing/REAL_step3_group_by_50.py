import os
import shutil


import subprocess
import numpy as np


import pdb
file_dir = '/work2/08834/tg881334/stampede2/CHA_preproc/2010-2021_acutal_preproc_files/2.dcm2niix/'
things_to_run = [i for i in os.listdir(file_dir) if i[:3] == "age"] #extract list of BIDS folders (ageXX)

save_dir = '/scratch/08834/tg881334/CHA_preproc/2.BIDS/'


for i,age_group in enumerate(things_to_run):
    #print(i)
    sub_list = os.listdir(file_dir+age_group)
    num_sub = len(sub_list)
    low_age, high_age = age_group.split('_')[1], age_group.split('_')[3]
    
    if num_sub > 50:
        num_divide = np.ceil(num_sub/50.) #. because float
        sub_split = np.array_split(np.array(sub_list),num_divide)
        for i,sub_sub_list in enumerate(sub_split):
            new_sub_group =  save_dir + age_group + '_' + str(i)
            os.makedirs(new_sub_group, exist_ok = True)
            subprocess.run('cp {}/dataset_description.json {}'.format(file_dir, new_sub_group), shell = True)
            for sub in sub_sub_list:
                ha = 3
                #print(sub)
                subprocess.run('cp -r {}/{}/{} {}'.format(file_dir,age_group,sub,new_sub_group) , shell = True)
            
        
    else:
        if low_age == '7' or low_age == '8':
            new_sub_group = save_dir+"age_7_to_9"
            os.makedirs(new_sub_group, exist_ok = True)
            subprocess.run('cp {}/dataset_description.json {}'.format(file_dir, new_sub_group), shell = True)
            subprocess.run('cp -r {}/{}/* {}'.format(file_dir,age_group,new_sub_group) , shell = True)

            #print("hi")
        elif float(low_age) > 9.9 : 
            new_sub_group = save_dir+"age_10_to_20"
            os.makedirs(new_sub_group, exist_ok = True)
            subprocess.run('cp {}/dataset_description.json {}'.format(file_dir, new_sub_group), shell = True)
            subprocess.run('cp -r {}/{}/* {}'.format(file_dir,age_group,new_sub_group) , shell = True)
        elif low_age == '9' or low_age == '5.5':
            new_sub_group = save_dir + age_group
            os.makedirs(new_sub_group, exist_ok = True)
            subprocess.run('cp {}/dataset_description.json {}'.format(file_dir, new_sub_group), shell = True)
            subprocess.run('cp -r {}/{}/* {}'.format(file_dir,age_group,new_sub_group) , shell = True)    


        #do nothing, 
    #print(low_age, high_age)
    #os.mkdir(save_dir + age_group)
    
    #subprocess.run(f"cp -r {file_dir}/age_group/"
    
    #break
pdb.set_trace()
