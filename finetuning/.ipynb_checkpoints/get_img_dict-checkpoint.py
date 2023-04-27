from glob import glob
from pathlib import Path
import os 
"""
given label_train
train_sub_data_list : csv상에서의 subject name sub_file_list = 실제 img파일의 이름
train_img_path_list : path of the subject images 
train_img_file_list : train_sub_data_list의 실제 folder 이름들 
(보통은 img_file_list와 train_sub_data_list가 동일할 것이다. ADNI가 독특한 경우
**must make sure that ALL these list have the same order!!
"""

def get_dict(config , label_train, label_valid, label_test):
    if "ADNI" in config.task or config.task == "test" or config.task == "test_age" : #change it so that this is the case for 
        print("this assumes that all the subjects in csv has corresopnding images") 
        print("make sure that this is the case!!!!!")            
        train_sub_data_list = list(label_train['SubjectID'])  
        train_img_path_list = [glob(os.path.join(config.data , f"sub-{sub}*",'brain_to_MNI_nonlin.nii.gz'))[0] for sub in train_sub_data_list] #glob등이 복잡하게 들어가는이유 : because very ADNI specific 
        
        train_img_file_list = [i.split('/')[-2] for i in train_img_path_list] #ADNI specific 
        ##valid/test : them over and over (repeating)
        
        #same with valid
        valid_sub_data_list = list(label_valid['SubjectID'])             
        valid_img_path_list = [glob(os.path.join(config.data , f"sub-{sub}*",'brain_to_MNI_nonlin.nii.gz'))[0] for sub in valid_sub_data_list] #glob등이 복잡하게 들어가는이유 : because very ADNI specific
        valid_img_file_list = [i.split('/')[-2] for i in valid_img_path_list] #ADNI specific
         
        #same with test
        test_sub_data_list = list(label_test['SubjectID'])             
        test_img_path_list = [glob(os.path.join(config.data , f"sub-{sub}*",'brain_to_MNI_nonlin.nii.gz'))[0] for sub in test_sub_data_list] #glob등이 복잡하게 들어가는이유 : because very ADNI specific
        test_img_file_list = [i.split('/')[-2] for i in test_img_path_list] #ADNI specific
        
    elif "ABCD" in config.task :
        train_sub_data_list = list(label_train['SubjectID'])      
        train_img_path_list = [Path(config.data) / (sub + ".npy") for sub in train_sub_data_list] #".npy" because we need the npy things
        train_img_file_list = train_sub_data_list  #they are the same in ABCD's case 
        
        valid_sub_data_list = list(label_valid['SubjectID'])      
        valid_img_path_list = [Path(config.data) / (sub + ".npy") for sub in valid_sub_data_list]
        valid_img_file_list = valid_sub_data_list  #they are the same in ABCD's case 
        
        test_sub_data_list = list(label_test['SubjectID'])      
        test_img_path_list = [Path(config.data) / (sub + ".npy") for sub in test_sub_data_list]
        test_img_file_list = test_sub_data_list  #they are the same in ABCD's case 
        
        
    elif "CHA" in config.task :
        import pdb ; pdb.set_trace()
    elif "UKB" in config.task :
        raise NotImplementedError("not done yet mf")
        
    else :  #other stuff (non-ADNI)
            raise NotImplementedError("Different thigns not done yet ") #do for when doing ABCD (아니다 그냥 glob으로 generally 가져가게 하기?) 
    
    #sanity check (just in case)
    assert train_sub_data_list[1] in str(train_img_path_list[1]), "the train_sub_data_list and img_path_list order must've changed..." #ddi str because the thing could be PosixPath 
    assert valid_sub_data_list[1] in str(valid_img_path_list[1]), "the valid_sub_data_list and img_path_list order must've changed..."
    assert test_sub_data_list[1] in str(test_img_path_list[1]), "the test_sub_data_list and img_path_list order must've changed..."
    assert len(train_sub_data_list) == len(train_img_file_list), "the size should be the same, but they're not, indicating that some subs exist in csv but not in img"
    assert len(valid_sub_data_list) == len(valid_img_file_list), "the size should be the same, but they're not, indicating that some subs exist in csv but not in img"
    assert len(test_sub_data_list) == len(test_img_file_list), "the size should be the same, but they're not, indicating that some subs exist in csv but not in img"
    

    ##defining them 
    train_img_dict = {sub : train_img_path_list[i] for i,sub in enumerate(train_sub_data_list)} 
    valid_img_dict = {sub : valid_img_path_list[i] for i,sub in enumerate(valid_sub_data_list)}    
    test_img_dict = {sub : test_img_path_list[i] for i,sub in enumerate(test_sub_data_list)} 
    
    return train_img_dict, valid_img_dict, test_img_dict