import random
import datetime
import json
import logging
import os
import shutil
import sys
import time
import warnings
from glob import glob
from collections import defaultdict

import optuna
import wandb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn import CrossEntropyLoss, MSELoss , BCEWithLogitsLoss

from utils import argument_setting, seed_all, get_dict, test_data_analysis_2, ensemble_prediction,  get_info_fold # get dict to create dataset
from config import Config, PRETRAINING, FINE_TUNING
from models.densenet import densenet121
from models.unet import UNet
from yAwareContrastiveLearning_optuna import yAwareCLModel
from dataset import MRI_dataset #save as ADNI datseet haha
from sk_multilearn_PR_reproducible.skmultilearn.model_selection import IterativeStratification # used PR version for reproducibility https://github.com/scikit-multilearn/scikit-multilearn/pull/248

def get_folds_original(config):
    label_name = config.label_name
    labels = pd.read_csv(config.label)
    
    if config.task_type == 'cls':
        task_include = config.task_name.split('/')
        config.task_include = task_include # needed later
        
        if config.binary_class == True:
            config.num_classes = 1 #config.num_classes 를 1로 다시 바꾸기 => 이러면 모델도 2가 아닌 하나의 output classification neuron을 가지게됨!
        
        labels = labels[labels[label_name].notna()] #해당 task데이터가 없는 아이들을 지우고 진행한다 
        labels[label_name] = labels[label_name].astype('str') #enforce str 

        ##do na removal and str conversion also to other labels if iter_strat
        if args.stratify in ["iter_strat", "balan_iter_strat"]: 
            for label_list in config.iter_strat_label.values():
                for label_i in label_list : 
                    labels = labels[labels[label_i].notna()] #해당 task데이터가 없는 아이들을 지우고 진행한다 
                    labels[label_i] = labels[label_i].astype('str') #enforce str 
        
        data_1 = labels[labels[label_name] == task_include[0]]
        data_2 = labels[labels[label_name] == task_include[1]]
        
        if 'balan' in args.stratify:
            limit = min(len(data_1), len(data_2))
            data_1 = random.sample(data_1, limit)
            data_2 = random.sample(data_2, limit)

        #getting number of test samples to keep
        test_rate = config.test_ratio #20% of the total data reserved for testing
        len_1_test = round(test_rate * len(data_1)) #shouldn't use labels, as labels may contain the third data that is not used (for example, AD, CN but also MCI)
        len_2_test = round(test_rate * len(data_2))
        
        #doing train/test split => NON DETERMINISTICALLY (must be same regardless of the seed #)
        data_1_rem , test1 = np.split(data_1, [-len_1_test])
        data_2_rem , test2 = np.split(data_2, [-len_2_test])
        #dataz_1_rem : test1 제거 후 남아있는 것 => this is the data 'pool' that we will sample from to create train/valid sets
        
        ratio = len(data_1) / (len(data_1) + len(data_2)) #ratio referred to during stratification
        ##label 1, 2, training/validation sample 갯수 정하기
        len_1_train = round(args.train_num*ratio)  
        len_2_train = args.train_num - len_1_train 
        len_1_valid = round(int(args.train_num*config.valid_ratio)*ratio)
        len_2_valid = int(args.train_num*config.valid_ratio) - len_1_valid
        if not args.eval :  #only check if not in eval mode
            assert args.train_num*(1+config.valid_ratio) < (len(data_1) + len(data_2) - len_1_test - len_2_test), 'Not enough valid data. Set smaller --train_num or smaller config.valid_ratio in config.py.'
        #import pdb ; pdb.set_trace() #(len(data_1) + len(data_2) - len_1_test - len_2_test) 이것보다 작아야함!
        #with bootstrapping하려면 밑에 split을 할때 df.sample을 할때 replace = True하면 with replacement로 할 수 있을듯?                
        train1, valid1, _ = np.split(data_1_rem.sample(frac=1), 
                                    [len_1_train, len_1_train + len_1_valid])
        train2, valid2, _ = np.split(data_2_rem.sample(frac=1),
                                    [len_2_train, len_2_train + len_2_valid]) #_ : remaining stuff
        #split된 것에서 train끼리 valid 끼리 test끼리 받은 후 shuffling하기!
        label_train = pd.concat([train1, train2]).sample(frac=1)
        label_valid = pd.concat([valid1, valid2]).sample(frac=1)
        label_test = pd.concat([test1, test2]).sample(frac=1)

        if "test" not in args.task : #i.e. if we're not running test version and actually running:
            assert len(label_test) >= 50, 'Not enough test data. (Total: {0})'.format(len(label_test))

        print('\nTrain data info:\n{0}\nTotal: {1}\n'.format(label_train[label_name].value_counts().sort_index(), len(label_train)))
        print('Valid data info:\n{0}\nTotal: {1}\n'.format(label_valid[label_name].value_counts().sort_index(), len(label_valid)))
        print('Test data info:\n{0}\nTotal: {1}\n'.format(label_test[label_name].value_counts().sort_index(), len(label_test)))
        
        #label (CN/AD for ex) 을 0,1로 숫자로 바꾸기 
        label_train[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)
        label_valid[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)
        label_test[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)

    else: # config.task_type = 'reg'
        raise NotImplementedError("not impmlemented yet (have to be done,... other finetuning is needed)")
        task_include = args.task_name.split('/')
        assert len(task_include) == 1, 'Set only one label.'
        assert config.num_classes == 1, 'Set config.num_classes == 1'
        
        labels = pd.read_csv(config.label)
        labels = labels[(np.abs(stats.zscore(labels[label_name])) < 3)] # remove outliers w.r.t. z-score > 3
        assert args.train_num*(1+config.valid_ratio) <= len(labels), 'Not enough valid data. Set smaller --train_num or smaller config.valid_ratio in config.py.'
        
        ####ADDED####
        #getting number of test samples to keep
        test_rate = 0.2 #20% of the total data reserved for testing
        len_test = round(test_rate * len(labels))
        
        #doing train/test split => NON DETERMINISTICALLY (must be same regardless of the seed #)
        data_rem , label_test = np.split(labels, [-len_test])
        ############
        label_train, label_valid , _ = np.split(data_rem.sample(frac = 1), 
                                                [args.train_num , int(args.train_num*(1+config.valid_ratio))])
        print('\nTrain data info:\nTotal: {0}\n'.format(len(label_train)))
        print('Valid data info:\nTotal: {0}\n'.format(len(label_valid)))
        print('Test data info:\nTotal: {0}\n'.format(len(label_test)))
    ###
    ###running cross valdiation
    #print("skf might not work if doing regression => look into it!!! (do normal KF if reg?)") #(https://stackoverflow.com/questions/54945196/sklearn-stratified-k-fold-cv-with-linear-model-like-elasticnetcv)    
    SPLITS = config.num_split #CHANGE TO 5!!!
    label_tv = pd.concat([label_train, label_valid]) #add up the train/valid datasets (어차피 다시 train/valid split할것이니)      
        
    ##add all the columns to see    
    cols_to_see = []
    for col_list in [i for i in config.iter_strat_label.values()]:
        cols_to_see = cols_to_see+col_list
    cols_to_see.append(config.label_name)
    cols_to_see = list(set(cols_to_see)) #remove redundancy    
    
    if args.stratify == 'strat'or args.stratify == "balan_strat": 
        kf = StratifiedKFold(n_splits = SPLITS) 
        skf_target =  [label_tv, label_tv[label_name]] 
        get_info_fold(kf.split(*skf_target), label_tv, cols_to_see)
    elif args.stratify == 'iter_strat' or args.stratify == "balan_iter_strat" : 
        kf = IterativeStratification(n_splits= SPLITS, order=10, random_state = np.random.seed(0))#0) #np.random.RandomState(0) #increasing order makes it similar if args.stratify == "iter_strat" 
        #https://github.com/scikit-multilearn/scikit-multilearn/pull/248 # makes iterative stratification reproducible 
        #if not config.iter_strat_label #make sure it's list 
        updated_binary_cols = list(set(config.iter_strat_label['binary'] + [label_name])) #add the label_name if it is new
        ##ASSUMES THAT the label_name (i.e. target) is binary!! (i.e. not age or sth)
        floatized_arr = multilabel_matrix_maker(label_tv,binary_cols= updated_binary_cols, 
                                        multiclass_cols= config.iter_strat_label['multiclass'],
                                       continuous_cols=config.iter_strat_label['continuous'], n_chunks=3)
        skf_target = [floatized_arr, floatized_arr]        
        
        get_info_fold(kf.split(*skf_target), label_tv, cols_to_see)
        
        ##call kf once again to make it deterministic
        kf = IterativeStratification(n_splits= SPLITS, order=10, random_state = np.random.seed(0))#0) #np.random.RandomState(0) 
    else : 
        raise NotImplementedError(f"{args.stratify} is not implemented!")
    ###ACTUALLY RUNNING CROSS-VALIDATION/TRAINING/EVALUATION (depending on mode)    
    
    #import pdb; pdb.set_trace()
    print("must implement the balan thing too (ABCD ADHD같은 것 하기 위해서)")
    print("change chunks!! (2 for ABCD 3 for sth else? / 실제로 lr_schedule 하려면 그 뭐지, tag를 쓰든지 해서, 그 without skf를 하던지 해야함! (아니다 stratified를 봐야하나?)")

    
    
#PR version for reproducibility https://github.com/scikit-multilearn/scikit-multilearn/pull/248
#get the boundaries to use to split into chunks with increasing value 
def slice_index(array, n_chunks ):
    partitioned_list = np.array_split(np.sort(array), n_chunks)
    return [i[-1] for i in partitioned_list]
    

def multilabel_matrix_maker(df, binary_cols=None, multiclass_cols=None , continuous_cols=None , n_chunks=3) :
    """
    returns matrix that will be used for multilabel, taking into account columns that are either multiclass or continuous
    * df : the dataframe to be split
    * binary_cols : LIST of cols (str)just cols that will be used (binarized)
    * multiclass_cols : LIST of the cols (str) that are multi class
    * continuous_cols : LIST of the cols (str) that will be split (continouous)
    * n_chunks : if using continouous cols are used, how many split?
    
    outputs matrix that has binarized binarized for all columns (only needs to be used during iskf to get the indices)
    """
    df = df.copy() #copy becasue we don't want to modify the original df
    if binary_cols == multiclass_cols == continuous_cols == None : #i.e. if all are None
        raise ValueError("at least one of the cols have to be put.. currently all cols are selected as None")
    
    if type(binary_cols)!= list or type(multiclass_cols)!= list or type(continuous_cols)!= list: 
        raise ValueError("the cols have to be lists!")
    #checking if NaN exist => raise error (sanity check)\
    if df[binary_cols+multiclass_cols+continuous_cols].isnull().values.any():
        raise ValueError("Your provided df had some NaN in columns that you are wanting to do iskf on")    
 
    
    #now adding binarized columns for each column types and aggregating them into total_cols
    total_cols = []
    if binary_cols : 
        for col in binary_cols :
            df[col] = pd.factorize(df[col], sort = True)[0] 
            total_cols.append(df[col].values) #or single []?  ([[]] : df 로 만드는 것, [] : series로 만듬) 
            
    if multiclass_cols :
        for col in multiclass_cols : 
            df_col = df[[col]] #[[]] not [] because of dims 
            ohe = OneHotEncoder()
            ohe.fit(df_col)
            binarized_col = ohe.transform(df_col).todense() 
            total_cols.append(binarized_col)
            
    if continuous_cols: 
        for col in continuous_cols:
            df[col] = df[col].astype('float') #change to float when doing 
            array = df[col].values
            boundaries = slice_index(array, n_chunks)  
            i_below = -np.infty
            for i in boundaries:
                extracted_df = (df[col]>i_below) & (df[col]<=i) 
                i_below = i #update i_below
                total_cols.append(extracted_df.values.astype(float))     
    
    #adding all together,
    final_arr = np.column_stack(total_cols)
    
    return final_arr


"""
#example of usage : 
DEBUG = False
from skmultilearn.model_selection import IterativeStratification
kf = StratifiedKFold(n_splits=5)
ikf = IterativeStratification(n_splits=5, order=50, random_state=np.random.seed(seed = 0)) #increasing order makes it similar

binary_col2view = ['sex', "Unspecified.Bipolar.and.Related.Disorder.x"]
multiclass_col2view = [] #['race.ethnicity']
continuous_col2view = ['age'] #['age'] #['age'] #없애고 싶으면 [] 로 하기 
###===setting complete====###



##config에서 넣어줄때, 종류를 미리 나눠서 ㄴ허어줘야할듯 

#col2view = binary_col2view + multiclass_col2view + continuous_col2view
col2view = binary_col2view + multiclass_col2view
#print(col2view)
print(calc_prop(label_tv[col2view].values)) if DEBUG else None




from iter_strat import multilabel_matrix_maker as maker
#floatized_arr = np.array(label_tv[i2view].values, dtype = float)
floatized_arr = maker(label_tv,binary_cols= binary_col2view, 
                                        multiclass_cols= multiclass_col2view,
                                       continuous_cols=continuous_col2view, n_chunks=3)#, contiuous_cols=['age', 'BMI'], n_chunks=)


set_thing = set(label_tv[col2view].values.flatten()) #un split된 상태에서의 set을 써야함
thing = []
haha = []
#single label
#label_name = ['Unspecified.Bipolar.and.Related.Disorder.x'] #sex로 고정
#for FOLD, (train_idx, valid_idx) in enumerate(kf.split(label_tv, label_tv[label_name])): 
#multilabel
for FOLD, (train_idx, valid_idx) in enumerate(ikf.split(floatized_arr, floatized_arr)): 
    print(f"===FOLD : {FOLD}===")
    train = label_tv.iloc[train_idx]
    valid = label_tv.iloc[valid_idx]

    if DEBUG : 
        print("with training")
        print(calc_prop_change(train[col2view].values, label_tv[col2view].values, set_thing = set_thing))
        
        print("with validation")
        print(calc_prop_change(valid[col2view].values, label_tv[col2view].values, set_thing = set_thing))
    #thing.append(valid[col2view[-2]])
    
    #thing.append(valid[binary_col2view[1]])
    #thing.append(valid[multiclass_col2view[0]])
    thing.append(valid[continuous_col2view[0]])
    haha.append(calc_prop_change(valid[col2view].values, label_tv[col2view].values, set_thing = set_thing))
    
plt.hist(thing,bins = 3, label = [i for i in range(5)])
plt.legend()
plt.show()


##continuous variables checking

##sometimes, the validation set might not have the same size, even if it can become like that  (so, for loop으로 통계계산)
print([i.mean() for i in thing])
print(np.array([i.mean() for i in thing]).std())
print("haha")

##discrete stuff checking
haha_arr = np.array(haha, dtype = float)
print(np.sqrt(np.square(haha_arr).sum(axis=0)/5))

"""


def get_images(labels, config):    
    if "CHA" in config.task : 
        total_subj_imgs = pd.DataFrame({'files': glob(config.data+"/*.nii.gz")})
        total_subj_imgs[config.subjectkey] = total_subj_imgs.files.map(lambda x: x.split('/')[-1].split(".")[0])

        images_split = defaultdict()
        for curr_label, split in zip([label_train, label_valid, label_test], ['train', 'valid', 'test']):
            images_split[split] = pd.merge(total_subj_imgs, curr_label, on=config.subjectkey, how='inner')
    
    return images_split


def get_dict(config , label_train, label_valid, label_test):
    """
    given label_train
    train_sub_data_list : csv상에서의 subject name sub_file_list = 실제 img파일의 이름
    train_img_path_list : path of the subject images 
    train_img_file_list : train_sub_data_list의 실제 folder 이름들 
    (보통은 img_file_list와 train_sub_data_list가 동일할 것이다. ADNI가 독특한 경우
    **must make sure that ALL these list have the same order!!
    """
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
        train_sub_data_list = list(label_train['subjectkey'])      
        train_img_path_list = [Path(config.data) / (sub + ".nii.gz") for sub in train_sub_data_list] #".npy" because we need the npy things
        train_img_file_list = train_sub_data_list  #they are the same in ABCD's case 
        
        valid_sub_data_list = list(label_valid['subjectkey'])      
        valid_img_path_list = [Path(config.data) / (sub + ".nii.gz") for sub in valid_sub_data_list]
        valid_img_file_list = valid_sub_data_list  #they are the same in ABCD's case 
        
        test_sub_data_list = list(label_test['subjectkey'])      
        test_img_path_list = [Path(config.data) / (sub + ".nii.gz") for sub in test_sub_data_list]
        test_img_file_list = test_sub_data_list  #they are the same in ABCD's case 
        
    elif "UKB" in config.task :
        raise NotImplementedError("not done yet mf")
        
    else :  #other stuff (non-ADNI)
            raise NotImplementedError("Different things not done yet ") #do for when doing ABCD (아니다 그냥 glob으로 generally 가져가게 하기?) 
    
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


def make_dataloaders(labels, config):
    label_name = config.label_name
    
    if config.task_type == 'cls' :
    ##assert that the train/valid datasets should have at least one of each classe's samples
        assert np.sum(labels['train'][label_name] == 0) * np.sum(labels['train'][label_name] == 1) > 0, "(probably) dataset too small, training dataset does not have at least one of each class"
        assert np.sum(labels['valid'][label_name] == 0) * np.sum(labels['valid'][label_name] == 1) > 0, "(probably) dataset too small, validation dataset does not have at least one of each class"

    ###label을 받도록 하기 
    dataloaders = defaultdict()
    images = get_images(labels, config)
    for i, split in enumerate(['train', 'valid', 'test']):
        curr_dataset = MRI_dataset(config, labels[split], images[split].files.to_dict(), data_type=split)
        batch_size = config.batch_size if split != 'test' else 1
        dataloaders[split] = DataLoader(curr_dataset,
                                        batch_size=batch_size,
                                        sampler=RandomSampler(curr_dataset), 
                                        pin_memory=config.pin_mem,
                                        num_workers=config.num_workers,
                                        persistent_workers = True)
                                        
    return dataloaders