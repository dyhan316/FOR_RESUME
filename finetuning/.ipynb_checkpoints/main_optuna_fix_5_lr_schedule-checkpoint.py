import os

import time
import datetime
###
import numpy as np


#print("change this below if needed!!")
print("if OOM occurs, dataset.__init__ might be too large...? (i.e. loading too much on memory (or during persist_workers = True)")
from dataset_BT import MRI_dataset  #save as ADNI datseet haha

from torch.utils.data import DataLoader, Dataset, RandomSampler
from yAwareContrastiveLearning_optuna_fix_5 import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss
from torch.nn import CrossEntropyLoss, MSELoss , BCEWithLogitsLoss# ADNI
from models.densenet import densenet121
from models.unet import UNet
import argparse
#from config import Config, PRETRAINING, FINE_TUNING
### ADNIc
import torch
import random
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from scipy import stats
import json
from glob import glob
from pathlib import Path
from iter_strat import multilabel_matrix_maker


#=====ADDED=====#
from sklearn.model_selection import StratifiedKFold, KFold
from finetune_utils import get_dict, test_data_analysis_2, ensemble_prediction,  get_info_fold #get dict to create dataset

def str2bool(v): #true, false를 argparse로 받을 수 있도록 하기!
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
import scikitplot as skplt
import optuna
import logging
import sys 
import shutil

import wandb 
#from skmultilearn.model_selection import IterativeStratification
from sk_multilearn_PR_reproducible.skmultilearn.model_selection import IterativeStratification 
#used PR version for reproducibility https://github.com/scikit-multilearn/scikit-multilearn/pull/248
#===============#

###
#====공통된 parser하자====#(optuna 쉽게하려고)
parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_path", type=str, required=True, # ADNI
                        help="Set the pretrained model path.")   
parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                    help="Set the training mode. Do not forget to configure config.py accordingly !")
parser.add_argument("--train_num", type=int, required=True, # ADNI
                    help="Set the number of training samples.")                        
#parser.add_argument("--task_name", type=str, required=False, # ADNI
#                    help="Set the name of the fine-tuning task. (e.g. AD/MCI)") #REMOVED BECAUSE WE DON'T NEED IT, WILL GET FROM CONFIG ITSELF
parser.add_argument("--layer_control", type=str, choices=['tune_all', 'freeze', 'tune_diff'], required=False, # ADNI
                    help="Set pretrained weight layer control option.")
parser.add_argument("--stratify", type=str, choices=["strat", "balan", "iter_strat"], required=False, # ADNI
                    help="Set training samples are stratified or balanced for fine-tuning task.")
parser.add_argument("--random_seed", type=int, required=False, default=0, # ADNI
                    help="Random seed for reproduction.")

#==========ADDED=========#
parser.add_argument("--task", type=str, required=True, default=0, # ADNI
                    help="which task within config.py to do")


#batchsize는 살림 왜냐면, batchsize가 달라지면 RAM이 달라지고, 보통 batchsize가 늘어나면 performance가 좋아지니
parser.add_argument('--batch_size', required = False, default = 8,type=int, metavar='N',
                help='mini-batch size')

  
parser.add_argument('--input_option', required = True, type = str, help='possible options : yAware, BT_org, or BT_unet,  which option top use when putting in input (reshape? padd or crop? at which dimensions?')

parser.add_argument('--binary_class',required = False, default = True, type= str2bool,
                help='whether to use binary classification or not ')
parser.add_argument('--save_path', required = False, default = './finetune_results_default_dir', type = str, 
                    help = "where to save the evaluation results (i.e. where is model.path2?)")
parser.add_argument('--run_where', required = False, default = 'sdcc', type = str, help= 'where to run the thing (decides whether to run config.py or config_lab.py, options : sdcc or lab')

parser.add_argument('--eval', required = False, default = False, type = str2bool, help= 'whether ot do evaluation (using the best trial) or not (True or False)')

parser.add_argument('--lr_schedule', required = False, default = None, type = str, help= 'what lr scheduler to use : lr_plateau, cosine_annealing,onecyclelr, custom_1, custom_2, custom_3, custom_4,5,6, cosine_annealing_faster, SGDR_1, SGDR_2, SGDR_3 (SGD+momentum+COSINEANNEALINGWARMRESTARTS), ... ')



##other things that can be changed if need be 
parser.add_argument('--verbose', required = False, default = False, type = str2bool, help= 'whether to use weight_tracker or not ')
parser.add_argument('--early_criteria' , required = False, default = 'none', type = str , help = "use valid AUROC or loss or none for early stopping criteria?(options : 'AUROC' or 'loss' or 'none' ")

#ability to specify lr, wd range directly
parser.add_argument('--lr_range' , required = False, type = str , help = "what optuna lr range to use, in the for of `1e-5/1e-2` ")

parser.add_argument('--wd_range' , required = False, type = str , help = "what optuna wd range to use, in the for of `1e-5/1e-2` ")


###BN 은 나중에 필요성이 느껴지면 implement 하기!
#parser.add_argument('--BN' , required = False, type = str , help = "what optuna BN option range to use, in the for of `1e-5/1e-2` ")


#=====finished adding the 공통된 parser==========

#######define args and so on ################### #the learning rates and so on will be add/updated within main tho
args = parser.parse_args() #parser은 이미 앞에서 정의했었다
if args.run_where == 'sdcc' : 
    from config import Config, PRETRAINING, FINE_TUNING
elif args.run_where == 'lab' : 
    from config_lab import Config, PRETRAINING, FINE_TUNING
else : 
    raise ValueError("run_where option should either be sdcc or lab")
    
mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING
config = Config(mode, args.task) 

#config.num_cpu_workers = 1  
#torch.set_num_threads(8)



####https://github.com/wandb/wandb/issues/386
#to enable python debugger (pdb)
#os.environ["WANDB_START_METHOD"]="thread" #pdb할때는 켜야하지만 multiple machine에 하면 안됨 https://github.com/wandb/wandb/issues/3565#issuecomment-1108044613

def main(trial):
    
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S') # ADNI
    print("[main.py started at {0}]".format(nowDatetime))
    start_time = time.time() # ADNI
    
    
    #####changed##### (add stuff so that yAware thing can look at this to get things)
    #OPTUNA hyperparam tuning
    #optuna lr
    if args.lr_range :  #if lr range was specified
        lr_range = [float(i) for i in args.lr_range.split('/')]
        args.learning_rate = trial.suggest_float("learning_rate", *lr_range, log = True)  
    elif args.layer_control == "tune_all" : #use default if using tune_all
        args.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log = True)  
    else : #default if using freeze
        args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e1, log = True)  
    config.lr = args.learning_rate
    
    #optuna wd
    if args.wd_range :  #if wd range was specified
        wd_range = [float(i) for i in args.wd_range.split('/')]
        args.weight_decay = trial.suggest_float("weight_decay", *wd_range, log = True) #1e-2
    else :  # use default
        args.weight_decay = trial.suggest_float("weight_decay", 1e-11 , 1e0, log = True) #1e-2
    config.weight_decay = args.weight_decay
    
    #optuna BN method
    args.BN = trial.suggest_categorical("BN_option", ['none','inst'])
    config.BN = args.BN
    
    
    config.batch_size = args.batch_size     
    args.task_name = config.task_name ##ADDED (so that we don't have to redefine task_name every time... just get from the config itself
    config.binary_class = args.binary_class
    config.save_path = args.save_path
    config.stratify = args.stratify
    config.layer_control = args.layer_control
    config.pretrained_path = Path(args.pretrained_path).parts[-1]
    config.train_num = args.train_num
    
    config.eval = args.eval
    config.lr_schedule = args.lr_schedule
    
    config.verbose = args.verbose
    config.early_criteria = args.early_criteria
    
    ######sanity check########
    if args.binary_class == False and config.num_classes == 2:
        print("you're trying to use use binary classification and set num_classes ==2? probably wrong ")
        #raise ValueError("you're trying to use use binary classification and set num_classes ==2? probably wrong ")
    
    #for input_option thing 
    if args.input_option == "yAware":
        config.resize_method = 'reshape'
        config.input_size = (1, 80, 80, 80)
        
    elif args.input_option == "BT_org":
        config.resize_method = None
        config.input_size = (1, 99, 117, 95)
        
    elif config.input_option == "BT_unet":
        raise NotImplementedError("do this after implementing brain specific augmentation")
        
    else : 
        raise ValueError("this input_option value is not expected, choose one of 'yAware, BT_org, or BT_unet'. ")
        
    ################
    pretrained_path = args.pretrained_path
    print('Pretrained path:', pretrained_path)
    ### ADNI
    label_name = config.label_name # 'Dx.new'
    # Control randomness for reproduction
    if args.random_seed != None:
        random_seed = args.random_seed
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        #below were not done because NIMA said it decreases speed
        #torch.backends.cudnn. deterministic = True 
        #torch.backends.cudnn.benchmark = False 
    ##added : add random seed to config
    config.random_seed = args.random_seed
    

    if config.mode == PRETRAINING:
        raise ValueError("this is not the code to run for pretraining")
    else: # config.mode == FINE_TUNING:
        labels = pd.read_csv(config.label)
        print('Task: Fine-tuning for {0}'.format(args.task_name))
        print('Task type: {0}'.format(config.task_type))
        print('N = {0}'.format(args.train_num))
        
        if config.task_type == 'cls':
            print('Policy: {0}'.format(args.stratify))
            task_include = args.task_name.split('/')
            config.task_include = task_include # needed later
            
            if args.binary_class == True:
                assert len(task_include) == 2, 'Set two labels.'
                config.num_classes = 1 #config.num_classes 를 1로 다시 바꾸기 => 이러면 모델도 2가 아닌 하나의 output classification neuron을 가지게됨!
                
            else : #i.e. binary_class = False (default
                assert len(task_include) == 2, 'Set two labels.'
                assert config.num_classes == 2, 'Set config.num_classes == 2'
            
            ###ADDED (using the ABCD data that has all the subjects)
            ##removing na and converting the things to str 
            
            labels = labels[labels[label_name].notna()] #해당 task데이터가 없는 아이들을 지우고 진행한다 
            labels[label_name] = labels[label_name].astype('str') #enforce str 
            
            
            ##do na removal and str conversion also to other labels if iter_strat
            if args.stratify == "iter_strat" : 
                for label_list in config.iter_strat_label.values():
                    for label_i in label_list : 
                        
                        labels = labels[labels[label_i].notna()] #해당 task데이터가 없는 아이들을 지우고 진행한다 
                        labels[label_i] = labels[label_i].astype('str') #enforce str 
            
            data_1 = labels[labels[label_name] == task_include[0]]
            data_2 = labels[labels[label_name] == task_include[1]]
            ####ADDED####
            #getting number of test samples to keep
            test_rate = 0.2 #20% of the total data reserved for testing
            len_1_test = round(test_rate * len(data_1)) #shouldn't use labels, as labels may contain the third data that is not used (for example, AD, CN but also MCI)
            len_2_test = round(test_rate * len(data_2))
            
            #doing train/test split => NON DETERMINISTICALLY (must be same regardless of the seed #)
            data_1_rem , test1 = np.split(data_1, [-len_1_test])
            data_2_rem , test2 = np.split(data_2, [-len_2_test])
            #dataz_1_rem : test1 제거 후 남아있는 것 => this is the data 'pool' that we will sample from to create train/valid sets
            ############
            
            
            
            if args.stratify == 'strat' or "iter_strat":
                ratio = len(data_1) / (len(data_1) + len(data_2)) #ratio referred to during stratification
                ##label 1, 2, training/validation sample 갯수 정하기
                len_1_train = round(args.train_num*ratio)  
                len_2_train = args.train_num - len_1_train 
                len_1_valid = round(int(args.train_num*config.valid_ratio)*ratio)
                len_2_valid = int(args.train_num*config.valid_ratio) - len_1_valid
                assert args.train_num*(1+config.valid_ratio) < (len(data_1) + len(data_2)), 'Not enough valid data. Set smaller --train_num or smaller config.valid_ratio in config.py.'
                
                #with bootstrapping하려면 밑에 split을 할때 df.sample을 할때 replace = True하면 with replacement로 할 수 있을듯?                
                train1, valid1, _ = np.split(data_1_rem.sample(frac=1, random_state=random_seed), 
                                          [len_1_train, len_1_train + len_1_valid])
                train2, valid2, _ = np.split(data_2_rem.sample(frac=1, random_state=random_seed),
                                          [len_2_train, len_2_train + len_2_valid]) #_ : remaining stuff
                #split된 것에서 train끼리 valid 끼리 test끼리 받은 후 shuffling하기!
                label_train = pd.concat([train1, train2]).sample(frac=1, random_state=random_seed)
                label_valid = pd.concat([valid1, valid2]).sample(frac=1, random_state=random_seed)
                label_test = pd.concat([test1, test2]).sample(frac=1, random_state=random_seed)
                
                if "test" not in args.task : #i.e. if we're not running test version and actually running:
                    assert len(label_test) >= 200, 'Not enough test data. (Total: {0})'.format(len(label_test))
                    
                    
            else: # args.stratify == 'balan'
                raise NotImplementedError("not impolemented yet!! (balanced 하려면 train/test split을 원래는 밖에서 했는데 여기서는 이 안에서하게 되어있어서 안될것임!! 이쪽 코드 고쳐야함!)")
                if len(data_1) <= len(data_2):
                    limit = len(data_1)
                else:
                    limit = len(data_2)
                #data_1,2의 갯수를 맞도록 맞추기!
                data_1 = data_1.sample(frac=1, random_state=random_seed)[:limit] 
                data_2 = data_2.sample(frac=1, random_state=random_seed)[:limit]
                
                len_1_train = round(args.train_num*0.5) #train = 50%! (valid : training num에 포함이 안됨!)
                len_2_train = args.train_num - len_1_train #50, 50
                len_1_valid = round(int(args.train_num*config.valid_ratio)*0.5) #valid ratio * 0.5  
                len_2_valid = int(args.train_num*config.valid_ratio) - len_1_valid #remaining valid things 
                
                assert args.train_num*(1+config.valid_ratio) < limit*2, 'Not enough data to make balanced set.'
                ##sampling amounts, and leaving the rest as test data 
                train1, valid1, test1 = np.split(data_1.sample(frac=1, random_state=random_seed), 
                                                 [len_1_train, len_1_train + len_1_valid]) #the rest : test data 로 둔다! 
                train2, valid2, test2 = np.split(data_2.sample(frac=1, random_state=random_seed), 
                                                 [len_2_train, len_2_train + len_2_valid])
                label_train = pd.concat([train1, train2]).sample(frac=1, random_state=random_seed)
                label_valid = pd.concat([valid1, valid2]).sample(frac=1, random_state=random_seed)
                label_test = pd.concat([test1, test2]).sample(frac=1, random_state=random_seed)
                
                if "test" not in args.task : #i.e. if we're not running test version and actually running:
                    assert len(label_test) >= 100, 'Not enough test data. (Total: {0})'.format(len(label_test))

            print('\nTrain data info:\n{0}\nTotal: {1}\n'.format(label_train[label_name].value_counts().sort_index(), len(label_train)))
            print('Valid data info:\n{0}\nTotal: {1}\n'.format(label_valid[label_name].value_counts().sort_index(), len(label_valid)))
            print('Test data info:\n{0}\nTotal: {1}\n'.format(label_test[label_name].value_counts().sort_index(), len(label_test)))
            
            #label (CN/AD for ex) 을 0,1로 숫자로 바꾸기 
            label_train[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)
            label_valid[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)
            label_test[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)

        else: # config.task_type = 'reg'
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
            label_train, label_valid , _ = np.split(data_rem.sample(frac = 1, random_state = random_seed), 
                                                    [args.train_num , int(args.train_num*(1+config.valid_ratio))])
            
            print('\nTrain data info:\nTotal: {0}\n'.format(len(label_train)))
            print('Valid data info:\nTotal: {0}\n'.format(len(label_valid)))
            print('Test data info:\nTotal: {0}\n'.format(len(label_test)))
    ###
    
    ###running cross valdiation
    #print("skf might not work if doing regression => look into it!!! (do normal KF if reg?)") #(https://stackoverflow.com/questions/54945196/sklearn-stratified-k-fold-cv-with-linear-model-like-elasticnetcv)    
    SPLITS = 5 #CHANGE TO 5!!!
    label_tv = pd.concat([label_train, label_valid]) #add up the train/valid datasets (어차피 다시 train/valid split할것이니)
    
    
    #keeping track of cross validaiton values across the folds
    aurocMean_list , aurocStd_list = [] , []
    loss_list , acc_list= [], []
    mse_list, mae_list, rmse_list, r2_list  = [], [], [],[]    
    last_epoch_list = [] #keep track of training epochs
    
    if args.eval :  #for majority voting
        outPRED_list = []
        outGT_list = []        
        
    ##add all the columns to see    
    cols_to_see = []
    for col_list in [i for i in config.iter_strat_label.values()]:
        cols_to_see = cols_to_see+col_list
    cols_to_see.append(config.label_name)
    
    cols_to_see = list(set(cols_to_see)) #remove redundancy    
    
    ##saving the label and predictions 
    label_list = []
    pred_list = []
    
    if args.stratify == 'strat'or args.stratify == "balan": 
        kf = StratifiedKFold(n_splits = SPLITS) 
        skf_target =  [label_tv, label_tv[label_name]] 
        get_info_fold(kf.split(*skf_target), label_tv, cols_to_see)
    elif args.stratify == 'iter_strat' : 
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
        
    ###ACTUALLY RUNNING CROSS-VALIDATION/TRAINING/EVALUATION (depending on mode)    
    
    #import pdb; pdb.set_trace()
    print("must implement the balan thing too (ABCD ADHD같은 것 하기 위해서)")
    print("change chunks!! (2 for ABCD 3 for sth else? / 실제로 lr_schedule 하려면 그 뭐지, tag를 쓰든지 해서, 그 without skf를 하던지 해야함! (아니다 stratified를 봐야하나?)")

    for FOLD, (train_idx, valid_idx) in enumerate(kf.split(*skf_target)):
        print(f"FOLD : {FOLD}",train_idx, valid_idx)
        #setting up wandb
        #https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893
        job_type = "train" if not args.eval else "eval"
        wandb.init(project = "LR_SCHEDULE_OPTIMIZE", #"LR_SCHEDULE_OPTIMIZE", #"VAE_ADHD_downstream_task",
                   config = config,
                   name = "fold" + str(FOLD),
                   job_type = job_type,
                   group = f"trial_{trial.number}_{args.task}-{args.layer_control}_{str(args.pretrained_path).split('/')[-1].split('.')[0]}_tr_{args.train_num}_batch_{config.batch_size}",
                   reinit = True) #becasue optuna,we have to redo the trials
        wandb.config.weight_used = str(args.pretrained_path).split('/')[-1].split('.')[0]
        
        
        config.fold = FOLD #which fold it is in
        print("FOLD" , FOLD)
        print('\n<<< StratifiedKFold: {0}/{1} >>>'.format(FOLD+1, SPLITS))
        label_train = label_tv.iloc[train_idx] #skf.split으로 한 train_idx, valid_idx를 실제로 넣어줘서 값들을 구한다  
        label_valid = label_tv.iloc[valid_idx]
        
        if config.task_type == 'cls' : 
        ##assert that the train/valid datasets should have at least one of each classe's samples
            assert np.sum(label_train[label_name] == 0) * np.sum(label_train[label_name] == 1) > 0, "(probably) dataset too small, training dataset does not have at least one of each class"
            assert np.sum(label_valid[label_name] == 0) * np.sum(label_valid[label_name] == 1) > 0, "(probably) dataset too small, validation dataset does not have at least one of each class"
        

        ###was INDENTED
        if config.mode == PRETRAINING:
            raise ValueError("this is not the code to run for pretraining")
            
        elif config.mode == FINE_TUNING:
            ###label을 받도록 하기 
            train_img_dict, valid_img_dict, test_img_dict = get_dict(config, label_train, label_valid, label_test)
                    
            dataset_train = MRI_dataset(config, label_train, train_img_dict, data_type = 'train') #label_train df 을 쓴다 
            dataset_valid = MRI_dataset(config, label_valid, valid_img_dict, data_type = 'valid') #label_train df 을 쓴다 
            dataset_test = MRI_dataset(config, label_test, test_img_dict, data_type = 'test') #label_test df 을 쓴다 
                    
            #I may need to do config too (for BT)
            loader_train = DataLoader(dataset_train,
                                  batch_size=config.batch_size,
                                  sampler=RandomSampler(dataset_train), #collate_fn=dataset_train.collate_fn,
                                  pin_memory=config.pin_mem,
                                  num_workers=config.num_cpu_workers, persistent_workers = True
                                  )
            loader_val = DataLoader(dataset_valid,
                                  batch_size=config.batch_size,
                                  sampler=RandomSampler(dataset_valid), #collate_fn=dataset_valid.collate_fn,
                                  pin_memory=config.pin_mem,
                                  num_workers=config.num_cpu_workers,  persistent_workers = True
                                  )
            loader_test = DataLoader(dataset_test,
                                  batch_size=1,
                                  sampler=RandomSampler(dataset_test), #collate_fn=dataset_test.collate_fn,
                                  pin_memory=config.pin_mem,
                                  num_workers=config.num_cpu_workers, persistent_workers = True
                                  )
            ####FOR CHECKING
            #train_batch, val_batch, test_batch = next(iter(loader_train)), next(iter(loader_val)), next(iter(loader_test))
            #np.save("./trash/ABCD_train_loader.npy", np.array(train_batch[0])) 
            #np.save("./trash/ABCD_val_loader.npy", np.array(val_batch[0]))
            #np.save("./trash/ABCD_test_loader.npy", np.array(test_batch[0]))    
            
            #choose model
            if config.model == "DenseNet":
                if args.BN == "none":
                    net = densenet121(mode = "classifier_no_BN", drop_rate= 0.0, num_classes = config.num_classes)
                elif args.BN == "inst":
                    net = densenet121(mode = "classifier_inst_BN", drop_rate = 0.0, num_classes= config.num_classes)
                elif args.BN == "default":
                    net = densenet121(mode="classifier", drop_rate=0.0, num_classes=config.num_classes)
                else :
                    raise ValueError("args.BN should be one of three")
                    
            elif config.model == "UNet":
                net = UNet(config.num_classes, mode="classif")
            else:
                raise ValueError("Unkown model: %s"%config.model)
            
            print("FIX FROM HERE ON OUT (THE DATASET LOADER AND ETC ARE JUST FINE I THINK")
            #=====여기를 바꿔야할듯??====(if args.binary_class = True)#
            #choose loss function
            if config.task_type == 'cls' and config.binary_class == False:
                loss = CrossEntropyLoss()
            elif config.task_type == 'cls' and config.binary_class == True:
                loss = BCEWithLogitsLoss()
            elif config.task_type == 'reg' : # config.task_type == 'reg': # ADNI
                loss = MSELoss()
            else : 
                raise ValueError("choose one of cls or reg for config task_type")    
            
            wandb.watch(models =net, criterion = loss)
            
            
            
            model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, args.task_name, args.train_num, args.layer_control, None, pretrained_path, FOLD, wandb, trial) # ADNI
            if args.eval : 
                ##load checkpoint => don't do fine_tuning, but just do the eval_model part (only evaluaiton) to get outGT outPRED and so on 
                ckpt_dir = glob(model.best_trial_path+f'/*{FOLD}.pt')[0] #this fold's best trial's ckpt dir
                model.model.load_state_dict(torch.load(ckpt_dir)) #model (yAware)내의 net 를 직접 꺼내기
                
                
                outGT, outPRED, loss ,acc , aurocMean = model.eval_model(mode = 'test')
                model.test_loss = loss
                model.test_acc = acc
                model.aurocMean = aurocMean
                
                #for majority voting
                outGT_list.append(outGT)
                outPRED_list.append(outPRED)
                
            else :  #OPTUNA (regular fine_tuning)
                outGT, outPRED, last_epoch = model.fine_tuning() # does actual finetuning #returns BEST validation results (after early stopping is reached or when all the epochs are reached)
                last_epoch_list.append(last_epoch) #last_epoch : last epoch of the training phaes
            
            #label list pred list append
            label_list.append(outGT.cpu().numpy())
            pred_list.append(torch.nn.Sigmoid()(outPRED).cpu().numpy()) #sigmoided!
            
            ###calculating mean AUROC and saving the AUC graph 
            if config.task_type == 'cls':
                if config.binary_class == False :  #original mode
                    raise NotImplementedError("have to redo this!! (so that test_data_analysis_2 is merged to the original  test_data_analysis!!")
                
                elif config.binary_class == True :                                         
                    #ACTIVATE IF IN VALID MODE
                    if args.eval : 
                        print(json.dumps({f"test_loss" : model.test_loss , f"test_acc" : model.test_acc}), file= model.eval_stats_file) #get what the setting was (this is actually the validaiton file and not the evaluation file, but whatever)
                        auroc_mean_arr = test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD)
                        #i.e. even in test mode, do plotting the AUC graph thing
                        print(json.dumps({"AUROC" : auroc_mean_arr}), file = model.eval_stats_file) #test 모드에서 이것을 저장할 이유는 없어보임.. 그냥 array형태로 eval_stats.txt 마지막에 할대만 해도 될듯?
                        aurocMean_list.append(np.mean(auroc_mean_arr))
                        aurocStd_list.append(np.std(auroc_mean_arr))
                        loss_list.append(model.test_loss)
                        acc_list.append(model.test_acc)
                        
                        
                    else : #optuna
                        print(json.dumps({f"val_loss" : model.val_loss , f"val_acc" : model.val_acc}), file= model.eval_stats_file) #get what the setting was (this is actually the validaiton file and not the evaluation file, but whatever)
                        auroc_mean_arr = test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD)
                        print(json.dumps({"AUROC" : auroc_mean_arr}), file = model.eval_stats_file) #test 모드에서 이것을 저장할 이유는 없어보임.. 그냥 array형태로 eval_stats.txt 마지막에 할대만 해도 될듯?
                        aurocMean_list.append(np.mean(auroc_mean_arr))
                        aurocStd_list.append(np.std(auroc_mean_arr))
                        loss_list.append(model.val_loss)
                        acc_list.append(model.val_acc)
                        
                        if auroc_mean_arr <=0.5 : 
                            print("pruned because fold's auroc <= 0.5")
                            wandb.config.state = "fold_auroc_below_0.5"
                            #os.rmdir(model.path2)
                            raise optuna.TrialPruned()
                                  
            else: # config.task_type == 'reg':
                #ACTIVATE IF IN EVAL MODE
                #mse, mae, rmse, r2 = test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD)
                #mse_list.append(mse) ; mae_list.append(mae) ; rmse_list.append(rmse) ; r2_list.append(r2)
                #loss_list.append(model.test_loss)
                raise NotImplementedError("reg is not implemented yet (has to be checked)")
                #activate if in valid mode
                mse, mae, rmse, r2 = test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD)
                print(json.dumps({f"val_loss" : str(model.loss), f"val_MSE" : str(mse), "val_MAE" : str(mae), "val_rmse" : str(rmse), "R2" : str(r2)}), file = model.eval_stats_file)
                mse_list.append(mse) ; mae_list.append(mae) ; rmse_list.append(rmse) ; r2_list.append(r2)
                loss_list.append(model.val_loss)
                print("regression was done baby~")
                
    
        else : 
            raise ValueError("config.mode should be either pretraining or finetuning!!")
            
            
        #fold마다 run이기에, fold마다 이렇게 끝내준다!
        wandb.run.summary['state'] = "success"
        
        if not args.eval :  #in optuna mode
            if config.task_type == 'cls' : 
                wandb.run.summary['final_val_AUROC'] = auroc_mean_arr
                wandb.run.summary['final_val_acc'] = model.val_acc
            wandb.run.summary['final_val_loss'] = model.val_loss
        
        wandb.finish()
            
    #FOLD다 돈 후에 되는 것 
    
    
    ##from here on out implement things like majority voting and so on 
    ##see https://machinelearningmastery.com/voting-ensembles-with-python/
    
    if args.eval : #only do if eval mode 
        with open(model.best_trial_path + "/val_stats.txt", "r") as file :
            data = [json.loads(line) for line in file]
            val_auroc_list = [d["mean_auroc"] for d in data if "mean_auroc" in d][0] #used as weight 
            val_auroc_list = [float(x) for x in val_auroc_list]

        cv_val_file = open(f'{model.path2}/test_stats_final.txt', 'a', buffering=1)  #cross validation eval file 
        
        mean= ensemble_prediction(config, outGT_list, outPRED_list, stat_measure = 'mean', weights = val_auroc_list, model = model, task_include = task_include)
        median= ensemble_prediction(config, outGT_list, outPRED_list, stat_measure = 'median', weights = val_auroc_list, model = model, task_include = task_include)
        weighted = ensemble_prediction(config, outGT_list, outPRED_list, stat_measure = 'weighted', weights = val_auroc_list, model = model, task_include = task_include)
        hard_mode = ensemble_prediction(config, outGT_list, outPRED_list, stat_measure = 'hard_mode', weights = val_auroc_list, model = model, task_include = task_include)
        hard_weighted = ensemble_prediction(config, outGT_list, outPRED_list, stat_measure = 'hard_weighted', weights = val_auroc_list, model = model, task_include = task_include)
        
        #voting all list 
        voting_list = [mean, median, weighted, hard_mode, hard_weighted]
        
        for thing in voting_list : 
            print(json.dumps(thing), file = cv_val_file)
            
        #copy the best trial's validation results to the best trial test results, so that we can look at em better
        shutil.copytree(model.best_trial_path, os.path.join(model.path2, "validation_results") )
        cv_val_file = open(f'{model.path2}/test_stats_final.txt', 'a', buffering=1)  #cross validation eval file 
    else : 
        cv_val_file = open(f'{model.path2}/val_stats.txt', 'a', buffering=1)  #cross validation eval file 
    print(json.dumps(config.__dict__), file= cv_val_file) #save config dict stuff 
    
    #since we were returned the validation, aurocMean_list will be from validation 
    print_CV(config, file_pth = cv_val_file, aurocMean_list = aurocMean_list, aurocStd_list= aurocStd_list, loss_list = loss_list, 
            acc_list = acc_list, last_epoch_list = last_epoch_list)
    
    end_time = time.time()
    print('\nTotal', (end_time - start_time) / 60, 'minutes elapsed.')
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S') # ADNI
    print("[main.py finished at {0}]".format(nowDatetime))
    ###
    
    
    #catchall pruner (objective func output에서 nan이 나오면 바로 prune)
    if torch.tensor(aurocMean_list).isnan().any(): #i.e. if there are any nans, in the output
        print("pruned due to NAN")
        raise optuna.TrialPruned() #prune this trial    
    
    #caluclating validation/testing (depending on mode) mean AUC from 5 folds.
    
    mean_AUC = np.array(aurocMean_list).mean() #for optuna #in val mode, returns the validaiton auroc

    if (study.trials_dataframe()['state'] == "COMPLETE").sum() == 0 :  #i.e. first (complete) trial
        pass 
    elif study.best_value > mean_AUC :  #i.e. not first trial, but not best trial
        [os.remove(pt_i) for pt_i in glob(model.path2+'/*.pt')]

    
    return mean_AUC
    

    
def print_CV(config, file_pth,**kwargs):
    print("<<<Cross Validation Result>>>")
    if config.task_type == 'cls':      
        print(f"Mean AUROC : {kwargs['aurocMean_list']}")
        print(f"stdev AUROC : {kwargs['aurocStd_list']}")
        print(json.dumps({"loss_list": kwargs['loss_list'] , "acc_list" : kwargs['acc_list'], "mean_auroc" : kwargs['aurocMean_list'], "std_auroc" : kwargs['aurocStd_list'], "last_epoch_list" : kwargs['last_epoch_list']}), file = file_pth)  
        
    elif config.task_type == 'reg' : 
        print(json.dumps({"mse_list " : str(kwargs['mse_list'])  , "mae_list " : str(kwargs['mae_list'])  , "rmse_list" : str(kwargs['rmse_list']) , "r2_list" : str(kwargs['r2_list']), "last_epoch_list" : kwargs['last_epoch_list']}), file = file_pth)  
        
    else : 
        raise ValueError("use either cls or reg!")   
        
if __name__ == "__main__":    
    #make db_folder if needed
    db_folder = os.makedirs(f"./{args.save_path}/optuna_db", exist_ok = True)
    
    
    db_dir = f"{args.save_path}/optuna_db/{args.task}-{args.layer_control}-{config.task_type}-{str(args.pretrained_path).split('/')[-1].split('.')[0]}-train_{args.train_num}-batch_{args.batch_size}"
    
    #{config.save_path}/{config.task}-{self.layer_control}-{config.task_type}
    url = "sqlite:///" + os.path.join(os.getcwd(), (db_dir + '.db'))
    print(url)
    # Add stream handler of stdout to show the messages => 이건 되어있던데 왜하는 건지 모르겠다 
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    
    storage = optuna.storages.RDBStorage( url = url, heartbeat_interval = 60, grace_period = 120 )
    
    study = optuna.create_study(study_name = "test_study_name", 
                                storage = storage,
                                load_if_exists = True,
                                pruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=20, interval_steps=1), #wrap with patience?
                                direction = 'maximize') #pruner, sampler도 나중에 넣기) 
    
    ##run this later in fix_4_scheduler_test
    lr_0 = 0.00001947807947898847
    wd_0 = 0.0028403780495017527
    
    #study.enqueue_trial(
    #{
    #    "BN_option": 'inst',
    #    "learning_rate": 0.1*lr_0,
    #    "weight_decay": 0.1*wd_0,
    #})
    study.enqueue_trial(
    {
        "BN_option": 'inst',
        "learning_rate": 0.1*lr_0,
        "weight_decay": wd_0,
    })
    #study.enqueue_trial(
    #{
    #    "BN_option": 'inst',
    #    "learning_rate": 0.1*lr_0,
    #    "weight_decay": 10*wd_0,
    #})
    study.enqueue_trial(
    {
        "BN_option": 'inst',
        "learning_rate": lr_0,
        "weight_decay": 0.1*wd_0,
    })
    study.enqueue_trial(
    {
        "BN_option": 'inst',
        "learning_rate": lr_0,
        "weight_decay": wd_0,
    })
    study.enqueue_trial(
    {
        "BN_option": 'inst',
        "learning_rate": lr_0,
        "weight_decay": 10*wd_0,
    })
    #study.enqueue_trial(
    #{
    #    "BN_option": 'inst',
    #    "learning_rate": 10*lr_0,
    #    "weight_decay": 0.1*wd_0,
    #})
    study.enqueue_trial(
    {
        "BN_option": 'inst',
        "learning_rate": 10*lr_0,
        "weight_decay": wd_0,
    })
    #study.enqueue_trial(
    #{
    #    "BN_option": 'inst',
    #    "learning_rate": 10*lr_0,
    #    "weight_decay": 10*wd_0,
    #})
    
    
    if args.eval :
        main(study.best_trial)
    else : 
        study.optimize(main, n_trials = 30, timeout = 86000) #8 hrs : 28800
    #https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
    
    
    ##using best model further inspired from  : https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/010_reuse_best_trial.html
    ##즉, 저기서 testing part만 뽑아서 돌리던지 해야할듯? (main 에 다 넣지 말고)
