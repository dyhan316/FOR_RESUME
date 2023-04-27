import os
import time
import datetime
###
import numpy as np

# python3 main_optuna_fix_2.py --pretrained_path None --mode finetuning --train_num 100 --layer_control tune_all --stratify strat --random_seed 0 --task ADNI_ALZ_ADCN --input_option yAware --binary_class True --save_path finetune_trash --run_where sdcc --eval_mode True --learning_rate 0.0003600924406843466 --weight_decay 0.00015499754224826388 --BN none --foreach False --batch_size 64

#print("change this below if needed!!")
print("if OOM occurs, dataset.__init__ might be too large...? (i.e. loading too much on memory (or during persist_workers = True)")
from dataset_BT import MRI_dataset  #save as ADNI datseet haha

from torch.utils.data import DataLoader, Dataset, RandomSampler
from yAwareContrastiveLearning_optuna_fix_2 import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss
from torch.nn import CrossEntropyLoss, MSELoss , BCEWithLogitsLoss# ADNI
from models.densenet import densenet121
from models.unet import UNet
import argparse
#from config import Config, PRETRAINING, FINE_TUNING
### ADNI
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


#=====ADDED=====#
from sklearn.model_selection import StratifiedKFold, KFold
from get_img_dict import get_dict #get dict to create dataset

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
parser.add_argument("--stratify", type=str, choices=["strat", "balan"], required=False, # ADNI
                    help="Set training samples are stratified or balanced for fine-tuning task.")
parser.add_argument("--random_seed", type=int, required=False, default=0, # ADNI
                    help="Random seed for reproduction.")

#==========ADDED=========#
parser.add_argument("--task", type=str, required=True, default=0, # ADNI
                    help="which dataset/data column to do")



#batchsize는 살림 왜냐면, batchsize가 달라지면 RAM이 달라지고, 보통 batchsize가 늘어나면 performance가 좋아지니
parser.add_argument('--batch_size', required = False, default = 8,type=int, metavar='N',
                help='mini-batch size')

  
parser.add_argument('--input_option', required = True, type = str, help='possible options : yAware, BT_org, or BT_unet,  which option top use when putting in input (reshape? padd or crop? at which dimensions?')

parser.add_argument('--binary_class',required = False, default = True, type= str2bool,
                help='whether to use binary classification or not ')
parser.add_argument('--save_path', required = False, default = './finetune_results_default_dir', type = str, 
                    help = "where to save the evaluation results (i.e. where is model.path2?)")
parser.add_argument('--run_where', required = False, default = 'sdcc', type = str, help= 'where to run the thing (decides whether to run config.py or config_lab.py, options : sdcc or lab')
parser.add_argument('--eval_mode', required = False, default = False, type = str2bool, help= 'whether eval_mode is true or not (if True, no optuna, if False, hyperparam optimization mode (training + validation) ')

#subparser같은 것 추가해서 해도 되지만, 복잡하다.. 그냥 lr wd 등을 받을 수 있게 하되, default None, and if the eval_mode is false, raise error하도록 하자 

#========ONLY REQUIRED IF DOING eval_mode, if provided even though not in eval_mode, raises error
parser.add_argument('--learning_rate', required = False, default = 0. ,type=float, metavar='LR',
                help='base learning rate')
parser.add_argument('--weight_decay',required = False, default = 0., type=float, metavar='W',
                help='weight decay')
parser.add_argument('--BN' , required = False, type = str, default = "ERROR", help = "what BN to use : none, inst, default")  
#parser.add_argument('--foreach' , required = False, type = str, default = False, help = "whether to use foreach or not ") #Default True because only implementation differs (see my pytorch question)

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

##ADDED
if args.eval_mode :  #if in eval mode, require the hyperparameters
    print("in eval mode, therefore the lr and so on will BE USED")
    if args.learning_rate == 0. or args.weight_decay == 0. or args.BN == "ERROR" :
        raise ValueError("lr, weightdecay, BN , and foreach need to be provided if in eval mode!")

else : #i.e. in optuna mode, if leanring rate or weight decya was changed => raise error 
    if args.learning_rate != 0. or args.weight_decay != 0. or args.BN != "ERROR" :
        raise ValueError("even in optuna mode, learning rate and weight decay or sth ewre provided")

#args = parser.parse_args()
config = Config(mode, args.task) 

config.num_cpu_workers = 1
torch.set_num_threads(16)
###


def main(trial):
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S') # ADNI
    print("[main.py started at {0}]".format(nowDatetime))
    start_time = time.time() # ADNI
    
    #####changed##### (add stuff so that yAware thing can look at this to get things)
    if args.eval_mode : 
        pass 
        
    else :  #use optuna
        #OPTUNA hyperparam tuning
        #optuna lr, wd ,BN ,foreach 여부
        if args.layer_control == "tune_all":
            args.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log = True)  #1e-4
        else  : # freeze
            args.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e1, log = True)  #1e-4
            
        
        #args.learning_rate = trial.suggest_float("learning_rate", 1e-7,1e2, log = True)  #1e-4
        args.weight_decay = trial.suggest_float("weight_decay", 1e-9 , 1e0, log = True) #1e-2
        args.BN = trial.suggest_categorical("BN_option", ['none','inst'])
        #args.foreach = trial.suggest_categorical("AdamW_foreach" , ["True", "False"])
    
    config.lr = args.learning_rate
    config.weight_decay = args.weight_decay    
    config.BN = args.BN
    #config.foreach = args.foreach    
        
    config.batch_size = args.batch_size     
    args.task_name = config.task_name ##ADDED (so that we don't have to redefine task_name every time... just get from the config itself
    config.binary_class = args.binary_class
    config.save_path = args.save_path
    config.stratify = args.stratify
    config.layer_control = args.layer_control
    config.pretrained_path = Path(args.pretrained_path).parts[-1]
    config.train_num = args.train_num
    config.eval_mode = args.eval_mode
    
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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
            
            if args.binary_class == True:
                assert len(task_include) == 2, 'Set two labels.'
                config.num_classes = 1 #config.num_classes 를 1로 다시 바꾸기 => 이러면 모델도 2가 아닌 하나의 output classification neuron을 가지게됨!
                
            else : #i.e. binary_class = False (default
                assert len(task_include) == 2, 'Set two labels.'
                assert config.num_classes == 2, 'Set config.num_classes == 2'


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
            
            
            if args.stratify == 'strat':
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
                raise NotImplementedError("not impolemented yet!! (the train/test split을 원래는 밖에서 했는데 여기서는 이 안에서하게 되어있어서 안될것임!! 이쪽 코드 고쳐야함!)")
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
    
    ##saving the label and predictions 
    label_list = []
    pred_list = []
    
    kf = StratifiedKFold(n_splits = SPLITS) if  config.task_type == 'cls' else KFold(n_splits = SPLITS) #kf could be stratified or not 
    
    ###ACTUALLY RUNNING CROSS-VALIDATION
    for FOLD, (train_idx, valid_idx) in enumerate(kf.split(label_tv, label_tv[label_name])): 
        print("FOLD" , FOLD)
        config.fold = FOLD #which fold it is in
        
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
                
            model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, args.task_name, args.train_num, args.layer_control, None, pretrained_path, FOLD) # ADNI
            
            outGT, outPRED, last_epoch = model.fine_tuning(args.eval_mode) # does actual finetuning 
            #if eval_mode = True, returns testing results 
            #if eval_mode = False,returns validation results after early stopping is reached (or when all the epochs are reached) 
            last_epoch_list.append(last_epoch) #last_epoch : last epoch of the training phaes
            #if Binary is True, outGT is not one hot encoded, and outPRED is only probabilty of true (not probabiltiy of True/False
    
            #label list pred list append
            label_list.append(outGT.cpu().numpy())
            pred_list.append(torch.nn.Sigmoid()(outPRED).cpu().numpy()) #sigmoided!
            
            ###calculating mean AUROC and saving the AUC graph 
            if config.task_type == 'cls':
                if config.binary_class == False :  #original mode
                    raise NotImplementedError("have to redo this!! (so that test_data_analysis_2 is merged to the original  test_data_analysis!!")
                    auroc_mean_arr = test_data_analysis(config, model, outGT, outPRED, task_include, FOLD)
                    aurocMean_list.append(np.mean(auroc_mean_arr))
                    aurocStd_list.append(np.std(auroc_mean_arr))
                    loss_list.append(model.test_loss)
                    acc_list.append(model.test_acc)
                
                elif config.binary_class == True : 
                    if args.eval_mode :  #the returned value should be the testing thing
                        #ACTIVATE IF IN EVAL MODE (not implem ented yet) (겹치는 부분은 빨ㄹ ㅣ할 수 있도록 만들기)
                        auroc_mean_arr = test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD, mode = "test")
                        aurocMean_list.append(np.mean(auroc_mean_arr))
                        aurocStd_list.append(np.std(auroc_mean_arr))
                        loss_list.append(model.test_loss)
                        acc_list.append(model.test_acc)
                        ###############################
                    
                    else : #i.e. in optuna mode, meaning we should give back the validation stuff
                        #ACTIVATE IF IN VALID MODE (if 문안으로 넣기)             
                        auroc_mean_arr = test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD, mode = "val")
                        aurocMean_list.append(np.mean(auroc_mean_arr))
                        aurocStd_list.append(np.std(auroc_mean_arr))
                        loss_list.append(model.val_loss)
                        acc_list.append(model.val_acc)
                    
                
            else: # config.task_type == 'reg':
                if args.eval_mode : 
                    mse, mae, rmse, r2 = test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD, mode = 'test')
                    mse_list.append(mse) ; mae_list.append(mae) ; rmse_list.append(rmse) ; r2_list.append(r2)
                    loss_list.append(model.test_loss)
                
                else : #activate if in valid mode
                    mse, mae, rmse, r2 = test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD, mode = 'val')
                    mse_list.append(mse) ; mae_list.append(mae) ; rmse_list.append(rmse) ; r2_list.append(r2)
                    loss_list.append(model.val_loss)
                print("regression was done baby~")
    
        else : 
            raise ValueError("config.mode should be either pretraining or finetuning!!")
            
    
    #FOLD다 돈 후에 되는 것    
    os.remove(glob(model.path2+'/*.pt')[0])
    
    if args.eval_mode : #i.e. evaluation mode
        #ACTIVATE THIS DURING EVAL MDOE 
        cv_eval_file = open(f'{model.path2}/eval_stats.txt', 'a', buffering=1)  #cross validation eval file 
        print(json.dumps(config.__dict__), file= cv_eval_file) #save config dict stuff 
        
        
        ##save the thing
        np.save(f'{model.path2}/labels_eval.npy',label_list, allow_pickle = True)
        np.save(f'{model.path2}/predictions_sigmoided_eval.npy',pred_list, allow_pickle = True)
        
        ##the aurocMean_list는 validation 을 yAware에서 받았을때는, validaiton의 mean값이 되지만, 
        ##testing을 받았을때는 testing의 mean값이됨
        #따라서 두 경우 모두 필요하니, 일단은 위에다가 복사해놓겠다    
        print("<<<Cross Validation Result>>>")
        if config.task_type == 'cls':        
            print(f"Mean AUROC : {aurocMean_list}")
            print(f"stdev AUROC : {aurocStd_list}")
            print(json.dumps({"loss_list": loss_list , "acc_list" : acc_list, "mean_auroc" : aurocMean_list, "std_auroc" : aurocStd_list, "last_epoch_list" : last_epoch_list}), file = cv_eval_file)  
            #print(json.dumps({"loss_list": loss_list , "acc_list" : acc_list, "mean_auroc" : aurocMean_list, "std_auroc" : aurocStd_list}), file = cv_eval_file)  
            
        elif config.task_type == 'reg' : 
            print(json.dumps({"mse_list " : str(mse_list)  , "mae_list " : str(mae_list)  , "rmse_list" : str(rmse_list) , "r2_list" : str(r2_list), "last_epoch_list" : last_epoch_list}), file = cv_eval_file)  
            
        else : 
            raise ValueError("use either cls or reg!")    
    
    
    else : #i.e. optuna mode
        ###ACTIVATE IF IN VALIDATION MODE
        ##use cv_val_file instead of cv_eval_file
        cv_val_file = open(f'{model.path2}/val_stats.txt', 'a', buffering=1)  #cross validation eval file 
        print(json.dumps(config.__dict__), file= cv_val_file) #save config dict stuff 
        
        #sincer we were returned the validation, aurocMean_list will be from validation 
        print("<<<Cross Validation Result>>>")
        if config.task_type == 'cls':        
            print(f"Mean AUROC : {aurocMean_list}")
            print(f"stdev AUROC : {aurocStd_list}")
            print(json.dumps({"loss_list": loss_list , "acc_list" : acc_list, "mean_auroc" : aurocMean_list, "std_auroc" : aurocStd_list, "last_epoch_list" : last_epoch_list}), file = cv_val_file)  
            #print(json.dumps({"loss_list": loss_list , "acc_list" : acc_list, "mean_auroc" : aurocMean_list, "std_auroc" : aurocStd_list}), file = cv_val_file)  
            
        elif config.task_type == 'reg' : 
            print(json.dumps({"mse_list " : str(mse_list)  , "mae_list " : str(mae_list)  , "rmse_list" : str(rmse_list) , "r2_list" : str(r2_list), "last_epoch_list" : last_epoch_list}), file = cv_val_file)  
            
        else : 
            raise ValueError("use either cls or reg!")
    
    

    
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S') # ADNI
    print("[main.py finished at {0}]".format(nowDatetime))
    ###
    
    #catchall pruner (objective func output에서 nan이 나오면 바로 prune)
    if not config.eval_mode : #if in optuna mode 
        if torch.tensor(aurocMean_list).isnan().any(): #i.e. if there are any nans, in the output
            raise optuna.TrialPruned() #prune this trial        
    
    return np.array(aurocMean_list).mean() #for optuna #in val mode, returns the validaiton auroc
    
##define function does the printing, plotting and AUROC MSE, MAE calculation to a seperate function
def test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD, mode ):
    """
    mode : either 'val' or 'test' 
    """
    if mode == "val":
        loss = model.val_loss
        acc = model.val_acc if config.task_type == "cls" else None #cls 일때만
    elif mode == 'test' : 
        loss = model.test_loss
        acc = model.test_acc if config.task_type == "cls" else None #cls 일때만
        #raise NotImplementedError("not tested yet")    
    else : 
        raise ValueError("Value ERror!")
    
    if config.task_type =='cls':
        #works only if binary_class is True!
        outGTnp = outGT.cpu().numpy()
        outPREDnp = outPRED.cpu().numpy()
        print('\n<<< Test Results: AUROC >>>')

        roc_score = roc_auc_score(outGTnp, outPREDnp)
        pred_arr = np.array([[1-pred_i, pred_i] for pred_i in outPREDnp])
        skplt.metrics.plot_roc(outGTnp, pred_arr,
                              title = f"task : {config.task}", #put in task names here
                      figsize = (6,6), title_fontsize="large",
                       plot_micro = False, plot_macro = False, 
                      classes_to_plot=[1])
        
        plt.legend([f'ROC curve for class {task_include[1]}, AUC : {roc_score : .2f}'])
        plt.savefig(model.path2  + f"/ROC_figure_{FOLD}.png" , dpi = 100) #그래도 일단 보기 위해 살려두자
        
        aurocMean = roc_score #이건 크게 필요없다 (그냥 원래 코드랑 비슷하게 보이려고 하는 것)
        print(json.dumps({f"{mode}_loss" : loss , f"{mode}_acc" : acc}), file= model.eval_stats_file) #get what the setting was (this is actually the validaiton file and not the evaluation file, but whatever)
        print(json.dumps({"AUROC" : aurocMean}), file = model.eval_stats_file)
        
        return aurocMean
        
        
    else : #reg일때
        ##calculating the MSE, MAE and so on  
        outGTnp = outGT.cpu().numpy()
        outPREDnp = outPRED.cpu().numpy()
        mse = mean_squared_error(outGTnp, outPREDnp)
        mae = mean_absolute_error(outGTnp, outPREDnp)
        rmse = np.sqrt(mean_squared_error(outGTnp, outPREDnp))
        r2 = r2_score(outGTnp, outPREDnp)
        
        print('\n<<< Test Results >>>')
        print('MSE: {:.2f}'.format(mse))
        print('MAE: {:.2f}'.format(mae))
        print('RMSE: {:.2f}'.format(rmse))
        print('R2-score: {:.4f}'.format(r2))
        #saving the file to eval_stats.txt
        print(json.dumps({f"{mode}_loss" : str(model.loss), f"{mode}_MSE" : str(mse), "test_MAE" : str(mae), "test_rmse" : str(rmse), "R2" : str(r2)}), file = model.eval_stats_file)
        return mse, mae, rmse, r2
        
        
if __name__ == "__main__":
    if args.eval_mode : #if in eval_mode
        main(trial = "not_doing_optuna")
        
    else : #i.e. if in optuna mode
        ##STUDY NAME이 같으면 다른 hyperparam option들이 겹칠 수 있으니, {study_name}받아서 하도록 만들기!
        
        #make db_folder if needed
        db_folder = os.makedirs(f"./{args.save_path}/optuna_db", exist_ok = True)
        
        
        db_dir = f"{args.save_path}/optuna_db/{args.task}-{args.layer_control}-{config.task_type}-{str(args.pretrained_path).split('/')[-1].split('.')[0]}-train_{args.train_num}-batch_{args.batch_size}"
        
        #{config.save_path}/{config.task}-{self.layer_control}-{config.task_type}
        url = "sqlite:///" + os.path.join(os.getcwd(), (db_dir + '.db'))
        print(url)
        # Add stream handler of stdout to show the messages => 이건 되어있던데 왜하는 건지 모르겠다 
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        
        storage = optuna.storages.RDBStorage(
            url = url,
            heartbeat_interval = 60,
            grace_period = 120    
        )
        
        study = optuna.create_study(study_name = "test_study_name", 
                                    storage = storage,
                                    load_if_exists = True,
                                    direction = 'maximize') #pruner, sampler도 나중에 넣기) 
        study.optimize(main, n_trials = 10, timeout = 60000) #8 hrs : 28800
        #https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
    
    
    
