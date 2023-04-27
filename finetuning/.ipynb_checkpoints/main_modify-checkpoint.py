
### ADNI
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
import time
import datetime
###
import numpy as np



#print("change this below if needed!!")
print("if OOM occurs, dataset.__init__ might be too large...? (i.e. loading too much on memory (or during persist_workers = True)")
from dataset_BT import MRI_dataset  #save as ADNI datseet haha

from torch.utils.data import DataLoader, Dataset, RandomSampler
from yAwareContrastiveLearning_modify import yAwareCLModel
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

#===============#


###

def main():
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S') # ADNI
    print("[main.py started at {0}]".format(nowDatetime))
    start_time = time.time() # ADNI
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
    parser.add_argument('--batch_size', required = False, default = 8,type=int, metavar='N',
                    help='mini-batch size')
    parser.add_argument('--learning_rate', required = False,default=1e-4, type=float, metavar='LR',
                    help='base learning rate')
    parser.add_argument('--weight_decay',required = False, default=5e-5, type=float, metavar='W',
                    help='weight decay')
    
    parser.add_argument('--input_option', required = True, type = str, help='possible options : yAware, BT_org, or BT_unet,  which option top use when putting in input (reshape? padd or crop? at which dimensions?')
    parser.add_argument('--BN' , required = True, type = str, help = "what BN to use : none, inst, default")
    
    parser.add_argument('--binary_class',required = False, default = True, type= str2bool,
                    help='whether to use binary classification or not ')
    parser.add_argument('--save_path', required = False, default = './finetune_results_default_dir', type = str, 
                        help = "where to save the evaluation results (i.e. where is model.path2?)")
    parser.add_argument('--run_where', required = False, default = 'sdcc', type = str, help= 'where to run the thing (decides whether to run config.py or config_lab.py, options : sdcc or lab')
    
    ##########################
    args = parser.parse_args()

    if args.run_where == 'sdcc' : 
        from config import Config, PRETRAINING, FINE_TUNING
    elif args.run_where == 'lab' : 
        from config_lab import Config, PRETRAINING, FINE_TUNING
    else : 
        raise ValueError("run_where option should either be sdcc or lab")
        
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING
    

    #####changed##### (add stuff so that yAware thing can look at this to get things)
    config = Config(mode, args.task) 
    args.task_name = config.task_name ##ADDED (so that we don't have to redefine task_name every time... just get from the config itself
    config.batch_size = args.batch_size
    config.lr = args.learning_rate
    config.weight_decay = args.weight_decay
    config.binary_class = args.binary_class
    config.save_path = args.save_path
    config.stratify = args.stratify
    config.layer_control = args.layer_control
    config.pretrained_path = Path(args.pretrained_path).parts[-1]
    config.train_num = args.train_num
    
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
            if args.stratify == 'strat':
                ratio = len(data_1) / (len(data_1) + len(data_2))
                len_1_train = round(args.train_num*ratio)
                len_2_train = args.train_num - len_1_train
                len_1_valid = round(int(args.train_num*config.valid_ratio)*ratio)
                len_2_valid = int(args.train_num*config.valid_ratio) - len_1_valid
                assert args.train_num*(1+config.valid_ratio) < (len(data_1) + len(data_2)), 'Not enough valid data. Set smaller --train_num or smaller config.valid_ratio in config.py.'
                train1, valid1, test1 = np.split(data_1.sample(frac=1, random_state=random_seed), 
                                                 [len_1_train, len_1_train + len_1_valid])
                train2, valid2, test2 = np.split(data_2.sample(frac=1, random_state=random_seed), 
                                                 [len_2_train, len_2_train + len_2_valid])
                label_train = pd.concat([train1, train2]).sample(frac=1, random_state=random_seed)
                label_valid = pd.concat([valid1, valid2]).sample(frac=1, random_state=random_seed)
                label_test = pd.concat([test1, test2]).sample(frac=1, random_state=random_seed)
                assert len(label_test) >= 100, 'Not enough test data. (Total: {0})'.format(len(label_test))
            else: # args.stratify == 'balan'
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
                assert len(label_test) >= 100, 'Not enough test data (should have at least 100). (Total: {0})'.format(len(label_test))

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

            label_train, label_valid, label_test = np.split(labels.sample(frac=1, random_state=random_seed), 
                                                            [args.train_num, int(args.train_num*(1+config.valid_ratio))])
            
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
        
        
    #for train_idx, valid_idx in kf.split(label_tv, label_tv[label_name]): #kf could be skf or just kf 
    #train_idx, valid_idx : stratified k-fold (무조건 stratified= strat을 써야함!) (왜냐하면 label_tv가 애초에 stratified되어있다는 가정을 깔고 가는 거니)
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
            ###############밑을 조금 많이 바꿔야할듯!!!(currently only done for (probably) ADNI#######################
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
            
            outGT, outPRED, last_epoch = model.fine_tuning() # does actual finetuning 
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
                    #modified version 
                    auroc_mean_arr = test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD)
                    aurocMean_list.append(np.mean(auroc_mean_arr))
                    aurocStd_list.append(np.std(auroc_mean_arr))
                    loss_list.append(model.test_loss)
                    acc_list.append(model.test_acc)
                    
                
            else: # config.task_type == 'reg':
                mse, mae, rmse, r2 = test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD)
                mse_list.append(mse) ; mae_list.append(mae) ; rmse_list.append(rmse) ; r2_list.append(r2)
                loss_list.append(model.test_loss)
                print("regression was done baby~")
    
        else : 
            raise ValueError("config.mode should be either pretraining or finetuning!!")
    #FOLD다 돈 후에 되는 것    
    cv_eval_file = open(f'{model.path2}/eval_stats.txt', 'a', buffering=1)  #cross validation eval file 
    os.remove(glob(model.path2+'/*.pt')[0])
    
    print(json.dumps(config.__dict__), file= cv_eval_file) #save config dict stuff 
    
    
    ##save the thing
    np.save(f'{model.path2}/labels.npy',label_list, allow_pickle = True)
    np.save(f'{model.path2}/predictions_sigmoided.npy',pred_list, allow_pickle = True)
    
    
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
    
    #below : same as main_cv.py
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S') # ADNI
    print("[main.py finished at {0}]".format(nowDatetime))
    ###
    
##define function does the printing, plotting and AUROC MSE, MAE calculation to a seperate function
def test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD):
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
        print(json.dumps({"test_loss" : model.test_loss , "test_acc" : model.test_acc}), file= model.eval_stats_file) #get what the setting was   
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
        print(json.dumps({"test_loss" : str(model.test_loss), "test_MSE" : str(mse), "test_MAE" : str(mae), "test_rmse" : str(rmse), "R2" : str(r2)}), file = model.eval_stats_file)
        #print(json.dumps({"test_loss" : str(model.test_loss / len(model.loader_test.dataset)), "test_MSE" : str(mse), "test_MAE" : str(mae), "test_rmse" : str(rmse), "R2" : str(r2)}), file = model.eval_stats_file)
        return mse, mae, rmse, r2
        
        
if __name__ == "__main__":
    main()
    
