import datetime
import json
import logging
import os
import shutil
import sys
import time
import warnings
from glob import glob
from copy import deepcopy
from collections import defaultdict
from functools import reduce

import optuna
import wandb
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from config import Config, PRETRAINING, FINE_TUNING
from data_utils import get_folds_original, get_psm_balan_iter_strat_folds, make_dataloaders
from models.densenet import densenet121
from models.unet import UNet
from utils import argument_setting, seed_all, test_data_analysis_2, ensemble_prediction, print_CV
from yAwareContrastiveLearning_optuna import yAwareCLModel


### Helper funcitons ###
def sanity_check(config):
    '''To check if configurations are ok'''
    if config.binary_class == False and config.num_classes == 2:
        print("[binary_class, num_classes] you aren't using binary classification and set num_classes ==2? probably wrong ") # ????
    if config.stratify in ['balan', "balan_iter_strat", "balan_strat"] and config.upweight == True :  #i.e. when two imbalanced thing are activate together
        raise ValueError("[stratify] Only raise either balan or upweight, not both!!") 
    if config.task_type not in ['cls','reg']: 
        raise ValueError("[task type] should use either cls or reg!") 
    if config.input_option not in ["yAware", "BT_org", "BT_unet", "resize128"]:
        raise ValueError("this input_option value is not expected, choose one of 'yAware, BT_org, or BT_unet'. ")
    if config.BN not in ["none","inst","default"]:
        raise ValueError("""[BN] - args.BN should be one of ["none","inst","default"] """)
    if config.model not in ["DenseNet", "UNet"]:
        raise ValueError(f"[model] - Unkown model: {config.model}")
    if config.run_where not in  ['sdcc', 'lab', 'kisti'] : 
        raise ValueError("[run_where] option should be in [sdcc, lab, kisti]")
    if config.task_type == 'cls':
        assert len(config.task_name.split('/')) == 2, 'Set two labels.'
        if config.binary_class == False:
            assert config.num_classes == 2, 'Set config.num_classes == 2'


def suggest_options(trial, config):
    # optuna lr
    if config.lr_range:  #if lr range was specified
        lr_range = [float(i) for i in config.lr_range.split('/')] 
    elif config.layer_control == "tune_all" : #use default if using tune_all
        lr_range = [1e-6, 1e0]
    else: #default if using freeze
        lr_range = [1e-5, 1e1]
    config.lr = trial.suggest_float("learning_rate", *lr_range, log = True) 

    # optuna wd
    wd_range = [float(i) for i in config.wd_range.split('/')]if config.wd_range else [1e-9 , 1e2]  #if wd range was specified
    config.weight_decay = trial.suggest_float("weight_decay", *wd_range, log=True)

    #optuna BN method
    #args.BN = trial.suggest_categorical("BN_option", ['none','inst'])


def select_model(config):
    if config.model == "DenseNet":
        mode = 'classifier'
        if config.BN == "none":
            mode = mode + '_no_BN'
        elif config.BN == "inst":
            mode = mode + '_inst_BN'
        net = densenet121(mode=mode, drop_rate=0.0, num_classes=config.num_classes)
    elif config.model == "UNet":
        net = UNet(config.num_classes, mode="classif")
        
    return net


def select_loss(train_labels, config):
    if config.task_type == 'cls' and config.binary_class == False:
        loss = CrossEntropyLoss()
    elif config.task_type == 'cls' and config.binary_class == True:
        if config.upweight : 
            pos_weight = (train_labels==0.).sum()/train_labels.sum() #i.e. the proportion of 0 to 1 (i.e. how much to boost 1 )
            loss = BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        else : #just default version (not upweighted)
            loss = BCEWithLogitsLoss()
    elif config.task_type == 'reg' : # config.task_type == 'reg': # ADNI
        loss = MSELoss()        
        
    return loss 
    
    
def run_experiment(trial, kf_split, label_tv, label_test):  
    for FOLD, (train_idx, valid_idx) in enumerate(kf_split):
        print(f'<<< StratifiedKFold: {FOLD+1}/{config.num_folds} >>>')
        config.fold = FOLD #which fold it is in

        #setting up wandb
        #https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893
        job_type = "train" if not config.eval else "eval"
        wandb.init(project=config.wandb_name, config=config, name=f"fold{FOLD}", job_type=job_type, reinit=True,
                   group = f"trial{trial.number}_-{config.layer_control}_{str(config.pretrained_path).split('/')[-1].split('.')[0]} \
                             _{config.lr_schedule}_{config.stratify}") #becasue optuna,we have to redo the trials
        wandb.config.weight_used = str(config.pretrained_path).split('/')[-1].split('.')[0]

        labels = {'train': label_tv.iloc[train_idx], 'valid': label_tv.iloc[valid_idx], 'test': label_test}
        dataloaders = make_dataloaders(labels, config)
        net = select_model(config)
        loss = select_loss(labels['train'], config)
        wandb.watch(models=net, criterion=loss)
        results = defaultdict(list)

        # wrap network into yAware Model?
        model = yAwareCLModel(net, loss, dataloaders['train'], dataloaders['valid'], dataloaders['test'],
                              config, config.task_name, config.train_num, config.layer_control,
                              None, config.pretrained_path, FOLD, wandb, trial) # ADNI

        # train or eval
        if config.eval : 
            ##load checkpoint => don't do fine_tuning, but just do the eval_model part (only evaluaiton) to get outGT outPRED and so on 
            ckpt_dir = glob(model.best_trial_path+f'/*{FOLD}.pt')[0] #this fold's best trial's ckpt dir
            model.model.load_state_dict(torch.load(ckpt_dir)) #model (yAware)내의 net 를 직접 꺼내기
            print(f"FOLD : {FOLD} checkpoint loading from {ckpt_dir} success")

            outGT, outPRED, loss, acc, aurocMean = model.eval_model(mode = 'test')
            model.test_loss = loss
            model.test_acc = acc
            model.aurocMean = aurocMean

            #for majority voting
            results['GT'].append(outGT)
            results['PRED'].append(outPRED)
        else:  #OPTUNA (regular fine_tuning)
            outGT, outPRED, last_epoch = model.fine_tuning() # does actual finetuning #returns BEST validation results (after early stopping is reached or when all the epochs are reached)
            results['last_epoch'].append(last_epoch) #last_epoch : last epoch of the training phaes

        ###calculating mean AUROC and saving the AUC graph 
        if config.task_type == 'cls':
            if config.binary_class == False :  #original mode
                raise NotImplementedError("have to redo this!! (so that test_data_analysis_2 is merged to the original  test_data_analysis!!")

            elif config.binary_class == True :                                         
                #ACTIVATE IF IN VALID MODE
                loss_ = model.test_loss if args.eval else model.val_loss
                acc_ = model.test_acc if args.eval else model.val_acc
                mode_ = 'test' if args.eval else 'val'

                auroc_mean_arr = test_data_analysis_2(config, model, outGT, outPRED, config.task_include, FOLD)
                results['loss'].append(loss_)
                results['acc'].append(acc_)
                results['aurocMean'].append(np.mean(auroc_mean_arr))
                results['aurocStd'].append(np.std(auroc_mean_arr))

                print(json.dumps({f"{mode_}_loss": loss_ , f"{mode_}_acc": acc_}), file= model.eval_stats_file) 
                print(json.dumps({"AUROC" : auroc_mean_arr}), file = model.eval_stats_file)
                if not args.eval and (auroc_mean_arr <=0.5) : 
                    print("== pruned because fold's auroc <= 0.5 ==")
                    wandb.config.state = "fold_auroc_below_0.5"
                    raise optuna.TrialPruned()

        else: # config.task_type == 'reg':
            #ACTIVATE IF IN EVAL MODE
            #mse, mae, rmse, r2 = test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD)
            #mse_list.append(mse) ; mae_list.append(mae) ; rmse_list.append(rmse) ; r2_list.append(r2)
            #loss_list.append(model.test_loss)
            raise NotImplementedError("reg is not implemented yet (has to be checked)")
            #activate if in valid mode
            mse, mae, rmse, r2 = test_data_analysis_2(config, model, outGT, outPRED, config.task_include, FOLD)
            print(json.dumps({f"val_loss" : str(model.loss), f"val_MSE" : str(mse), "val_MAE" : str(mae), "val_rmse" : str(rmse), "R2" : str(r2)}), file = model.eval_stats_file)

            for name, res in zip(['mse', 'mae', 'rmse', 'r2', 'loss'], [mse, mae, rmse, r2, model.val_loss]):
                result['name'].append(res)
            print("== regression was done baby~ ==")

        # wandb summary
        wandb.run.summary['state'] = "success"

        # wandb summary if evaluation
        if not args.eval :  #in optuna mode
            if config.task_type == 'cls' : 
                wandb.run.summary['final_val_AUROC'] = auroc_mean_arr
                wandb.run.summary['final_val_acc'] = model.val_acc
            wandb.run.summary['final_val_loss'] = model.val_loss

        wandb.finish()
        
    return model, results


def ensemble_voting(trial, model, results, config):
    #FOLD다 돈 후에 되는 것 
    ##from here on out implement things like majority voting and so on 
    ##see https://machinelearningmastery.com/voting-ensembles-with-python/
    if config.eval :
        with open(model.best_trial_path + "/val_stats.txt", "r") as file :
            data = [json.loads(line) for line in file]
            val_auroc_list = [d["mean_auroc"] for d in data if "mean_auroc" in d][0] #used as weight 
            val_auroc_list = [float(x) for x in val_auroc_list]

        cv_val_file = open(f'{model.path2}/test_stats_final.txt', 'a', buffering=1)  #cross validation eval file 

        voting_list = []
        for measure in ['mean', 'median', 'weighted', 'hard_mode', 'hard_weighted']:
            ensembled = ensemble_prediction(config, results['GT'], results['PRED'], stat_measure=measure,
                                            weights=val_auroc_list, model=model, task_include=config.task_include)
            voting_list.append(ensembled)
            print(json.dumps(ensembled), file=cv_val_file)

        # copy the best trial's validation results to the best trial test results, so that we can look at em better
        shutil.copytree(model.best_trial_path, os.path.join(model.path2, "validation_results") )
        cv_val_file = open(f'{model.path2}/test_stats_final.txt', 'a', buffering=1)  #cross validation eval file 
    else: 
        cv_val_file = open(f'{model.path2}/val_stats.txt', 'a', buffering=1)  #cross validation eval file 
        print(json.dumps(config.__dict__), file=cv_val_file) #save config dict stuff 

    #since we were returned the validation, aurocMean_list will be from validation 
    print_CV(config, cv_val_file, results)

    # prune current trial
    # catchall pruner (objective func output에서 nan이 나오면 바로 prune)
    if torch.tensor(results['aurocMean']).isnan().any(): #i.e. if there are any nans, in the output
        print("pruned due to NAN")
        raise optuna.TrialPruned() #prune this trial    

    # caluclating validation/testing (depending on mode) mean AUC from 5 folds.
    mean_AUC = np.array(results['aurocMean']).mean() #for optuna #in val mode, returns the validaiton auroc
    if (study.trials_dataframe()['state'] == "COMPLETE").sum() == 0 :  #i.e. first (complete) trial
        pass 
    elif study.best_value > mean_AUC :  #i.e. not first trial, but not best trial
        [os.remove(pt_i) for pt_i in glob(model.path2+'/*.pt')]

    return mean_AUC
    
    
def main(trial):
    # Set start time
    now = datetime.datetime.now()
    print(f"[main.py started at {now.strftime('%Y-%m-%d %H:%M:%S')}]")
    start_time = time.time()       

    # OPTUNA hyperparam tuning(lr, wd)
    suggest_options(trial, config)
    
    # Split dataset
    get_folds = get_folds_original if 'psm' not in config.stratify else get_psm_balan_iter_strat_folds
    kf_split, label_tv, label_test = get_folds(config)
    
    # Run k-fold experiments, ensemble voting
    model, results = run_experiment(trial, kf_split, label_tv, label_test)
    mean_AUC = ensemble_voting(trial, model, deepcopy(results), config)
    
    end_time = time.time()
    print('\nTotal', (end_time - start_time) / 60, 'minutes elapsed.')
    now = datetime.datetime.now() # ADNI
    print(f"[main.py finished at {now.strftime('%Y-%m-%d %H:%M:%S')}]")

        
if __name__ == "__main__":    
    print("*** Experiment Starts ***")
    # Configuration setting, Sanity check, Set seed for reproducibility
    args = argument_setting()    
    config = Config(args) # changed to update config using all attributes of args(lr_range, num_workers, task, ...). we will use only "config" from here.
    sanity_check(config)
    seed_all(config.random_seed) # setting random_seed of np.random is sufficient because pandas's random things are governed by seed of np.random
    
    # optuna setting
    db_folder = os.makedirs(f"./{config.save_path}/optuna_db", exist_ok = True)
    db_dir = f"{config.save_path}/optuna_db/{config.task}-{config.layer_control}-{config.task_type}-{str(config.pretrained_path).split('/')[-1].split('.')[0]}-train_{config.train_num}-batch_{config.batch_size}"
    url = "sqlite:///" + os.path.join(os.getcwd(), (db_dir + '.db'))
    print(' - db_dir:',url)
    # Add stream handler of stdout to show the messages => 이건 되어있던데 왜하는 건지 모르겠다 
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage = optuna.storages.RDBStorage( url = url, heartbeat_interval = 60, grace_period = 120 )
    study = optuna.create_study(study_name = "test_study_name", 
                                storage = storage,
                                load_if_exists = True,
                                pruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=5, interval_steps=1), #wrap with patience?
                                direction = 'maximize') #pruner, sampler도 나중에 넣기) 
    
    # run experiment(evaluation only or finetuning)
    if args.eval :
        main(study.best_trial)
    else: 
        study.optimize(main, n_trials = 200, timeout = 160000) #~2day
        
    print("*** Experiment Ended ***")
    ##using best model further inspired from  : https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/010_reuse_best_trial.html
    ##즉, 저기서 testing part만 뽑아서 돌리던지 해야할듯? (main 에 다 넣지 말고)