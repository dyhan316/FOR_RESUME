import argparse
import os 
import json 
import math
import random
from collections import defaultdict

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scikitplot as skplt
import torch 
from scipy import stats
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, precision_score, accuracy_score, recall_score
from torch.optim.lr_scheduler import _LRScheduler


def str2bool(v): #true, false를 argparse로 받을 수 있도록 하기!
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
def argument_setting():
    parser = argparse.ArgumentParser()
    # environment setting
    parser.add_argument('--run_where', choices=['sdcc', 'lab', 'kisti'], type = str,
                        help='where to run the thing (decides whether to run config.py or config_lab.py')
    parser.add_argument('--save_path', required = False, default = './finetune_results_default_dir', type = str, 
                        help="where to save the evaluation results (i.e. where is model.path2?)")
    parser.add_argument('--wandb_name',required = False, default = "default_WANDB_file", type = str,
                        help="wandb where to save" )
    parser.add_argument('--gpus', type=str)
    # experiment setting
    parser.add_argument("--mode", type=str, default="FINE_TUNING", choices=["FINE_TUNING"], required=False,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Set the pretrained model path.")  
    parser.add_argument("--dataset", type=str, choices=['ABCD','CHA','ADNI','UKB'], 
                        help="which dataset to use")
    parser.add_argument("--data_type", type=str, choices=['raw', 'resize128'],
                        help="data type to use.")
    parser.add_argument("--task", type=str, required=True, default=0, 
                        help="which task within config.py to do")
    parser.add_argument('--binary_class',required = False, default = True, type= str2bool,
                        help='whether to use binary classification or not ')
    parser.add_argument("--stratify", type=str, choices=["strat", "balan", "iter_strat", "balan_iter_strat",
                                                         "balan_strat", "psm_balan_iter_strat"],
                        help="Set training samples are stratified or balanced for fine-tuning task.") #balan_iter_strat : first balances the number of samples, then does iterative stratification, "balan_strat" : just does balancing (so that not imbalanced), then does stratification
    parser.add_argument("--balanced_testset", type=str2bool, default=True, 
                        help="make balanced test dataset")
    parser.add_argument("--random_seed", type=int, required=False, default=0,
                        help="Random seed for reproduction.")
    parser.add_argument("--num_folds", type=int, default=5,
                        help="number of folds to split")
    parser.add_argument("--train_num", type=str, default='all', required=False, 
                        help="Set the number of training samples.")      
    parser.add_argument("--test_ratio", type=float, default=0.2, choices=np.arange(0.01,0.5,0.01),
                        help="test set ratio of dataset split")      
    parser.add_argument('--num_workers', required = False, default = 8, type = int,
                        help="number of workers to use (default of 8)")
    # model training setting
    parser.add_argument('--input_option', required = True, type = str,
                        help='possible options : yAware, BT_org, or BT_unet,  which option top use when putting in input (reshape? padd or crop? at which dimensions?')
    parser.add_argument('--eval', required = False, default = False, type = str2bool,
                        help='whether ot do evaluation (using the best trial) or not (True or False)')
    parser.add_argument('--nb_epochs', required = False, type = int, default = None,
                        help="custom set epoch if wanted." )    
    parser.add_argument('--batch_size', required = False, default = 8,type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--lr_range', required = False, type = str,
                        help= "what optuna lr range to use, in the for of `1e-5/1e-2` ")
    parser.add_argument("--layer_control", type=str, choices=['tune_all', 'freeze', 'tune_diff'], required=False, 
                        help="Set pretrained weight layer control option.")     
    parser.add_argument("--weight_decay", type=float, default=1e-2)    
    parser.add_argument('--wd_range', required = False, type = str,
                        help="what optuna wd range to use, in the for of `1e-5/1e-2` ")
    parser.add_argument('--lr_schedule', required = False, default = None, type = str, 
                        help='what lr scheduler to use : lr_plateau, cosine_annealing_decay, cosine_annealing,onecyclelr, custom_1, custom_2, custom_3, custom_4,5,6, cosine_annealing_faster, SGDR_1, SGDR_2, SGDR_3 (SGD+momentum+COSINEANNEALINGWARMRESTARTS), ... ')
    parser.add_argument('--patience',required = False, default = 100, type = int, 
                        help="how much patience?" )
    parser.add_argument('--early_criteria' , required = False, default = 'loss', type = str ,
                        help="use valid AUROC or loss or none for early stopping criteria?(options : 'AUROC' or 'loss' or 'none' ")   
    parser.add_argument('--AMP', required = False, default = True, type = str2bool,
                        help="if True, uses AMP, if False does not")
    parser.add_argument('--prune', required = False, default = False, type = str2bool, 
                        help="whether to prune based on AUROC or not (only prunes based on first fold)")
    parser.add_argument('--upweight', required = False, default = False, type = str2bool, 
                        help="whether or not to do pruning")
    parser.add_argument('--verbose', required = False, default = False, type = str2bool,
                        help='whether to use weight_tracker or not ')    
    #parser.add_argument('--BN' , required = False, type = str , help = "what optuna BN option range to use, in the for of `1e-5/1e-2` ")
    args = parser.parse_args() 

    return args


def seed_all(random_seed):
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False 
    return random_seed



###finding out the contents of the fold without going through the whole training procedure (미리 for loop enumerate 해서 분포를 보기)
def get_info_fold(kf_split, df, target_col): #get info from fold
    """
    * kf_split : the `kf.split(XX)`된것
    * df : the dataframe with the metadata I will use
    * target_col : the columns in the df that I'll take statistics of 
    """
    train_dict = defaultdict(list)
    valid_dict = defaultdict(list)
    
    for FOLD, (train_idx, valid_idx) in enumerate(kf_split): 
        label_train = df.iloc[train_idx]
        label_valid = df.iloc[valid_idx]
        for col in target_col:
            if df[col].nunique()<=10: # case: categorical variable
                keys=list(map(lambda x: f'{col}[{x}]', label_train[col].unique()))
                train_counts=label_train[col].value_counts()
                valid_counts=label_valid[col].value_counts()
                for i, key in enumerate(keys):
                    train_dict[key].append(train_counts[i])
                    valid_dict[key].append(valid_counts[i])
            else: # case: continuous variable
                train_dict[f'{col}-mean/std'].append(f'{label_train[col].mean():.2f} / {label_train[col].std():.2f}')
                valid_dict[f'{col}-mean/std'].append(f'{label_valid[col].mean():.2f} / {label_valid[col].std():.2f}')
                    
    print("=== Fold-wise categorical values information of training set ===")
    print(pd.DataFrame(train_dict))
    print("=== Fold-wise categorical values information of validation set ===")
    print(pd.DataFrame(valid_dict))
    
    
def print_CV(config, file_pth, results_):
    print("<<< Cross Validation Result >>>")
    if config.task_type == 'cls':      
        print(f"Mean AUROC : {results_.pop('aurocMean')}")
        print(f"stdev AUROC : {results_.pop('aurocStd')}")
        print(json.dumps(results_), file = file_pth)  

    elif config.task_type == 'reg':
        for k in results_:
            results_[k] = list(map(lambda x: str(x), results_[k]))
        print(json.dumps(results_), file = file_pth) 
        
        
##define function does the printing, plotting and AUROC MSE, MAE calculation to a seperate function
def test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD, compute_only = False ):
    """
    if `compute_only` is True : roc graph 그린다던지, stats filedㅔ 적는다던지 안하고 그냥 roc값만 주는 것 
    """
    if config.task_type =='cls':
        #works only if binary_class is True!
        outGTnp = outGT.cpu().numpy()
        outPREDnp = outPRED.cpu().numpy()

        roc_score = roc_auc_score(outGTnp, outPREDnp)
        pred_arr = np.array([[1-pred_i, pred_i] for pred_i in outPREDnp])
        aurocMean = roc_score #이건 크게 필요없다 (그냥 원래 코드랑 비슷하게 보이려고 하는 것)
        if not compute_only : 
            print('\n<<< Test Results: AUROC >>>')
            skplt.metrics.plot_roc(outGTnp, pred_arr,
                                  title = f"task : {config.task}", #put in task names here
                          figsize = (6,6), title_fontsize="large",
                           plot_micro = False, plot_macro = False, 
                          classes_to_plot=[1])
            
            plt.legend([f'ROC curve for class {task_include[1]}, AUC : {roc_score : .2f}'])
            plt.savefig(model.path2  + f"/ROC_figure_{FOLD}.png" , dpi = 100) #그래도 일단 보기 위해 살려두자
        
        return aurocMean
        
    else : #reg일때
        ##calculating the MSE, MAE and so on  
        outGTnp = outGT.cpu().numpy()
        outPREDnp = outPRED.cpu().numpy()
        mse = mean_squared_error(outGTnp, outPREDnp)
        mae = mean_absolute_error(outGTnp, outPREDnp)
        rmse = np.sqrt(mean_squared_error(outGTnp, outPREDnp))
        r2 = r2_score(outGTnp, outPREDnp)
        
        if not compute_only : 
            print('\n<<< Test Results >>>')
            print('MSE: {:.2f}'.format(mse))
            print('MAE: {:.2f}'.format(mae))
            print('RMSE: {:.2f}'.format(rmse))
            print('R2-score: {:.4f}'.format(r2))
            
        return mse, mae, rmse, r2


def ensemble_prediction(config, outGT_list, outPRED_list, stat_measure, weights, model, task_include):
    """
    https://machinelearningmastery.com/voting-ensembles-with-python/
    https://jermwatt.github.io/machine_learning_refined/notes/11_Feature_learning/11_9_Bagging.html
    * stat_measure : 'mean', 'median', 'weighted', 'hard_mode', or 'hard_weighted'
    * weights = weights to use if weighted
    
    soft voting
    * mean
    * median
    * weighted
    
    hard voting
    * mode (vote)
    * weighted
    
    """
    #converting to np.array in case it's list
    PRED_stack = torch.stack(outPRED_list) 
    PRED_stack = PRED_stack.cpu().numpy() 
    weights = np.array(weights)
    outGT = outGT_list[0].cpu().numpy() #because we only need one of the five (same ordering)
    
    #SOFT VOTING
    if stat_measure in ["mean", "median", "weighted"]: 
        if stat_measure == "mean" : 
            PRED_summary = PRED_stack.mean(axis=0)
        elif stat_measure == "median" : 
            PRED_summary = np.median(PRED_stack, axis = 0)
        elif stat_measure == "weighted" :
            PRED_summary = (PRED_stack.T @ weights)/weights.sum()
            
        #calculating statistics
        auroc_value = test_data_analysis_2(config, model, torch.Tensor(outGT), torch.Tensor(PRED_summary), task_include, 0, compute_only = True)
        
        pred_arr = np.array(PRED_summary > 0.5, dtype = float)  #binarized pred (0.5 threshold)
        acc = accuracy_score(outGT, pred_arr) #precision
        prec = precision_score(outGT, pred_arr)
        recall = recall_score(outGT, pred_arr)
        
        #final results to be returned as output 
        final_results = dict(stat_measure = stat_measure,
                             auroc_value = auroc_value,
                             acc = acc,
                             prec = prec,
                             recall = recall)
    
    #HARD VOTING
    #https://vitalflux.com/hard-vs-soft-voting-classifier-python-example/
    #위의 예시보고 하기!
    elif stat_measure in ["hard_mode", "hard_weighted"]:
        predictions = np.array(PRED_stack > 0.5, dtype = float) #pred of all five 
        
        if stat_measure == "hard_mode" : 
            pred_arr = stats.mode(predictions, axis = 0)[0][0] #vote based on mode
            
        elif stat_measure == "hard_weighted":
            ###DO FROM HERE 
            logit_arr = (predictions.T @ weights)/np.sum(weights)
            pred_arr = np.array(logit_arr>0.5, dtype = float)
               
        acc = accuracy_score(outGT, pred_arr) #precision
        prec = precision_score(outGT, pred_arr)
        recall = recall_score(outGT, pred_arr)
        
        #NO auroc_value! (but still has acc and prec)
        #final results to be returned as output 
        final_results = dict(stat_measure = stat_measure,
                             acc = acc,
                             prec = prec,
                             recall = recall)
    else : 
        raise ValueError(f"{stat_measure} is not one of the stat measures that can be used")

    return final_results
        
    #elif stat_measure == "hard_mode" : 
    #    pass
    #    근데 true_arr로는 AUROC못구하잖아... logit value가 아닌, 0 or 1 value이니 
    #    어떻게 하지? accuracy만 구할까? 
    #    true_arr
    #    see the codes from the websites on IPAD! (check and see if I can find something)(also that book is very very good!)

    
##FROM https://gaussian37.github.io/dl-pytorch-lr_scheduler/ 
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
            
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt', direction = "max"):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
            direction : whether to try to maximize/minimize the criteria 
                * if using `val_score` : use "min"
                * if using `val_auroc` : use "max"
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        if direction == "min" :
            self.raw_best_score = np.Inf
        elif direction == "max" : 
            self.raw_best_score = -np.Inf
        self.delta = delta
        self.path = path
        
        self.direction = direction

    def __call__(self, val_score, model):
        #direction에 따라 score을 줄지 말지 고르기 
        if self.direction == "max": 
            score = val_score
        elif self.direction == "min":
            score = -val_score
        else : 
            raise ValueError("use either max/min for direction")
        ####
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else: #i.e. score is higher or SAME than the best score #즉,same score이면 early stoping counter 을 안씀
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.raw_best_score:.6f} --> {val_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.raw_best_score = val_score