from glob import glob
from pathlib import Path
import os 
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, precision_score, accuracy_score, recall_score
import scikitplot as skplt
import numpy as np 
import json 
import matplotlib.pyplot as plt 
import torch 
from scipy import stats


import math
from torch.optim.lr_scheduler import _LRScheduler
import pandas as pd 

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

###finding out the contents of the fold without going through the whole training procedure (미리 for loop enumerate 해서 분포를 보기)
def get_info_fold(kf_split, df, target_col): #get info from fold
    """
    * kf_split : the `kf.split(XX)`된것
    * df : the dataframe with the metadata I will use
    * target_col : the columns in the df that I'll take statistics of 
    """
    print("일일히 print하지 않고, list of things 로 되도록 하기! (not individual)")
    #기존처럼 그 2D array로 하기? 
    
    
    train_dict = {}
    valid_dict = {}
    
    
    for FOLD, (train_idx, valid_idx) in enumerate(kf_split): 
        print(train_idx, valid_idx)
        #print(f"\n=====FOLD : {FOLD}=====")
        label_train = df.iloc[train_idx] #skf.split으로 한 train_idx, valid_idx를 실제로 넣어줘서 값들을 구한다  
        label_valid = df.iloc[valid_idx]
        
        for col in target_col :             
            if len(set(label_train[col])) > 10 : 
                print(f"column {col} had more than 10 labels, and therefore will be treated as a continuous value and not be printed")
            else :  #assume that its actual labels (not sth like age)                
                #print(f"---column : {col}---")
                classes = set(label_train[col])
                for class_i in classes:
                    key = f"{col}_{class_i}"
                    if FOLD == 0 : 
                        train_dict[key] = []
                        valid_dict[key] = []
                    
                    
                    num_class_i = (label_train[col] == class_i).sum()
                    train_dict[key].append(num_class_i)
                    #print(f"training, # of class {class_i} : {num_class_i}")
                    
                    num_class_i = (label_valid[col] == class_i).sum()
                    valid_dict[key].append(num_class_i)
                    #print(f"validation, # of class {class_i} : {num_class_i}")
                    
    train_df = pd.DataFrame(train_dict)
    valid_df = pd.DataFrame(valid_dict)
    print("===training===")
    print(train_df)
    print("===validation===")
    print(valid_df)
                    
                
    return train_df, valid_df
        
        #print(f"FOLD : {FOLD} | "
        
#def print_corresponding(df, ):
    

#if size(set(xX))> 20, don't do it 



##define function does the printing, plotting and AUROC MSE, MAE calculation to a seperate function
def test_data_analysis_2(config, model, outGT, outPRED, task_include, FOLD, compute_only = False ):
    """
    if `compute_only` is True : roc graph 그린다던지, stats filedㅔ 적는다던지 안하고 그냥 roc값만 주는 것 
    """
    #if mode == 'test' : 
    #    raise NotImplementedError("not tested yet")
    
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
                        acc = acc ,
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
                             acc = acc ,
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