
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
from yAwareContrastiveLearning_DEBUG import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss
from torch.nn import CrossEntropyLoss, MSELoss # ADNI
from models.densenet import densenet121
from models.unet import UNet
import argparse
from config_DEBUG import Config, PRETRAINING, FINE_TUNING
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
###

if __name__ == "__main__":
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
    
    ##########################
    args = parser.parse_args()

    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING

    #####changed#####
    config = Config(mode, args.task) 
    args.task_name = config.task_name ##ADDED (so that we don't have to redefine task_name every time... just get from the config itself
    config.batch_size = args.batch_size
    config.lr = args.learning_rate
    config.weight_decay = args.weight_decay
    
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
    #    data = pd.read_csv(config.label)
    #    for i in range(len(label_name)): # ["PTAGE", "PTGENDER"]
    #        if config.label_type[i] != 'cont': # convert str object to numbers
    #            data[label_name[i]] = pd.Categorical(data[label_name[i]])
    #            data[label_name[i]] = data[label_name[i]].cat.codes
#
    #    assert args.train_num*(1+config.valid_ratio) <= len(data), 'Not enough valid data. Set smaller --train_num or smaller config.valid_ratio in config.py.'
    #    label_train, label_valid, label_test = np.split(data.sample(frac=1, random_state=random_seed), 
    #                                                    [args.train_num, int(args.train_num*(1+config.valid_ratio))])
    #    
    #    print('Task: Pretraining')
    #    print('N = {0}'.format(args.train_num))
    #    print('meta-data: {0}\n'.format(label_name))
    #    assert len(label_name) == len(config.alpha_list), 'len(label_name) and len(alpha_list) should match.'
    #    assert len(label_name) == len(config.label_type), 'len(label_name) and len(label_type) should match.'
    #    assert len(label_name) == len(config.sigma), 'len(alpha_list) and len(sigma) should match.'
    #    assert sum(config.alpha_list) == 1.0, 'Sum of alpha list should be 1.'
#
    else: # config.mode == FINE_TUNING:
        labels = pd.read_csv(config.label)
        print('Task: Fine-tuning for {0}'.format(args.task_name))
        print('Task type: {0}'.format(config.task_type))
        print('N = {0}'.format(args.train_num))
        
        if config.task_type == 'cls':
            print('Policy: {0}'.format(args.stratify))
            task_include = args.task_name.split('/')
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
    
    ### ADNI
    if config.mode == PRETRAINING:
        dataset_train = ADNI_Dataset(config, label_train, data_type='train')
        dataset_val = ADNI_Dataset(config, label_valid, data_type='valid')
        dataset_test = ADNI_Dataset(config, label_test, data_type='test')
    elif config.mode == FINE_TUNING:
        ###############밑을 조금 많이 바꿔야할듯!!!(currently only done for (probably) ADNI#######################
        ###label을 받도록 하기 
        """
        given label_train
        train_sub_data_list : csv상에서의 subject name sub_file_list = 실제 img파일의 이름
        train_img_path_list : path of the subject images 
        train_img_file_list : train_sub_data_list의 실제 folder 이름들 
        (보통은 img_file_list와 train_sub_data_list가 동일할 것이다. ADNI가 독특한 경우
        **must make sure that ALL these list have the same order!!
        """
        if "ADNI" in config.task or config.task == "test" : #change it so that this is the case for 
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
        train_img_dict = {sub : train_img_path_list[i] for i,sub in enumerate(train_sub_data_list)} #possible because sub_data and img_path have same ordering        
        dataset_train = MRI_dataset(config, label_train, train_img_dict, data_type = 'train') #label_train df 을 쓴다 


        valid_img_dict = {sub : valid_img_path_list[i] for i,sub in enumerate(valid_sub_data_list)}
        dataset_valid = MRI_dataset(config, label_valid, valid_img_dict, data_type = 'valid') #label_train df 을 쓴다 

        test_img_dict = {sub : test_img_path_list[i] for i,sub in enumerate(test_sub_data_list)} 
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
    
    if config.mode == PRETRAINING:
        if config.model == "DenseNet":
            net = densenet121(mode="encoder", drop_rate=0.0)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="simCLR") # ?? why num_classes?
        else:
            raise ValueError("Unkown model: %s"%config.model)
    else: # config.mode == FINETUNING:
        if config.model == "DenseNet":
            net = densenet121(mode="classifier", drop_rate=0.0, num_classes=config.num_classes)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="classif")
        else:
            raise ValueError("Unkown model: %s"%config.model)

    if config.mode == PRETRAINING:
        loss = GeneralizedSupervisedNTXenLoss(config=config, # ADNI
                                              temperature=config.temperature,
                                              sigma=config.sigma,
                                              return_logits=True)
    elif config.mode == FINE_TUNING:
        if config.task_type == 'cls':
            loss = CrossEntropyLoss()
        else: # config.task_type == 'reg': # ADNI
            loss = MSELoss()

    if config.mode == PRETRAINING:
        model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, "no", 0, "no", None, pretrained_path) # ADNI
    else:
        ####FOR CHECKING
        #train_batch, val_batch, test_batch = next(iter(loader_train)), next(iter(loader_val)), next(iter(loader_test))
        #np.save("./trash/ABCD_train_loader.npy", np.array(train_batch[0])) 
        #np.save("./trash/ABCD_val_loader.npy", np.array(val_batch[0]))
        #np.save("./trash/ABCD_test_loader.npy", np.array(test_batch[0]))
        
        model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, args.task_name, args.train_num, args.layer_control, None, pretrained_path) # ADNI

    if config.mode == PRETRAINING:
        model.pretraining()
    else:
        outGT, outPRED = model.fine_tuning() # ADNI

    
    ### ADNI
    if config.mode == FINE_TUNING:
        if config.task_type == 'cls':
            outGTnp = outGT.cpu().numpy()
            outPREDnp = outPRED.cpu().numpy()
            print('\n<<< Test Results: AUROC >>>')
            outAUROC = []
            for i in range(config.num_classes):
                outAUROC.append(roc_auc_score(outGTnp[:, i], outPREDnp[:, i]))
            aurocMean = np.array(outAUROC).mean()
            print('MEAN', ': {:.4f}'.format(aurocMean))
            print(json.dumps({"MEAN_auroc" : aurocMean}), file = model.eval_stats_file)
            fig, ax = plt.subplots(nrows = 1, ncols = config.num_classes)
            ax = ax.flatten()
            fig.set_size_inches((config.num_classes * 10, 10))
            for i in range(config.num_classes):
                print(task_include[i], ': {:.4f}'.format(outAUROC[i]))
                print(json.dumps({task_include[i] : outAUROC[i]}), file = model.eval_stats_file)
                fpr, tpr, threshold = metrics.roc_curve(outGT.cpu()[:, i], outPRED.cpu()[:, i])
                roc_auc = metrics.auc(fpr, tpr)
                ax[i].plot(fpr, tpr, label = 'AUC = %0.2f' % (roc_auc))
                ax[i].set_title('ROC for {0}'.format(task_include[i]))
                ax[i].legend(loc = 'lower right')
                ax[i].plot([0, 1], [0, 1], 'r--')
                ax[i].set_xlim([0, 1])
                ax[i].set_ylim([0, 1])
                ax[i].set_ylabel('True Positive Rate')
                ax[i].set_xlabel('False Positive Rate')
            if args.layer_control == 'tune_all':
                control = 'a'
            elif args.layer_control == 'freeze':
                control = 'f'
            else:
                control = 'd'
            plt.savefig(model.path2  + "/ROC_figure.png" , dpi = 100)
            plt.close()
            
            ############################# rename stats.txt ###########################################
            #os.rename('./stats.txt', "./"+args.task_name.replace('/', '')+"_stats.txt") #removed, will save stats file in the directory directly instead
            #stats_file = open("./"+args.task_name.replace('/', '')+"_stats.txt", 'a', buffering=1)
            #stats_file.write("until here_"+args.task_name.replace('/', '')+"_"+str(args.train_num)[0]+"_"+args.stratify[0]+"_"+control)
            #stats_file.write("#########################################################################")
            #stats_file.close()
            
            
        else: # config.task_type == 'reg':
            outGTnp = outGT.cpu().numpy()
            outPREDnp = outPRED.cpu().numpy()
            print('\n<<< Test Results >>>')
            print('MSE: {:.2f}'.format(mean_squared_error(outGTnp, outPREDnp)))
            print('MAE: {:.2f}'.format(mean_absolute_error(outGTnp, outPREDnp)))
            print('RMSE: {:.2f}'.format(np.sqrt(mean_squared_error(outGTnp, outPREDnp))))
            print('R2-score: {:.4f}'.format(r2_score(outGTnp, outPREDnp)))

    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S') # ADNI
    print("[main.py finished at {0}]".format(nowDatetime))
    ###
