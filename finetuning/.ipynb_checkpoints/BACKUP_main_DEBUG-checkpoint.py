### ADNI
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
import time
import datetime
###
import numpy as np
from dataset import ADNI_Dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from BACKUP_yAwareContrastiveLearning_DEBUG import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss
from torch.nn import CrossEntropyLoss, MSELoss # ADNI
from models.densenet import densenet121
from models.unet import UNet
import argparse
from BACKUP_config_DEBUG import Config, PRETRAINING, FINE_TUNING
### ADNI
import torch
import random
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from scipy import stats
import json
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
    parser.add_argument('--learning-rate', required = False,default=1e-4, type=float, metavar='LR',
                    help='base learning rate')
    parser.add_argument('--weight-decay',required = False, default=5e-5, type=float, metavar='W',
                    help='weight decay')
    
    ##########################
    args = parser.parse_args()

    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING

    #####changed#####
    config = Config(mode, args.task) 
    args.task_name = config.task_name ##ADDED (so that we don't have to redefine task_name every time... just get from the config itself
    config.batch_size = args.batch_size
    config.lr = args.learning_rate
    config.weight_decay = args.weight_decay
    
    #################
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

    if config.mode == PRETRAINING:
        raise ValueError("this is not the code to run for pretarining")
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
        dataset_train = ADNI_Dataset(config, label_train, data_type='train')
        dataset_val = ADNI_Dataset(config, label_valid, data_type='valid')
        dataset_test = ADNI_Dataset(config, label_test, data_type='test')

        ###
    loader_train = DataLoader(dataset_train,
                              batch_size=config.batch_size,
                              sampler=RandomSampler(dataset_train),
                              collate_fn=dataset_train.collate_fn,
                              pin_memory=config.pin_mem,
                              num_workers=config.num_cpu_workers
                              )
    loader_val = DataLoader(dataset_val,
                            batch_size=config.batch_size,
                            sampler=RandomSampler(dataset_val),
                            collate_fn=dataset_val.collate_fn,
                            pin_memory=config.pin_mem,
                            num_workers=config.num_cpu_workers
                            )

    ### ADNI
    loader_test = DataLoader(dataset_test,
                             batch_size=1,
                             sampler=RandomSampler(dataset_test),
                             collate_fn=dataset_test.collate_fn,
                             pin_memory=config.pin_mem,
                             num_workers=config.num_cpu_workers
                             )
    ###

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
        #train_batch, val_batch, test_batch = next(iter(loader_train)), next(iter(loader_val)), next(iter(loader_test))
        #np.save("./trash/original_ADNI_train_loader.npy", np.array(train_batch[0]))
        #np.save("./trash/original_ADNI_val_loader.npy", np.array(val_batch[0]))
        #np.save("./trash/original_ADNI_test_loader.npy", np.array(test_batch[0]))
        #import pdb; pdb.set_trace()
        model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, args.task_name, args.train_num, args.layer_control, None, pretrained_path) # ADNI

    if config.mode == PRETRAINING:
        model.pretraining()
    else:
        outGT, outPRED = model.fine_tuning() # ADNI
        #print('outGT:', outGT)
        #print('outPRED:', outPRED)
    
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