import numpy as np
from dataset import MRIDataset, UKBDataset

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

#DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import DataParallel
import builtins


from utils import get_scheduler

from yAwareContrastiveLearning import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss, NTXenLoss
from torch.nn import CrossEntropyLoss
from models.densenet import densenet121
from models.unet import UNet
import argparse

import pandas as pd
import os
import random
from sklearn.preprocessing import MinMaxScaler


import wandb
import optuna
import logging
import sys
import time
from sklearn.preprocessing import StandardScaler
    
def drop_n_normalize(df, drop, scale):
    """
    #added to streamline dropping and normalizing!
    drops and normalizes the given def 
    drop, scale : the cols in df to drop if Na and scale
    hint from : https://stackoverflow.com/questions/28576540/how-can-i-normalize-the-data-in-a-range-of-columns-in-my-pandas-dataframe
    """
    df = df.dropna(subset = drop) 
    df[scale] = StandardScaler().fit_transform(df[scale])
    
    return df

def subj_meta_maker(subjects, meta_data, column, args): 
    """
    * subjects : list of subject directories 
    * metadata : the df that contains the file name
    * column : which column in the df to use as label
    * args : relevant args
    
    so, depending on the data (CHA or UKB), decides how much to remove from the end (i.e. [:7] for UKB [:-7] for CHA), and outputs a list of tuples (data.nii.gz , label) (subj_meta_maker)
    """
    if "UKB" in args.dataset2use: 
        eid_set = meta_data['eid'].tolist() #faster than list
        subj_meta = [(subj,meta_data[meta_data['eid']==int(subj[:7])][column].values[0]) for subj in subjects if int(subj[:7]) in eid_set]  #extracts data only if exists in df, and so on 
    elif args.dataset2use in ["CHA", "CHA_secHC", "test"]: 
        eid_set = meta_data['eid'].tolist()
        subj_meta = [(subj,meta_data[meta_data['eid']==subj[:-7]][column].values[0]) for subj in subjects if subj[:-7] in meta_data['eid'].tolist() if subj[:-7] in eid_set]
    
    return subj_meta

def str2bool(v): #true, false를 argparse로 받을 수 있도록 하기!
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    parser.add_argument("--framework", type=str, choices=["yaware", "simclr"], required=True)
    parser.add_argument("--kernel", type=str, default = 'rbf',choices=["rbf", "XOR"], help="rbf(continuous), XOR(binary categorical variable)")
    parser.add_argument("--ckpt_dir", type=str, default = './checkpoint',
                        help="select which dir to save the checkpoint!")
    parser.add_argument("--tb_dir", type=str, default = './tb',
                       help="select which dir to save the tensorboard log")
    parser.add_argument("--tf", type=str, default = 'all_tf',choices=['all_tf','cutout','crop'],
                       help="select which transforms to apply")
    parser.add_argument("--nb_epochs", type=int, default = 100, help="number of epochs") #default number of epochs to do (not goes through all at once)
    
    
    ##hyperparams
    parser.add_argument("--lr_policy",type=str,default='None' ,choices=['onecyclelr','lambda','step','multi-step','plateau','cosine','SGDR', "cosine_WR_1","cosine_WR_2","cosine_WR_3",'None', 'cosine_decay', 'cosine_annealing'], help='learning rate policy: lambda|step|multi-step|plateau|cosine|SGDR')
    parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument("--gamma",default=0.1,type=float,help='multiply by a gamma every lr_decay_iters iterations')
    
    parser.add_argument("--lr", type=float, default = 1e-4, help="initial learning rate for optimizer")
    parser.add_argument('--wd', type = float, default = 5e-5, help = "weight decay")
    parser.add_argument("--sigma",default=0.6176507443170335,type=float,help='Hyperparameters for our y-Aware InfoNCE Loss depending on the meta-data at hand, default was 5, but changed as we will do the thing')                    
    parser.add_argument("--temp", type = float, default = 0.1, help = "temperatuer to use for the InfoNCE loss")
    
    # DDP configs:
    parser.add_argument('--world_size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--batch_size', default=64, type=int, 
                        help='batch_size')
    parser.add_argument('--wandb_name' , default = "yAwa_pretraining", type = str, help = "what project name to use for wandb")
    
    #which data to use and which label to use as y 
    parser.add_argument("--label_name", type=str, default = 'age', choices= ['age', 'sex', 'intelligence_gps', 'intelligence'], help="target meta info")
    parser.add_argument("--dataset2use", default = "UKB", choices = ["UKB", "CHA", "CHA_secHC", 'test'],type = str, help = "which dataset and label_name to use for pretraining")
    
    
    parser.add_argument("--run_where", default = 'lab', choices = ['lab','SDCC'], type = str, help = "whether it's runnign on lab or sdcc")
    
    
    #specify lr, wd range directly for optuna
    #parser.add_argument('--lr_range' , required = False, type = str , help = "what optuna lr range to use, in the for of `1e-5/1e-2` ")
    #parser.add_argument('--wd_range' , required = False, type = str , help = "what optuna wd range to use, in the for of `1e-5/1e-2` ")
    #parser.add_argument('--sigma_range' , required = False, type = str , help = "what optuna sigma range to use, in the for of `1e-5/1e-2` ")
    
    
    #parser.add_argument('--y_normalize', required = False, type = str2bool, default = True, help = "whether to do y_normalize or not.. (i.e. normalize to N(0,1))")  #not needed, normalizes itself
    args = parser.parse_args()
    
    if args.run_where == "lab" : 
        from config_lab import Config
        wandb_dir = './wandb'
    elif args.run_where == "SDCC" : 
        from config_SDCC import Config
        wandb_dir = './wandb_SDCC'
        
    config = Config(args)
    
#def main(trial) :  #only activate when trying to implement optuna
    
    #FOR optuna, not done yet
    #lr_range = [float(i) for i in args.lr_range.split('/')]
    #lr_range = [float(i) for i in args.wd_range.split('/')]
    #lr_range = [float(i) for i in args.sigma_range.split('/')]
    
    meta_data = pd.read_csv(config.label)
    
    subjects = sorted(os.listdir(config.data)) 
    
    
    #deciding which col to use for data thing     
    if config.label_name == "sex" : 
        col2use = "sex"
    elif config.label_name == "age" : 
        col2use = "age"
    elif config.label_name == "intelligence_gps":
        col2use = "SCORE_auto"
    elif config.label_name == "intelligence" : 
        col2use = "fluid"
    else : 
        raise NotImplementedError(f"the label you asked, {config.label_name} is not one that can be used")
    
    meta_data = drop_n_normalize(meta_data, drop = [col2use], scale = [col2use]) 
    subj_meta = subj_meta_maker(subjects, meta_data, col2use, args)
        
    
    print(f'training {len(subj_meta)} {args.dataset2use} subjects')
    
    
    #subj_meta is a list consisting of tuple (filename, label(int))
     
    #random.shuffle(subj_meta)
    random.Random(42).shuffle(subj_meta) #for reproducible shuffling
    
    num_total = len(subj_meta)
    num_train = int(num_total*(1 - config.val_size))
    num_val = int(num_total*config.val_size)
    
    subj_train= subj_meta[:num_train]
    subj_val = subj_meta[num_train:]
    
    #initialize wandb 
    
    wandb.init(project = args.wandb_name, config = config, dir = wandb_dir,
               name = f"{args.framework}_{args.label_name}_k{args.kernel}_b{args.batch_size}_lr_{args.lr:.2e}_wd_{args.wd:.2e}_sigma_{args.sigma:.2e}_temp_{args.temp:.2e}") #f"{trial.number}_{args.label_name}_k{args.kernel}_b{args.batch_size}")
    
    
    
    if config.mode == 'pretraining':
        if config.model == "DenseNet":
            net = densenet121(mode="encoder", drop_rate=0.0)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="simCLR")
        else:
            raise ValueError("Unkown model: %s"%config.model)
    else:
        if config.model == "DenseNet":
            net = densenet121(mode="classifier", drop_rate=0.0, num_classes=config.num_classes)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="classif")
        else:
            raise ValueError("Unkown model: %s"%config.model)
     

    # def _get_sync_file():
    #     """Logic for naming sync file using slurm env variables"""
    #     sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
    #     # You need to export 'SCRATCH' as a shared directory between nodes for torch.distributed.launch
    #     os.makedirs(sync_file_dir, exist_ok=True)
    #     sync_file = 'file://%s/pytorch_sync.%s.%s' % (
    #         sync_file_dir,os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
    #     # You need to export 'SLURM_JOB_ID' and 'SLURM_STEP_ID' as any 5-digit numbers for torch.distributed.launch
    #     return sync_file
    
    ### DDP         
    # sbatch script에서 WORLD_SIZE를 지정해준 경우 (노드 당 gpu * 노드의 수)
    if "WORLD_SIZE" in os.environ:
        config.world_size = int(os.environ["WORLD_SIZE"])
    # 혹은 슬럼에서 자동으로 ntasks per node * nodes 로 구해줌
    elif 'SLURM_NTASKS' in os.environ:
        config.world_size = int(os.environ['SLURM_NTASKS'])
        
    config.distributed = config.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if config.distributed:
        if config.local_rank != -1: # for torch.distributed.launch
            config.rank = config.local_rank
            config.gpu = config.local_rank
        elif 'RANK' in os.environ: # for torchrun
            config.rank = int(os.environ['RANK'])
            config.gpu = int(os.environ['LOCAL_RANK'])
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            config.rank = int(os.environ['SLURM_PROCID'])
            config.gpu = config.rank % torch.cuda.device_count()

        sync_file = "env://" #_get_sync_file()
        dist.init_process_group(backend=config.dist_backend, init_method=sync_file,
                            world_size=config.world_size, rank=config.rank)
    else:
        config.rank = 0
        config.gpu = 0

    # suppress printing if not on master gpu
    if config.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
        
    ### model 
    if config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            config.device = torch.device('cuda:{}'.format(config.gpu))
            torch.cuda.set_device(config.gpu)
            net.cuda(config.gpu)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[config.gpu], broadcast_buffers=False)
            net_without_ddp = net.module
        else:
            config.device = torch.device("cuda" if config.cuda else "cpu")
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
            model_without_ddp = net.module
    else:
        config.device = torch.device("cuda" if config.cuda else "cpu")
        net = DataParallel(net).to(config.device)        
            
    torch.backends.cudnn.benchmark = True        
    if config.mode == 'pretraining':
        dataset_train = UKBDataset(config, subj_train) #MRIDataset(config, training=True)
        dataset_val = UKBDataset(config,subj_val)  #MRIDataset(config, validation=True)
    else:
        ## Fill with your target dataset
        dataset_train = Dataset()
        dataset_val = Dataset()
        
    if config.distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        valid_sampler = DistributedSampler(dataset_val, shuffle=True)
    else:
        train_sampler = RandomSampler(dataset_train)
        valid_sampler = RandomSampler(dataset_val)
        
    loader_train = DataLoader(dataset_train,
                              batch_size=config.batch_size,
                              sampler=train_sampler,
                              collate_fn=dataset_train.collate_fn,
                              pin_memory=config.pin_mem,
                              num_workers=config.num_cpu_workers
                              )
    loader_val = DataLoader(dataset_val,
                            batch_size=config.batch_size,
                            sampler=valid_sampler,
                            collate_fn=dataset_val.collate_fn,
                            pin_memory=config.pin_mem,
                            num_workers=config.num_cpu_workers
                            )
    if config.mode == 'pretraining':
        if config.framework == 'simclr':
            loss = NTXenLoss(temperature=config.temperature,return_logits=True)
        elif config.framework == 'yaware':
            loss = GeneralizedSupervisedNTXenLoss(config = config, temperature=config.temperature,
                                              kernel=config.kernel,
                                              sigma=config.sigma,
                                              return_logits=True)

    elif config.mode == FINE_TUNING:
        loss = CrossEntropyLoss()
    wandb.watch(models = net, criterion = loss)
    
    model = yAwareCLModel(net, loss, loader_train, loader_val, config, trial=None) #if optuna, remove None thing
    
    
    
                
    if config.mode == 'pretraining':
        if config.framework == 'simclr':
            model.pretraining_simclr()
        elif config.framework == 'yaware':
            model.pretraining_yaware()
    else:
        model.fine_tuning
    
    wandb.finish()
    
    print("HIHIHIHIHI++")
    #return None #for optuna, must be changed
    
    #return XXX (mean_AUC)


#OPTUNA, do'nt activate yet
#if __name__ == "__main__" : 
#    os.makedirs(args.ckpt_dir, exist_ok = True)
#    db_dir = f"{args.ckpt_dir}/optuna"
#    url = "sqlite:///" + os.path.join(os.getcwd(), (db_dir + '.db'))
#    # Add stream handler of stdout to show the messages => 이건 되어있던데 왜하는 건지 모르겠다 
#    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
#    
#    storage = optuna.storages.RDBStorage( url = url, heartbeat_interval = 60, grace_period = 120 )
#    
#    study = optuna.create_study(study_name = "test_study_name", 
#                                storage = storage,
#                                load_if_exists = True,
#                                pruner = optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=0, interval_steps=1), #wrap with patience?
#                                direction = 'minimize') #pruner, sampler도 나중에 넣기) 
#    study.optimize(main, n_trials = 200, timeout = 160000) #~2day




    
    
#GDD
#sub-210029 (0_1_3)
#sub-200072 (0_1_2)
#sub-200103 (1_1.5_2)
#/scratch/connectome/dyhan316/CHA_QSIPREP_2/result_sample_changing_T1_mask
#BIDS, FS ,QSIPREP


#ASd
#sub-170425 (2_2.5_1)
#sub-150758 (1_1.5_2)
#sub-130917 (1.5_2_1)
