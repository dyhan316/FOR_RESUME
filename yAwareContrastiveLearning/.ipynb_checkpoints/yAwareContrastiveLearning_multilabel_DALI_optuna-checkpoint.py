##for testing (SDCC)
#save_path=/hpcgpfs01/scratch/dyhan316/yAware_results_save/

#on slurm sbatch
#python main_multilabel_DALI_optuna.py --mode pretraining --framework yaware --ckpt_dir $save_path/ckpt_TEST --batch_size 8 --dataset2use test --run_where SDCC --wandb_name test --lr 1e-3 --sigma 0.05 

##interactively  (DP instead of DDP)
#python main_multilabel_DALI_optuna.py --mode pretraining --framework yaware --ckpt_dir $save_path/ckpt_TEST --batch_size 8 --dataset2use test --run_where SDCC --wandb_name test --lr 1e-3 --sigma 0.05 --DP_DEBUG True #set 

#DDP : 참조 https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904


import os
import numpy as np
import torch
from torch.nn import DataParallel
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from utils import get_scheduler, CosineAnnealingWarmUpRestarts
import random

from tqdm import tqdm
import logging
import wandb 
import time 


def lr_scheduler(scheduler_name, optimizer, config,**kwargs) :
    """
    * scheduler_name : name of the scheduler given in main.py parser
        * options : onecyclelr, cosine, cosine_decay, plateau, cosine_annealing
    * optimizer : the optimizer object to be scheduled?
    
    returns : scheduler
    """
    
    if scheduler_name == "None":
        return None
    
    elif scheduler_name == "onecyclelr":
        optimizer.param_groups[0]['lr'] = 0.0 #have to be reset to zero (instead the 
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, epochs = config.nb_epochs + 1, steps_per_epoch = 1)         
    
    elif scheduler_name == "plateau": 
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min', factor = 0.1, patience = 5) #could vary patience
    
    elif scheduler_name == "cosine" : 
        return NotImplementedError()
    
    elif scheduler_name == "cosine_decay" : 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = config.nb_epochs)
    
    elif scheduler_name == "cosine_annealing" : 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
        
        
    elif scheduler_name == "cosine_WR_1":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 1, T_mult=2)
                    
    elif scheduler_name == "cosine_WR_2":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult=1)
                    
    elif scheduler_name == "cosine_WR_3":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult=2)
        
    ###custom WR (reducing thing)
    ##
    elif "custom_WR" in scheduler_name : 
        optimizer.param_groups[0]['lr'] = config.lr  * 1e-4#have to be reset to zero
        if scheduler_name == "custom_WR_1" : 
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0 = 10, T_mult = 1, eta_max = config.lr, T_up = 2, gamma = 0.5)
            
        elif scheduler_name == "custom_WR_2" : 
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0 = 20, T_mult = 1, eta_max = config.lr, T_up = 4, gamma = 0.5)
        elif scheduler_name == "custom_WR_3" : 
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0 = 50, T_mult = 1, eta_max = config.lr, T_up = 10, gamma = 0.5)
        
        elif scheduler_name == "custom_WR_4" : 
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0 = 100, T_mult = 1, eta_max = config.lr, T_up = 20, gamma = 0.5)
        
    #tmult = 2같은 것들도 해보기!
    #elif scheduler_name == "custom_WR_1"
    #    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0 = 50, T_mult = 1, eta_max = config.lr, T_up = 10, gamma = 0.5)
        
    
    else : 
        return NotImplementedError("not done yet")
    
    return scheduler
class yAwareCLModel:

    def __init__(self, net, loss, dali_loader_train, dali_loader_val, config, trial, scheduler=None):
        """
        Parameters
        ----------
        net: subclass of nn.Module
        loss: callable fn with args (y_pred, y_true)
        loader_train, loader_val: pytorch DataLoaders for training/validation
        config: Config object with hyperparameters
        scheduler (optional)
        """
        super().__init__()
        self.logger = logging.getLogger("yAwareCL")
        self.loss = loss
        self.model = net
        #self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.wd)
        self.optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.wd) #switched to AdamW
        self.scheduler = get_scheduler(self.optimizer, config)
        self.loader = dali_loader_train
        self.loader_val = dali_loader_val
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")
        self.config = config
        self.device = config.device
        self.rank = config.rank
        self.gpu = config.gpu
        self.metrics = {}
        
        if hasattr(config, 'pretrained_path') and config.pretrained_path is not None:
            self.load_model(config.pretrained_path)
            
        os.makedirs(config.checkpoint_dir, exist_ok=True)    
        os.makedirs(config.tb_dir, exist_ok=True)
        
        self.st_epoch = 0
        #if config.train_continue == 'on' and any([".pth" in file for file in config.checkpoint_dir ]):
        if config.train_continue == 'on' and any([".pth" in file for file in os.listdir(config.checkpoint_dir) ]):
            self.load_checkpoint(config.checkpoint_dir)
            print("===weight was loaded!!===")
            
        self.writer_train = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'train'))
        self.writer_val = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'val'))
        
        self.trial = trial
        
        
        self.scheduler = lr_scheduler(config.lr_policy, self.optimizer, config)
        

    def pretraining_yaware(self):
        print(self.loss)
        print(self.optimizer)
        
        DALI_BATCH_SIZE = self.config.dali_batch_size
        BATCH_SIZE = self.config.batch_size
            
        start_time = time.time()
        scaler = torch.cuda.amp.GradScaler(enabled = True) 
        for epoch in range(self.st_epoch, self.config.nb_epochs):
            #import pdb;  pdb.set_trace()
            print("put in optuna")
            np.random.seed(epoch)
            random.seed(epoch)
            
            ## Training step
            self.model.train()
            
            nb_batch = len(self.loader)//int(BATCH_SIZE/DALI_BATCH_SIZE) 
            training_loss = 0
            print("===training====")
            pbar = tqdm(total= nb_batch, desc=f"Training epoch : {epoch}")
            
            #import pdb ; pdb.set_trace()
            #todo :  4. 그 del인가 해보기  5. put in removing pipeline, and shard_seed = epoch too! (or instead do reset after epoch True)  6. dali loader 에서 그 갯수, thread수 등이, DDP를 쓰면 두배가 되고 이런식인가? => 확인해보기!(아닐듯... since shard) 7. GPU util is STILL LOW ==> maybe, dali batch size를 더 늘리던지 해서 더 GPU가 쓰이도록 하기? 8. val도 DDP?(probs not) 
            
            ##DONE : 1. valid에도 dali 돌리도록 하기,5. wandb rank0에서 하도록 하기!2. model/dali batch size달라도 되도록 하기 (jupyter 그 X.5.4.보면됨) 3. DDP에서 되는지 보기
            images_list, labels_list = [],[]
            for i, data in enumerate(self.loader):
                inputs_i = data[0]["data"]
                labels_i = data[0]["label"]
                
                images_list.append(inputs_i)
                labels_list.append(labels_i)
                
                if len(images_list) != int(BATCH_SIZE/DALI_BATCH_SIZE) :
                    #if batch is not made yet, don't do the things below, if batch is made, then do below 
                    continue 
                    
                ##below will only be activated once a proper batch has been made
                #if inputs, labels list of tensors is properly made : 
                inputs = torch.cat(images_list,0) 
                labels = torch.cat(labels_list,0)
                self.optimizer.zero_grad()
                
                #doing AMP, changing to fp16 automatically
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled = True):
                    z_i = self.model(inputs[:, 0, :])
                    z_j = self.model(inputs[:, 1, :])
                    batch_loss, logits, target = self.loss(z_i, z_j, labels)
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(batch_loss).backward()
                    
                # unscaled optimizer thing 
                scaler.step(self.optimizer)
                scaler.update()
                
                training_loss += float(batch_loss) / nb_batch
                pbar.update()
                
                images_list , labels_list = [],[] #reset    
                
            pbar.close()
            print(torch.cuda.device_count())
            print(self.model.module.features.transition1.norm.running_mean[0] , self.model.module.features.transition1.norm.running_var[0])
            
            print("===validation====")
            #raise NotImplementedError("VALIDAIONT NOT IMPLEMENTED YET FOR DALI") 
            if self.rank == 0: #done only on single GPU
                ## Validation step
                nb_batch = len(self.loader_val)//int(BATCH_SIZE/DALI_BATCH_SIZE)  #has to be changed (like training case)
                pbar = tqdm(total=nb_batch, desc=f"Validation Epoch : {epoch}")
                val_loss = 0
                val_values = {}
                
                with torch.no_grad():
                    images_list, labels_list = [], []
                    self.model.eval()
                    #for (inputs, labels) in self.loader_val:
                        #inputs = inputs.to(self.gpu)
                        #labels = labels.to(self.gpu)
                    for i, data in enumerate(self.loader_val):
                        inputs_i = data[0]["data"]
                        labels_i = data[0]["label"]
                        
                        images_list.append(inputs_i)
                        labels_list.append(labels_i)
                        
                        if len(images_list) != int(BATCH_SIZE/DALI_BATCH_SIZE):
                            continue
                        #not making one large batch to actually run the thing 
                        inputs = torch.cat(images_list,0)
                        labels = torch.cat(labels_list,0)
                        
                        z_i = self.model(inputs[:, 0, :])
                        z_j = self.model(inputs[:, 1, :])
                        batch_loss, logits, target = self.loss(z_i, z_j, labels)
                        val_loss += float(batch_loss) / nb_batch
                        for name, metric in self.metrics.items():
                            if name not in val_values:
                                val_values[name] = 0
                            val_values[name] += metric(logits, target) / nb_batch
                        pbar.update()
                        
                        images_list, labels_list = [], [] #reset

                pbar.close()
            
                metrics = "\t".join(["Validation {}: {:.4f}".format(m, v) for (m, v) in val_values.items()])
                
                print(f'Epoch [{epoch+1}/{self.config.nb_epochs}] Training loss = {training_loss:.4f}\t Validation loss = {val_loss:.4f}\t lr = {self.optimizer.param_groups[0]["lr"]}\t time = {(time.time()-start_time):.4f}\t'+metrics)

                print("=========")
                wandb.log({"base_lr" : self.optimizer.param_groups[0]['lr'],"training_loss" : training_loss, "validation_loss" : val_loss}, step = epoch)
                #self.trial.report(val_loss, epoch) #undo if trial is not needed
                #if self.trial.should_prune():
                #    print("PRUNED BABY")
                #    raise optuna.TrialPruned()
                
                self.writer_train.add_scalar('training_loss', training_loss, epoch+1)
                self.writer_val.add_scalar('validation_loss', val_loss, epoch+1)
                self.writer_val.add_scalar('lr', self.optimizer.param_groups[0]["lr"], epoch+1)
                
                if self.config.lr_policy == "plateau": 
                    self.scheduler.step(val_loss)
                elif self.scheduler is not None:
                    self.scheduler.step()

                    
                if (epoch % self.config.nb_epochs_per_saving == 0 or epoch == self.config.nb_epochs - 1):
                    torch.save({
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict()},
                        os.path.join(self.config.checkpoint_dir, "{name}_epoch_{epoch}.pth".
                                     format(name="y-Aware_Contrastive_MRI", epoch=epoch)))
            
            #pbar.close()
        self.writer_train.close()
        self.writer_val.close()
        
        

    def pretraining_simclr(self):
        print(self.loss)
        print(self.optimizer)
        raise NotImplementedError("NOT IMPLEMENTED FOR DALI AND SO ON (rewrite it or see if yAWare can be used for simCLR)")
        #pbar = tqdm(total=self.config.nb_epochs, desc="Training")
        for epoch in range(self.st_epoch, self.config.nb_epochs):
            
            np.random.seed(epoch)
            random.seed(epoch)
            # fix sampling seed such that each gpu gets different part of dataset
            if self.config.distributed:
                self.loader.sampler.set_epoch(epoch)
            #print("epoch : {}".format(epoch))
            #pbar.update()
            
            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = 0
            
            for (inputs, labels) in self.loader:
                
                inputs = inputs.to(self.gpu)
                labels = labels.to(self.gpu)
                self.optimizer.zero_grad()
                z_i = self.model(inputs[:, 0, :])
                z_j = self.model(inputs[:, 1, :])
                batch_loss, logits, target = self.loss(z_i, z_j)
                batch_loss.backward()
                self.optimizer.step()
                training_loss += float(batch_loss) / nb_batch
            
            
            if self.rank == 0:
                ## Validation step
                nb_batch = len(self.loader_val)
                #pbar = tqdm(total=nb_batch, desc="Validation")
                val_loss = 0
                val_values = {}
                with torch.no_grad():
                    self.model.eval()
                    for (inputs, labels) in self.loader_val:
                        inputs = inputs.to(self.gpu)
                        labels = labels.to(self.gpu)
                        z_i = self.model(inputs[:, 0, :])
                        z_j = self.model(inputs[:, 1, :])
                        batch_loss, logits, target = self.loss(z_i, z_j)
                        val_loss += float(batch_loss) / nb_batch
                        for name, metric in self.metrics.items():
                            if name not in val_values:
                                val_values[name] = 0
                            val_values[name] += metric(logits, target) / nb_batch
                
            
                metrics = "\t".join(["Validation {}: {:.4f}".format(m, v) for (m, v) in val_values.items()])
                print("Epoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                    epoch+1, self.config.nb_epochs, training_loss, val_loss)+metrics) #flush=True
                #self.trial.report(val_loss, epoch) #do if doing optuna
                #if self.trial.should_prune():
                #    print("PRUNED BABY")
                #    raise optuna.TrialPruned()
                    
                    
                self.writer_train.add_scalar('training_loss', training_loss, epoch+1)
                self.writer_val.add_scalar('validation_loss', val_loss, epoch+1)
                
                if self.config.lr_policy == "plateau": 
                    self.scheduler.step(val_loss)
                elif self.scheduler is not None:
                    self.scheduler.step()

                if (epoch % self.config.nb_epochs_per_saving == 0 or epoch == self.config.nb_epochs - 1):
                    torch.save({
                        "epoch": epoch,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict()},
                        os.path.join(self.config.checkpoint_dir, "{name}_epoch_{epoch}.pth".
                                     format(name="Simclr_Contrastive_MRI", epoch=epoch)))
            #pbar.close()
        self.writer_train.close()
        self.writer_val.close()

    def fine_tuning(self):
        print(self.loss)
        print(self.optimizer)

        for epoch in range(self.config.nb_epochs):
            ## Training step
            self.model.train()
            nb_batch = len(self.loader)
            training_loss = []
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, labels) in self.loader:
                pbar.update()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(inputs)
                batch_loss = self.loss(y,labels)
                batch_loss.backward()
                self.optimizer.step()
                training_loss += float(batch_loss) / nb_batch
            pbar.close()

            ## Validation step
            nb_batch = len(self.loader_val)
            pbar = tqdm(total=nb_batch, desc="Validation")
            val_loss = 0
            with torch.no_grad():
                self.model.eval()
                for (inputs, labels) in self.loader_val:
                    pbar.update()
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    y = self.model(inputs)
                    batch_loss = self.loss(y, labels)
                    val_loss += float(batch_loss) / nb_batch
            pbar.close()

            print("Epoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                epoch+1, self.config.nb_epochs, training_loss, val_loss), flush=True)
            print("=========")
            wandb.log({"base_lr" : self.optimizer.param_groups[0]['lr'],"training_loss" : training_loss, "validation_loss" : val_loss}, step = epoch)
            if self.scheduler is not None:
                self.scheduler.step()

    def load_model(self, path):
        checkpoint = None
        try:
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        except BaseException as e:
            self.logger.error('Impossible to load the checkpoint: %s' % str(e))
        if checkpoint is not None:
            try:
                if hasattr(checkpoint, "state_dict"):
                    unexpected = self.model.load_state_dict(checkpoint.state_dict())
                    self.logger.info('Model loading info: {}'.format(unexpected))
                elif isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        unexpected = self.model.load_state_dict(checkpoint["model"], strict=False)
                        self.logger.info('Model loading info: {}'.format(unexpected))
                else:
                    unexpected = self.model.load_state_dict(checkpoint)
                    self.logger.info('Model loading info: {}'.format(unexpected))
            except BaseException as e:
                raise ValueError('Error while loading the model\'s weights: %s' % str(e))
    
    #developed for train_continue
    def load_checkpoint(self, ckpt_dir):
        if not os.path.exists(ckpt_dir) or len(os.listdir(ckpt_dir))==0:
            self.st_epoch = 0
            
        else:
            ckpt_lst = os.listdir(ckpt_dir)
            ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
            ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  

            # 가장 에포크가 큰 모델을 불러옴
            dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=self.device)

            self.model.load_state_dict(dict_model['model'])
            self.optimizer.load_state_dict(dict_model['optimizer'])
            self.st_epoch = dict_model['epoch'] + 1
        




