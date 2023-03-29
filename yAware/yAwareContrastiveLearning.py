##for testing (SDCC)
#save_path=/hpcgpfs01/scratch/dyhan316/yAware_results_save/
#python main.py --mode pretraining --framework yaware --ckpt_dir $save_path/ckpt_TEST --batch_size 8 --dataset2use test --run_where SDCC --wandb_name test --lr 1e-3 --sigma 0.05

import os
import numpy as np
import torch
from torch.nn import DataParallel
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from utils import get_scheduler
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
        
        
    else : 
        return NotImplementedError("not done yet")
    
    return scheduler
class yAwareCLModel:

    def __init__(self, net, loss, loader_train, loader_val, config, trial, scheduler=None):
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
        self.loader = loader_train
        self.loader_val = loader_val
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
        
        start_time = time.time()
        #scaler = torch.cuda.amp.GradScaler(enabled = True)
        scaler = torch.cuda.amp.GradScaler(enabled = True) 
        for epoch in range(self.st_epoch, self.config.nb_epochs):
            print("put in the  AMP, wandb, optuna, AdamW, WR")
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
            print("===training====")
            pbar = tqdm(total= len(self.loader), desc=f"Training epoch : {epoch}")
            for (inputs, labels) in self.loader:
                inputs = inputs.to(self.gpu)
                labels = labels.to(self.gpu)
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
                
                
            pbar.close()
            
            print("===validation====")
            if self.rank == 0: #done only on single GPU
                ## Validation step
                nb_batch = len(self.loader_val)
                pbar = tqdm(total=len(self.loader_val), desc=f"Validation Epoch : {epoch}")
                val_loss = 0
                val_values = {}
                with torch.no_grad():
                    self.model.eval()
                    for (inputs, labels) in self.loader_val:
                        inputs = inputs.to(self.gpu)
                        labels = labels.to(self.gpu)
                        z_i = self.model(inputs[:, 0, :])
                        z_j = self.model(inputs[:, 1, :])
                        batch_loss, logits, target = self.loss(z_i, z_j, labels)
                        val_loss += float(batch_loss) / nb_batch
                        for name, metric in self.metrics.items():
                            if name not in val_values:
                                val_values[name] = 0
                            val_values[name] += metric(logits, target) / nb_batch
                        pbar.update()
                        

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
        
        #pbar = tqdm(total=self.config.nb_epochs, desc="Training")
        for epoch in range(self.st_epoch, self.config.nb_epochs):
            import pdb ; pdb.set_trace()
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
        




