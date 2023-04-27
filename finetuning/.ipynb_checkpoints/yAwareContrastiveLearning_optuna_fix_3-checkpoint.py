#may need to change run_where to lab or SDCC!

###python3 main_optuna_fix_3.py --pretrained_path ./weights/y-Aware_Contrastive_MRI_epoch_99.pth --mode finetuning --train_num 20 --layer_control tune_all --stratify strat --random_seed 0 --task test --input_option yAware --binary_class True --save_path finetune_trash --run_where lab


###python3 main_optuna_fix_3.py --pretrained_path ./weights/y-Aware_Contrastive_MRI_epoch_99.pth --mode finetuning --train_num 20 --layer_control tune_all --stratify strat --random_seed 0 --task ADNI_sex --input_option yAware --binary_class True --save_path finetune_test --run_where sdcc #이게 그나마 잘됨 #--BN none도 없앰 (also optuna)


import os
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import logging
from Earlystopping import EarlyStopping # ADNI
import sys
import pathlib
import json 

##ADDED
import sys # for sys.argv
import optuna
from finetune_utils import test_data_analysis_2
import time
import wandb
##


class yAwareCLModel:

    def __init__(self, net, loss, loader_train, loader_val, loader_test, config, task_name, train_num, layer_control, n_iter, pretrained_path, FOLD, wandb, trial, scheduler=None): # ADNI
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
        self.FOLD = FOLD
        
        
        random_seed = config.random_seed
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        #turned off because decreases computation speed
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        self.wandb = wandb
        self.trial = trial
        
        ### ADNI
        if config.mode == 0: # PRETRAINING
            self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        else: # config.mode == 1: # FINE_TUNING
            if layer_control == 'tune_all':
                self.optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)#, foreach = True)
            elif layer_control == 'freeze':
                self.optimizer = torch.optim.AdamW(net.classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)#, foreach = True)

            else: # layer_control == 'tune_diff':
                if config.model == 'DenseNet':
                    self.optimizer = torch.optim.Adam([
                        {"params": net.features.conv0.parameters(), "lr": config.lr*1e-3},
                        {"params": net.features.denseblock1.parameters(), "lr": config.lr*1e-3},
                        {"params": net.features.transition1.parameters(), "lr": config.lr*1e-3},

                        {"params": net.features.denseblock2.parameters(), "lr": config.lr*1e-2},
                        {"params": net.features.transition2.parameters(), "lr": config.lr*1e-2},

                        {"params": net.features.denseblock3.parameters(), "lr": config.lr*1e-1},
                        {"params": net.features.transition3.parameters(), "lr": config.lr*1e-1},

                        {"params": net.features.denseblock4.parameters(), "lr": config.lr},
                        {"params": net.classifier.parameters(), "lr": config.lr},
                        ], lr=config.lr, weight_decay=config.weight_decay)
                else: # config.model == 'UNet':
                    self.optimizer = torch.optim.Adam([
                        {"params": net.up.parameters(), "lr": config.lr*1e-2},
                        {"params": net.down.parameters(), "lr": config.lr*1e-1},
                        {"params": net.classifier.parameters(), "lr": config.lr},
                        ], lr=config.lr, weight_decay=config.weight_decay)
        ###
        self.layer_control = layer_control
        self.scheduler = scheduler
        self.loader = loader_train #train loader!
        self.loader_val = loader_val
        self.loader_test = loader_test # ADNI
        self.device = torch.device("cuda" if config.cuda else "cpu")
        if config.cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: set cuda=False parameter.")
        self.config = config
        self.metrics = {}
        ### ADNI
        if train_num != 0:
            self.task_name = task_name
            self.train_num = train_num
        self.n_iter = n_iter
        self.pretrained_path = pretrained_path


        if self.config.eval: 
            self.path2 = f"{config.save_path}/best_trial_eval/{config.task}-{self.layer_control}-{str(self.pretrained_path).split('/')[-1].split('.')[0]}-train_{self.train_num}-batch_{config.batch_size}-lr_{config.lr}-wd_{config.weight_decay}-tf_{config.tf}-seed_{config.random_seed}"
            self.best_trial_path = f"{config.save_path}/{config.task}-{self.layer_control}-{config.task_type}/{str(self.pretrained_path).split('/')[-1].split('.')[0]}-train_{self.train_num}/batch_{config.batch_size}-lr_{config.lr}-wd_{config.weight_decay}-tf_{config.tf}/seed_{config.random_seed}"
            
        else : #i.e. optuna
            self.path2 = f"{config.save_path}/{config.task}-{self.layer_control}-{config.task_type}/{str(self.pretrained_path).split('/')[-1].split('.')[0]}-train_{self.train_num}/batch_{config.batch_size}-lr_{config.lr}-wd_{config.weight_decay}-tf_{config.tf}/seed_{config.random_seed}"
        
        if FOLD == 0: #i.e. first fold에서만 확인한다!
            try :
                os.makedirs(self.path2, exist_ok = False) 
            except : #if it already exists, then raise error
                raise ValueError(f"the path {self.path2} already exsits!!! (probably already done?)")
        
        self.stats_file = open(f'{self.path2}/stats_{FOLD}.txt', 'a', buffering=1) #여기 수정 
        self.eval_stats_file = open(f'{self.path2}/eval_stats_{FOLD}.txt', 'a', buffering=1) #여기 수정 
        
        ###
        
        if pretrained_path != 'None':
            print(f"DID use pretrained path :{pretrained_path}")
            self.load_model(pretrained_path)
        else :
            print("FROM SCRATCH")

        self.model = DataParallel(self.model).to(self.device)

    def fine_tuning(self):
        print(self.loss)
        print(self.optimizer)
        if self.n_iter is not None:
            n_iter = 'CV' + str(self.n_iter)
        else:
            n_iter = 'model'
        
        tmp = str(self.pretrained_path).split('/')[-1].split('.')[0]
        ckpt_dir = f'{self.path2}/{tmp}_{self.layer_control[0]}_{self.train_num}_{n_iter}_fold_{self.FOLD}.pt' 
        
        if self.config.early_criteria == "AUROC" : 
            early_stopping = EarlyStopping(patience = self.config.patience, 
                                       path = ckpt_dir, direction = "max") #may change if loss 로 early stopping
        elif self.config.early_criteria == "loss" : 
            early_stopping = EarlyStopping(patience = self.config.patience, 
                                       path = ckpt_dir, direction = "min") #may change if loss 로 early stopping
        elif self.config.early_criteria == "none":
            early_stopping = None
        else : 
            raise ValueError("--early_criteria should be either AUROC or loss or none")
        ###########ADDED######
        ##put requires grad = False, so that we don't carry computational trees not needed (memory footprint down 33%)
        if self.layer_control == "freeze":
            self.model.requires_grad_(False) #changed
            self.model.module.classifier.requires_grad_(True) #module이 앞에 붙는 이유 : DP여서
        #disable BN, droput tracking (may need to be disabled if we want renormalization)
            self.model.eval() #because eval mode still enables backprop (different from setting grad to False remember?)(i.e. running avg and stuff like that are gonna be fixed now, but not the backprop)(그래서 굳이 model.module.classifier.train()이라고 할 필요가없다
        elif self.layer_control == "tune_all":
            self.model.requires_grad_(True) 
            self.model.train()
        
        else : 
            print(f"not implemented yet for layer_control method {self.layer_control}")
        ##################### 
        start_time = time.time()
        #training/validation loop
        ############
        ##====training/validation step====##
        for epoch in range(self.config.nb_epochs):
            ###############CHANGED###########
            weight_tracker(self.model, module = True, verbose =  self.config.verbose)
            ###############
            
            
            ##training loop## (one epoch)
            nb_batch = len(self.loader)
            training_loss = 0 ### ADNI [] replaced to 0 (original code had error)
            training_acc = 0 # ADNI
            
            pbar = tqdm(total=nb_batch, desc="Training")
            for (inputs, labels) in self.loader:
                
                pbar.update()
                labels = torch.flatten(labels).type(torch.LongTensor) # ADNI
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                y = self.model(inputs)
                ##changed : change labels to float if not only reg but also binary class (because BCE can only do float even if 0 or 1)
                if self.config.task_type == 'reg' or self.config.binary_class == True : 
                    labels = labels.to(torch.float32) # ADNI
                
                batch_loss = self.loss(y, labels)
                batch_loss.backward()
                
                ###ANALYZING###
                #밑의 weight grad값들이 다 0이 아닌 것은 당연함!! (b/c didn't kill the weights)
                #print("classifier weight grad")
                #print(torch.sum(self.model.module.classifier.weight.grad))
                #print("conv layer weight grad")
                #print(torch.sum(self.model.module.features[0].weight.grad))
                ###ANALYZING###
                self.optimizer.step()
                training_loss += batch_loss.item()*inputs.size(0) # add up the batch losses
                
                
                ##ACC계산하는데 detach해야할듯한데 안함 => 고쳐야할듯? 
                if self.config.task_type == 'cls' :
                    if self.config.binary_class == False: 
                        _, predicted = torch.max(y, 1) # _ : acutal things, predicted : the indices where it occured 
                    
                    #if doing binary classification, calculate accuracy with threshold at 0.0
                    elif self.config.binary_class == True:     
                        predicted = (y > 0).float() #since labels is float
                    training_acc += (predicted == labels).sum().item() # ADNI    
                    
                
                
                
            if self.config.binary_class == False:
                training_loss = training_loss / len(self.loader.dataset) #should be false if BCEWithLogitLoss is used
            training_acc = 100 * training_acc / len(self.loader.dataset)
            pbar.close()
            
            wandb.log({"training_loss" : training_loss, "training_acc" : 0.01*training_acc}, step = epoch)
            
            ##=== Validation step (loop for one eopoch (multiple steps))====##
            outGT, outPRED, val_loss, val_acc, val_auroc = self.eval_model(mode = "valid")

            wandb.log({"validation_loss" : val_loss,
                      "validation_accuracy" : 0.01*val_acc, "validation_auroc" : val_auroc}, step = epoch) #log for each epoch
            #multiple values in the same epoch might confuse the system => 
            
            
            
            #======================TRYING TO IMPLEMENT PURNER========#
            ###trying to use smart pruner => not work yet
            if self.FOLD == 0 : #i.e. first fold 에서만 일단 pruning 
                pass
                                
                
                ###trying to prune with val_loss or val_auroc => NOT WORK BECAUSE direction (will build custom pruner later)
                #self.trial.report(val_loss, epoch)
                ##prune 
                #if self.trial.should_prune():
                #    print("pruend becasue weird val_auroc value")
                #    wandb.config.pruned = True
                #    raise optuna.TrialPruned()
                #    # Prune if sub-optimal
            
            ###trying to use dumb pruner (used regardless of the fold)
            if val_loss > 1e10 and epoch > 10 : #at least run for 10 epochs then see if pruning shouold be done 
                print("pruned because val_loss > 1e10")
                wandb.config.pruned = "large_val_loss"
                raise optuna.TrialPruned()
            #========================================================#
            
            
            #printing some statistics
            print("\nEpoch [{}/{}] Training loss = {:.4f}\t Validation loss = {:.4f}\t".format(
                  epoch+1, self.config.nb_epochs, training_loss, val_loss))
            if self.config.task_type == 'cls': # ADNI
                print("Training accuracy: {:.2f}%\t Validation accuracy: {:.2f}%\t Validation AUROC: {:.4f}\t".format(
                      training_acc, val_acc, val_auroc, flush=True))
            
            #removing this and change to sth else since we're doing optuna 
            #early_stopping(val_loss, self.model)
            if self.config.early_criteria in ["AUROC", "loss"] : 
                if self.config.early_criteria == "AUROC" : 
                    early_stopping(val_auroc, self.model) #early stopping based on auroc
                elif self.config.early_criteria == "loss":
                    early_stopping(val_loss, self.model)
                
                if early_stopping.early_stop: #i
                    print("[ Early stopped ]")
                    break
            else : #i.e. 'none'
                pass #do nothing
            
            ###
            if self.scheduler is not None:
                self.scheduler.step()
                
            ############################### stat.txt ###########################################
            #os.mkdir(parents=True, exist_ok=True)
            #pathlib.Path("./output").mkdir(exit_ok = True) 
            #stats_file = open(f'{self.path2}/stats.txt', 'a', buffering=1) #여기 수정 
            
            if epoch == 0: #added so that we save the typed system input (python XXX)
                print(json.dumps(' '.join(sys.argv)), file= self.stats_file)
            stats = dict(epoch=epoch, #step=step,
                         training_loss = (training_loss),
                         training_acc = round(training_acc, 2),
                         val_loss = (val_loss),
                         val_acc = (val_acc ), 
                        val_auroc = round(val_auroc, 3),
                        time = round(time.time()-start_time, 2))
            
            print(json.dumps(stats))
            print(json.dumps(stats), file=self.stats_file)
            ####################################################################################
        
        last_epoch = epoch + 1  #i.e. save the training phase's epoch (+1 because starts from zero)
        
        #===loading the early stopped model, and using it one last time to calculate the validaiton and so on===#
        
        if self.config.early_criteria == "none" : #i.e. 'none', then save the model points
            torch.save(self.model.state_dict(), ckpt_dir)
        
        ##the load_state_dict below comes from the save from the earlyg stopping (saves the model when early stopping reached)
        self.model.load_state_dict(torch.load(ckpt_dir)) # ADNI #reload the best thing!
        print(f"========after loading from the best ckpt (for testing (if early_critiera is none, then reloads the thing ) =======")
        weight_tracker(self.model, module = True, verbose = self.config.verbose)
        
        ##redoing validation with best model state (early stopped)
        ######CHANGE THIS TO "TEST" OR STH IF IN TEST MODE!
        outGT, outPRED, val_loss, val_acc, val_auroc = self.eval_model(mode = "valid")
        
            
        print("\nBEST EPOCH Validation loss = {:.4f}\t".format(val_loss))
        if self.config.task_type == 'cls': # ADNI
            print("Validation accuracy: {:.2f}%\t Validation AUROC: {:.2f}\t".format(val_acc, val_auroc,flush=True))
        ###CHANGE THIS TO "test" or sth if in eval mode!!
        
        self.val_loss = val_loss #saving the validation loss for future reference
        
        
        if self.config.task_type == 'cls' :
            self.val_acc = val_acc
        
        #for saving eval results and other data that are relevant
        print(json.dumps(self.config.__dict__), file= self.eval_stats_file) #save config dict stuff 
                                    #어? batch_size같은 것은 바뀌었지 않나=> NO! because we update the config itself
        #save training data
        print(json.dumps(stats), file = self.eval_stats_file) #remember taht this stats is the LAST stats (not the best performing one) from the TRAINING/VALIDATION
        
        """
        여기를 mode 가 evaluatino모드면, 밑에 commented out된것을 return시키고, 아니면 validation 것을 return 하도록 하기?
        """
        return outGT, outPRED, last_epoch #returns the last validation's outGT and outPRED! #also provide the value of the last epoch
    

    ##module.XXX 인, DDP용은 원래 대로 loading하면 안되서 https://stackoverflow.com/questions/70386800/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data 여기 방식대로 해서 함!
    def eval_model(self, mode):
        ## evaluating (forward prop once and so on)
        """
        input : self, mode (test or valid)
        output : outGT, outPRED, loss, acc (loss : normalized, acc : in terms of percentage)
        """
        ##selecting loader2use based on whether test or valid
        loader2use = self.loader_test if mode == "test" else self.loader_val if mode == "valid" else None
        
        
        nb_batch = len(loader2use)
        pbar = tqdm(total=nb_batch, desc=mode)
        loss = 0
        acc = 0
        outGT = torch.FloatTensor().cuda() #empty tensors
        outPRED = torch.FloatTensor().cuda()
        with torch.no_grad():
            self.model.eval()
            for (inputs, labels) in loader2use:
                pbar.update()
                labels = torch.flatten(labels).type(torch.LongTensor)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                y = self.model(inputs)
                
                
                #pruning if in valid mode (i.e. when using optuna)
                if y.isnan().any(): #i.e. if there are any nans, in the output
                    wandb.config.pruned = "y_nan_pruned"
                    print("purned because NAN")
                    raise optuna.TrialPruned() #prune this trial      
                
                if self.config.task_type == 'reg': 
                    labels = labels.to(torch.float32) 
                    outGT = torch.cat((outGT, labels), 0) # i.e. model regression actual answer
                    outPRED = torch.cat((outPRED, y), 0) 
                
                if self.config.task_type == 'cls':
                    if self.config.binary_class == False:
                        raise NotImplementedError("probalby not well done.. .jsut don't use multilabel")
                        #=======이 파트 : num_classes를 받아서 더 flexible하게 되도록 하기(지금은 안하고 나중에)=========
                        #also, 밑에서 한 것보다 더 smart하게 할 수 있을 것 같다! (CrossEntropyLoss자체적으로 label들을 받을 수 있는 것 
                        m = torch.nn.Softmax(dim=1)
                        output = m(y)
                        if int(labels) == 0:
                            onehot = torch.LongTensor([[1, 0]])
                        elif int(labels) == 1:
                            onehot = torch.LongTensor([[0, 1]])
                        onehot = onehot.cuda()
                        outGT = torch.cat((outGT, onehot), 0) #answers
                        outPRED = torch.cat((outPRED, output), 0) #model predictions
                        _, predicted = torch.max(y, 1)
                        acc += (predicted == labels).sum().item()
                        
                    elif self.config.binary_class == True:
                        labels = labels.to(torch.float32) #becasue BCE only works with float label values
                        
                        #m = torch.nn.Softmax(dim=1)
                        m = torch.nn.Sigmoid() #sigmoid instead of softmax remember
                        output = m(y) #output : logit이 아닌 0~1사이의 probability값!
                        
                        #OPTION 1 not forced (will use this) 
                        outGT = torch.cat((outGT, labels), 0)
                        outPRED = torch.cat((outPRED, y), 0)
                        predicted = (y > 0.5).float() #outPRED : not logit (but sigmoided) therefore the boundary 
                        acc += (predicted == labels).sum().item()
                        
        
                batch_loss = self.loss(y, labels)
                loss += batch_loss.item()*inputs.size(0)
                    
        pbar.close()
        
        ##normalizing the results by the number of samples used and so on 
        if self.config.task_type == 'cls':    
            if not self.config.binary_class : 
                loss = loss / len(loader2use.dataset)
                #why only if binary class is False? because CE와 달리 BCEWithLogitLoss는 이미 reduce = mean으로 하기때문에 avearge해줄 필요가 없다! 
            acc = 100 * acc / len(loader2use.dataset) #to be called up in the main.py
            
        else: #i.e. 'reg'
            loss = loss / len(loader2use.dataset)
            
        aurocMean = test_data_analysis_2(self.config, self.model, outGT, outPRED, self.config.task_include, self.FOLD, compute_only = True)
        
        return outGT, outPRED, loss , acc , aurocMean
        
    def load_model(self, path):
        checkpoint = None
        
        try: #실제로 사용되는 것 
            checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
            print("try in load_model succeeded") 
        except BaseException as e:
            self.logger.error('Impossible to load the checkpoint: %s' % str(e))
            raise ValueError("loading checkpoint failed!")
        if checkpoint is not None:
            try:
                if hasattr(checkpoint, "state_dict"):
                    print("hasattr has been used")
                    unexpected = self.model.load_state_dict(checkpoint.state_dict())
                    self.logger.info('Model loading info: {}'.format(unexpected))
                elif isinstance(checkpoint, dict): #실제로 사용되는 것 (for both BT AND checkpoint)
                    print("isinstance has been used") 
                    if "model" in checkpoint:
                        print("model in checkpoint = True") #실제로 사용되는 것 (for junbeom)
                        
                        #first attept at loading, assuming no `module.` thing (i.e. saving DDP model done properly) (https://stackoverflow.com/questions/70386800/what-is-the-proper-way-to-checkpoint-during-training-when-using-distributed-data)
                        unexpected = self.model.load_state_dict(checkpoint["model"], strict=False) #original method
                        
                        if len(unexpected.missing_keys) > 100 : #i.e. if not done properly, where more than 100 are not matched
                            ####LOAD USING THE SPECIAL METHOD, PROPOSED IN THE SITE ABOVE#####
                            state_dict = checkpoint["model"]
                            from collections import OrderedDict
                            new_state_dict = OrderedDict()
                            for k, v in state_dict.items():
                                name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                                new_state_dict[name] = v
                            unexpected = self.model.load_state_dict(new_state_dict, strict = False) ###new attempt
                                                
                        self.logger.info('Model loading info: {}'.format(unexpected))
                        
                        if len(unexpected.missing_keys) > 8 : #if still not proper,
                            raise NotImplementedError("probably improper loading was done, as more than 100 keys are missing")
                        ####
                        print(f"========inside load_model =======")
                        weight_tracker(self.model, verbose = self.config.verbose)
                    else :
                        print("===tried loading directly becasue model no tin the thing===")
                        unexpected = self.model.load_state_dict(checkpoint, strict = False)
                        weight_tracker(self.model, verbose = self.config.verbose)
                        
                        if len(unexpected.missing_keys) > 7 : #if still not proper,
                            raise NotImplementedError("probably improper loading was done, as more than 100 keys are missing")
                    print(f"===unexpected keys : {unexpected.missing_keys}===")
                else:
                    print("===else has been used====")
                    unexpected = self.model.load_state_dict(checkpoint)
                    self.logger.info('Model loading info: {}'.format(unexpected))
            #except BaseException as e:
            except Exception as e :
                print("weight loading failed")
                
                raise ValueError('Error while loading the model\'s weights: %s' % str(e))


def weight_tracker(model, module = False, verbose = None): #module : if model.module of just model?
    if verbose : 
        print(f" training mode? : {model.training}")
        
        if module == True:
            model = model.module #i.e. model을 model.module로 바꾸기
        
        print("conv0.weight       : ", torch.std(model.features.conv0.weight)) #conv3d 1 64 777 
        print("norm2.runninv_var : ", torch.std(model.features.denseblock4.denselayer16.norm2.running_var))
        print("denselayer16.conv2.weight : ", torch.std(model.features.denseblock4.denselayer16.conv2.weight))
        
        if model.mode == "classifier" : 
            print("norm5.runninv_var : ",torch.std(model.features.norm5.running_var))
            print("norm5.weight : ",torch.std(model.features.norm5.weight))
        elif model.mode == "classifier_no_BN" : 
            print("no norm, only classifier")
        elif model.mode == "classifier_inst_BN" : 
            #이거는 none이라서 std를 하지 않는다! 
            print("inst_norm5.running_var : ",model.features.inst_norm5.running_var)
            print("inst_nrom5.weight : ",model.features.inst_norm5.weight)
        else : 
            raise ValueError("here sth wrong")
        print("classifier.weight std : ",torch.std(model.classifier.weight))
        print("classifier.bias   : ",model.classifier.bias) #don't take std becasue only onevalue
    else : 
        return None