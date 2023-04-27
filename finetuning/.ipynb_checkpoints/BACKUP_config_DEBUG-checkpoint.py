from cmath import nan

PRETRAINING = 0
FINE_TUNING = 1

class Config:

    def __init__(self, mode, task):
        assert mode in {PRETRAINING, FINE_TUNING}, "Unknown mode: %i"%mode

        self.mode = mode

        if self.mode == PRETRAINING:
            raise NotImplementedError("shouldn't use this as pretarining part of this code is wrong ")
            #self.batch_size = 8 # ADNI
            #self.nb_epochs_per_saving = 1
            #self.pin_mem = True
            #self.num_cpu_workers = 8
            #self.nb_epochs = 100 # ADNI #####
            #self.cuda = True
            ## Optimizer
            #self.lr = 1e-4
            #self.weight_decay = 5e-5
            ## Hyperparameters for our y-Aware InfoNCE Loss
            #self.temperature = 0.1
            #self.tf = 'cutout' # ADNI
            #self.model = 'DenseNet' # 'UNet'
            #### ADNI
            #self.data = './adni_t1s_baseline' # ADNI
            #self.label = './csv/fsdat_baseline_CN.csv' # ADNI
            #self.valid_ratio = 0.25 # ADNI (valid set ratio compared to training set)
            #self.input_size = (1, 80, 80, 80) # ADNI #####
            #
            #self.label_name = ['PTAGE', 'PTGENDER'] # ADNI
            #self.label_type = ['cont', 'cat'] # ADNI
            #self.cat_similarity = [nan, 0] # similarity for mismatched categorical meta-data. set nan for continuous meta-data
            #self.alpha_list = [0.5, 0.5] # ADNI # sum = 1
            #self.sigma = [5, 5] # ADNI # depends on the meta-data at hand
            #
            #self.checkpoint_dir = './ckpts' # ADNI
            #self.patience = 20 # ADNI
        
        ##trash values (only for debugging)
        elif self.mode == FINE_TUNING:
            ## We assume a classification task here
            ##default, true for all
            self.nb_epochs_per_saving = 10
            self.pin_mem = True
            self.num_cpu_workers = 1
            self.nb_epochs = 200 # ADNI #####
            self.cuda = True
            self.task = task # for future reference

            
            ##training hyperparams
            self.tf = 'cutout' # ADNI #option for none? 
            self.model = 'DenseNet' # 'UNet'
            self.patience = 20 # ADNI
            self.valid_ratio = 0.25 # ADNI (valid set ratio compared to training set)
            self.input_size = (1, 80, 80, 80) # ADNI

            
            #these are now deifned in parser
            #self.batch_size = 32            
            #self.lr = 1e1 #1e-4
            #self.weight_decay = 5e-5
            
            
            if task == "test":
                self.nb_epochs = 5
                self.data = '/scratch/connectome/study_group/VAE_ADHD/data' #'./adni_t1s_baseline' # 
                self.label = './csv/fsdat_baseline.csv' # where the label file is 
                self.task_type = 'cls' # ADNI # 'cls' or 'reg' #####
                self.label_name = 'Dx.new' # ADNI # `Dx.new` #####
                self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####
                self.task_name = "AD/CN"

            ###task 정해주기###
            if "ADNI" in task: #i.e. if ADNI
                ### ADNI
                self.data = '/scratch/connectome/study_group/VAE_ADHD/data' #'./adni_t1s_baseline' # ADNI #ADNI dataset path 
                self.label = './csv/fsdat_baseline.csv' # where the label file is 
                
                if task == "ADNI_ALZ_ADCN":
                    self.task_type = 'cls' # ADNI # 'cls' or 'reg' #####
                    self.label_name = 'Dx.new' # ADNI # `Dx.new` #####
                    self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####
                    self.task_name = "AD/CN"
                    
                elif task == "ADNI_ALZ_ADMCI":
                    self.task_type = 'cls' # ADNI # 'cls' or 'reg' #####
                    self.label_name = 'Dx.new' # ADNI # `Dx.new` #####
                    self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####
                    self.task_name = "AD/MCI"
                    
                elif task == "ADNI_ALZ_CNMDI":
                    self.task_type = 'cls' # ADNI # 'cls' or 'reg' #####
                    self.label_name = 'Dx.new' # ADNI # `Dx.new` #####
                    self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####
                    self.task_name = "CN/MCI"
                ###could get more variants here###    
                elif task == "ADNI_sex":
                    self.task_type = 'cls' # ADNI # 'cls' or 'reg' #####
                    self.label_name = 'PTGENDER' # ADNI # `Dx.new` #####
                    self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####
                    self.task_name = "M/F"
                #not sure if it works, but whatever haha 
                elif task == "ADNI_age":
                    self.task_type = 'reg'
                    self.label_name = "PTAGE"
                    self.num_classes = 1
                    self.task_name = "AGE"
                    
            
            ##possible ABCD options
            ##not sure if doing it like this is a good idea!! but whatever !
            #"ABCD" + "HC" (or not) + "sex" or "ADHD" (이들의 조합으로 만들기 가능)
            elif "ABCD" in task : #if ABCD task 
                
                if "HC" in task : #i.e. if including healthy subs only
                    ###
                    raise NotImplementedError("healthy only version notimplemented yet")
                else : #i.e. include all subjects,
                    self.data = '/scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked'
                #'/scratch/connectome/3DCNN/data/1.ABCD/1.sMRI_fmriprep/preprocessed_masked'
                    self.label = '/scratch/connectome/dyhan316/VAE_ADHD/barlowtwins/csv/updated_ABCD.csv'
                if "sex" in task : #i.e. if sex classification
                    self.task_type = 'cls'
                    self.label_name = 'sex'
                    self.num_classes = 2
                    self.task_name = "M/W"
                if "ADHD" in task : 
                    self.task_type = 'cls'
                    raise NotImplementedError("ADHD not implemented yet ")
                    
                    
                    
                
    
                #self.pretrained_path = './weights/BHByAa64c.pth' # ADNI #####
                #self.layer_control = 'tune_all' # ADNI # 'freeze' or 'tune_diff' (whether to freeze pretrained layers or not) #####
            
         #below : for barlow twins (where the input_size is different)
        """elif self.mode == FINE_TUNING:
            ## We assume a classification task here
            self.batch_size = 8
            self.nb_epochs_per_saving = 10
            self.pin_mem = True
            self.num_cpu_workers = 1
            self.nb_epochs = 100 # ADNI #####
            self.cuda = True
            # Optimizer
            self.lr = 1e-4
            self.weight_decay = 5e-5
            self.tf = 'cutout' # ADNI
            self.model = 'DenseNet' # 'UNet'
            ### ADNI
            self.data = '/scratch/connectome/study_group/VAE_ADHD/data' #'./adni_t1s_baseline' # ADNI
            self.label = './csv/fsdat_baseline.csv' # ADNI
            self.valid_ratio = 0.25 # ADNI (valid set ratio compared to training set)
            self.input_size = (1, 99, 117, 95) # ADNI

            self.task_type = 'cls' # ADNI # 'cls' or 'reg' #####
            self.label_name = 'Dx.new' # ADNI # `Dx.new` #####
            self.num_classes = 2 # ADNI - AD vs CN or MCI vs CN or AD vs MCI or reg #####

            #self.pretrained_path = './weights/BHByAa64c.pth' # ADNI #####
            #self.layer_control = 'tune_all' # ADNI # 'freeze' or 'tune_diff' (whether to freeze pretrained layers or not) #####
            self.patience = 20 # ADNI
       """