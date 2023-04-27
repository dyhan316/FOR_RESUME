from pathlib import Path

PRETRAINING = 0
FINE_TUNING = 1

# Dataset & csv files directory
kisti_data = {"ABCD": '/scratch/x2519a05/data/1.ABCD/T1_MNI_resize128', # should be implemented for multimodal data i.e. FA
              "CHA": '/scratch/x2519a05/data/CHA_bigdata/sMRI_resize128'}
kisti_csv = {"ABCD": '/scratch/x2519a05/data/1.ABCD/4.demo_qc/ABCD_phenotype_total_balanced_multitarget.csv',
             "CHA": '/scratch/x2519a05/data/CHA_bigdata/metadata/CHA_sMRI_brain.csv'}

lab_data = {"ADNI": '/scratch/connectome/study_group/VAE_ADHD/data',
            "ABCD": '/scratch/connectome/3DCNN/data/1.ABCD/2.sMRI_freesurfer',
            "UKB": '/scratch/connectome/3DCNN/data/2.UKB/1.sMRI_fs_cropped',
            "CHA": '/storage/bigdata/CHA_bigdata/sMRI_brain'}
lab_csv = {"ADNI": './csv/fsdat_baseline.csv',
           "ABCD": '/scratch/connectome/dyhan316/VAE_ADHD/junbeom_finetuning/csv/ABCD_csv/ABCD_phenotype_total_ONLY_MRI.csv',
           "ABCD_ADHD_strict": '/scratch/connectome/dyhan316/VAE_ADHD/junbeom_finetuning/csv/ABCD_csv/BT_ABCD_dataset_brain_cropped.csv',
           "UKB": None,
           "CHA": '/storage/bigdata/CHA_bigdata/metadata/CHA_sMRI_brain.csv'}

sdcc_data = {"test": '/hpcgpfs01/scratch/dyhan316/VAE_ADHD_data/ADNI_T1_data_SAMPLE_for_try',
             "ADNI": '/hpcgpfs01/scratch/dyhan316/VAE_ADHD_data/ADNI_T1_data',
             "ABCD": '/hpcgpfs01/scratch/dyhan316/VAE_ADHD_data/ABCD_MNI_CROP',
             "UKB": '/hpcgpfs01/scratch/dyhan316/VAE_ADHD_data/UKB_MNI_CROP',
             "CHA": None}
sdcc_csv = {"test": './csv/fsdat_try_sample.csv',
            "ADNI": './csv/fsdat_baseline.csv',
            "ABCD": '/sdcc/u/dyhan316/VAE_ADHD/barlowtwins/csv/updated_ABCD.csv',
            "UKB": None,
            "CHA": None}  

# dictionary used for class config
dataset_dict = {'kisti': kisti_data,
                'lab': lab_data,
                'sdcc': sdcc_data}
csv_dict = {'kisti': kisti_csv,
            'lab': lab_csv,
            'sdcc': sdcc_csv}


class Config:
    """
    Configurations for finetuning.
    Default attributes are defined initially,
    and args(input arguments from command line) are used for updating other configurations.
    """
    def __init__(self, args):
        self.mode = args.mode
        self.__dict__.update(args.__dict__) # get all attributes from args and inherit them to config
        self._set_default_config(args.task)
        self.nb_epochs = args.nb_epochs if args.nb_epochs else self.nb_epochs
        self.pretrained_path = args.pretrained_path
        
        self.data_path = dataset_dict[args.run_where]
        self.label_path = csv_dict[args.run_where]
        self._set_task_config(args.task)
        
        self._print_config()


    def _print_config(self):
        print('*** Configurations Info ***')
        print(f" - Data & metadata path: {self.data} & {self.label}")
        print(' - Pretrained path:', self.pretrained_path)
        print(f' - Task: Fine-tuning for {self.task_name}')
        print(f' - Task type: {self.task_type}')
        print(f' - N of train: {self.train_num}')
        print(f' - Policy: {self.stratify}')


    def _set_default_config(self, task):
        ## We assume a classification task here
        if self.train_num.isnumeric():
            self.train_num = int(self.train_num)
        elif self.train_num != 'all':
            raise ValueError(f"Not supported option for train_num {self.train_num}")
        self.nb_epochs_per_saving = 10
        self.pin_mem = False # when using persistent workers, pin_memory False is faster - jubin modified
        self.nb_epochs = 300
        self.cuda = True
        self.task = task
        self.tf = 'cutout'
        self.model = 'DenseNet'
        self.valid_ratio = 0.25
        self.BN = 'default' # - jubin modified
        if self.input_option == "yAware":
            self.resize_method = 'reshape'
            self.input_size = (1, 80, 80, 80)
        elif self.input_option == "BT_org":
            self.resize_method = None
            self.input_size = (1, 99, 117, 95)
        elif self.input_option == "BT_unet":
            raise NotImplementedError("do this after implementing brain specific augmentation")
        ### special, must change by the weights we use 
        # self.resize_method = 'reshape'    #"padcrop", None (three cases, resizing or pad/cropping or none)        
        # self.input_size = (1, 80, 80, 80) # junbeom weights        
        # below are now deifned in parser
        # self.batch_size = 32            
        # self.lr = 1e1 #1e-4
        # self.weight_decay = 5e-5
        
        
    def _set_task_config(self, task):
        self.data = self.data_path[self.dataset]
        self.label = self.label_path[self.dataset] if task != 'ABCD_ADHD_strict' else self.label_path[task]
        self.task_type = 'cls' if 'age' not in task else 'reg' # should be modified later. using cat_target or num_target would be better.
        self.num_classes = 2 if 'age' not in task else 1 # should be modified later. using cat_target or num_target would be better.
        
        task_config_func = {'ADNI': self._set_ADNI_config,
                            'ABCD': self._set_ABCD_config,
                            'UKB': self._set_UKB_config,
                            'CHA': self._set_CHA_config}
        task_config_func[self.dataset](task)
        
        
    def _set_ADNI_config(self, task):
        self.subjectkey = 'SubjectID'
        self.iter_strat_label = {"binary" : ['PTGENDER'], "multiclass" : [] ,"continuous" : ["PTAGE"]} 
        if 'test' in task:
            self.nb_epochs = 5
            self.label_name = 'PTAGE' if 'age' in task else "Dx.new"
            self.task_name = 'AGE' if 'age' in task else "AD/CN"
        elif 'ALZ' in task:
            self.label_name = 'Dx.new'
            self.task_name = task.split("_")[-1].insert(1,'/') # "AD/CN", "AD/MCI", "CN/MCI"
        elif 'sex' in task:
            self.label_name = 'PTGENDER' # ADNI # `Dx.new` #####
            self.task_name = "M/F"
        elif 'age' in task:
            self.label_name = "PTAGE"
            self.task_name = "AGE"   
        else:
            raise NotImplementedError(f"Invalid task - {task}")

            
    def _set_ABCD_config(self, task):
        if "HC" in task : #i.e. if including healthy subs only
            raise NotImplementedError("healthy only version notimplemented yet")
        self.subjectkey = 'SubjectID'
        self.iter_strat_label = {"binary" : ['sex'], "multiclass" : [] ,"continuous" : ["age", "nihtbx_totalcomp_uncorrected"]} #MUST BE IN LIST FORM, 이 label들을 정한이유 : 
        self.task_type = 'cls'
        self.num_classes = 2
        if 'sex' in task:
            self.label_name = 'sex'
            self.task_name = "1.0/2.0"
        elif "ADHD_strict" in task : #train_num = 7265 |||| is max (if we take out intelligence from the iter strat label)
            self.iter_strat_label = {"binary" : ['sex'], "multiclass" : [] ,"continuous" : ["age"]} #removed intelligence because it took out too much samples (NAN)
            self.label_name = 'ADHD'
            self.task_name = "HC/ADHD"      
        else:
            raise NotImplementedError(f"Invalid task - {task}")
            
            
    def _set_UKB_config(self, task):
        raise NotImplementedError("UKB not done yet, also BT, yAware weights에 따라 resize method등등 다르게 해서 하기!")
        self.subjectkey = 'eid'
        
    
    def _set_CHA_config(self, task):
        self.subjectkey = 'subjectkey'
        self.iter_strat_label = {"binary" : ['sex'], "multiclass" : [] ,"continuous" : ["age(days)"]}
        if "ASDGDD" in task : 
            self.task_type = 'cls'
            self.num_classes = 2
            self.label_name = "ASDvsGDD"
            self.task_name = "ASD/GDD"
    

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