class Config:

    def __init__(self, args):

        #args에 있는 속성들을 config에 등록
        for name, value in vars(args).items():
            setattr(self, name, value)
        
        # self.mode = mode
        # self.framework = args.framework
        # self.world_size = args.world_size
        # self.rank = args.rank
        # self.dist_backend = args.dist_backend
        # self.local_rank = args.local_rank
        # self.batch_size = args.batch_size
        # self.lr_policy = args.lr_policy
        # self.lr_decay_iters = args.lr_decay_iters
        # self.gamma = args.gamma
        
        
        if self.mode == 'pretraining':
            self.nb_epochs_per_saving = 1 # 몇 에포크에 한 번씩 저장할 것인가
            self.pin_mem = True
            self.num_cpu_workers = 16
            self.cuda = True
            
            # Hyperparameters for our y-Aware InfoNCE Loss
            self.temperature = args.temp #add this to the hyperparam!
            
            self.model = "DenseNet"
            
            self.checkpoint_dir = args.ckpt_dir
            self.tb_dir = args.tb_dir
            self.tf = args.tf
            self.label_name = args.label_name
            
                        
            self.val_size = 0.1
            
            
            #this MAY NEED TO BE CHANGED IF DOING THE TALARICH SPACE
            self.input_size = (1, 80, 80, 80) #(1, 121, 145, 121)

            self.checkpoint_dir = args.ckpt_dir
            
            self.train_continue = "on"
            
            if "UKB" == args.dataset2use : 
                self.data = '/scratch/connectome/3DCNN/data/2.UKB/1.sMRI_fs_cropped/' 
                self.label = '/scratch/connectome/dyhan316/VAE_ADHD/junbeom_finetuning/csv/UKB_phenotype_gps_included.csv' 
            elif "CHA" in args.dataset2use : 
                self.data = "/storage/bigdata/CHA_bigdata/sMRI_brain"
                
                if "secHC" in args.dataset2use : #i.e. CHA_PSM #
                    #onlyl with secHC subjects
                    self.label = "/scratch/connectome/dyhan316/VAE_ADHD/yAwareContrastiveLearning/csv/CHA_pretrain.csv"
                else :  #label that has all 
                    raise NotImplementedYet("eid, age로 column 값들이 바뀌여야함!! (i.e. for pretraining).. 아직 안함!")
                    self.label = '/storage/bigdata/CHA_bigdata/metadata/CHA_sMRI_brain.csv'
                

       # elif self.mode == 'finetuning':
       #     ## We assume a classification task here
       #     self.batch_size = 8
       #     self.nb_epochs_per_saving = 10
       #     self.pin_mem = True
       #     self.num_cpu_workers = 1
       #     self.nb_epochs = 100
       #     self.cuda = True
       #     # Optimizer
       #     self.lr = 1e-4
       #     self.weight_decay = 5e-5
#
       #     self.pretrained_path = "/path/to/model.pth"
       #     self.num_classes = 2
       #     self.model = "DenseNet"
