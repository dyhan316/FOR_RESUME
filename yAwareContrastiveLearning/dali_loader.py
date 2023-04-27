##original code in VAE_ADND/NVIDIA_DALLI_AUGMENTATION

##trying to just do regular stuff
from nvidia.dali.pipeline import Pipeline
from nvidia import dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

##trying to just do regular stuff
from nvidia.dali.pipeline import Pipeline
from nvidia import dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np 

data_path = "/hpcgpfs01/scratch/dyhan316/testing_4_dali"

def random_augmentation(probability, augmented, original):
    """
    * probability : prob of returning augmented (instead of original)
    * augmented : augmented image (fn거친 것)
    * original : original image (before going through augmented thing)
    """
    condition = fn.cast(fn.random.coin_flip(probability=probability), dtype=types.DALIDataType.BOOL)
    neg_condition = condition ^ True
    return condition * augmented + neg_condition * original

class GenericPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, imgs, lbls, aug_seed, shard_seed, shard_id, num_shards, **kwargs):
        super().__init__(batch_size, num_threads, device_id, seed = aug_seed)
        #my additions
        
        self.multi_channel = kwargs["multi_channel"]
        self.shard_seed = shard_seed
        self.shard_id = shard_id
        self.num_shards = num_shards
        if self.multi_channel : 
            raise NotImplementedError("make sure that the multi-channel (DWI) works with dali (especially the crop and cutout)")
        self.resize_method = kwargs["resize_method"]
        self.resize_shape = kwargs["resize_shape"]
        
        if self.resize_method : #i.e. resize shape is not None
            if not self.resize_shape : #sanity check
                raise ValueError(f"you specified resize method {self.resize_method}, and therefore the shape should've been specified, but got {self.resize_shape}")
                
            self.resize_shape = types.Constant(np.array(self.resize_shape), dtype=types.INT64) #make it into int
            self.resize_shape_float = 1.0*self.resize_shape #floatized resize_shape DALI data thing 
        else : #i.e. resize_method is None or False :
            if self.resize_shape : 
                raise ValueError(f"since resize_method is None, the resize_shape should also be None! ")
        
        self.kwargs = kwargs
        self.dim = kwargs["dim"]
        self.device = device_id
        #self.layout = kwargs["layout"]
        self.load_to_gpu = kwargs["load_to_gpu"]
        self.input = self.get_reader(imgs, name = "ReaderX") #list of image directories
        self.label = self.get_reader(lbls, name = "ReaderY")
        #self.labels = kwargs["labels"] # list of labels not done yet, as version 1.25 is neede
        #self.cdhw2dhwc = ops.Transpose(device="gpu", perm=[1, 2, 3, 0])            
                
    def get_reader(self, data, name):
        """
        * data : LIST of abs dir of data files 
        """
        return fn.readers.numpy(files = data, #gets the list of files 
                                device = 'cpu',
                                shard_id = self.shard_id, num_shards = self.num_shards,
                                seed = self.shard_seed, random_shuffle = True, name = name,
                                pad_last_batch = True, read_ahead = True) #random_shuffle없으면, seed가 달라도 똑같은 데이터를 sharding한다  
        
        """
        ADD the options below when needed (일단은 loading하는 것만 함)
        """
        #return ops.readers.Numpy(
        #    files=data,
        #    device="cpu",
        #    read_ahead=True,
        #    dont_use_mmap=True,
        #    pad_last_batch=True,
        #    shard_id=self.device,
        #    seed=self.kwargs["seed"],
        #    num_shards=self.kwargs["gpus"],
        #    shuffle_after_epoch=self.kwargs["shuffle"],
        #)

    def load_resize_data(self):
        img_cpu = self.input
        #img_cpu = fn.cast(img_cpu, dtype = types.FLOAT) #npy가 double로 로딩된 경우, 에러가 떠서 32fp로 부까기 
            
        if self.load_to_gpu:
            img = img_cpu.gpu()
        img = fn.cast(img, dtype = types.FLOAT) #npy가 double로 로딩된 경우, 에러가 떠서 32fp로 부까기 
        
        if not self.kwargs['multi_channel'] :
            img = img[dali.newaxis] #add one additional axis if not multi channel
        img = fn.reshape(img, layout = "CDHW") #resahpe the metadata
        
        ###여기다다 resize하면 됨 
        #일단 좀따 해보자 
        ##https://github.com/NVIDIA/DALI/issues/2492
        if self.resize_method == "reshape" :             
            img = fn.resize(img, size = self.resize_shape_float, mode = "default")#[1.0*i for i in self.resize_shape]
        elif self.resize_method == "padcrop" : 
            raise NotImplementedError("Not done yet ")
        elif self.resize_method == None : 
            pass
        else : 
            raise ValueError(f"{self.resize_method} is not one of the possible options")
            
        if self.resize_method : 
            org_shape = self.resize_shape_float
        else : 
            org_shape = fn.shapes(img_cpu) #get org shape from img_cpu, (not resized)        
        
        return img, org_shape # img_cpu #, label #reutrn img_cpu cuz sometimes it's needed

    def make_dhwc_layout(self, img, lbl):
        img, lbl = self.cdhw2dhwc(img), self.cdhw2dhwc(lbl)
        return img, lbl

    def crop(self, data):
        return fn.crop(data, crop=self.patch_size, out_of_bounds_policy="pad")

    def crop_fn(self, img, lbl):
        img, lbl = self.crop(img), self.crop(lbl)
        return img, lbl

    def transpose_fn(self, img, lbl):
        img, lbl = fn.transpose(img, perm=(1, 0, 2, 3)), fn.transpose(lbl, perm=(1, 0, 2, 3))
        return img, lbl
        

class TrainPipeline_IMPROVED(GenericPipeline):
    def __init__(self, batch_size, num_threads, device_id, imgs, lbls, aug_seed, shard_seed, 
                 shard_id, num_shards, tf = "all_tf", mode = "valid", **kwargs):
        """
        * tf : which tf to use (all_tf, cutout, crop, none)
        * mode : "CL_train", "valid" (add more) (CL_train : returns two different views of the same input
        """
        super().__init__(batch_size, num_threads, device_id, imgs, lbls, aug_seed, shard_seed, 
                         shard_id, num_shards, **kwargs)
        #self.oversampling = kwargs["oversampling"]
        self.tf = tf
        self.mode = mode
        
        
    def flips_fn(self, img, prob = 0.5):
        #prob : 0.5, as in before
        kwargs = {
            "horizontal": fn.random.coin_flip(probability=0.5),
            "vertical": fn.random.coin_flip(probability=0.5),
        }
        if self.dim == 3:
            kwargs.update({"depthwise": fn.random.coin_flip(probability=0.5)})
        return random_augmentation(prob, fn.flip(img, **kwargs), img) #kwargs : horizontal, vertical and so on (whether to flip in those dir or not)
    
    def blur_fn(self, img, prob = 0.5):
        #prob, sigma range를 yAware에 맞춤
        img_blurred = fn.gaussian_blur(img, sigma=fn.random.uniform(range=(0.1, 1)))
        return random_augmentation(prob, img_blurred, img)
    
    def noise_fn(self, img, prob = 0.5):
        #random noise level (between 0.1 and 1 becasue that's what was used in org)
        std = fn.random.uniform(range = [0.1,1]) 
        img_noised = fn.noise.gaussian(img , stddev = std)
        return random_augmentation(prob, img_noised, img) 
    
    def crop_fn(self,img, org_shape, crop_size, type = 'random', resize = False, prob = 0.5):
        """
        * img : original image
        * org_shape : original shape (cpu shape data node)(gpu면 안됨, gpu node끼리는 못주기에)
        * crop_size : size of the crop (single float value b/w 0~1) (dim 마다 다르게 하는 거는 몰라서 적용안함)
        * type : 
            * "random" : random start point 
            * "center" : not implemented (안쓰임)
        * resize : resize여부 resize시켜서 original img와 같은 dim을 가지게 하는가?        
        """
        #need to add zoom later
        patch_dim = crop_size*org_shape #0.75면 shape자체가 0.75배씩됨 #float임! (그래도 되더라)
        if type == "center" : 
            raise NotImplementedError("center not implemented yet")
        elif type == "random" : 
            kwargs = {
                "crop_pos_x" : fn.random.uniform(range = [0.0, 1.0]),
                "crop_pos_y" : fn.random.uniform(range = [0.0, 1.0]),
                "crop_pos_z" : fn.random.uniform(range = [0.0, 1.0]), #random값이 0~1이어도 되는 것이, 1예가 알아서 해주는 듯 
            } 
        else : #default (not used)
            kwargs = {}
            
        img_cropped = fn.crop(img, crop = patch_dim, **kwargs)
        
        if resize : 
            img_cropped_resized = fn.resize(img_cropped, size = fn.cast(org_shape, dtype = types.FLOAT)[0:3])#org_shape)
            return random_augmentation(prob, img_cropped_resized, img)#random_augmentation(0.5, img_cropped_resized, img)
        else : 
            return img_cropped
            #주의 : random_augmentation은 안됨 왜냐하며,ㄴ "will not return random_aug because the output shape will be different, so random_aug cannot be applied (cannot broadcast)")
    
    def cutout_fn(self, img, org_shape, cutout_size, anchor_type = "random", prob = 0.5):
        """
        * img : original image
        * org_shape : origina shape
        * cutout_size : float or "random", if float (0~1)사이, makes a cube to use 
        * anchor_type : type of anchor, "center" or "random"
        #dali does not have native cutout implemented, but took taken inspiration from the erase operator (https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/erase.html) 
        """
        if type(cutout_size) == float : 
            cutout_dim = cutout_size * org_shape
        elif cutout_size == "random":
            cutout_size = fn.random.uniform(range = [0.0, 1.0], shape = 1) 
            cutout_dim = cutout_size * org_shape
        else : 
            raise ValueError(f"provided cutout_size argument {cutout_size} is invalid")
            
        cutout_dim = cutout_size * org_shape 
        
        if anchor_type == "random": 
            anchor = fn.random.uniform(range = [0.0, 1.0], shape = self.dim) #3차웜이면 TensorList of 3 float values를 만듬
        elif anchor_type == "center" :
            None
        img_cutout = fn.erase(img, anchor = anchor, shape = cutout_dim,
                             normalized_anchor = True,
                             normalized_shape = False,
                             fill_value = 1.0) #fill_value 1.0 for testing purposes
        return random_augmentation(prob, img_cutout, img)
        
    def all_tf(self,img, org_shape):
        img = self.flips_fn(img)
        img = self.blur_fn(img)
        img = self.noise_fn(img)
        img = self.cutout_fn(img, org_shape, cutout_size = 0.25) #cutout_size : random가능, but not recommended
        img = self.crop_fn(img, org_shape, crop_size = 0.75, type = "random", resize = True) 
        return img
        
    def define_graph(self):
        #numpy reading은 이미 generic __init__에서 함 
        
        #loading and stuff
        img, org_shape = self.load_resize_data() #move to GPU
        
        #acutal aug, prob of doing each : 0.5, except normalize, which is 100%
        img = fn.normalize(img, axis_names = "DHW") #"DHW" so that channel-wise normalization is not done 
        
        
        #copy tensor if CL_train mode 
        if self.mode == "CL_train" : 
            img_prime = fn.copy(img) #diff aug, same img
        else : #i.e. valid or sth else
            img_prime = None 
        
        #apply aug based on tf type
        if self.tf == "all_tf" : 
            img = self.all_tf(img, org_shape)
            if self.mode == "CL_train" :  #i.e. if CL_train mode, do seoncd aug too 
                img_prime = self.all_tf(img_prime, org_shape)
                                
        elif self.tf == "cutout" : 
            """
             cutout has prob 100% in "cutout" because 1. how MICCAI paper did it , 2. cutout paper found 100% aug all the time was best, assuming that the cutout box could partially outside of the image itself (which is true in our version of cutout implementation)"""
            img = self.cutout_fn(img, org_shape, cutout_size = 0.25, prob = 1.0) 
            if self.mode == "CL_train" : 
                img_prime = self.cutout_fn(img_prime, org_shape, cutout_size = 0.25, prob = 1.0)        
                
        elif self.tf == "crop" : 
            img = self.crop_fn(img, org_shape, crop_size = 0.75, type = "random", resize = True, prob = 1.0) 
            if self.mode == "CL_train" : 
                img_prime = self.crop_fn(img_prime, org_shape, crop_size = 0.75, type = "random", resize = True, prob = 1.0) 
  
        elif self.tf == "none" : 
            pass
        else : 
            raise ValueError(f"provided {self.tf} option is not valid...") 
            
        
        print("flip here enalbes any combinations of three types of flipping, while the original only had one")
        print("therefore, ipmlement the original yAware version of flipping too (or not..")
        print("in normalizaiton, make sure that even when doing multichannel, channel-wise normalization is not done (only single channel-wise normalization(즉, axis_names = 'DHW'가 잘작동하는지 보기 ")
        print("also see if dali batch size and gpu batch size have to be the same (즉, 16짜리 dali를 두번돌려서 2*16 = 32W짜리 model batch size에 맞춰도 되나?")
        print("DDP + DALI? (data sampler이 DDP스면 rank마다 다르게 주는 걸로 알고있는데, 이거에 맞게 DALI는 어떻게 되는거지?")
        print("also crop/cutout patch size could also be made random if needed? ")
        
        if self.mode == "CL_train" : 
            return fn.stack(img, img_prime, axis = 0), self.label
        else: 
            return img, self.label #fn.get_property(img_cpu, key = "source_info") #didn't work, asked NVIDIA
    
#wheng making the other versions, just super.__init__ from the original Training thing and add onto it !



def fetch_dali_loader(config, data_dir_list, lbl_dir_list, file_type, mode,  *args, **kwargs):
    """
    이거 만들때 여기 참조하기!! : https://github.com/NVIDIA/DeepLearningExamples/blob/1e103522fef1066bf1ae94d5b9c8b2cee72a5d92/PyTorch/Segmentation/nnUNet/data_loading/dali_loader.py#L240
    
    """
    if file_type != "npy" : 
        raise ValueError("dali loader can only load npy as of moment!!")
    
    #####setting kwargs and so on for pipeline create####
    kwargs = {"load_to_gpu" : True,  "multi_channel" : False,  #if DWI : maybe multichannel?
              "dim": 3, "resize_method" : "reshape", #reshape, padcrop, None 세가지 가능 
              "resize_shape" : config.input_size[1:]} #[1:] because need to cutout the channel dimension    #[80,80,80]}
    
    DALI_BATCH_SIZE = config.dali_batch_size #내가 임의로 일단 정해둠
    num_threads = config.num_cpu_workers
    epoch = 0 
    if mode == "train" : 
        shuffle_after_epoch = True
        if config.DP_DEBUG  :  #DEBUG 모드에서는 직접 지정해줘야함 (since DP 에서 world size = -1)
            rank = 0
            world_size = 1
        else : 
            rank = config.rank
            world_size = config.world_size     
    elif mode == "valid":
        shuffle_after_epoch = False
        rank = 0
        world_size = 1        
    else : 
        raise ValueError()
        
    print("need to make shuffle_after_epoch = True for numpy reader since I won't be resetting the pipeline!)")
    print("아니다... 이미 random shuffle = True로 했기에, shuffle after epoch을 쓰지 못함.. 둘중 하나 포기해야한대.. 근데 그러면 어떻게 할지는 아직 생각안해서 일단 보류")
    ##first pipe (sharded)
    pipe = TrainPipeline_IMPROVED(batch_size = DALI_BATCH_SIZE, num_threads = num_threads,
                                    device_id = rank, 
                                    imgs = data_dir_list, lbls = lbl_dir_list,
                                    aug_seed = 0, #controls what augs will be done 
                                    shard_seed = epoch, shard_id = rank, num_shards = world_size, #controls which data each shard will get (random sampler)
                                    tf = "all_tf", mode = "CL_train",
                                  shuffle_after_epoch = shuffle_after_epoch,**kwargs)
    
    print(f"rank : {config.rank}, world+size {config.world_size}")
    #may not need to delete it every epoch? (doesn't take up that much...)
    #delete every epoch하고 epoch마다 shard_seed다르게 하려면 복잡해짐 (need ot make pipeline every epoch)
    #대신, shuffle_after_epoch 을 True로 해야함
    
    #일단은, 그냥 shuffle_after_epoch = True로 해보고 memory contraint많이 가져가면 그 del 하고 하자
    #optimize the parameters for numpy reader
    
    #implement labels (1.25)
    pipe.build()
    
    #TODO : 
    #import pdb ; pdb.set_trace()
    #위의 kwargs도... 다시 만들어야함 #(config에 "reshape"옵션을 넣는 등등)
    #실제 transform 결과 확인하기!(npy로 저장해서) (epoch 별로)
    #time it해서, numpy reader option조합을 어덯게 하면 가장 빠른지 조사하기 
    #dataset.py에 맞게 바꿔야함
    #also using del to remove the pipeline and so on 도 하기
    #also dali batch size diff도 하기!
    #densenet.py 다시 고치기
    #config내에서 input_size를 무조건 지정해주도록 했는데.. 이렇게 하는게 맞을지, talarich space면 바꾸게 해야할지 등등을 고려해야할듯 
        #만약 바꿔야한다면, config_SDCC, dali_loader를 바꿔야함
        #also, config내에서 channel dim까지 주도록 했음 (1,80,80,80)이런식으로... 이가정이 무조건 맞아야함! 왜냐하면 shape를 config.input_size[1:] 로 가지도록 만들었기때문!
    dali_iter = DALIGenericIterator([pipe], ['data', 'label'], 
                                    reader_name = "ReaderX",
                                    last_batch_policy = LastBatchPolicy.DROP)
    
    return dali_iter
            
    #TODO :
    # * 밑의 사이트 보기
    # * dali reader 두개 (reader X, reader Y)인데 그 두개로 어떻게 그 불러올지 고민해보기
    # drop last batch등등 어떻게 세팅할지 고민해보기
    # https://github.com/NVIDIA/DeepLearningExamples/blob/1e103522fef1066bf1ae94d5b9c8b2cee72a5d92/PyTorch/Segmentation/nnUNet/data_loading/dali_loader.py#L240
    #위에 참조하기!!
    #그 pth저장되는 곳에다가 label저장시키고, continued running이 되도록 exist_ok = True로 하기?

#wheng making the other versions, just super.__init__ from the original Training thing and add onto it !

    #5min : 264
    #5min : 277 #with read ahead
    
    #with dali batch isze 16
    #