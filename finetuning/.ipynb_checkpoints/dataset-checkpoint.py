import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset
import monai
from monai.data import ImageDataset
from monai.transforms import ScaleIntensity, ResizeWithPadOrCrop, NormalizeIntensity, ToTensor, EnsureChannelFirst, ToNumpy, Resize

from transforms import transform_yAware_all, transform_yAware_cutout, transform_yAware_crop

class basic_transform():
    """
    basic transform for creating the dataset in the first place 
    (MNI로 fixed input size 면 not resize하고 등등)
    
    MNI : if true, does not apply ResizeWithPadOrCrop, but if false, applies the spatial size that was given by **kwargs
    """
    def __init__(self, resize_method, **kwargs):
        transform_list = [EnsureChannelFirst()]  #removed scale intensity
                    
        #add padding/cropping if not MNI (i.e. shape varies)
        shape_no_channel = kwargs['shape'][1:] #shape without the channel dimension, as that should be used when doing Resizewithpadorcrop
        
        if resize_method == "reshape": 
            transform_list.append(Resize(spatial_size = shape_no_channel))
        elif resize_method == "padcrop":
            transform_list.append(ResizeWithPadOrCrop(spatial_size =  shape_no_channel, method = "symmetric", mode = "edge"))
        else : 
            assert resize_method == None, "resize method must be either None reshape or padcrop"
                    
        #adding the normalize intensity To Tensor things 
        transform_list = transform_list + [NormalizeIntensity(), ToNumpy()]
            #ToNumpy instead of tensor because transform yAware is implemnetned in numpy
        self.transform = monai.transforms.Compose(transform_list)        

    def __call__(self, x):
        return self.transform(x)          

    
##changed function UKB_T1 to MRI_dataset
class MRI_dataset(Dataset):
    """
    split = 'train' or 'test'
    split_prop = '0~1'사이, how much to take as training for splitting
    transform = transform to be performed
    shape = shape of the image (MUST INCLUDE CHANNEL)
        * if MNI==True, does not do cropping or anything
        * if MNI==False, apply ResizeWithPadOrCrop
    MNI = fixed shape or not
    """
    
    #def __init__(self,data_path, split, split_prop, transform, shape, MNI= True): #기존 것 
    def __init__(self, config, data_csv, img_dict, data_type):
        """
        TODO (all done, except the implementing config for BT part):
        0. data_path가, 이미지들이 들어있는 파일이 아닌, 이미지들의 path들이 있는 DICT 받도록 하자! (for flexibility)
            * dict : {"img 이름 in csv" : "실제 img path" }
        1. remove split_prop (100% no split) (split은 main에서 하도록) (어차피 CL은 split을 안할 것이고 (valid용으로는 할수도...? 근데 일단은 스킵), 만약 한다고 하더라도, main함수 단에서 test/train 을 구분해서 넣도록 하기  
        2. split = 'train'을, mode = "PRETRAINING" "FINETUNING" 이든지 이런식으로 바꾸게 하기 
        3. data csv를 받도록 하기 
            * this data csv has to inclue only the images 
            * glob을 받도록 하기?
                * 아니다, 그냥 datatype을 받아서, ADNI 쓰라고 하면 nii.gz쓰던지 그런식으로 할까? 
            * data csv should be a subset of the total image (data에는 있는데 img는 없는 것은 안됨! 다신 그 역은 해도 괜찮도록 main 함수를 짤것이다)
        4. UKB, 등등 => flip을 잘해서 해야함! (이것도 implement해야함)
        5. MNI 여부를 config으로 받아야함!!! (MNI 여부가 중요함)
        6. config.tf을 받아서 transformation을 적용하도록 하기!
        7. BT에서 self.config.mode==0: #config.mode : PRETRAINING일떄 을 하기!
        8. 실제로 이미지 프린트 해서 보기!!!
        """
        self.shape = config.input_size #make it into channel
        self.split = data_type #split여부 tracking하기 
        self.config = config #for reference, in case it's needed
        self.data_type = data_type #train/val/test중 어느건지

        assert len(self.shape) == 4, "shape must be given WITH the channel too! (C, H, W, D)"
        
        ##getting list of img paths ==> 이미 image_dict해서 가져온다!        
        labels = data_csv[config.label_name].values.tolist() #change to list
        ##labels : must be corresponding to the dict order (iamge file order) #img_dict가 있으니 이건 쉬울듯? (응! 순서가 똑같다!! 원래 코드처러 ㅁ할 필요없다!

        self.dataset = ImageDataset(image_files=list(img_dict.values()), labels = labels, transform = basic_transform(resize_method = config.resize_method,shape=self.shape))
        

        #defining transformations to use 
        if config.tf == "all_tf" :
            transform = transform_yAware_all
        elif config.tf == "cutout" :
            transform = transform_yAware_cutout
        elif config.tf == "crop" : 
            transform = transform_yAware_crop
        elif config.tf == None:
            transform = None #will be not used 
        
        if self.split == 'train' or self.config.mode==0: #config.mode : PRETRAINING일떄
            self.transform = transform(shape = self.shape) #the transform to use 
            self.transform_prime = transform(shape = self.shape) 
        #import pdb ; pdb.set_trace()
        
        #raise NotImplementedError("어떤 data 냐에 따라 (flip을 어떻게 하는지 등등을 적어야한다!) ")
        
    #def collate_fn():#see later (dataloader할때 batch가 이상하면 이쪽을 고쳐야할수도)
    
    def __getitem__(self,idx):
        sub_img, sub_label = self.dataset[idx] #해당 idx subject의 img뽑기
        
        if self.config.mode == 0 : #i.e. pretraining
            y1 = from_numpy(self.transform(sub_img).copy()).float() #change from numpy to tensor, minding the strider errors and dtype errors
                    #copy done to avoid Torch.from_numpy not support negative strides errors
                    #.float to avoid dtype mismatch https://stackoverflow.com/questions/44717100/pytorch-convert-floattensor-into-doubletensor
            y2 = from_numpy(self.transform_prime(sub_img).copy()).float()
            return (y1, y2), sub_label
        
        else : #finetuning
            if self.config.tf and self.data_type == "train" : #i.e. config is not none and data type is train
                return from_numpy(self.transform(sub_img.copy())).float(), sub_label    
            else: #i.e. the rest (valid/eval이거나 tf가 none일때는 무조건 no transform)
                return from_numpy(sub_img.copy()).float(), sub_label    
            
    def __len__(self):
        return len(self.dataset)