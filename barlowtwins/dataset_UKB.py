import numpy as np
import monai
from monai.data import ImageDataset
from monai.transforms import *
from pathlib import Path
import os
import pandas as pd
import numpy as np

##define dataset (through kinda wrapping over the ImageDataset)
def UKB_T1(data_path, split, transform, **kwargs):
    """주의 : train/test split rate : 0.75로 preset (희환쌤 코드 처럼 분리할 수도 있을 것 같으나, 이것은 나중에 하자 )"""
    

    #==============BELOW : CHANGED because npy경우같은 것 못하게되서 밑에처러하면======#
    #list imgs list inside the directory, with suffixes ending in .gz (list of nii.gz files) ==> sorted 
    #imgs = sorted([f for f in Path(data_path).iterdir() if f.suffix == '.gz']) # f.suffixes로 하면 nii.gz까지 다 있으나, .crop도 suffix로 인정되어 복잡해짐..
    imgs = sorted([f for f in Path(data_path).iterdir() if f.suffix != ".txt"]) #txt가 들어있는 경우가 있어서 제외

    #=======REMOVED THIS=========#
    #TRAIN_TEST SPLIT (take first 75% or last 25% of iamge list depending on split)
    #imgs = imgs[:int(len(imgs)*0.75)] if split == "train" else imgs[int(len(imgs)*0.75): ] #i.e. 앞의 75% if train/뒤의 25% if test
    #============================#
    
    
    """
    
    LABEL : 
    barlow twins에서는, https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 
    위에서 나오는 crossentropyloss를 쓴다. 우리같은 경우에는, each sample can only belong to one of the classes인 경우 (그중에서도 num_classes = 1인경우(if sex classification)이다.
    
    따라서, 밑에서와 같이, 
    
    
    sex가 아닌 general한 classification 할만한 것을 찾도록 하기!
    
    LABEL은 다음과 같아야 한다! 만약 5개의 class가 있다면, lbls = [2,3,0,3,4] 이런식으로, 해당 ith sample이 속하는 class의 index값이 lbls에 있어야 한다! (위의 사이트 가서 example몇번 만지작 거리면 알게 됨) => 그리고 이 값들이 integer이어야 한다!!!!!
    
    
    
    
    """
    #밑 : commented out because ABCD 로 하면 달라서... => 이 파트 저 generalize해야할듯#
    #file_list = os.listdir(data_path) 
    #phenotype = pd.read_csv("./UKB_phenotype.csv") ###이거는 위치에 따라 달라질 수 있음 (같은 폴더내에있다고 가정)
    #phenotype_real = phenotype[["eid","sex"]]
    #phenotype_real = phenotype_real.astype({"eid":'str'})
    #lbls_list = []
    #for i in file_list:
    #    number = i[:7]
    #    #index = phenotype_real.index[phenotype_real["eid"]==number]
    #    index2 = np.where(phenotype_real["eid"]==number)
    #    select_indices = list(index2)[0]
    #    select_df = phenotype_real.iloc[select_indices]
    #    a = list(select_df["sex"])[0]
    #    a = int(a)
    #    lbls_list.append(a)
    #lbls = np.array(lbls_list)
    
    
    ##밑 : 임시 처방전. 일단 이것 작동하는 지보고, 정윤쌤의 sex class가져다가 쓰기!!!##
    lbls = np.zeros(len(imgs))
    lbls = lbls.astype(int)

    return ImageDataset(image_files=imgs, labels = lbls, transform = transform)


class Transform_yAware:
    def __init__(self, MNI = True, **kwargs):
        ##MNI : if true, does not apply ResizeWithPadOrCrop, but if false, applies the spatial size that was given by **kwargs
        
        #transform normalize , flip, blur, noise, (cutout, crop (only if True))
        transform_list = [
            ScaleIntensity(), AddChannel(),
            RandFlip(prob = 0.5),
            RandGaussianSharpen(sigma1_x=(0.1, 1.0), sigma1_y=(0.1, 1.0), sigma1_z=(0.1, 1.0), sigma2_x=0.1, sigma2_y=0.1, sigma2_z=0.1,prob=0.5)]
        
        #add padding/cropping if not MNI (i.e. shape varies)
        transform_list.append(ResizeWithPadOrCrop(spatial_size =  kwargs['shape'], method = "symmetric", mode = "constant")) if MNI == False else None
                              
        transform_list = transform_list + [NormalizeIntensity(),ToTensor()] #append two things at once
        
        self.transform_list = transform_list
        
        self.transform = monai.transforms.Compose(transform_list)
        self.transform_prime = monai.transforms.Compose(transform_list)
        

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1,y2    

"""
원래 transform
        [
            #normalize , flip, blur, noise, cutout, crop
            ScaleIntensity(), AddChannel(),
            RandFlip(prob = 0.5),
            RandGaussianSharpen(sigma1_x=(0.1, 1.0), sigma1_y=(0.1, 1.0), sigma1_z=(0.1, 1.0), sigma2_x=0.1, sigma2_y=0.1, sigma2_z=0.1,prob=0.5),
            ResizeWithPadOrCrop(spatial_size =  (182,218,182), method = "symmetric", mode = "constant"),
            NormalizeIntensity()
            ,ToTensor()
        ]
            

"""
#=======ADDED NO TRANSFORM (only adding channels and stuff, but no augmentaiotn(used for evaulation)====#

class No_Augmentation:
    def __init__(self):
        
        self.transform = monai.transforms.Compose([
            #normalize , flip, blur, noise, cutout, crop
            ScaleIntensity(), AddChannel(),
            NormalizeIntensity() #이파트 지워야할지 모르겠다
            ,ToTensor()
        ])
        
   
        ##어디에 뭐가 들어있는지 확인하기 위해서, (182,20,182)로 함
        #self.transform_prime = monai.transforms.Compose([
        #    #normalize , flip, blur, noise, cutout, crop
        #    ScaleIntensity(), AddChannel(),
        #    NormalizeIntensity()
        #    ,ToTensor()
        #])
        
         
            
    def __call__(self, x):
        y1 = self.transform(x)
        #y2 = self.transform_prime(x) #shouldn't do this, as we only need to send one output
        return y1 #,y2  
