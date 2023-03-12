    
from multiprocessing import Pool
import multiprocessing 
from IMP_pacakges_step0 import *
import os 
#p = Pool(160)

def untar_parallel(args):
    a = tar_ABCD(args, do = "T1_DTI")
    a.create_list()
    a.extract_move()
    a.untar_BIDS()
    
    b = tar_ABCD(args, do = "T2")
    b.create_list()
    b.extract_move()
    b.untar_BIDS()
    

##make the list of globs to run

add_list = ['0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

glob_list = ['Z'+i for i in add_list]# THIS PART MUST BE CHANGED



if __name__ == "__main__":
#try : 
    T1_DWI_dir = "/scratch/08834/tg881334/ABCD_preproc/1.data/T1_DWI/NDARINVY-Z.tar"
    T2_dir = "/scratch/08834/tg881334/ABCD_preproc/1.data/T2/NDARINVU-Z.tar"
    save_dir = "/scratch/08834/tg881334/ABCD_preproc/2.BIDS/"
               
    arg_list = []

    for i in glob_list:
        sub_save_dir = os.path.join(save_dir, str(i))
        os.mkdir(sub_save_dir)
        args = Args(T1_DWI_dir = T1_DWI_dir, T2_dir = T2_dir, save_dir = sub_save_dir,
                glob = "NDARINV"+i)
        arg_list.append(args)
        print(args.glob)
    

    #not do multiprocessing becuase lots of erros to wkqrl 
    p = Pool(96)
    for s in p.imap(untar_parallel, arg_list):
        print("-=====")
#except : 
#    import pdb; pdb.set_trace()        

#########DO THE SAVE DIR TO SEPERATE PATHS (MAKING SUBFOLDERS AND STUFF) 
## THEN AFTER ALL THE THIBNGS ARE DONE, ADD EM UP 


#####MULTIPROCESSING BACUP
#if __name__ == "__main__":
#
#    T1_DWI_dir = "/scratch/08834/tg881334/ABCD_preproc/1.data/T1_DWI/NDARINV0.tar"
#    T2_dir = "/scratch/08834/tg881334/ABCD_preproc/1.data/T2/NDARINV0-5.tar"
#    save_dir = "/scratch/08834/tg881334/ABCD_preproc/2.BIDS/"
#    args = Args(T1_DWI_dir = T1_DWI_dir, T2_dir = T2_dir, save_dir = save_dir,
#               glob = "NDARINV0AB")
#               
#    arg_list = []
#
#    for i in glob_list:
#        args = Args(T1_DWI_dir = T1_DWI_dir, T2_dir = T2_dir, save_dir = save_dir,
#               glob = "NDARINV"+i)
#        arg_list.append(args)
#        print(args.glob)
#    
#
#    #not do multiprocessing becuase lots of erros to wkqrl 
#    p = Pool(96)
#    for s in p.imap(untar_parallel, arg_list):
#        print("-=====")
##########DO THE SAVE DIR TO SEPERATE PATHS (MAKING SUBFOLDERS AND STUFF) 
### THEN AFTER ALL THE THIBNGS ARE DONE, ADD EM UP 






##IF TRYING SINGLE THING (DON'T USE!)    
#T1_DWI_dir = "/scratch/08834/tg881334/ABCD_preproc/1.data/T1_DWI/NDARINV0.tar"
#T2_dir = "/scratch/08834/tg881334/ABCD_preproc/1.data/T2/NDARINV0-5.tar"
#save_dir = "/scratch/08834/tg881334/ABCD_preproc/2.BIDS/"
#args = Args(T1_DWI_dir = T1_DWI_dir, T2_dir = T2_dir, save_dir = save_dir,
#           glob = "NDARINV0")    
#untar_parallel(args)

