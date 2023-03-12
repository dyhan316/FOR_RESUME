import os 

from REAL_step4_modules import * 
for BIDS_folder in os.listdir("/scratch/08834/tg881334/CHA_preproc/2.BIDS_data/"):
    args = Args(BIDS_folder= BIDS_folder, alloc = 'skx-normal') #make args for each BIDS_folder
    #skx-normal
    running_it = qsiprep_shell(args, operation = "preproc")
    
    running_it.make_sh_files()
    running_it.run() #have to be run inside of the login node
