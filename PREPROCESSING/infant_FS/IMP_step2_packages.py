import json
import os 
from glob import glob
from multiprocessing import Pool
import tarfile

#looked at : https://stackoverflow.com/questions/23111625/how-to-add-a-key-value-to-json-data-retrieved-from-a-file


class whole_class():
    def __init__(self,tar_dir,untar_dir,BIDS_dir):
        self.tar_dir = tar_dir
        self.untar_dir = untar_dir
        self.BIDS_dir = BIDS_dir

    ######base function for untarring one thing ####
    def extract_tar(self,file_dir):
        file = tarfile.open(file_dir)
        file.extractall(self.untar_dir)

    def parallel_untar(self):
        p = Pool(80) #let's set to 80 haha
        tar_list = [os.path.join(self.tar_dir, file_dir) for file_dir in os.listdir(self.tar_dir)]
        for s in p.imap(self.extract_tar, tar_list):
            print("========")

    #######modifying json so that it works#######
    def modify_json(self,json_dir):
        with open(json_dir, 'r') as f:
            data = json.loads(f.read())
        data["PhaseEncodingDirection"] = 'i'
        
        with open(json_dir, 'w') as json_file:
            json.dump(data, json_file)
    
    
    def modify_whole_bids(self):
        """
        iteratively applies modify_json to all the dwi json files inside of a directory
        """
        BIDS_dir = self.BIDS_dir
        json_dir_list = glob(BIDS_dir + '/sub*/*/dwi/*.json')
        for json_dir in json_dir_list:
            self.modify_json(json_dir)
      
        