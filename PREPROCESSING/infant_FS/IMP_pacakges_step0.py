
import tarfile
import os 
import shutil
import glob 
import subprocess

class Args():
    def __init__(self,T1_DWI_dir, T2_dir, save_dir, glob):
        self.T1_DWI_dir = T1_DWI_dir
        self.T2_dir = T2_dir
        self.save_dir = save_dir
        self.glob = glob

class tar_ABCD():
    def __init__(self, args, do="T1_DTI"):
        """
        do  : "T1_DTI" or "T2"
        """
        if do == "T1_DTI":
            self.tar_dir = args.T1_DWI_dir
        elif do == "T2":
            self.tar_dir = args.T2_dir
        else : 
            raise ValueError("not one of the modalities that can be done")
            
            
        self.file_to_untar = []
        self.last_member_names = []
        self.args = args
   

    def create_list(self):
        args = self.args
        tar = tarfile.open(self.tar_dir)
        for i,member in enumerate(tar):
            file_name = member.name.split('/')[-1]
            if not member.isfile(): #skip if not file
                print("NOT", member)
                continue 
            elif member.name.split('_')[-3] == "baselineYear1Arm1": #skip if baseline
                continue 
            elif args.glob not in file_name: #skip if doesn't match glob 
                continue 
            
            self.file_to_untar.append(member)
            
            #if i > 1000:
            #    break
        
        self.last_member_names.append(member)
        
        #self.file_to_untar = file_to_untar
        #self.last_member_names = last_member_names
        self.tar = tar
        print(self.tar_dir)                    

    def extract_move(self):
        args = self.args
        print(self.file_to_untar)
        self.tar.extractall(members = self.file_to_untar, path = args.save_dir) #extract files
        for i,member in enumerate(self.last_member_names):
            #raise ValueError("muyst be changed")
            upper_dir = os.path.split(member.name)[0] #get uppder dir name
            [shutil.move(file, args.save_dir) for file in glob.glob(os.path.join(args.save_dir, f"{upper_dir}/*"))] #move them to the actual directory I wanted
                
    def untar_BIDS(self):
        args = self.args
        os.chdir(args.save_dir)

        #unzip the .tgz files
        for file in os.listdir(args.save_dir):
            print(file)
            if "tgz" in file: #i.e. if tgz
                subprocess.run(f'tar zxvf {file}', shell = True)
                subprocess.run(f'rm {file}', shell = True)
            elif not "sub-" in file and "dataset_description.json" != file: #i.e. 이미 unzip된 sub-파일이 아니라면, 또는 dataset_description이 아니라면 
                shutil.rmtree(file) #remove the nuisance stuff

  