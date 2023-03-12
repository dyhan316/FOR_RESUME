import os
import subprocess
import argparse
from pathlib import Path
import pdb
import warnings

#######ASSUMPTIONS######
curr_dir = os.getcwd()
class Args():
    def __init__(self, BIDS_folder,alloc = "skx-normal"):
        self.alloc = alloc #also possible skx-normal#what to be allocated with 
        self.BIDS_folder = BIDS_folder#'age_4.5_to_5'
        self.base_dir = Path(curr_dir)
        self.scratch_dir = Path('/scratch/08834/tg881334/CHA_preproc') #where the BIDS and outputs are and will be saved
        self.BIDS_dir = Path(self.scratch_dir/'2.BIDS_data/{}'.format(self.BIDS_folder)) #MUST BE MODIFIED TO THE INPUT I WANT TO RUN WITH
        self.save_dir = Path(self.scratch_dir/'3.qsipreproc_results_REAL/{}'.format(self.BIDS_folder))
        self.supp_dir = Path('/work2/08834/tg881334/stampede2/CHA_preproc/CHA_preproc_supplementary_files') #directory where the singularity images and etc are held
        #self.shell_dir = Path(self.base_dir / 'step4_shell_outputs/{}/'.format(self.BIDS_folder)) #where the shelll scripts for each subject will be saved
        #self.log_dir = Path(self.base_dir/'step4_shell_outputs/{}/'.format(self.BIDS_folder))
        self.sub_list = sorted(os.listdir(self.BIDS_dir))
        #age 에 관한 것이 있어야 할듯! (특히, freesurfer infant 할떄)
        ages = self.BIDS_folder.split('_')
        print(ages)
        if float(ages[3]) <= 4.5:
            self.infant = True
        elif float(ages[1]) >= 4.5:
            self.infant = False
        else : 
            raise ValueError("something is werid!!! ")
        print(f"infant ? : {self.infant}")

class qsiprep_shell():
    """
    shell script for performing preproc or reconstruction for infants AND adults given their args    
    """
    
    def __init__(self, args, operation ):
        """
        args : 앞에서 정의했어야 했는 것
        perform : what mode to perform, out of ['preproc', 'recon'] 
            * adult preproc : use 0.14.3, full T1 and dwi
            * infant preproc : use 0.14.3, dwi only 
            * adult_recon, infant_recon : not done yet (use 0.16.1)
        """
        if operation == "preproc":
            if args.infant == True:
                self.perform = "infant_preproc"
            else:
                self.perform = "adult_preproc"
        elif operation == "recon":
            raise NotImplementedError("reconstruction not implemented yet")
        else :
            raise ValueError("perform is not either 'preproc' or 'recon'")
        
        #공통된 것들 (preproc이든 recon이든 동일한 것 적용)
        time_lim = {"skx-normal": 48, "skx-dev" : 2}
        
        self.shell_dir = f"{str(args.base_dir)}/step4_shell_output/{self.perform}/{args.BIDS_folder}"#where shell scripts and output/errors will be saved
        os.makedirs(self.shell_dir, exist_ok=True)        
        self.bash_shell = ["#! /bin/bash",
                     f"#SBATCH -J {self.perform}_{args.BIDS_folder}", f"#SBATCH -p {args.alloc}",
                     f"#SBATCH -o {self.shell_dir}/output.%j",
                     f"#SBATCH -e {self.shell_dir}/error.%j",
                     f"#SBATCH -t {time_lim[args.alloc]}:00:00", "#SBATCH -n 1", "#SBATCH -N 1",
                     "#SBATCH -c 96", "module load tacc-singularity"]
        self.args = args
    
    def make_sh_files(self):
        """
        creates sh files for execution
        """
        args = self.args #불러오기
        ##perform mode에 따라서 다르게 학시 
        perform = self.perform #bad coding style, but whatever..ㅋㅋㅋ
        if perform in ['adult_preproc', 'infant_preproc'] : 
            print(f"will perform preprocessing for folder {args.BIDS_folder}")
            #updating bash_shell with the singularity command
            self.sing_img_dir = args.supp_dir/"qsiprep-0.14.3.sif"#singularity image directory (0.14.3.sif for this one )
            os.makedirs(args.save_dir, exist_ok = False)
            self.singularity_command = f"singularity run --cleanenv -B {args.BIDS_dir}/:/data:ro,{args.save_dir}/:/out,{str(args.supp_dir)}:/freesurfer {str(self.sing_img_dir)} /data /out participant -w /out/tmp --output_resolution 1.2 --denoise_after_combining --unringing_method mrdegibbs --b0_to_t1w_transform Affine --intramodal_template_transform SyN --do_reconall --fs-license-file /freesurfer/license.txt --skip_bids_validation --omp-nthreads 4"
            
            #밑에 infant할때 --infant이런것들 넣었는데, --dwi-only도 넣어야할듯?
            #(조금 복잡해짐... 일단은... 두고보자 (그리고 args.infant 를 받도록 했는데, 이것은 좋은 것이 아닐듯..
            #왜냐하면 args.infant = True인데 adult_preproc하라고 내가 말하면 애러가 뜰 것이니 ))
            #below : add dwi-only and --infant if infant 
            if args.infant == True : self.singularity_command = self.singularity_command + ' --infant' + ' --dwi-only' #add infant option of doing infant 
                
            #밑에 warning : 지우기 (나중에 freesurfer도 하고 나면)
            if args.infant == True:
                warnings.warn('infant is half implemented! must do the fressurefr infant, THAN add them up!')      
            
            self.bash_shell.append(self.singularity_command)
            
            #saving the shell files
            
            
            self.shell_shell = f"{self.shell_dir}/run_{perform}_{args.BIDS_folder}.sh"
            with open(self.shell_shell, "x") as f: #"x" so that creates only if it didn't exist
                for line in self.bash_shell:
                    f.write(f"{line}\n")
            
        elif perform in ['adult_recon','infant_recon'] :
            raise NotImplementedError("reconstruction not implemented yet")
            self.sing_img_dir = args.supp_dir/"qsiprep-0.16.1.sif"
        else:
            raise ValueError(f"incorrect perform option : {perform}")
            
        #assert self.infant == False , "NOT AN INFANT! (or infant range is not set correctly in args)"
    
    def run(self):
        """
        runs the created sh files using sbatch 
        """
        args = self.args
        subprocess.run(f"chmod 755 {self.shell_shell}", shell = True)
        
        subprocess.run(f"sbatch {self.shell_shell}", shell = True)
        
        
