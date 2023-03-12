import os
import subprocess
import argparse
from pathlib import Path
import pdb
#parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet')
#parser.add_argument('data', type=Path, metavar='DIR',
#                    help='path to dataset')
#args = parser.parse_args()

######IMPORTANT TO SELECT THIS########
ntasks_per_node = 24#48 #node당 몇명을 할지 (have to be optimized so memory error doeens't occur)
node = "skx"#"skx" #or "icx"

age_list = ['age_9_to_10','age_13_to_16','age_18_to_20','age_16_to_18'] #,'age_18_to_20']
           #['age_4.5_to_5','age_5_to_5.5']
           #below : not done yet
           #,'age_5.5_to_6','age_6_to_7','age_7_to_8','age_8_to_9','age_9_to_10']
           #['age_0_to_1',  'age_1.5_to_2','age_1_to_1.5','age_3.5_to_4','age_4_to_4.5','age_10_to_13',
           #'age_16_to_18','age_2.5_to_3','age_3_to_3.5','age_13_to_16','age_18_to_20','age_2_to_2.5']
######################################
if node	== "skx":
    total_cpu =	96 #count of total cpus
elif node == "icx":
    total_cpu =	160
elif node == "flat-quadrant":
    total_cpu = 240
qsiprep_nthreads_per_sub = int(total_cpu/ntasks_per_node)

#######ASSUMPTIONS######
curr_dir = os.getcwd()
class Args():
    def __init__(self, age_range):
        self.age_range = age_range#'age_4.5_to_5'
        self.base_dir = Path(curr_dir)
        self.scratch_dir = Path('/scratch/08834/tg881334/CHA_preproc') #where the BIDS and outputs are and will be saved
        self.BIDS_dir = Path(self.scratch_dir/'2.one_BIDS/{}'.format(self.age_range)) #MUST BE MODIFIED TO THE INPUT I WANT TO RUN WITH
        self.save_dir = Path(self.scratch_dir/'3.qsipreproc_results/{}'.format(self.age_range))
        self.supp_dir = Path('/work2/08834/tg881334/stampede2/CHA_preproc/CHA_preproc_supplementary_files') #directory where the singularity images and etc are held
        self.shell_dir = Path(self.base_dir / 'step4_shell_outputs/run_this_test/{}'.format(self.age_range)) #where the shelll scripts for each subject will be saved
        self.log_dir = Path(self.base_dir/'step4_shell_outputs/log_outputs')
        self.sub_list = sorted(os.listdir(self.BIDS_dir))
        #age 에 관한 것이 있어야 할듯! (특히, freesurfer infant 할떄)
        ages = self.age_range.split('_')
        print(ages)
        if float(ages[3]) <= 4.5:
            self.infant = True
        elif float(ages[1]) >= 4.5:
            self.infant = False
        else : 
            raise ValueError("something is werid!!! ")
        print(f"infant ? : {self.infant}")
#print("change __init__ to make sure that 'infant' is only turned on when we ARE dealing with infants")
#os.chdir(args.base_dir)

#age_list = ['age_0_to_1', 'age_6_to_7']

num_subs = 0
for age_range in age_list:
    args = Args(age_range=age_range)
    print(args.sub_list)
    
    ####shell file folders
    
    ##위의 것들을 필요한 부분들만 들어서 txt file로 만들자
    #단, mkdir, file_path등은 python내에서 자체적으로 하기 
    
    sing_img_dir = args.supp_dir/"qsiprep-0.14.3.sif"#singularity image directory (0.14.3.sif for this one )
    
    os.makedirs(args.shell_dir, exist_ok = True)
    os.makedirs(args.save_dir, exist_ok = True)
    for sub in args.sub_list:
        print(f"creating qsiprep .sh for subject : {sub}")
        
        sub_BIDS_dir = str(args.BIDS_dir / sub)
        sub_save_dir = str(args.save_dir / sub)
        bash_shell = ["#! /bin/bash", "module load tacc-singularity"]
        os.mkdir(args.save_dir/sub)
        
        singularity_command = f"singularity run --cleanenv -B {sub_BIDS_dir}/:/data:ro,{sub_save_dir}/:/out,{str(args.supp_dir)}:/freesurfer {sing_img_dir} /data /out participant -w /out/tmp --output_resolution 1.2 --denoise_after_combining --unringing_method mrdegibbs --b0_to_t1w_transform Affine --intramodal_template_transform SyN --do_reconall --fs-license-file /freesurfer/license.txt --skip_bids_validation" #removeed nthreads because we want to use all available
        if args.infant == True : singularity_command = singularity_command + ' --infant' #add infant option of doing infant 
        ###tr this
        singularity_command = singularity_command + f' --nthreads {qsiprep_nthreads_per_sub}'
        
        bash_shell.append(singularity_command)
        #print(f"bash_shell {bash_shell}")
        
        ##saving the shell files
        with open(f"{args.shell_dir}/run_shell_{sub}.sh", "a") as f:
            for line in bash_shell:
                f.write(f"{line}\n")
    
    subprocess.run(f"chmod 755 {args.shell_dir}/*", shell = True)
    
    ##############IMPLEMENT : ACTUALLLY SAVING THE THING############3
    #print("hihi")
    os.makedirs(args.log_dir, exist_ok = True)
    for sub_shell in os.listdir(args.shell_dir):
        shell_to_run = os.path.join(args.shell_dir, sub_shell)
        sub_id = sub_shell.split('-')[-1][:-3]
        command_to_add = shell_to_run + f" >> {args.log_dir}/output-${{LAUNCHER_TSK_ID}}_sub-{sub_id}"
        #print(command_to_add)
        with open(f"{args.base_dir}/step4_shell_outputs/qsipreproc_command_list.txt", "a") as f:
            f.write(f"{command_to_add}\n")
    
    num_subs+=len(args.sub_list) #addition over all age groups 
import math

#num_subs = num_subs + 1 #had to do +1 to get every subject to be done (one process seems to be reserfved for running the thing as a whole)
num_nodes = int(math.ceil(float(num_subs)/ntasks_per_node)) #number of nodes to take, with ntasks per node satisfiedceiling으로 필요한 만큼 node가져가기
subprocess.run(f"sbatch -n {num_subs} -N {num_nodes} {args.base_dir}/REAL_activate_step4.sh", shell = True)
#added -n {num_subs} so that it automatically chooses the right number of jobs (slurm number of jobs)
pdb.set_trace()
    #pdb.set_trace()




