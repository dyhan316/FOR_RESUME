import subprocess
import os
file_dir = '/work2/08834/tg881334/stampede2/CHA_preproc/2010-2021_acutal_preproc_files/2.dcm2niix/'
things_to_run = [i for i in os.listdir(file_dir) if i[:3] == "age"] #extract list of BIDS folders (ageXX)

print("2.one_BIDS has to have been made before running")

for i,age_group in enumerate(things_to_run):
    print(i)
    subprocess.run(f"./REAL_activate_step3.sh {age_group}", shell = True)
    #if i==1:
        #break
