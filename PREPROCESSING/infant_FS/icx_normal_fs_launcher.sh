#!/bin/bash

#SBATCH -J run_qsiprep_preproc#name of the job #SBATCH -N 2  #24 sub per node, so for 128 subjects, 5.01 neede,d so doing 6  #how many nodes you need #SBATCH -n $1  #how many jobs 
#SBATCH -p icx-normal #skx-dev for faster allocation#flat-quadrant #the queue on stampede 2 to use
#SBATCH -o ./step1_shell_outputs/skx_normal_QSIPREP_preproc.o%j  #change according to you job name
#SBATCH -e ./step1_shell_outputs/skx_normal_QSIPREP_preproc.e%j #change according to you job name
#SBATCH -t 20:00:00     #number of hours for the job to run. 48hr is the maximum for normal queue.

module load launcher #맞는지 모르겠다

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=./step1_shell_outputs/qsipreproc_command_list.txt

$LAUNCHER_DIR/paramrun

#EOT #looked at https://stackoverflow.com/questions/27708656/pass-command-line-arguments-via-sbatch
