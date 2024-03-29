#!/bin/bash
# Run 2 consecutive jobs of cycleGAN

iterations=5 # 총 몇 번이나 연속으로 돌릴 것인지
jobid=$(sbatch --parsable /global/cfs/cdirs/m3898/yAwareContrastiveLearning/run_main_simclr.slurm)

for((i=0; i<$iterations; i++)); do            
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency /global/cfs/cdirs/m3898/yAwareContrastiveLearning/run_main_simclr.slurm)
    dependency=",${dependency}afterany:${jobid}"
done