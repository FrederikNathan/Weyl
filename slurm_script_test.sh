#!/bin/sh

#SBATCH --job-name=slurm_test    # Job name
#SBATCH --output=test.log     # Standard output and error log
#SBATCH --mem=500mb   
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=2            # Number of CPU cores per task

#export OMP_NUM_THREADS=2
#export MKL_NUM_THREADS=2


echo "$SLURM_JOB_NAME"

python3 -c "print('hej')"


