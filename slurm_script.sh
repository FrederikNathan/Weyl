#!/bin/sh

#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --output=test.log     # Standard output and error log
#SBATCH --mem=500mb   
#SBATCH --nodes=1
#SBATCH --ntasks=1                   # Run a single task		
#SBATCH --cpus-per-task=2            # Number of CPU cores per task
cd $PBS_O_WORKDIR
#cd Weyl/Code

#export OMP_NUM_THREADS=2
#export MKL_NUM_THREADS=2


#echo $JOBNAME
python3 pbs_pyscript.py $LF "$PRESTR" $Q $SL


