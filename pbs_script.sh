#!/bin/sh
# Bash script to exectue by queuing software. Is just set to run pbs_pyscript.py with some resource specifications, and with some argstring arguments detailing the run.


#PBS -e /home/frederik_nathan/Out/${PBS_JOBID%%.*}.$PBS_JOBNAME.err
#PBS -o /home/frederik_nathan/Out/${PBS_JOBID%%.*}.$PBS_JOBNAME.out
#PBS -l nodes=1:ppn=2
#PBS -l mem=700mb


cd $PBS_O_WORKDIR
#cd Weyl/Code

#export OMP_NUM_THREADS=2
#export MKL_NUM_THREADS=2


#echo $JOBNAME
python3 pbs_pyscript.py $LF "$PRESTR" $Q $SL


