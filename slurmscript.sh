#!/bin/bash

job_directory=$PWD/jobs
mkdir -p $job_directory

for i in {-20..20..1}
do
	for j in {-20..20..1}
  do
		job_file="${job_directory}/run-$i-$j.job"

    echo "#!/bin/bash
          #SBATCH --job-name=run-$i-$j
          #SBATCH --time=00:05:00
          #SBATCH --mem-per-cpu=100
          python mysimulation.py $i $j" > $job_file
    sbatch $job_file
  done
done
