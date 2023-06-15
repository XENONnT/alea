#!/bin/sh

#SBATCH --account=pi-lgrandi
#SBATCH --ntasks=1
#SBATCH --partition=xenon1t
#SBATCH --qos=xenon1t


toymc_script=$(python -c 'import pkg_resources; print(pkg_resources.resource_filename("binference","/scripts/run_toymc.py"))')
execution_cmd="python $toymc_script $@"
echo $execution_cmd
echo "-------------------"

echo "loading singularity"
module load singularity

echo "executing command"
singularity exec --bind /project2 --bind /dali --bind /home /project2/lgrandi/xenonnt/singularity-images/xenonnt-development.simg $execution_cmd
