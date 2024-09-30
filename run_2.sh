#!/bin/bash
#SBATCH --job-name=bbdm2       # Job name
#SBATCH --output=log_slurm/result_cuongdm_2.txt      # Output file
#SBATCH --error=log_slurm/error_cuongdm_2.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=40                             # Number of CPU cores per task
sh ./test-bash-2.sh


