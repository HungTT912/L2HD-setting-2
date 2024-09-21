#!/bin/bash
#SBATCH --job-name=cuongdm        # Job name
#SBATCH --output=log_slurm/result_cuongdm_1.txt      # Output file
#SBATCH --error=log_slurm/error_cuongdm_1.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --gpus=1                 # Number of GPUs per node
#SBATCH --cpus-per-task=10                              # Number of CPU cores per task
sh ./test-bash-2.sh
