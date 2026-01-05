#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=1
#SBATCH --error=/info/etu/m1/s2501728/AutoML/automl_logs/%j_0_log.err
#SBATCH --job-name=submitit
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/info/etu/m1/s2501728/AutoML/automl_logs/%j_0_log.out
#SBATCH --partition=common
#SBATCH --signal=USR2@90
#SBATCH --time=60
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /info/etu/m1/s2501728/AutoML/automl_logs/%j_%t_log.out --error /info/etu/m1/s2501728/AutoML/automl_logs/%j_%t_log.err /info/etu/m1/s2501728/miniconda3/envs/autoML_env/bin/python -u -m submitit.core._submit /info/etu/m1/s2501728/AutoML/automl_logs
