#!/bin/bash
#SBATCH --job-name=AutoML
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --time=1-00:00:00
#SBATCH --output=log_%j.txt

# commande srun equivalente :
# srun --job-name=P --ntasks=1 --exclude=gpue[01-05,08-12] --gres=gpu:1 --cpus-per-task=20 --mem=50G --pty bash

# Pour eviter les problemes de conda dans les scripts batch
#shellcheck source=/dev/null
source ~/miniconda3/etc/profile.d/conda.sh

# Active l'environnement conda
conda activate autoML_env

# Va dans le dossier du projet ou échoue
cd /info/etu/m1/s2501728/AutoML || exit

python -u scores.py