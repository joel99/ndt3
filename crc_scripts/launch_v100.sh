#!/bin/bash
#SBATCH --job-name=ndt3
#SBATCH --cluster gpu
#SBATCH -p v100
#SBATCH --gres gpu:1
#SBATCH -c 6
#SBATCH -t 24:00:00
#SBATCH --mem 48G
#SBATCH --output=slurm_logs/%j.out

echo $@
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
source ~/load_env.sh
python -u run.py $@

