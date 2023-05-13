#!/bin/bash
#SBATCH --job-name=ndt2_2x
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH -p gpu
#SBATCH -t 36:00:00
#SBATCH --mem 60G
#SBATCH -x mind-1-5
#SBATCH --output=slurm_logs/%j.out

echo 'tasks'
echo $SLURM_NTASKS
echo 'per node'
export SLURM_NTASKS_PER_NODE=2
echo $SLURM_NTASKS_PER_NODE
# set slurm tasks to 2
hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
source ~/load_env.sh
srun python -u run.py $1

