#!/bin/bash
#SBATCH --job-name=ndt2_2x2
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH -p gpu
#SBATCH -t 36:00:00
#SBATCH --mem 40G
#SBATCH -x mind-1-30
#SBATCH --output=slurm_logs/%j.out

# CPUs requested conservative, trying to slip these jobs in, unclear if cpu is limiting factor
# Mem is
# Multinode notes
# From https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html
# # debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface

# Currently (7/24/22) IB isn't working well on mind due to memlock limits
export NCCL_IB_DISABLE=1
# # export NCCL_SOCKET_IFNAME=^docker0,lo

# # might need the latest CUDA
# # module load NCCL/2.4.7-1-cuda.10.0

# srun python3 artifacts/artifact_estimator.py --num-nodes 1 -c $@ # Multinode
echo 'tasks'
echo $SLURM_NTASKS
echo 'per node'
export SLURM_NTASKS_PER_NODE=2
echo $SLURM_NTASKS_PER_NODE

hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
source ~/load_env.sh
srun python -u run.py nodes=2 $1

