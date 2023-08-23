#!/bin/bash
#SBATCH --job-name=ndt2_4x
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH -p gpu
#SBATCH -t 36:00:00
#SBATCH --mem 90G
#SBATCH --output=slurm_logs/%j.out
#SBATCH -x mind-1-13

# Multinode notes
# From https://pytorch-lightning.readthedocs.io/en/stable/clouds/cluster.html
# # debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# # on your cluster you might need these:
# # set the network interface

# # Currently (7/24/22) IB isn't working well on mind due to memlock limits
# export NCCL_IB_DISABLE=1
# # export NCCL_SOCKET_IFNAME=^docker0,lo

# # might need the latest CUDA
# # module load NCCL/2.4.7-1-cuda.10.0

# srun python3 artifacts/artifact_estimator.py --num-nodes 1 -c $@ # Multinode
echo 'tasks'
echo $SLURM_NTASKS
echo 'per node'
export SLURM_NTASKS_PER_NODE=4
echo $SLURM_NTASKS_PER_NODE

hostname
source ~/.bashrc # Note bashrc has been modified to allow slurm jobs
# echo "loaded bashrc"
# echo $TEST_PROFILE # did profile run?
source ~/load_env.sh
# echo "done loading env"
srun python -u run.py $1

