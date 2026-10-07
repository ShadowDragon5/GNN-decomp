#!/bin/bash -l
#SBATCH --account="prep45"
#SBATCH --nodes=1                   # number of nodes
#SBATCH --gpus-per-task=1           # 1 GPU per MPI rank
#SBATCH --cpus-per-task=3
#SBATCH --partition=normal
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00             # maximum execution time requested (HH:MM:SS)
#SBATCH --environment=gnndecomp


# Container NCCL version: 2.30.7-1
export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE
export NCCL_DEBUG=INFO
export NCCL_IB_CUDA_SUPPORT=1
export SLURM_MPI_TYPE=pmix

# module load pytorch-gpu/py3/2.6.0
source .venv/bin/activate

# python3 src/main.py "$@"
# https://docs.pytorch.org/docs/2.14/elastic/run.html#important-notices
torchrun --nproc-per-node "$SLURM_NTASKS_PER_NODE" src/main.py "$@"
