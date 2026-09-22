#!/bin/bash
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1                   # number of nodes
#SBATCH --cpus-per-task=3           # number of CPUs per task (1/8 of 24 cores on gpu_p2)
#SBATCH --hint=nomultithread        # hyperthreading disabled
#SBATCH --time=20:00:00             # maximum execution time requested (HH:MM:SS)


module purge

# jean-zay NCCL version: 2.27.3
export OMP_NUM_THREADS=$SLURM_NTASKS_PER_NODE
export NCCL_DEBUG=INFO
export NCCL_IB_CUDA_SUPPORT=1
# run ibstat to get interfaces
# export NCCL_IB_HCA=ipogif0

# module load pytorch-gpu/py3/2.6.0
source .venv/bin/activate

# python3 src/main.py "$@"
torchrun --nproc-per-node "$SLURM_NTASKS_PER_NODE" src/main.py "$@"
