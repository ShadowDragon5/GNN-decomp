#!/bin/bash -l
#SBATCH --account="prep45"
#SBATCH --nodes=1                   # number of nodes
#SBATCH --cpus-per-task=24
#SBATCH --partition=normal
#SBATCH --hint=nomultithread
#SBATCH --time=8:00:00             # maximum execution time requested (HH:MM:SS)

export GPUS="$1"
shift

# Container NCCL version: 2.30.7-1
export OMP_NUM_THREADS=$GPUS
export NCCL_DEBUG=WARN
export NCCL_IB_CUDA_SUPPORT=1
export SLURM_MPI_TYPE=pmix
export SLURM_NETWORK=disable_rdzv_get

cd /iopsstor/scratch/cscs/lvarnas/GNN-decomp/ || exit

# https://docs.pytorch.org/docs/2.14/elastic/run.html#important-notices
srun -u --environment=cscs bash -c '
    source .venv/bin/activate
    python -m torch.distributed.run --nproc-per-node $GPUS src/main.py "$@"
' bash "$@"
