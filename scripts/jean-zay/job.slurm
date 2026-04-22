#!/bin/bash
#SBATCH --partition=gpu_p2
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MPI tasks per node (= number of GPUs per node)
#SBATCH --gres=gpu:1                # number of GPUs per node (max 8 with gpu_p2, gpu_p5)
#SBATCH --cpus-per-task=2        	# number of CPUs per task (1/4 of the 4-GPU V100 node here)
#SBATCH --hint=nomultithread        # hyperthreading disabled
#SBATCH --time=20:00:00             # maximum execution time requested (HH:MM:SS)


module purge

module load pytorch-gpu/py3/2.6.0

python3 src/main.py "$@"
