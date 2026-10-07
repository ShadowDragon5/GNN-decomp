#!/bin/sh

OUT_PREF="$SCRATCH/out/"
SCRIPT_PATH=$(dirname "$(realpath "$0")")

run_job() {
	NAME="$1"
	shift
	# max 4 GPUs per node on GH200
	GPUS="$1"
	shift

	# --ntasks-per-node - number of MPI tasks per node (= number of GPUs per node)
	sbatch --job-name="${NAME}" --error="${OUT_PREF}err_${NAME}" --output="${OUT_PREF}out_${NAME}" \
		--ntasks-per-node="$GPUS" \
		"$SCRIPT_PATH/job.slurm" dev=jean_zay "$@"
}

for seed in 12 35 41 95;
do
	run_job "CIF2r${seed}" 2 partitions=2 seed=$seed epochs=1000 --config-name cifar10_dist
done
