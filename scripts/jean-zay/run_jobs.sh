#!/bin/sh

OUT_PREF="${SCRATCH}/out"

run_job() {
	NAME="$1"
	shift
	sbatch --job-name="GNN_${NAME}" --error="${OUT_PREF}err_${NAME}" --output="${OUT_PREF}out_${NAME}" \
		job.sh q=true dev=jean_zay "$@"
}

run_job "CIF2T1" full_batches=1 --config-name cifar10_Adag
run_job "CIF2Tall" full_batches=all --config-name cifar10_Adag
