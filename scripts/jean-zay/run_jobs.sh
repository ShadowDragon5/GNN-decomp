#!/bin/sh

OUT_PREF="$SCRATCH/out/"
SCRIPT_PATH=$(dirname "$(realpath "$0")")

run_job() {
	NAME="$1"
	shift
	sbatch --job-name="${NAME}" --error="${OUT_PREF}err_${NAME}" --output="${OUT_PREF}out_${NAME}" \
		"$SCRIPT_PATH/job.slurm" dev=jean_zay "$@"
}

run_job "CIF2T1" full_batches=1 --config-name cifar10_Adag
run_job "CIF2Tall" full_batches=all --config-name cifar10_Adag

run_job "C2p10Tall" pre_epochs=10 full_batches=all --config-name cifar10_Adag
run_job "C2p10T1" pre_epochs=10 full_batches=1 --config-name cifar10_Adag
run_job "C2p10T10" pre_epochs=10 full_batches=10 --config-name cifar10_Adag
