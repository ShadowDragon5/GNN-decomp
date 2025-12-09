#!/bin/sh
# CIFAR10

. .venv/bin/activate

for PRE in 1 3 10 40;
do
	for SEED in 12 35 41 95;
	do

		# Naive combination (AS/MS)
		python src/main.py seed=$SEED partitions=2 epochs=100 pre_epochs=$PRE gamma_algo=none ASM=true --config-name cifar10_bat
		python src/main.py seed=$SEED partitions=2 epochs=100 pre_epochs=$PRE gamma_algo=none ASM=false --config-name cifar10_bat

		# backtracking line search (AS/MS)
		python src/main.py seed=$SEED partitions=2 epochs=100 pre_epochs=$PRE gamma_algo=backtracking ASM=true --config-name cifar10_bat
		python src/main.py seed=$SEED partitions=2 epochs=100 pre_epochs=$PRE gamma_algo=backtracking ASM=false --config-name cifar10_bat

		# # SGD (AS)
		# python src/main.py seed=$SEED partitions=2 epochs=100 pre_epochs=$PRE gamma_algo=SGD ASM=true target=train --config-name cifar10_bat
		# python src/main.py seed=$SEED partitions=2 epochs=100 pre_epochs=$PRE gamma_algo=SGD gamma_strat=inverse ASM=true target=train --config-name cifar10_bat
		# python src/main.py seed=$SEED partitions=2 epochs=100 pre_epochs=$PRE gamma_algo=SGD gamma_strat=inverse ASM=true target=valid --config-name cifar10_bat

		# # partitions
		# for PARTS in 3 5;
		# do
		# 	python src/main.py seed=$SEED partitions=$PARTS epochs=100 pre_epochs=$PRE gamma_algo=SGD gamma_strat=inverse ASM=true target=train --config-name cifar10_bat
		# 	python src/main.py seed=$SEED partitions=$PARTS epochs=100 pre_epochs=$PRE gamma_algo=backtracking ASM=false --config-name cifar10_bat
		# done

	done
done
