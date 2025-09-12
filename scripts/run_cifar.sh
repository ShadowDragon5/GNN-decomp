#!/bin/sh
# CIFAR10

. .venv/bin/activate

# 09-12
# SEED=12
SEED=35
# SEED=41
# SEED=95

DESC="Gamma optim. initialized to 0s instead of 1s"

python src/main.py seed=$SEED description="$DESC" partitions=2 epochs=100 pre_epochs=40 gamma_algo=SGD ASM=true target=train --config-name cifar10_bat
python src/main.py seed=$SEED description="$DESC" partitions=3 epochs=100 pre_epochs=40 gamma_algo=SGD ASM=true target=train --config-name cifar10_bat
python src/main.py seed=$SEED description="$DESC" partitions=5 epochs=100 pre_epochs=40 gamma_algo=SGD ASM=true target=train --config-name cifar10_bat
