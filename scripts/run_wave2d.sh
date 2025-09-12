#!/bin/sh
# Wave2D

. .venv/bin/activate

# 07-28
# python src/main.py dev.batch=32 model.hidden_dim=32 --config-name wave2d_base

# 07-30
python src/main.py description="Gamma optim. allowed to converge. Loss is w.r.t. gt" epochs=100 trainer=pre-batched dev.batch=256 model.hidden_dim=32 pre_epochs=40 gamma_algo=SGD ASM=true target=train --config-name wave2d_acc
# python src/main.py description="Gamma optim. allowed to converge. Loss is w.r.t. gt" partitions=3 epochs=100 trainer=pre-batched dev.batch=32 model.hidden_dim=32 pre_epochs=10 gamma_algo=SGD ASM=true target=train --config-name wave2d_acc
