# Graph Neural Network Domain Decomposition Experiments

This repository implements a preconditioning algorithm inspired by domain
decomposition methods for GNNs, aiming to improve convergence behavior without
modifying the network architecture.

**Key idea**

1. Partition each graph dataset into `partitions` subsets
2. Perform `pre_epochs` local optimization steps on each partition
3. Compute weight differences per partition (local contributions)
4. Compute optimal scaling factors $\gamma$ for each contribution
5. Aggregate the scaled contributions
6. Perform a global optimizer step on the full dataset


## Tested Datasets

* [AirfRANS](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.AirfRANS.html#torch_geometric.datasets.AirfRANS)
* [CIFAR10](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GNNBenchmarkDataset.html#torch_geometric.datasets.GNNBenchmarkDataset)
* [MNIST](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GNNBenchmarkDataset.html#torch_geometric.datasets.GNNBenchmarkDataset)

Considered (not tested with the current version):

* [PATTERN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GNNBenchmarkDataset.html#torch_geometric.datasets.GNNBenchmarkDataset)
* [Wave2D](https://github.com/computational-imaging/GraphPDE)


## Running Experiments

```sh
# Create virtual environment and install dependencies
uv sync --no-build-isolation

# AirfRANS
uv run src/main.py description="a demo run" dev.batch=1 --config-name airfrans_SGD.yaml

# CIFAR10
uv run src/main.py --config-name cifar10_SGD.yaml

# MNIST
uv run src/main.py dataset=MNIST --config-name cifar10_SGD.yaml
```

Experiment result dashboard:

```sh
uv run mlflow ui
```


## Configurable Parameters

| Name               | Values (default)                        | Description                                                                                                                                    |
| ------------------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `description`      | string (`""`)                           | Text experiment description                                                                                                                    |
| `seed`             | integer (`42`)                          | Random seed                                                                                                                                    |
| `epochs`           | integer (`100`)                         | Total number of training epochs                                                                                                                |
| `u`                | `true/false`                            | Force dataset re-download/update                                                                                                               |
| `dataset`          | `MNIST/CIFAR10/PATTERN/Wave2D/AirfRANS` | Dataset                                                                                                                                        |
| `full_epochs`      | integer (`1`)                           | Number of full-dataset passes after each preconditioning phase                                                                                 |
| `partitions`       | `2/3/5` (`2`)                           | Number of dataset partitions                                                                                                                   |
| `pre_epochs`       | integer (`40`)                          | Number of preconditioner epochs                                                                                                                |
| `ASM`              | `true/false`                            | `true` - Additive Schwarz variant, `false` - Multiplicative Schwarz variant                                                                    |
| `gamma_algo`       | `none/backtracking/brent/SGD`           | Algorithm used to compute partition scaling factors $\gamma$ (SGD only valid with `ASM=true`)                                                  |
| `gamma_strat`      | `direct/clipped/inverse`                | Strategy for combining $\gamma$ values (only when `ASM=true` and `gamma_algo=SGD`)                                                             |
| `gamma_lr`         | float (`0.01`)                          | Learning rate for SGD-based $\gamma$ optimization                                                                                              |
| `target`           | `train/valid`                           | Dataset split used for $\gamma$ optimization (any line search or SGD)                                                                          |
| `optim`            | `Adam/SGD/SGDm/RMSprop`                 | Optimizer used in Additive Schwarz variant with `gamma_algo=SGD`                                                                               |
| `max_evals`        | integer (`1`)                           | Number of evaluations when running hyperparameter search                                                                                       |
| `ll_resolution`    | integer (`0`)                           | Loss landscape grid resolution (`0` disables evaluation; see `Preconditioned.loss_landscape()`)                                                |
| `dev.batch`        | integer (`128`)                         | Batch size                                                                                                                                     |
| `dev.data_dir`     | string (`./datasets`)                   | Directory for dataset download/storage                                                                                                         |
| `dev.q`            | `true/false`                            | Quiet mode (suppress standard output)                                                                                                          |
| `dev.num_workers`  | integer (`2`)                           | Number of PyTorch DataLoader workers                                                                                                           |
| `model.base`       | `GCN_CG/GraphSAGE/GCN_CN/MeshGraphNet`  | Backbone model (`GCN_CG` for graph classification on CIFAR10/MNIST; `GraphSAGE` for AirfRANS; `GCN_CN` for PATTERN; `MeshGraphNet` for Wave2D) |
| `model.hidden_dim` | integer (`146`)                         | Hidden feature dimension                                                                                                                       |
| `model.out_dim`    | integer (`146`)                         | Decoder dimension                                                                                                                              |
| `model.dropout`    | float `0–1` (`0.0`)                     | Dropout probability                                                                                                                            |
| `model.lr`         | float (`0.001`)                         | Optimizer learning rate                                                                                                                        |
| `model.wd`         | float (`0.0`)                           | Optimizer weight decay                                                                                                                         |
| ~~`model.pre_lr`~~ | float (`0.0`)                           | Preconditioner learning rate (currently ignored)                                                                                               |
| ~~`model.pre_wd`~~ | float (`0.0`)                           | Preconditioner weight decay (currently ignored)                                                                                                |
