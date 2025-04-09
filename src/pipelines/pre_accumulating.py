from copy import deepcopy

import mlflow
import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from pipelines.common import Pipeline
from utils import PartitionedData


class PreAccumulating(Pipeline):
    """Partitioned graph preconditioner, gradient accumulation variation"""

    def __init__(
        self,
        pre_epochs: int,
        part_trainloader: DataLoader,
        num_parts: int,
        pre_lr: float,
        pre_wd: float,
        ASM: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pre_epochs = pre_epochs
        self.num_parts = num_parts
        self.part_trainloader = part_trainloader
        self.pre_lr = pre_lr
        self.pre_wd = pre_wd
        self.ASM = ASM

    def precondition(self, model: Module, i: int, epoch: int) -> Module:
        pre_optimizer = torch.optim.Adam(
            model.parameters(), lr=self.pre_lr, weight_decay=self.pre_wd
        )

        pre_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            pre_optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        for pre_epoch in range(self.pre_epochs):
            pre_train_loss = 0

            for data in tqdm(
                self.part_trainloader,
                desc=f"P{i} Epoch: {pre_epoch:03}",
                dynamic_ncols=True,
                leave=False,
                disable=self.quiet,
            ):
                data: PartitionedData = data
                x = data.get_x(i, self.device).to(self.device)
                edge_index = data.get_edge_index(i, self.device)
                batch = data.get_batch(i, self.device)
                y = data.get_y(i, self.device)

                out = model(x, edge_index, batch)
                loss = model.loss(out, y)

                loss = loss / len(self.part_trainloader)
                pre_train_loss += loss.detach().item()
                loss.backward()

            pre_optimizer.step()
            pre_optimizer.zero_grad()
            pre_scheduler.step(pre_train_loss)

            mlflow.log_metrics(
                {
                    f"pre_train/loss_p{i}": pre_train_loss,
                    f"pre_train/lr_p{i}": pre_scheduler.get_last_lr()[0],
                },
                step=epoch * self.pre_epochs + pre_epoch,
            )
        return model

    def run(self) -> float:
        """Main training loop"""
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.model.to(self.device)

        valid_loss = 0
        for epoch in range(self.epochs):
            train_loss = 0
            self.model.train()

            # Preconditioning step
            if self.ASM:  # Additive Schwarz
                models = []
                for i in range(self.num_parts):
                    models.append(
                        self.precondition(
                            deepcopy(self.model).to(self.device), i, epoch
                        ).state_dict()
                    )

                # TODO: try other strategies
                # average model weights
                w_avg = models[0]
                for key in w_avg:
                    if w_avg[key].data.dtype == torch.float:
                        for m in models[1:]:
                            w_avg[key].data += m[key].data.clone()
                        w_avg[key] /= len(models)
                self.model.load_state_dict(w_avg)

            else:  # Multiplicative Schwarz
                for i in range(self.num_parts):
                    self.precondition(self.model, i, epoch)

            # Full pass
            for data in tqdm(
                self.trainloader,
                desc=f"Epoch: {epoch:03}",
                dynamic_ncols=True,
                leave=False,
                disable=self.quiet,
            ):
                x = data.x.to(self.device)
                y = data.y.to(self.device)

                edge_index = data.edge_index.to(self.device)
                batch = data.batch.to(self.device)

                out = self.model(x, edge_index, batch)
                loss = self.model.loss(out, y)

                loss = loss / len(self.trainloader)
                train_loss += loss.detach().item()
                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Validation
            valid_loss = 0
            correct = 0
            total = 0
            self.model.eval()
            with torch.no_grad():
                for data in tqdm(
                    self.validloader,
                    dynamic_ncols=True,
                    leave=False,
                    disable=self.quiet,
                ):
                    x = data.x.to(self.device)
                    y = data.y.to(self.device)

                    edge_index = data.edge_index.to(self.device)
                    batch = data.batch.to(self.device)

                    out = self.model(x, edge_index, batch)
                    loss = self.model.loss(out, y)
                    valid_loss += loss.detach().item()

                    # Validation accuracy
                    pred = out.argmax(dim=1)  # Predicted labels
                    correct += (pred == y).sum().item()
                    total += y.size(0)

            valid_loss /= len(self.validloader)

            scheduler.step(valid_loss)

            if not self.quiet:
                print(f"Epoch: {epoch:03} | " f"Valid Loss: {valid_loss}")

            mlflow.log_metrics(
                {
                    "train/loss": train_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "validate/loss": valid_loss,
                    "validate/accuracy": correct / total,
                },
                step=epoch,
            )

        accuracy = self.test()
        if not self.quiet:
            print(f"Accuracy: {accuracy}")

        mlflow.log_metric("test/accuracy", accuracy)

        return valid_loss
