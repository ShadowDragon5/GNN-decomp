import mlflow
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from pipelines.common import Pipeline


class PreAccumulating(Pipeline):
    """Partitioned graph preconditioner, gradient accumulation variation"""

    def __init__(
        self,
        pre_epochs: int,
        part_trainloader: DataLoader,
        num_parts: int,
        pre_lr: float,
        pre_wd: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pre_epochs = pre_epochs
        self.num_parts = num_parts
        self.part_trainloader = part_trainloader
        self.pre_lr = pre_lr
        self.pre_wd = pre_wd

    def run(self) -> float:
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        pre_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.pre_lr, weight_decay=self.pre_wd
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        # pre_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     pre_optimizer,
        #     mode="min",
        #     factor=0.5,
        #     patience=5,
        # )

        self.model.to(self.device)

        valid_loss = 0
        for epoch in range(self.epochs):
            train_loss = 0
            self.model.train()

            # Preconditioning step
            for pre_epoch in range(self.pre_epochs):
                for i in range(self.num_parts):
                    pre_train_loss = 0
                    for data in self.part_trainloader:
                        x = getattr(data, f"x_{i}").to(self.device)
                        edge_index = getattr(data, f"edge_index_{i}").to(self.device)
                        batch = getattr(data, f"x_{i}_batch").to(self.device)

                        y = data.y.to(self.device)

                        out = self.model(x, edge_index, batch)
                        loss = self.model.loss(out, y)

                        loss = loss / len(self.part_trainloader)
                        pre_train_loss += loss.detach().item()
                        loss.backward()

                    pre_optimizer.step()
                    pre_optimizer.zero_grad()

                    mlflow.log_metrics(
                        {
                            f"pre_train/loss_p{i}": pre_train_loss,
                        },
                        step=epoch * self.pre_epochs + pre_epoch,
                    )

            # Full pass
            for data in tqdm(
                self.trainloader,
                desc=f"Epoch: {epoch:03}",
                dynamic_ncols=True,
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
                    self.validloader, dynamic_ncols=True, disable=self.quiet
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
            # pre_scheduler.step(valid_loss)  # NOTE: maybe check with pre_train loss?

            if not self.quiet:
                print(f"{self.name} Epoch: {epoch:03} | " f"Valid Loss: {valid_loss}")

            mlflow.log_metrics(
                {
                    "train/loss": train_loss,
                    "validate/loss": valid_loss,
                    "validate/accuracy": correct / total,
                },
                step=epoch,
            )

        accuracy = self.test()
        if not self.quiet:
            print(f"{self.name} Accuracy: {accuracy}")

        mlflow.log_metric("test/accuracy", accuracy)

        return valid_loss
