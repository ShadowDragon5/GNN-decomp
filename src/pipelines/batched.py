import mlflow
import torch
from tqdm import tqdm

from pipelines.common import Pipeline


class Batched(Pipeline):
    """
    State of the art re-implementation from (Dwivedi et al., 2022).
    Mini batches of full graphs.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(self) -> float:
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

        for epoch in range(self.epochs):
            train_loss = 0
            self.model.train()

            for data in tqdm(
                self.trainloader,
                desc=f"Epoch: {epoch:02}",
                dynamic_ncols=True,
                disable=self.quiet,
                leave=False,
                position=2,
            ):
                x = data.x.to(self.device)
                y = data.y.to(self.device)

                edge_index = data.edge_index.to(self.device)
                batch = data.batch.to(self.device)

                optimizer.zero_grad()
                out = self.model(x, edge_index, batch)
                loss = self.model.loss(out, y)

                train_loss += loss.detach().item()

                loss.backward()
                optimizer.step()

            train_loss /= len(self.trainloader)

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
                    position=2,
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
        print(f"{self.name} Accuracy: {accuracy}")
        mlflow.log_metric("test/accuracy", accuracy)

        return accuracy
