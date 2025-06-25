import mlflow
import torch
from tqdm import tqdm

from .common import Trainer


class Batched(Trainer):
    """
    State of the art re-implementation from (Dwivedi et al., 2022).
    Mini batches of full graphs.
    """

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

        valid_loss = 0
        for epoch in range(self.epochs):
            train_loss = 0
            self.model.train()

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

                train_loss += loss.detach().item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            train_loss /= len(self.trainloader)

            # Validation
            accuracy, valid_loss = self.validate(self.model)
            scheduler.step(valid_loss)

            if not self.quiet:
                print(f"{self.name} Epoch: {epoch:03} | " f"Valid Loss: {valid_loss}")

            mlflow.log_metrics(
                {
                    "train/loss": train_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "validate/loss": valid_loss,
                    "validate/accuracy": accuracy,
                },
                step=epoch,
            )

        accuracy = self.test()
        if not self.quiet:
            print(f"Accuracy: {accuracy}")

        mlflow.log_metric("test/accuracy", accuracy)

        return valid_loss
