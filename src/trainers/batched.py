from collections import defaultdict

import mlflow
import torch
from tqdm import tqdm

from utils import get_data

from .common import Trainer


class Batched(Trainer):
    """
    Mini batches of full graphs.
    """

    def run(self) -> float:
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        scheduler = self.scheduler(
            optimizer, self.lr, len(self.trainloader) * self.epochs
        )

        self.model.to(self.device)

        valid_loss = defaultdict(float)
        for epoch in range(self.epochs):
            train_loss = 0
            self.model.train()

            for data in tqdm(
                self.trainloader,
                desc=f"Epoch: {epoch:03}",
                dynamic_ncols=True,
                disable=self.quiet,
            ):
                data.to(self.device)

                out, y = self.model(**get_data(data))
                loss = self.model.loss(out, y)["loss"]

                train_loss += loss.detach().item()

                loss.backward()
                optimizer.step()
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                optimizer.zero_grad()

            train_loss /= len(self.trainloader)

            # Validation
            valid_loss = self.validate(self.model)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_loss["loss"])

            if not self.quiet:
                print(f"Epoch: {epoch:03} | Valid Loss: {valid_loss['loss']}")

            mlflow.log_metrics(
                {
                    "train/loss": train_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    **{f"validate/{k}": v for k, v in valid_loss.items()},
                },
                step=epoch,
            )

        if self.need_acc:
            accuracy = self.test()
            if not self.quiet:
                print(f"Accuracy: {accuracy}")

            mlflow.log_metric("test/accuracy", accuracy)

        return valid_loss["loss"]
