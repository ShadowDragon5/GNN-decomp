import mlflow
import numpy as np
import torch
from tqdm import tqdm

from .common import Trainer


class MGN_trainer(Trainer):
    """
    Mesh Graph Net trainer
    Batched
    """

    def run(self) -> float:
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, gamma=np.exp(np.log(1e-4) / self.epochs * 20), last_epoch=-1
        # )

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
                data.to(self.device)

                optimizer.zero_grad()

                out = self.model(data)
                loss = self.model.loss(out, data)

                train_loss += loss.detach().item()

                loss.backward()
                optimizer.step()

            train_loss /= len(self.trainloader)

            # Validation
            _, valid_loss = self.validate(self.model)
            # scheduler.step(valid_loss)

            if not self.quiet:
                print(f"{self.name} Epoch: {epoch:03} | Valid Loss: {valid_loss}")

            mlflow.log_metrics(
                {
                    "train/loss": train_loss,
                    "validate/loss": valid_loss,
                },
                step=epoch,
            )

        return self.test()

    def validate(self, model, dataloader=None) -> tuple[float, float]:
        if dataloader is None:
            dataloader = self.validloader

        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for data in tqdm(
                dataloader,
                desc="Validation",
                dynamic_ncols=True,
                leave=False,
                disable=self.quiet,
            ):
                data.to(self.device)

                out = model(data)
                loss = model.loss(out, data)
                valid_loss += loss.detach().item()

        valid_loss /= len(self.validloader)
        return np.inf, valid_loss

    def test(self) -> float:
        _, loss = self.validate(self.model)
        return loss
