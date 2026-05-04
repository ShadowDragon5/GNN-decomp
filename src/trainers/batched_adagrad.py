from collections import defaultdict

import mlflow
from tqdm import tqdm

from optimizers.dd_adagrad import DD_Adagrad
from utils import get_data

from .common import Trainer


class BatchedAdagrad(Trainer):
    def __init__(
        self,
        optim_params: dict,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.optim_params = optim_params

    def run(self) -> float:
        self.model.to(self.device)
        optimizer = DD_Adagrad(self.model.parameters(), **self.optim_params)

        k_iter = 0
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

                def closure():
                    out, y = self.model(**get_data(data))
                    return self.model.loss(out, y)["loss"]

                loss = optimizer.step(closure)

                train_loss += loss.detach().item()

                optimizer.zero_grad()
                k_iter += 1

            train_loss /= len(self.trainloader)

            # Validation
            valid_loss = self.validate(self.model)

            if not self.quiet:
                print(f"Epoch: {epoch:03} | Valid Loss: {valid_loss['loss']}")

            mlflow.log_metrics(
                {
                    "train/loss": train_loss,
                    **{f"validate/{k}": v for k, v in valid_loss.items()},
                },
                step=k_iter,
            )

        if self.need_acc:
            accuracy = self.test()
            if not self.quiet:
                print(f"Accuracy: {accuracy}")

            mlflow.log_metric("test/accuracy", accuracy)

        return valid_loss["loss"]
