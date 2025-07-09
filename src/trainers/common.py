from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.common import GNN


class Trainer(ABC):
    """A wrapper to unify trainer signature"""

    def __init__(
        self,
        name: str,
        model: GNN,
        trainloader: DataLoader,
        validloader: DataLoader,
        testloader: DataLoader,
        device: torch.device,
        epochs: int,
        lr: float,
        wd: float,
        quiet: bool = False,
        **_,
    ) -> None:
        self.name = name
        self.model = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.quiet = quiet

    @abstractmethod
    def run(self) -> float:
        """Main training loop"""
        pass

    # TODO: change return to dict[str, float] for more dynamic measurement
    def validate(self, model, dataloader=None) -> tuple[float, float]:
        """
        Validates the model w.r.t. given dataloader (uses validation set by default)
        returns: (accuracy, loss)
        """
        if dataloader is None:
            dataloader = self.validloader
        valid_loss = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in tqdm(
                dataloader,
                desc="Validation",
                dynamic_ncols=True,
                leave=False,
                disable=self.quiet,
            ):
                x = data.x.to(self.device)
                y = data.y.to(self.device)

                edge_index = data.edge_index.to(self.device)
                batch = data.batch.to(self.device)

                out = model(x, edge_index, batch)
                loss = model.loss(out, y)
                valid_loss += loss.detach().item()

                # Validation accuracy
                pred = out.argmax(dim=1)  # Predicted labels
                correct += (pred == y).sum().item()
                total += y.size(0)

        valid_loss /= len(self.validloader)
        return correct / total, valid_loss

    def test(self) -> float:
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                x = data.x.to(self.device)
                y = data.y.to(self.device)

                edge_index = data.edge_index.to(self.device)
                batch = data.batch.to(self.device)

                out = self.model(x, edge_index, batch)
                pred = out.argmax(dim=1)  # Predicted labels

                correct += (pred == y).sum().item()
                total += y.size(0)

        return correct / total


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-6,
        mode: Literal["min", "max"] = "min",
    ):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self._init_is_better(mode, min_delta)

    def step(self, metrics: float):
        if self.best is None:
            self.best = metrics
            return False

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode: Literal["min", "max"], min_delta: float):
        if mode == "min":
            self.is_better = lambda a, best: a < best - min_delta
        elif mode == "max":
            self.is_better = lambda a, best: a > best + min_delta
