from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.common import GNN
from utils import get_data


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
        need_acc: bool = False,
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
        self.need_acc = need_acc

    @abstractmethod
    def run(self) -> float:
        """Main training loop"""
        pass

    # TODO: change return to dict[str, float] for more dynamic measurement
    def validate(self, model, dataloader=None):
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
                data.to(self.device)

                out, y = model(**get_data(data))
                loss = model.loss(out, y)
                valid_loss += loss.detach().item()
                if loss.detach().item() == torch.nan:
                    print("NaN loss")
                    print(out.detach())
                    print(y.detach())

                # Validation accuracy
                if self.need_acc:
                    pred = out.argmax(dim=1)  # Predicted labels
                    correct += (pred == y).sum().item()
                    total += y.size(0)

        valid_loss /= len(self.validloader)
        acc = None
        if self.need_acc:
            acc = correct / total
        return acc, valid_loss

    def test(self) -> float:
        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                data.to(self.device)

                out, y = self.model(**get_data(data))

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
        else:
            self.is_better = lambda a, best: a > best + min_delta
