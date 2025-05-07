from abc import ABC, abstractmethod

import torch
from torch_geometric.loader import DataLoader


class Trainer(ABC):
    """A wrapper to unify trainer signature"""

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
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
        pass

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
