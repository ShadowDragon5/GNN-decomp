import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, global_mean_pool

from .common import GNN, ResidualGCNLayer


class GCN_CG(GNN):
    def __init__(self, n_classes: int, **kwargs):
        super().__init__(**kwargs)

        self.embedding_h = nn.Linear(self.in_dim, self.hidden_dim)

        self.conv = Sequential(
            "x, edge_index",
            [
                (ResidualGCNLayer(self.hidden_dim, self.dropout), "x, edge_index -> x"),
                (ResidualGCNLayer(self.hidden_dim, self.dropout), "x, edge_index -> x"),
                (ResidualGCNLayer(self.hidden_dim, self.dropout), "x, edge_index -> x"),
                (ResidualGCNLayer(self.hidden_dim, self.dropout), "x, edge_index -> x"),
            ],
        )

        self.MLP_layer = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim >> 1),
            nn.ReLU(),
            nn.Linear(self.out_dim >> 1, self.out_dim >> 2),
            nn.ReLU(),
            nn.Linear(self.out_dim >> 2, n_classes),
        )

    def forward(self, x, y, edge_index, batch, **_):
        h = self.embedding_h(x)

        h = self.conv(h, edge_index)

        # per node features [n, f] -> graph features [n_batches, f]
        h = global_mean_pool(h, batch)

        h = self.MLP_layer(h)
        return h, y

    def loss(self, pred, label) -> dict[str, torch.Tensor]:
        criterion = nn.CrossEntropyLoss()
        return {"loss": criterion(pred, label)}


if __name__ == "__main__":
    model = GCN_CG(
        hidden_dim=146,
        out_dim=146,
        in_dim=5,
        n_classes=10,
        dropout=0,
        device="cpu",
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # cloned = deepcopy(model)
    # cloned_dict = deepcopy(model.state_dict())
    model.train()

    print(sum(p.numel() for p in model.parameters()))

    # print("Initial")
    # print(model.state_dict()["embedding_h.weight"][:10])
    # # print(next(model.parameters())[:10])
    #
    # x = torch.ones(5, 5)
    # y = torch.tensor([1], dtype=torch.int64)
    #
    # edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])
    #
    # out, y = model(x, y, edge_index, None)
    # loss = model.loss(out, y)
    # print(loss.item())
    # loss.backward()
    # optimizer.step()
    # model.eval()
    #
    # out, y = model(x, y, edge_index, None)
    # loss = model.loss(out, y)
    # print(loss.item())
    # loss.backward()
    # optimizer.step()
    #
    # print("Optimized")
    # print(model.state_dict()["embedding_h.weight"][:10])
    # print(next(model.parameters())[:10])
    #
    # print("Cloned")
    # print(cloned_dict["embedding_h.weight"][:10])
    # print(next(cloned.parameters())[:10])

    # d = model.state_dict()
    # for k, v in d.items():
    #     print(v.data.dtype)
    #     if v.data.dtype != torch.float:
    #         print(k, v)
