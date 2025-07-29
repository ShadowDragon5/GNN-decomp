import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from .common import GNN


def init_weights_normal(m):
    if m is nn.Linear:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(
                m.weight, a=0.0, nonlinearity="relu", mode="fan_out"
            )


class FCBlock(nn.Module):
    """A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int,
        hidden_features: int,
        layer_norm=False,
    ):
        super().__init__()

        # Create the net
        net = [
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_hidden_layers):
            net.append(nn.Linear(hidden_features, hidden_features))
            net.append(nn.ReLU(inplace=True))

        net.append(nn.Linear(hidden_features, out_features))
        if layer_norm:
            net.append(nn.LayerNorm([out_features]))

        self.net = nn.Sequential(*net)

        self.net.apply(init_weights_normal)

    def forward(self, coords):
        output = self.net(coords)
        return output


class MeshGraphNet(GNN):
    def __init__(
        self,
        edge_dim,
        num_steps,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.encoder_edge = FCBlock(
            in_features=edge_dim,
            out_features=self.hidden_dim,
            num_hidden_layers=2,
            hidden_features=self.hidden_dim,
            layer_norm=True,
        )
        self.encoder_nodes = FCBlock(
            in_features=self.in_dim,
            out_features=self.hidden_dim,
            num_hidden_layers=2,
            hidden_features=self.hidden_dim,
            layer_norm=True,
        )

        self.num_steps = num_steps
        self.diffMLP = True
        # message passing with different MLP for each steps
        self.processors = torch.nn.Sequential(
            *[
                Processor(self.hidden_dim, self.hidden_dim, layer_norm=True)
                for _ in range(num_steps)
            ]
        )

        self.decoder_node = FCBlock(
            in_features=self.hidden_dim,
            out_features=self.out_dim,
            num_hidden_layers=3,
            hidden_features=self.hidden_dim,
            layer_norm=False,
        )

        self.normalizer_node_feature = Normalizer(self.in_dim, self.device)
        self.normalizer_edge_feature = Normalizer(edge_dim, self.device)
        self.normalizer_v_gt = Normalizer(1, self.device)

    def encoder(self, graph):
        graph.x[:, :-1] = graph.x[:, :-1]
        graph.noise = graph.x[:, [-1]] * 0

        graph.x = self.encoder_nodes(graph.x)
        graph.edge_attr = self.encoder_edge(graph.edge_attr)
        return graph

    def decoder(self, graph):
        graph.x = self.decoder_node(graph.x)
        return graph

    def forward(self, graph):
        # normalize the dataset
        graph.x = self.normalizer_node_feature.update(graph.x, self.training)
        graph.edge_attr = self.normalizer_edge_feature.update(
            graph.edge_attr, self.training
        )
        graph.v_gt = self.normalizer_v_gt.update(graph.v_gt, self.training)

        # ecode edges and nodes to latent dim
        graph_latent = self.encoder(graph)

        # message passing steps with different MLP each time
        for i in range(self.num_steps):
            graph_latent = self.processors[i](graph_latent)

        # decoding
        graph_latent = self.decoder(graph_latent)
        graph_latent.x = graph_latent.x / 10  # div 10 for 46, div 100 for 31
        if not self.training:
            graph_latent.x = self.normalizer_v_gt.reverse(graph_latent.x)
        graph_latent.eval = graph_latent.x
        return graph_latent

    def loss(self, pred, label) -> torch.Tensor:
        """MSE"""
        gnn_prdiction = pred.eval[:, :]
        gt = pred.v_gt[:, :].to(self.device) - pred.noise[:, 0]
        # else:
        #     gnn_prdiction = pred.eval[:, :]
        #     gt = pred.v_gt[:, :].to(self.device)
        return ((gnn_prdiction[:, 0] - gt[:, 0]) ** 2).mean()


class Processor(MessagePassing):
    def __init__(self, in_channels, out_channels, layer_norm=False):
        super(Processor, self).__init__(aggr="add")  # "Add" aggregation.
        self.edge_encoder = FCBlock(
            in_features=in_channels * 3,
            out_features=out_channels,
            num_hidden_layers=2,
            hidden_features=in_channels,
            layer_norm=layer_norm,
        )
        self.node_encoder = FCBlock(
            in_features=in_channels * 2,
            out_features=out_channels,
            num_hidden_layers=2,
            hidden_features=in_channels,
            layer_norm=layer_norm,
        )
        self.latent_dim = out_channels

    def forward(self, graph):
        edge_index = graph.edge_index
        # cat features together (eij,vi,ei)
        x_receiver = torch.gather(
            graph.x, 0, edge_index[0, :].unsqueeze(-1).repeat(1, graph.x.shape[1])
        )
        x_sender = torch.gather(
            graph.x, 0, edge_index[1, :].unsqueeze(-1).repeat(1, graph.x.shape[1])
        )
        edge_features = torch.cat([x_receiver, x_sender, graph.edge_attr], dim=-1)
        # edge processor
        edge_features = self.edge_encoder(edge_features)

        # aggregate edge_features
        node_features = self.propagate(edge_index, x=graph.x, edge_attr=edge_features)
        # cat features for node processor (vi,\sum_eij)
        features = torch.cat([graph.x, node_features[:, self.latent_dim :]], dim=-1)
        # node processor and update graph
        graph.x = self.node_encoder(features) + graph.x
        graph.edge_attr = edge_features
        return graph

    def message(self, x_i, edge_attr):
        z = torch.cat([x_i, edge_attr], dim=-1)
        return z


class Normalizer(nn.Module):
    def __init__(self, dim, device, max_acc=60 * 600):
        super().__init__()
        self.device = device
        self.acc_sum = nn.Parameter(torch.zeros(dim).to(device), requires_grad=False)
        self.acc_sum_squared = nn.Parameter(
            torch.zeros(dim).to(device), requires_grad=False
        )
        self.mean = nn.Parameter(torch.zeros(dim).to(device), requires_grad=False)
        self.std = nn.Parameter(torch.ones(dim).to(device), requires_grad=False)

        self.total_acc = 0
        self.max_acc = max_acc

    def update(self, value, train):
        if self.total_acc < self.max_acc * value.shape[0] and train:
            self.total_acc += value.shape[0]
            self.acc_sum += torch.sum(value, 0).data
            self.acc_sum_squared += torch.sum(value**2, 0).data
            safe_count = max(1, self.total_acc)
            self.mean = nn.Parameter(self.acc_sum / safe_count)
            self.std = nn.Parameter(
                torch.maximum(
                    torch.sqrt(self.acc_sum_squared / safe_count - self.mean**2),
                    torch.tensor(1e-5).to(self.device),
                )
            )
        return (value - self.mean.data) / self.std.data

    def reverse(self, value):
        return value * self.std.data + self.mean.data
