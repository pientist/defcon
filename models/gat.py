from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        node_in_dim = args["node_in_dim"]
        edge_in_dim = args["edge_in_dim"]
        node_emb_dim = args["node_emb_dim"]
        graph_emb_dim = args["graph_emb_dim"]
        n_layers = args["gnn_layers"]
        n_heads = args["gnn_heads"]

        self.node_in_fc = nn.Sequential(nn.Linear(node_in_dim, node_emb_dim), nn.ReLU(inplace=True))

        self.gat_layers = nn.ModuleList()
        self.node_layernorms = nn.ModuleList()
        self.activation = nn.ELU()

        for _ in range(n_layers):
            gat_in_dim = node_in_dim + node_emb_dim if args["skip_conn"] else node_emb_dim
            self.gat_layers.append(GATConv(gat_in_dim, node_emb_dim // n_heads, heads=n_heads, edge_dim=edge_in_dim))
            self.node_layernorms.append(nn.LayerNorm(node_emb_dim))

        self.node_out_fc = nn.Sequential(nn.Linear(node_emb_dim, node_emb_dim), nn.ReLU(inplace=True))
        self.node_layernorms.append(nn.LayerNorm(graph_emb_dim))

        if graph_emb_dim > 0:
            self.graph_in_fc = nn.Sequential(nn.Linear(node_emb_dim, graph_emb_dim), nn.ReLU(inplace=True))
            self.graph_layernorm = nn.LayerNorm(graph_emb_dim)

    def forward(self, batch_graphs: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            batch_graphs: a DataBatch object containing B graphs and N nodes with shape [N, x].
        Returns:
            node_embeddings: a float tensor with shape [N, z].
            graph_embeddings: a float tensor with shape [B, z].
        """
        node_embeddings = self.node_in_fc(batch_graphs.x)  # [N, z]
        for i, gat_layer in enumerate(self.gat_layers):
            if self.args["skip_conn"]:
                gat_inputs = torch.cat([batch_graphs.x, node_embeddings], -1)  # [N, x + z]
                node_embeddings = gat_layer(gat_inputs, batch_graphs.edge_index, batch_graphs.edge_attr)  # [N, z]
            else:
                node_embeddings = gat_layer(node_embeddings, batch_graphs.edge_index, batch_graphs.edge_attr)  # [N, z]
            node_embeddings = self.activation(self.node_layernorms[i](node_embeddings))

        if self.args["graph_emb_dim"] > 0:
            graph_embeddings = self.graph_layernorm(self.graph_in_fc(node_embeddings))  # [N, z]
            graph_embeddings = global_mean_pool(graph_embeddings, batch_graphs.batch)  # [B, z]
        else:
            graph_embeddings = None

        node_embeddings = self.node_layernorms[-1](self.node_out_fc(node_embeddings))
        return node_embeddings, graph_embeddings


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        node_in_dim = args["node_emb_dim"]
        graph_in_dim = args["graph_emb_dim"]

        if args["skip_conn"]:
            node_in_dim += args["node_in_dim"]

        if args["target"].startswith("dest"):
            node_in_dim += 2
            graph_in_dim += 2

        if args["task"] == "node_selection":  # intent, receiver, {intent/failure/dest}_receiver
            self.nodewise_mlp = nn.Sequential(
                nn.Linear(node_in_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
                nn.GLU(),
            )
            if args["residual"]:  # To estimate probabilities for the ball going out of bounds
                self.graphwise_mlp = nn.Sequential(
                    nn.Linear(graph_in_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 8),
                    nn.ReLU(),
                    nn.Linear(8, 2),
                    nn.GLU(),
                )

        elif args["task"] == "node_binary":  # {intent/receiver}_{scoring/conceding}
            out_dim = 1 if args["target"].startswith("receiver") else 2
            self.nodewise_mlp = nn.Sequential(
                nn.Linear(node_in_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, out_dim),
                # nn.Sigmoid(),
            )
            if args["residual"]:  # To estimate probabilities for the ball going out of bounds
                self.graphwise_mlp = nn.Sequential(
                    nn.Linear(graph_in_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 8),
                    nn.ReLU(),
                    nn.Linear(8, out_dim),
                    # nn.Sigmoid(),
                )

        elif args["task"] == "graph_binary":
            # intent_success, scoring/conceding, dest_{success/scoring/conceding}, shot_blocking
            out_dim = 2 if args["target"] in ["dest_scoring", "dest_conceding"] else 1
            self.graphwise_mlp = nn.Sequential(
                nn.Linear(graph_in_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, out_dim),
                # nn.Sigmoid(),
            )

    def forward(
        self,
        node_features: torch.FloatTensor,
        node_embeddings: torch.FloatTensor,
        graph_embeddings: torch.FloatTensor,
        batch_indices: torch.LongTensor = None,
        batch_dests: torch.FloatTensor = None,
    ) -> torch.Tensor:
        """
        Arguments:
            node_features: a float tensor with shape [N, x].
            node_embeddings: a float tensor with shape [N, z].
            graph_embeddings: a float tensor with shape [B, z] or None.
            batch_indices: a long tensor with shape [N,] indicating which batch each node belongs to.
            batch_dests: a float tensor with shape [B, 2] or None.
        Returns:
            if target in ["receiver", "dest_receiver", "scoring", "conceding"]:
                out: a float tensor with shape [N + B,] if residual else [N,].
            elif target in ["dest_scoring", "dest_conceding"]:
                out: a float tensor with shape [B,].
        """

        if self.args["task"].startswith("node"):  # node_selection, node_binary
            node_inputs = torch.cat([node_features, node_embeddings], -1) if self.args["skip_conn"] else node_embeddings

            if self.args["target"] == "dest_receiver":
                assert batch_dests is not None and batch_indices is not None
                nodewise_dests = batch_dests[batch_indices]
                node_inputs = torch.cat([node_inputs, nodewise_dests], -1)  # [N, x + z + 2] or [N, z + 2]

            node_outputs = self.nodewise_mlp(node_inputs)  # [N, 1] or [N, 2]
            if node_outputs.shape[-1] == 1:
                node_outputs = node_outputs.squeeze(-1)  # [N,]

            if not self.args["residual"]:
                return node_outputs

            else:
                assert graph_embeddings is not None
                if self.args["target"] == "dest_receiver":
                    residual_inputs = torch.cat([graph_embeddings, batch_dests], -1)  # [B, z + 2]
                else:
                    residual_inputs = graph_embeddings  # [B, z]
                residual_outputs = self.graphwise_mlp(residual_inputs).squeeze(-1)  # [B,]
                return torch.cat([node_outputs, residual_outputs], -1)  # [N + B,]

        else:  # graph_binary
            assert graph_embeddings is not None

            if self.args["target"].startswith("dest"):
                graph_inputs = torch.cat([graph_embeddings, batch_dests], -1)  # [B, z + 2]
                return self.graphwise_mlp(graph_inputs)  # [B, 2]

            else:  # intent_success, shot_blocking
                return self.graphwise_mlp(graph_embeddings).squeeze(-1)  # [B,]


class GAT(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, batch_graphs: Batch, batch_dests: torch.Tensor = None) -> torch.Tensor:
        """
        Arguments:
            batch_graphs: a DataBatch object with B graphs and N nodes.
            batch_dests: a float tensor with shape [B, 2] or None.
        Returns:
            out: a float tensor with shape [N+B,] if residual else [N,].
        """
        node_embeddings, graph_embeddings = self.encoder(batch_graphs)
        return self.decoder(batch_graphs.x, node_embeddings, graph_embeddings, batch_graphs.batch, batch_dests)
