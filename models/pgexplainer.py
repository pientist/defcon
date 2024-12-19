import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.explain.config import ExplanationType, ModelMode, ModelTaskLevel
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.utils import get_embeddings, subgraph

from models.gat import GAT


class PGExplainer(ExplainerAlgorithm):
    r"""The PGExplainer model from the `"Parameterized Explainer for Graph
    Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper.

    Internally, it utilizes a neural network to identify subgraph structures
    that play a crucial role in the predictions made by a GNN.
    Importantly, the :class:`PGExplainer` needs to be trained via
    :meth:`~PGExplainer.train` before being able to generate explanations:

    .. code-block:: python

        explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=30, lr=0.003),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=ModelConfig(...),
        )

        # Train against a variety of node-level or graph-level predictions:
        for epoch in range(30):
            for index in [...]:  # Indices to train against.
                loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                                 target=target, index=index)

        # Get the final explanations:
        explanation = explainer(x, edge_index, target=target, index=0)

    Args:
        epochs (int): The number of epochs to train.
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.003`).
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.PGExplainer.coeffs`.
    """

    coeffs = {
        "edge_size": 0.05,
        "edge_ent": 1.0,
        "temp": [5.0, 2.0],
        "bias": 0.01,
    }

    def __init__(self, epochs: int, lr: float = 0.003, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)

        self.mlp = nn.Sequential(Linear(-1, 64), nn.ReLU(), Linear(64, 1))
        self.optimizer = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self._curr_epoch = -1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.mlp)

    def train(self, epoch: int, model: GAT, batch_graphs: Batch, *, target: torch.Tensor, **kwargs):
        r"""Trains the underlying explainer model.
        Needs to be called before being able to make predictions.

        Args:
            epoch (int): The current epoch of the training phase.
            model (torch.nn.Module): The model to explain.
            batch (torch_geometric.data.Batch): The input batch consisting of homogeneous graphs.
            target (torch.Tensor): The target of the model.
            index (int or torch.Tensor, optional): The index of the model output to explain.
                Needs to be a single index. (default: :obj:`None`)
            **kwargs (optional): Additional keyword arguments passed to :obj:`model`.
        """
        self.optimizer.zero_grad()
        temperature = self._get_temperature(epoch)

        index_range = torch.unique(batch_graphs.batch)
        batch = torch.cat([batch_graphs.batch, index_range])

        node_embeddings, graph_embeddings = model.encoder(batch_graphs)  # [N, z], [B, z]
        out = model.decoder(batch_graphs.x, node_embeddings, graph_embeddings, batch_graphs.batch)  # [N + B,]
        embeddings = torch.cat([node_embeddings, graph_embeddings], dim=0)  # [N + B, z]

        batch_loss = 0

        for graph_index in index_range:
            node_mask = batch_graphs.batch == graph_index
            x = batch_graphs.x[node_mask]  # [N_i, x]
            edge_index, edge_attr = subgraph(node_mask, batch_graphs.edge_index, relabel_nodes=True)
            graph = Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_attr)])

            z = embeddings[batch == graph_index]  # [N_i + 1, z]
            y_graph = F.softmax(out[batch == graph_index], dim=-1)  # [N_i + 1,]

            graph_loss = 0

            for class_index in range(y_graph.shape[0]):
                edge_embeddings = self._get_inputs(z, edge_index, class_index)
                edge_logits = self.mlp(edge_embeddings).view(-1)
                edge_mask = self._concrete_sample(edge_logits, temperature)
                set_masks(model, edge_mask, edge_index, apply_sigmoid=True)

                # if self.model_config.task_level == ModelTaskLevel.node:
                #     _, hard_edge_mask = self._get_hard_masks(model, index, edge_index, num_nodes=x.size(0))
                #     edge_mask = edge_mask[hard_edge_mask]

                y_subgraph = model(graph, **kwargs)
                class_ce_loss = -y_graph[class_index] * torch.log_softmax(y_subgraph, dim=-1)[class_index]
                class_reg_loss = self._regularization_loss(edge_mask)
                graph_loss += class_ce_loss + class_reg_loss
                clear_masks(model)

            batch_loss += graph_loss / y_graph.shape[0]

        batch_loss /= batch_graphs.num_graphs
        batch_loss.backward()
        self.optimizer.step()

        self._curr_epoch = epoch

        return float(batch_loss)

    def forward(self, model: GAT, graph: Data, class_index: int) -> Explanation:
        # hard_edge_mask = None
        # if self.model_config.task_level == ModelTaskLevel.node:
        #     if index is None:
        #         raise ValueError(
        #             f"The 'index' argument needs to be provided "
        #             f"in '{self.__class__.__name__}' for "
        #             f"node-level explanations"
        #         )
        #     if isinstance(index, torch.Tensor) and index.numel() > 1:
        #     raise ValueError(f"Only scalars are supported for the 'index' argument in '{self.__class__.__name__}'")

        #     # We need to compute hard masks to properly clean up edges and
        #     # nodes attributions not involved during message passing:
        #     _, hard_edge_mask = self._get_hard_masks(model, index, edge_index, num_nodes=x.size(0))

        graph = Batch.from_data_list([graph])  # [N, x]
        node_embeddings, graph_embeddings = model.encoder(graph)  # [N, z], [1, z]
        embeddings = torch.cat([node_embeddings, graph_embeddings], dim=0)  # [N + 1, z]

        # inputs = self._get_inputs(z, edge_index, index)
        edge_embeddings = self._get_inputs(embeddings, graph.edge_index, class_index)
        edge_logits = self.mlp(edge_embeddings).view(-1)

        # edge_mask = self._post_process_mask(logits, hard_edge_mask, apply_sigmoid=True)
        edge_mask = self._post_process_mask(edge_logits, apply_sigmoid=True)

        return Explanation(edge_mask=edge_mask)

    def supports(self) -> bool:
        explanation_type = self.explainer_config.explanation_type
        if explanation_type != ExplanationType.phenomenon:
            logging.error(
                f"'{self.__class__.__name__}' only supports "
                f"phenomenon explanations "
                f"got (`explanation_type={explanation_type.value}`)"
            )
            return False

        task_level = self.model_config.task_level
        if task_level not in {ModelTaskLevel.node, ModelTaskLevel.graph}:
            logging.error(
                f"'{self.__class__.__name__}' only supports "
                f"node-level or graph-level explanations "
                f"got (`task_level={task_level.value}`)"
            )
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(
                f"'{self.__class__.__name__}' does not support "
                f"explaining input node features "
                f"got (`node_mask_type={node_mask_type.value}`)"
            )
            return False

        return True

    def _get_inputs(self, embedding: torch.Tensor, edge_index: torch.Tensor, index: Optional[int]) -> torch.Tensor:
        zs = [embedding[edge_index[0]], embedding[edge_index[1]]]
        zs.append(embedding[index].view(1, -1).repeat(zs[0].size(0), 1))
        return torch.cat(zs, dim=-1)

    def _get_temperature(self, epoch: int) -> float:
        temp = self.coeffs["temp"]
        return temp[0] * pow(temp[1] / temp[0], epoch / self.epochs)

    def _concrete_sample(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        bias = self.coeffs["bias"]
        eps = (1 - 2 * bias) * torch.rand_like(logits) + bias
        return (eps.log() - (1 - eps).log() + logits) / temperature

    def _regularization_loss(self, edge_mask: torch.Tensor) -> torch.Tensor:
        # if self.model_config.mode == ModelMode.binary_classification:
        #     loss = self._loss_binary_classification(y_hat, y)
        # elif self.model_config.mode == ModelMode.multiclass_classification:
        #     loss = self._loss_multiclass_classification(y_hat, y)
        # elif self.model_config.mode == ModelMode.regression:
        #     loss = self._loss_regression(y_hat, y)

        # Regularization loss:
        mask = edge_mask.sigmoid()
        size_loss = mask.sum() * self.coeffs["edge_size"]
        mask = 0.99 * mask + 0.005
        mask_ent = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent.mean() * self.coeffs["edge_ent"]

        return size_loss + mask_ent_loss  # + loss
