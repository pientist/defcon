import argparse
import json
from typing import List

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import Delaunay
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.gat import GAT


def num_trainable_params(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        count = 1
        for s in p.size():
            count *= s
        total += count
    return total


def parse_model_params(model_arg_keys: List[str], args_dict: dict, parser: argparse.ArgumentParser):
    if parser is None:
        return args_dict

    for key in model_arg_keys:
        if key.startswith("n_") or key.endswith("_dim"):
            parser.add_argument("--" + key, type=int, required=True)
        elif key == "dropout":
            parser.add_argument("--" + key, type=float, default=0)
        else:
            parser.add_argument("--" + key, action="store_true", default=False)
    model_args, _ = parser.parse_known_args()

    for key in model_arg_keys:
        args_dict[key] = getattr(model_args, key)

    return args_dict


def get_args_str(keys, args_dict: dict) -> str:
    ret = ""
    for key in keys:
        if key in args_dict:
            ret += " {} {} |".format(key, args_dict[key])
    return ret[1:-2]


def get_losses_str(losses: dict) -> str:
    ret = ""
    for key, value in losses.items():
        ret += " {}: {:.4f} |".format(key, np.mean(value))
    # if len(losses) > 1:
    #     ret += " total_loss: {:.4f} |".format(sum(losses.values()))
    return ret[:-2]


def printlog(line: str, trial_path: str) -> None:
    print(line)
    with open(trial_path + "/log.txt", "a") as file:
        file.write(line + "\n")


def l1_regularizer(model, lambda_l1=0.1):
    l1_loss = 0
    for model_param_name, model_param_value in model.named_parameters():
        if model_param_name.endswith("weight"):
            l1_loss += lambda_l1 * model_param_value.abs().sum()
    return l1_loss


def encode_onehot(labels, classes=None):
    if classes:
        classes = [x for x in range(classes)]
    else:
        classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def calc_speed(xy):
    x = xy[:, :, 0]
    y = xy[:, :, 1]
    vx = torch.diff(x, prepend=x[:, [0]]) / 0.1
    vy = torch.diff(y, prepend=y[:, [0]]) / 0.1
    speed = torch.sqrt(vx**2 + vy**2 + torch.tensor(1e-6).to(xy.device))
    return torch.stack([x, y, speed], -1)


def sparsify_edges(graph: Data, how="distance", passer_index: int = None, max_dist=10) -> Data:
    if how == "distance":
        edge_index = graph.edge_index
        if passer_index is not None:
            passer_edges = (edge_index[0] == passer_index) | (edge_index[1] == passer_index)
        close_edges = graph.edge_attr[:, 0] <= max_dist

        graph.edge_index = edge_index[:, passer_edges | close_edges]
        graph.edge_attr = graph.edge_attr[passer_edges | close_edges]

    elif how == "delaunay":
        xy = graph.x[:, 1:3] if graph.x.shape[1] == 16 else graph.x[:, 3:5]
        tri_pts = Delaunay(xy.cpu().detach().numpy()).simplices
        tri_edges = np.concatenate((tri_pts[:, :2], tri_pts[:, 1:], tri_pts[:, ::2]), axis=0)
        tri_edges = np.unique(tri_edges, axis=0).tolist()

        for [i, j] in tri_edges:
            if [j, i] not in tri_edges:
                tri_edges.append([j, i])

        complete_edges = graph.edge_index.cpu().detach().numpy().T
        complete_edge_dict = {tuple(e): i for i, e in enumerate(complete_edges)}
        tri_edge_index = np.sort([complete_edge_dict[tuple(e)] for e in tri_edges]).tolist()

        graph.edge_index = graph.edge_index[:, tri_edge_index]
        graph.edge_attr = graph.edge_attr[tri_edge_index]

    return graph


def load_model(model_id="intent/01", device="cuda") -> GAT:
    model_path = f"saved/{model_id}"
    with open(f"{model_path}/args.json", "r") as f:
        args = json.load(f)

    model = GAT(args).to(device)
    weights_path = f"{model_path}/best_weights.pt"
    state_dict = torch.load(weights_path, weights_only=False, map_location=lambda storage, _: storage)
    model.load_state_dict(state_dict)

    return model


def estimate_likelihoods(dataset, model_id="intent/01", device="cuda", min_clip=0.01) -> torch.Tensor:
    model = load_model(model_id, device)
    loader = DataLoader(dataset, batch_size=2056, shuffle=False, pin_memory=True)
    likelihoods = []

    for batch_graphs, batch_labels in tqdm(loader, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            batch_graphs.x = batch_graphs.x[:, : model.args["node_in_dim"]]
            out: torch.Tensor = model(batch_graphs)
            for graph_index in range(batch_graphs.num_graphs):
                logits = out[(batch_graphs.batch == graph_index) & (batch_graphs.x[:, 0] == 1)]
                probs = nn.Softmax(dim=0)(logits).cpu().detach().numpy()
                likelihoods.append(probs[int(batch_labels[graph_index, 2].item())])

    return torch.Tensor(likelihoods).clip(min_clip)


def calc_pos_error(pred_xy, target_xy, aggfunc="mean"):
    if aggfunc == "mean":
        return torch.norm(pred_xy - target_xy, dim=-1).mean().item()
    else:  # if aggfunc == "sum":
        return torch.norm(pred_xy - target_xy, dim=-1).sum().item()


def calc_class_accuracy(y, y_hat, aggfunc="mean"):
    if aggfunc == "mean":
        return (torch.argmax(y_hat, dim=1) == y).float().mean().item()
    else:  # if aggfunc == "sum":
        return (torch.argmax(y_hat, dim=1) == y).float().sum().item()


def calc_binary_metrics(y, y_hat):
    y_pred = y_hat > 0.5
    precision = precision_score(y, y_pred) if np.sum(y_pred) > 0 else 0
    recall = recall_score(y, y_pred) if np.sum(y) > 0 else 0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(y, y_pred) if precision > 0 and recall > 0 else 0,
        "roc_auc": roc_auc_score(y, y_hat) if np.sum(y) > 0 else 0.5,
        "brier": brier_score_loss(y, y_hat),
        "log_loss": log_loss(y, y_hat) if np.sum(y) > 0 else np.nan,
    }
    return {k: round(v, 4) for k, v in metrics.items()}
