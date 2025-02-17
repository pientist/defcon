import argparse
import json
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Batch
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
        if key == "count":
            continue
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


def load_model(model_id="intent/01", device="cuda") -> GAT:
    if model_id is None:
        return None

    else:
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
                likelihoods.append(probs[int(batch_labels[graph_index, 5].item())])

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


def calc_binary_metrics(y, y_hat, threshold=0.5):
    y_pred = y_hat > threshold
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


def run_epoch(
    args: argparse.Namespace,
    model: nn.DataParallel,
    loader: DataLoader,
    optimizer: torch.optim.Adam = None,
    device: str = "cuda",
    pos_weight: float = 1,
    train: bool = False,
):
    # torch.autograd.set_detect_anomaly(True)
    model.train() if train else model.eval()
    n_batches = len(loader)

    if "binary" in args.task:  # node_binary, graph_binary
        metrics = {"count": 0, "ce_loss": 0, "l1_loss": 0, "f1": 0, "roc_auc": 0, "brier": 0}
    else:  # node_selection
        metrics = {"count": 0, "ce_loss": 0, "l1_loss": 0, "accuracy": 0}

    for batch_index, (batch_graphs, batch_labels) in enumerate(loader):
        batch_graphs: Batch = batch_graphs.to(device)
        index_range = torch.unique(batch_graphs.batch)

        metrics["count"] += batch_graphs.num_graphs

        if args.residual:
            batch = torch.cat([batch_graphs.batch, index_range])
            # One node per player and one residual node per graph instance
        else:
            batch = batch_graphs.batch

        batch_labels: torch.Tensor = batch_labels.to(device)
        batch_labels[batch_labels[:, 6] == -1, 6] = batch_labels[batch_labels[:, 6] == -1, 4]  # -1 to n_players
        if args.target.startswith("dest"):
            batch_dests = batch_labels[:, 8:10]  # if args.use_intent else batch_labels[:, 6:8]
        else:
            batch_dests = None

        if train:
            out: torch.Tensor = model(batch_graphs, batch_dests)
        else:
            with torch.no_grad():
                out: torch.Tensor = model(batch_graphs, batch_dests)

        if args.task == "node_selection":  # intent, receiver, {intent/dest/failed_pass/blocked_shot}_receiver
            if args.target.endswith("intent"):
                target = batch_labels[:, 5].clone().long()
            elif args.target.endswith("receiver"):
                target = batch_labels[:, 6].clone().long()

            loss_fn = nn.CrossEntropyLoss()
            ce_loss = 0
            accuracy = 0

            for graph_index in index_range:
                if args.target in ["intent", "oppo_agn_intent"]:
                    assert not args.residual
                    pred_i = out[(batch == graph_index) & (batch_graphs.x[:, 0] == 1)]
                    target_i = target[graph_index]

                elif args.target in ["failed_pass_receiver", "blocked_shot_receiver", "failure_receiver"]:
                    if args.residual:
                        residual_flags = torch.ones(batch_graphs.num_graphs).bool().to(device)
                        failure_flags = torch.cat([batch_graphs.x[:, 0] == 0, residual_flags])
                    else:
                        failure_flags = batch_graphs.x[:, 0] == 0
                    pred_i = out[(batch == graph_index) & failure_flags]
                    n_teammates = ((batch_graphs.batch == graph_index) & (batch_graphs.x[:, 0] == 1)).sum()
                    target_i = target[graph_index] - n_teammates

                else:  # receiver, intent_receiver, dest_receiver
                    pred_i = out[batch == graph_index]
                    target_i = target[graph_index]

                ce_loss += loss_fn(pred_i.unsqueeze(0), target_i.unsqueeze(0))
                accuracy += (torch.argmax(pred_i) == target_i).float()

            ce_loss /= index_range.shape[0]
            metrics["accuracy"] += accuracy.item()

        elif args.task == "node_binary":  # {intent/receiver}_{scoring/conceding}
            pred = []

            if args.target.startswith("intent_"):
                intent = batch_labels[:, 5].clone().long()
                success = batch_labels[:, -5].clone().long()
                for graph_index in index_range:
                    pred.append(out[batch == graph_index][intent[graph_index], success[graph_index]])

            elif args.target.startswith("receiver_"):
                receiver = batch_labels[:, 6].clone().long()
                for graph_index in index_range:
                    pred.append(out[batch == graph_index][receiver[graph_index]])

            pred = torch.stack(pred).unsqueeze(0)

            if args.target.endswith("scoring"):
                target = batch_labels[:, -3].unsqueeze(0) if args.use_xg else batch_labels[:, -4].unsqueeze(0)
            elif args.target.endswith("conceding"):
                target = batch_labels[:, -1].unsqueeze(0) if args.use_xg else batch_labels[:, -2].unsqueeze(0)

            ce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))(pred, target)

            y_hat = nn.Sigmoid()(pred.squeeze()).cpu().detach().numpy()
            y = batch_labels[:, -4] if args.target.endswith("scoring") else batch_labels[:, -2]
            y = y.cpu().detach().numpy()

            threshold = 0.1 if args.target.split("_")[-1] in ["scoring", "conceding"] else 0.5
            batch_metrics = calc_binary_metrics(y, y_hat, threshold)
            metrics["f1"] += batch_metrics["f1"] * batch_graphs.num_graphs
            metrics["roc_auc"] += batch_metrics["roc_auc"] * batch_graphs.num_graphs
            metrics["brier"] += batch_metrics["brier"] * batch_graphs.num_graphs

        elif args.task == "graph_binary":
            # intent_success, scoring/conceding, and dest_{success/scoring/conceding}, shot_blocking
            if args.target in ["intent_success", "dest_success", "shot_blocking"]:
                target = batch_labels[:, -5] if args.target.endswith("success") else batch_labels[:, -6]
                ce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))(out, target)

                y_hat = nn.Sigmoid()(out).cpu().detach().numpy()
                y = target.cpu().detach().numpy()

            elif args.target in ["scoring", "conceding", "dest_scoring", "dest_conceding"]:
                if args.target.startswith("dest"):
                    success = batch_labels[:, -5].clone().long()
                    pred = out[tuple([list(range(batch_graphs.num_graphs)), success])]
                else:
                    pred = out

                if args.target.endswith("scoring"):
                    target = batch_labels[:, -3] if args.use_xg else batch_labels[:, -4]
                elif args.target.endswith("conceding"):
                    target = batch_labels[:, -1] if args.use_xg else batch_labels[:, -2]

                ce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))(out, target)

                y_hat = nn.Sigmoid()(pred).cpu().detach().numpy()
                y = batch_labels[:, -4] if args.target.endswith("scoring") else batch_labels[:, -2]
                y = y.cpu().detach().numpy()

            threshold = 0.1 if args.target.split("_")[-1] in ["scoring", "conceding"] else 0.5
            batch_metrics = calc_binary_metrics(y, y_hat, threshold)
            metrics["f1"] += batch_metrics["f1"] * batch_graphs.num_graphs
            metrics["roc_auc"] += batch_metrics["roc_auc"] * batch_graphs.num_graphs
            metrics["brier"] += batch_metrics["brier"] * batch_graphs.num_graphs

        l1_loss = l1_regularizer(model, lambda_l1=args.lambda_l1)
        metrics["ce_loss"] += ce_loss.item() * batch_graphs.num_graphs
        metrics["l1_loss"] += l1_loss.item() * batch_graphs.num_graphs

        if train:
            optimizer.zero_grad()
            loss = ce_loss + l1_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters(), args.clip)
            optimizer.step()

        if train and batch_index % args.print_freq == 0:
            interim_metrics = dict()
            for key, value in metrics.items():
                if key == "count":
                    continue
                interim_metrics[key] = value / metrics["count"]
            print(f"[{batch_index:>{len(str(n_batches))}d}/{n_batches}]  {get_losses_str(interim_metrics)}")

    for key, value in metrics.items():
        if key == "count":
            continue
        metrics[key] = value / metrics["count"]

    return metrics
