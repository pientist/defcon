import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from dataset import ActionDataset
from models.gat import GAT
from utils import (
    calc_binary_metrics,
    calc_class_accuracy,
    estimate_likelihoods,
    get_args_str,
    get_losses_str,
    l1_regularizer,
    num_trainable_params,
    printlog,
)

# Modified from https://github.com/ezhan94/multiagent-programmatic-supervision/blob/master/train.py


# For one epoch
def run_epoch(args, model: nn.DataParallel, optimizer: torch.optim.Adam, loader: DataLoader, train=False):
    # torch.autograd.set_detect_anomaly(True)
    model.train() if train else model.eval()
    n_batches = len(loader)

    if "binary" in args.task:  # node_binary, graph_binary
        metrics = {"ce_loss": [], "l1_loss": [], "f1": [], "roc_auc": [], "brier": []}
    else:  # node_selection
        metrics = {"ce_loss": [], "l1_loss": [], "accuracy": []}

    for batch_index, (batch_graphs, batch_labels) in enumerate(loader):
        batch_graphs: Batch = batch_graphs.to(device)
        index_range = torch.unique(batch_graphs.batch)

        if args.residual:
            batch = torch.cat([batch_graphs.batch, index_range])
            # One node per player and one residual node per graph instance
        else:
            batch = batch_graphs.batch

        batch_labels: torch.Tensor = batch_labels.to(device)
        batch_labels[batch_labels[:, 3] == -1, 3] = batch_labels[batch_labels[:, 3] == -1, 1]  # -1 to n_players
        if args.target.startswith("dest"):
            batch_dests = batch_labels[:, 4:6] if args.use_intent else batch_labels[:, 6:8]
        else:
            batch_dests = None

        if train:
            out: torch.Tensor = model(batch_graphs, batch_dests)
        else:
            with torch.no_grad():
                out: torch.Tensor = model(batch_graphs, batch_dests)

        if args.task == "node_selection":  # intent, receiver, {intent/failure/dest}_receiver
            if args.target.endswith("intent"):
                target = batch_labels[:, 2].clone().long()
            elif args.target.endswith("receiver"):
                target = batch_labels[:, 3].clone().long()

            loss_fn = nn.CrossEntropyLoss()
            ce_loss = 0
            accuracy = 0

            for graph_index in index_range:
                if args.target == "intent":
                    assert not args.residual
                    pred_i = out[(batch == graph_index) & (batch_graphs.x[:, 0] == 1)]
                    target_i = target[graph_index]

                elif args.target == "failure_receiver":
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
            metrics["accuracy"] += [accuracy.item() / index_range.shape[0]]

        elif args.task == "node_binary":  # {intent/receiver}_{scoring/conceding}
            pred = []

            if args.target.startswith("intent"):
                intent = batch_labels[:, 2].clone().long()
                success = batch_labels[:, -5].clone().long()
                for graph_index in index_range:
                    pred.append(out[batch == graph_index][intent[graph_index], success[graph_index]])

            elif args.target.startswith("receiver"):
                receiver = batch_labels[:, 3].clone().long()
                for graph_index in index_range:
                    pred.append(out[batch == graph_index][receiver[graph_index]])

            pred = torch.stack(pred)

            if args.target.endswith("scoring"):
                target = batch_labels[:, -3] if args.use_xg else batch_labels[:, -4]
            elif args.target.endswith("conceding"):
                target = batch_labels[:, -1] if args.use_xg else batch_labels[:, -2]

            ce_loss = nn.BCELoss()(pred.unsqueeze(0), target.unsqueeze(0))

            y_hat = pred.cpu().detach().numpy()
            y = batch_labels[:, -4] if args.target.endswith("scoring") else batch_labels[:, -2]
            y = y.cpu().detach().numpy()
            batch_metrics = calc_binary_metrics(y, y_hat)

            metrics["f1"] += [batch_metrics["f1"]]
            metrics["roc_auc"] += [batch_metrics["roc_auc"]]
            metrics["brier"] += [batch_metrics["brier"]]

        elif args.task == "graph_binary":  # intent_success and dest_{success/scoring/conceding}
            if args.target.endswith("success"):  # {intent/dest}_success
                target = batch_labels[:, -5]
                ce_loss = nn.BCELoss()(out, target)

                y_hat = out.cpu().detach().numpy()
                y = target.cpu().detach().numpy()

            else:  # dest_{scoring/conceding}
                success = batch_labels[:, -5].clone().long()
                pred = out[tuple([list(range(len(batch_labels))), success])]

                if args.target.endswith("scoring"):
                    target = batch_labels[:, -3] if args.use_xg else batch_labels[:, -4]
                elif args.target.endswith("conceding"):
                    target = batch_labels[:, -1] if args.use_xg else batch_labels[:, -2]

                ce_loss = nn.BCELoss()(pred, target)

                y_hat = pred.cpu().detach().numpy()
                y = batch_labels[:, -4] if args.target.endswith("scoring") else batch_labels[:, -2]
                y = y.cpu().detach().numpy()

            batch_metrics = calc_binary_metrics(y, y_hat)
            metrics["f1"] += [batch_metrics["f1"]]
            metrics["roc_auc"] += [batch_metrics["roc_auc"]]
            metrics["brier"] += [batch_metrics["brier"]]

        l1_loss = l1_regularizer(model, lambda_l1=args.lambda_l1)
        metrics["ce_loss"] += [ce_loss.item()]
        metrics["l1_loss"] += [l1_loss.item()]

        if train:
            optimizer.zero_grad()
            loss = ce_loss + l1_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.module.parameters(), args.clip)
            optimizer.step()

        if train and batch_index % args.print_freq == 0:
            print(f"[{batch_index:>{len(str(n_batches))}d}/{n_batches}]  {get_losses_str(metrics)}")

    for key, value in metrics.items():
        metrics[key] = np.mean(value)  # /= len(loader.dataset)

    return metrics


# Main starts here
parser = argparse.ArgumentParser()

parser.add_argument("--trial", type=int, required=True)
parser.add_argument("--model", type=str, required=True, default="gat")
parser.add_argument("--target", type=str, required=True, help="intent, receiver, success, scoring, conceding")

parser.add_argument("--pass_only", action="store_true", default=False, help="not include ball carries")
parser.add_argument("--xy_only", action="store_true", default=False, help="only use xy locations as features")
parser.add_argument("--passer_aware", action="store_true", default=False, help="use passer features")
parser.add_argument("--one_touch_aware", action="store_true", default=False, help="use one-touch flags as a feature")
parser.add_argument("--use_intent", action="store_true", default=False, help="use intended receivers' locations")
parser.add_argument("--use_xg", action="store_true", default=False, help="use xG instead of actual goal labels")
parser.add_argument("--residual", action="store_true", default=False, help="attach a component for ball out of play")
parser.add_argument("--sparsify", type=str, choices=["distance", "delaunay", "none"], help="how to filter edges")
parser.add_argument("--max_edge_dist", type=int, default=10, help="max distance between off-ball nodes")
parser.add_argument("--oversampling", type=str, default="none", help="model ID to obtain oversampling probabilities")
# parser.add_argument("--seq_len", type=int, required=False, default=30, help="num recent frames to observe")

parser.add_argument("--edge_in_dim", type=int, required=False, default=0, help="num edge features")
parser.add_argument("--node_emb_dim", type=int, required=False, default=0, help="node embedding dim")
parser.add_argument("--graph_emb_dim", type=int, required=False, default=0, help="graph embedding dim")
parser.add_argument("--gnn_layers", type=int, required=False, default=0, help="num GNN layers")
parser.add_argument("--gnn_heads", type=int, required=False, default=0, help="num heads of GNN layers")
# parser.add_argument("--rnn_dim", type=int, required=True, help="RNN hidden dim")
# parser.add_argument("--rnn_layers", type=int, required=False, default=2, help="num layers of RNN")
parser.add_argument("--dropout", type=float, required=False, default=0, help="dropout prob")
parser.add_argument("--skip_conn", action="store_true", default=False, help="adopt skip-connection")

parser.add_argument("--n_epochs", type=int, required=False, default=200, help="num epochs")
parser.add_argument("--batch_size", type=int, required=False, default=32, help="batch size")
parser.add_argument("--lambda_l1", type=float, required=False, default=0, help="coeff of L1 regularizer")
parser.add_argument("--start_lr", type=float, required=False, default=0.0001, help="starting learning rate")
parser.add_argument("--min_lr", type=float, required=False, default=0.0001, help="minimum learning rate")
parser.add_argument("--clip", type=int, required=False, default=10, help="gradient clipping")
parser.add_argument("--print_freq", type=int, required=False, default=50, help="periodically print performance")
parser.add_argument("--seed", type=int, required=False, default=128, help="PyTorch random seed")

parser.add_argument("--cont", action="store_true", default=False, help="continue training previous best model")
parser.add_argument("--best_loss", type=float, required=False, default=0, help="best loss")
parser.add_argument("--best_acc", type=float, required=False, default=0, help="best accuracy")

args, _ = parser.parse_known_args()


if __name__ == "__main__":
    args.cuda = torch.cuda.is_available()
    device = "cuda:0" if args.cuda else "cpu"

    # Set manual seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.xy_only:
        args.node_in_dim = 8 if args.passer_aware else 7
    else:
        args.node_in_dim = 18 if args.passer_aware else 12

    if not args.one_touch_aware:
        args.node_in_dim -= 2

    intent_aware = args.target in ["intent_success", "failure_receiver"]
    if intent_aware:
        args.node_in_dim += 1

    if args.target in ["intent", "receiver", "intent_receiver", "failure_receiver", "dest_receiver"]:
        args.task = "node_selection"
    elif args.target in ["intent_scoring", "intent_conceding", "receiver_scoring", "receiver_conceding"]:
        args.task = "node_binary"
    elif args.target in ["intent_success", "dest_success", "dest_scoring", "dest_conceding"]:
        args.task = "graph_binary"
    # elif args.target.startswith("intent_") and args.target.split("_")[1] in ["success", "scoring", "conceding"]:
    #     args.task = "graph_binary"

    # Parameters to save
    args_dict = {
        "model": args.model,
        "target": args.target,
        "task": args.task,
        "pass_only": args.pass_only,
        "xy_only": args.xy_only,
        "passer_aware": args.passer_aware,
        "one_touch_aware": args.one_touch_aware,
        "use_intent": args.use_intent,
        "use_xg": args.use_xg,
        "residual": args.residual,
        "sparsify": args.sparsify,
        "max_edge_dist": args.max_edge_dist,
        "oversampling": args.oversampling,
        # "seq_len": args.seq_len,
        "node_in_dim": args.node_in_dim,
        "edge_in_dim": args.edge_in_dim,
        "node_emb_dim": args.node_emb_dim,
        "graph_emb_dim": args.graph_emb_dim,
        "gnn_layers": args.gnn_layers,
        "gnn_heads": args.gnn_heads,
        # "rnn_dim": args.rnn_dim,
        # "rnn_layers": args.rnn_layers,
        "skip_conn": args.skip_conn,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lambda_l1": args.lambda_l1,
        "start_lr": args.start_lr,
        "min_lr": args.min_lr,
        "seed": args.seed,
        "best_loss": args.best_loss,
        "best_acc": args.best_acc,
    }

    # Load model
    model = GAT(args_dict).to(device)
    model = nn.DataParallel(model)

    # Update params with model parameters
    args_dict["total_params"] = num_trainable_params(model)

    # Create save path and saving parameters
    # if args.dataset in ["success", "failure"]:
    #     target_type = f"{args.target}_if_{args.dataset}"

    trial_path = f"saved/{args.target}/{args.trial:02d}"
    os.makedirs(trial_path, exist_ok=True)
    with open(f"{trial_path}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    # Continue a previous experiment, or start a new one
    if args.cont:
        state_dict = torch.load(f"{trial_path}/best_weights.pt", weights_only=False)
        model.module.load_state_dict(state_dict)
    else:
        title = f"{args.target} {args.trial:02d} | {args.task} {args.model}"
        dataset_arg_keys = ["pass_only", "passer_aware", "one_touch_aware", "sparsify", "oversampling"]
        model_arg_keys = ["node_in_dim", "node_emb_dim", "gnn_heads", "skip_conn", "total_params"]
        train_arg_keys = ["batch_size", "lambda_l1", "start_lr"]

        printlog(title, trial_path)
        printlog(get_args_str(dataset_arg_keys, args_dict), trial_path)
        printlog(get_args_str(model_arg_keys, args_dict), trial_path)
        printlog(get_args_str(train_arg_keys, args_dict), trial_path)
        # printlog("n_params {:,}".format(args_dict["total_params"]), trial_path)

    printlog("############################################################", trial_path)

    print("\nGenerating datasets...")

    train_match_ids = [f.split(".")[0] for f in np.sort(os.listdir("data/train_features"))][:160]
    valid_match_ids = [f.split(".")[0] for f in np.sort(os.listdir("data/train_features"))][160:]
    include_ball_outs = (args.target.endswith("receiver") and args.residual) or (args.target.endswith("blocking"))

    dataset_args = {
        "split": "train",
        "pass_only": args.pass_only,
        "intended_only": "intent" in args.target or args.target.startswith("failure_"),
        "inplay_only": not include_ball_outs,
        "xy_only": args.xy_only,
        "passer_aware": args.passer_aware,
        "one_touch_aware": args.one_touch_aware,
        "intent_aware": intent_aware,
        "sparsify": args.sparsify,
        "max_edge_dist": args.max_edge_dist,
    }
    if args.target.startswith("success_") or args.target.startswith("failure_"):
        dataset_args["outcome"] = args.target.split("_")[0]  # success or failure

    train_dataset = ActionDataset(train_match_ids, **dataset_args)
    valid_dataset = ActionDataset(valid_match_ids, **dataset_args)

    loader_args = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 16, "pin_memory": True}
    if args.oversampling != "none":
        print("\nCalculating oversampling probabilities...")
        scores = 1 / estimate_likelihoods(train_dataset, model_id="intent/01", device=device)
        sampling_probs = scores / scores.sum()
        sampler = WeightedRandomSampler(sampling_probs, num_samples=len(train_dataset) * 2, replacement=True)
        loader_args["shuffle"] = False
        loader_args["sampler"] = sampler

    train_loader = DataLoader(train_dataset, **loader_args)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    # Train loop
    best_loss = args.best_loss
    best_acc = args.best_acc
    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr)

    for epoch in np.arange(args.n_epochs) + 1:
        # Set a custom learning rate schedule
        if epochs_since_best == 3 and lr > args.min_lr:
            # Load previous best model
            path = f"{trial_path}/best_weights.pt"
            state_dict = torch.load(path, weights_only=False)

            # Decrease learning rate
            lr = max(lr * 0.5, args.min_lr)
            printlog(f"########## lr {lr} ##########", trial_path)
            epochs_since_best = 0

        else:
            epochs_since_best += 1

        # Remove parameters with requires_grad=False (https://github.com/pytorch/pytorch/issues/679)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)

        printlog(f"\nEpoch: {epoch:d}", trial_path)
        start_time = time.time()

        train_metrics = run_epoch(args, model, optimizer, train_loader, train=True)
        printlog("Train:\t" + get_losses_str(train_metrics), trial_path)

        valid_metrics = run_epoch(args, model, optimizer, valid_loader, train=False)
        printlog("Valid:\t" + get_losses_str(valid_metrics), trial_path)

        epoch_time = time.time() - start_time
        printlog("Time:\t {:.2f}s".format(epoch_time), trial_path)

        # valid_loss = sum([value for key, value in valid_metrics.items() if key.endswith("loss")])
        epoch_metrics = valid_metrics if args.oversampling == "none" else train_metrics
        epoch_loss = epoch_metrics["ce_loss"]
        epoch_acc = epoch_metrics["accuracy"] if "accuracy" in valid_metrics else valid_metrics["f1"]

        # Best model on test set
        if best_loss == 0 or epoch_loss < best_loss:
            epochs_since_best = 0
            best_loss = epoch_loss
            if epoch_acc > best_acc:
                best_acc = epoch_acc

            torch.save(model.module.state_dict(), f"{trial_path}/best_weights.pt")
            printlog("######## Best Loss ########", trial_path)

        elif epoch_acc > best_acc:
            epochs_since_best = 0
            best_acc = epoch_acc

            torch.save(model.module.state_dict(), f"{trial_path}/best_acc_weights.pt")
            printlog("###### Best Accuracy ######", trial_path)

    printlog(f"Best loss: {best_loss:.4f}", trial_path)
