import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from dataset import ActionDataset
from datatools import config
from models.gat import GAT
from models.utils import (
    estimate_likelihoods,
    get_args_str,
    get_losses_str,
    num_trainable_params,
    printlog,
    run_epoch,
)

parser = argparse.ArgumentParser()

parser.add_argument("--target", type=str, required=True, help="intent, receiver, success, scoring, conceding")
parser.add_argument("--trial", type=int, required=True)
parser.add_argument("--model", type=str, required=True, default="gat")

parser.add_argument("--augment", action="store_true", default=False, help="include augmented data")
parser.add_argument("--oversampling", type=str, default="none", help="model ID to obtain oversampling probs")
parser.add_argument("--weight_bce", action="store_true", default=False, help="use weighted BCE to balance classes")

parser.add_argument("--min_duration", type=float, default=0, help="min duration of a valid action")
parser.add_argument("--xy_only", action="store_true", default=False, help="only use xy locations as features")
parser.add_argument("--possessor_aware", action="store_true", default=False, help="use possessor features")
parser.add_argument("--keeper_aware", action="store_true", default=False, help="distinguish keeper & goal nodes")
parser.add_argument("--ball_z_aware", action="store_true", default=False, help="consider the ball height")
parser.add_argument("--poss_vel_aware", action="store_true", default=False, help="consider possessor's velocity")
# parser.add_argument("--one_touch_aware", action="store_true", default=False, help="use one-touch flags as a feature")

parser.add_argument("--use_intent", action="store_true", default=False, help="use intended receivers' locations")
parser.add_argument("--use_xg", action="store_true", default=False, help="use xG instead of actual goal labels")
parser.add_argument("--residual", action="store_true", default=False, help="attach a component for ball out of play")
parser.add_argument("--filter_blockers", action="store_true", default=False, help="only include potential blockers")
parser.add_argument("--sparsify", type=str, choices=["distance", "delaunay", "none"], help="how to filter edges")
parser.add_argument("--max_edge_dist", type=int, default=10, help="max distance between off-ball nodes")

parser.add_argument("--edge_in_dim", type=int, required=False, default=0, help="num edge features")
parser.add_argument("--node_emb_dim", type=int, required=False, default=0, help="node embedding dim")
parser.add_argument("--graph_emb_dim", type=int, required=False, default=0, help="graph embedding dim")
parser.add_argument("--gnn_layers", type=int, required=False, default=0, help="num GNN layers")
parser.add_argument("--gnn_heads", type=int, required=False, default=0, help="num heads of GNN layers")
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
        args.node_in_dim = 9 if args.possessor_aware else 8
    else:
        args.node_in_dim = 19 if args.possessor_aware else 13

    intent_given = args.target in ["intent_success", "failed_pass_receiver", "failure_receiver"]
    if intent_given:
        args.node_in_dim += 1

    if args.target in config.NODE_SELECTION:
        args.task = "node_selection"
    elif args.target in config.NODE_BINARY:
        args.task = "node_binary"
    elif args.target in config.GRAPH_BINARY:
        args.task = "graph_binary"
    # elif args.target.startswith("intent_") and args.target.split("_")[1] in ["success", "scoring", "conceding"]:
    #     args.task = "graph_binary"

    if args.target in ["scoring", "conceding"]:
        args.action_type = "all"
    elif args.target.startswith("failure"):
        args.action_type = "failure"
    elif args.target == "oppo_agn_intent":
        args.action_type = "pass_shot"
    elif args.target.startswith("blocked_shot"):
        args.action_type = "blocked_shot"
    elif args.target.startswith("failed_pass"):
        args.action_type = "failed_pass"
    elif args.target.startswith("shot"):
        args.action_type = "shot"
    else:
        args.action_type = "pass"

    # Load model
    args_dict = vars(args)
    model = GAT(args_dict).to(device)
    model = nn.DataParallel(model)
    args_dict["total_params"] = num_trainable_params(model)

    # Create a path to save model arguments and parameters
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
        dataset_arg_keys = ["oversampling", "ball_z_aware", "poss_vel_aware", "sparsify"]
        model_arg_keys = ["node_in_dim", "node_emb_dim", "gnn_heads", "skip_conn", "total_params"]
        train_arg_keys = ["batch_size", "weight_bce", "lambda_l1", "start_lr", "seed"]

        printlog(title, trial_path)
        printlog(get_args_str(dataset_arg_keys, args_dict), trial_path)
        printlog(get_args_str(model_arg_keys, args_dict), trial_path)
        printlog(get_args_str(train_arg_keys, args_dict), trial_path)

    printlog("############################################################", trial_path)

    print("\nGenerating datasets...")
    action_type = args.action_type.split("_")[-1] if args.action_type != "pass_shot" else "pass_shot"
    if args.augment:
        feature_dir = f"data/ajax/pyg/{action_type}_aug_features"
    else:
        feature_dir = f"data/ajax/pyg/{action_type}_features"
    game_ids = [f.split(".")[0] for f in np.sort(os.listdir(feature_dir)) if f.endswith(".pt")]

    lineups = pd.read_parquet("data/ajax/lineup/line_up.parquet").sort_values("game_date", ignore_index=True)
    lineups["game_date"] = pd.to_datetime(lineups["game_date"])
    lineups["game_id"] = lineups["stats_perform_match_id"]

    game_dates = lineups[["game_id", "game_date"]].drop_duplicates().set_index("game_id")["game_date"]

    np.random.seed(0)
    game_ids = game_dates[(game_dates.index.isin(game_ids)) & (game_dates < datetime(2024, 6, 1))].index
    train_game_ids = np.sort(np.random.choice(game_ids, 200, replace=False))
    valid_game_ids = np.sort([id for id in game_ids if id not in train_game_ids])

    dataset_args = {
        "feature_dir": feature_dir,
        "label_dir": feature_dir.replace("features", "labels"),
        "action_type": args.action_type,
        "intended_only": "intent" in args.target or args.target.startswith("fail"),
        "inplay_only": args.target.endswith("receiver") and not args.residual,
        "min_duration": args.min_duration,
        "xy_only": args.xy_only,
        "possessor_aware": args.possessor_aware,
        "keeper_aware": args.keeper_aware,
        "ball_z_aware": args.ball_z_aware,
        "poss_vel_aware": args.poss_vel_aware,
        "intent_given": intent_given,
        "drop_opponents": args.target.startswith("oppo_agn"),
        "drop_non_blockers": args.filter_blockers,
        "sparsify": args.sparsify,
        "max_edge_dist": args.max_edge_dist,
    }
    train_dataset = ActionDataset(train_game_ids, **dataset_args)
    valid_dataset = ActionDataset(valid_game_ids, **dataset_args)

    if args.action_type == "failed_pass" and args.augment:
        train_dataset.balance_real_and_augmented()
        valid_dataset.balance_real_and_augmented()

    if args.target.endswith("_success") and args.weight_bce:
        n_positives = train_dataset.labels[train_dataset.labels[:, -5] == 1].shape[0]
        n_negatives = train_dataset.labels[train_dataset.labels[:, -5] == 0].shape[0]
        pos_weight = n_negatives / n_positives
    else:
        pos_weight = 1

    loader_args = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 16, "pin_memory": True}
    if args.oversampling != "none":
        print("\nCalculating oversampling probabilities...")
        scores = 1 / estimate_likelihoods(train_dataset, model_id=args.oversampling, device=device)
        sampling_probs = scores / scores.sum()
        sampler = WeightedRandomSampler(sampling_probs, num_samples=len(train_dataset) * 2, replacement=True)
        loader_args["shuffle"] = False
        loader_args["sampler"] = sampler

    train_loader = DataLoader(train_dataset, **loader_args)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

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

        train_metrics = run_epoch(args, model, train_loader, optimizer, device, pos_weight, train=True)
        printlog("Train:\t" + get_losses_str(train_metrics), trial_path)

        valid_metrics = run_epoch(args, model, valid_loader, optimizer, device, pos_weight, train=False)
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
