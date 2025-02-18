import argparse
import os
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from dataset import ActionDataset
from models import utils
from models.utils import get_losses_str, run_epoch

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, required=True, help="target/trial, e.g., intent/01")
args, _ = parser.parse_known_args()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = utils.load_model(args.model_id, device)
    model_args = argparse.Namespace(**model.args)
    # pprint(model.args)

    print("\nGenerating test datasets...")
    feature_dir = f"data/ajax/pyg/{model_args.action_type.split('_')[-1]}_features"  # Only use real data for evaluation
    game_ids = [f.split(".")[0] for f in np.sort(os.listdir(feature_dir)) if f.endswith(".pt")]

    lineups = pd.read_parquet("data/ajax/lineup/line_up.parquet").sort_values("game_date", ignore_index=True)
    lineups["game_date"] = pd.to_datetime(lineups["game_date"])
    lineups["game_id"] = lineups["stats_perform_match_id"]

    game_dates = lineups[["game_id", "game_date"]].drop_duplicates().set_index("game_id")["game_date"]
    test_game_ids = game_dates[(game_dates.index.isin(game_ids)) & (game_dates >= datetime(2024, 8, 1))].index

    intent_given = model_args.target in ["intent_success", "failed_pass_receiver", "failure_receiver"]
    dataset_args = {
        "feature_dir": feature_dir,
        "label_dir": feature_dir.replace("features", "labels"),
        "action_type": model_args.action_type,
        "intended_only": "intent" in model_args.target or model_args.target.startswith("fail"),
        "inplay_only": model_args.target.endswith("receiver") and not model_args.residual,
        "min_duration": model_args.min_duration,
        "xy_only": model_args.xy_only,
        "possessor_aware": model_args.possessor_aware,
        "keeper_aware": model_args.keeper_aware,
        "ball_z_aware": model_args.ball_z_aware,
        "poss_vel_aware": model_args.poss_vel_aware,
        "intent_given": intent_given,
        "drop_non_blockers": model_args.filter_blockers,
        "sparsify": model_args.sparsify,
        "max_edge_dist": model_args.max_edge_dist,
    }
    test_dataset = ActionDataset(test_game_ids, **dataset_args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    test_metrics = run_epoch(model_args, model, test_loader, device=device, train=False)
    print("Test:\t" + get_losses_str(test_metrics))
