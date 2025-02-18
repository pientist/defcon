import argparse
import os
import sys
from datetime import datetime

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from datatools.defcon import DEFCON
from datatools.feature import FeatureEngineer
from datatools.xg_model import XGModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=False, default="data/ajax/player_scores.parquet")
    args, _ = parser.parse_known_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    event_files = np.sort(os.listdir("data/ajax/event_synced"))
    game_ids = np.sort([f.split(".")[0] for f in event_files if f.endswith(".csv")])

    lineups = pd.read_parquet("data/ajax/lineup/line_up.parquet").sort_values("game_date", ignore_index=True)
    lineups["game_date"] = pd.to_datetime(lineups["game_date"])
    lineups["game_id"] = lineups["stats_perform_match_id"]

    game_dates = lineups[["game_id", "game_date"]].drop_duplicates().set_index("game_id")["game_date"]
    test_game_ids = game_dates[(game_dates.index.isin(game_ids)) & (game_dates >= datetime(2024, 8, 1))].index

    xg_model = XGModel(unblocked=True)
    xg_model.train(verbose=False)

    for i, game_id in enumerate(test_game_ids):
        events = pd.read_csv(f"data/ajax/event_synced/{game_id}.csv", header=0, parse_dates=["utc_timestamp"])
        traces = pd.read_parquet(f"data/ajax/tracking_processed/{game_id}.parquet")
        game_lineup = lineups.loc[lineups["stats_perform_match_id"] == game_id]

        eng = FeatureEngineer(events, traces, game_lineup, xg_model, action_type="all", include_goals=True)
        game_date = eng.lineup["game_date"].iloc[0].date()
        # home_name = eng.lineup.set_index("object_id").at[np.sort(eng.keepers)[1], "contestant_name"]
        # away_name = eng.lineup.set_index("object_id").at[np.sort(eng.keepers)[0], "contestant_name"]
        print(f"\n[{i}] {game_id}: {game_lineup['game'].iloc[0]} on {game_date}")

        eng.labels = eng.generate_label_tensors()

        print("Generating features for moments at actions...")
        eng.features = eng.generate_feature_graphs(flip_tackle_poss=True)
        print("Generating features for moments after actions...")
        eng.features_receiving = eng.generate_feature_graphs(at_receiving=True)

        print("\nValuing defensive contributions...")
        defcon = DEFCON(
            eng,
            scoring_model_id="scoring/32",
            pass_intent_model_id="intent/30",
            pass_success_model_id="intent_success/30",
            pass_scoring_model_id="intent_scoring/32",
            shot_blocking_model_id="shot_blocking/30",
            posterior_model_id="failure_receiver/30",
            likelihood_model_id="oppo_agn_intent/02",
            device=device,
        )
        defcon.evaluate(mask_likelihood=0.03)

        if not os.path.exists(args.result_path):
            defcon.player_scores.to_parquet(args.result_path, engine="pyarrow", index=False)
        else:
            saved_scores = pd.read_parquet(args.result_path, engine="pyarrow")
            saved_scores = pd.concat([saved_scores, defcon.player_scores], ignore_index=True)
            saved_scores.to_parquet(args.result_path, engine="pyarrow", index=False)
