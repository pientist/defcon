import os
import sys
from fnmatch import fnmatch
from typing import Any, Dict, List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

import datatools.trace_processing as tp
from datatools import config, utils
from datatools.xg_model import XGModel


class Game:
    def __init__(
        self,
        events: pd.DataFrame,
        traces: pd.DataFrame,
        lineup: pd.DataFrame,
        xg_model: XGModel = None,
        include_keepers=True,
        include_goals=False,
        fps: int = 25,
    ):
        self.events = events.copy()
        self.traces = traces.copy()
        self.lineup = lineup.copy()

        self.xg_model = xg_model
        self.fps = fps

        keeper_ids = lineup.loc[lineup["position"] == "Goalkeeper"].index
        self.keepers = events.loc[events["player_id"].isin(keeper_ids), "object_id"].unique().tolist()

        if "episode_id" not in self.traces.columns:
            self.traces = tp.label_frames_and_episodes(self.traces, self.fps).set_index("frame")

    def filter_passes(self, events: pd.DataFrame, filter_failures=False) -> pd.DataFrame:
        passes: pd.DataFrame = events[
            events["spadl_type"].isin(config.PASS)
            & events[["frame", "receive_frame"]].notna().all(axis=1)
            & (~(events["frame"].diff() < self.fps * 0.5) | (events["object_id"] == events["object_id"].shift(1)))
        ]
        passes: pd.DataFrame = passes[
            (passes["receiver_id"] == passes["next_player_id"])
            | (passes["receiver_id"] == "out")
            | (passes["next_type"].isin(["foul", "freekick_short"]))
        ].copy()
        passes["action_type"] = "pass"
        passes["blocked"] = False  # To be updated

        pass_team = passes["object_id"].apply(lambda x: x[:4])
        receive_team = passes["receiver_id"].apply(lambda x: x[:4])
        passes.loc[pass_team == receive_team, "outcome"] = True
        passes.loc[pass_team != receive_team, "outcome"] = False

        if filter_failures:
            return passes[~passes["outcome"]].astype({"frame": int}).copy()
        else:
            return passes.astype({"frame": int})

    def filter_chances(self, events: pd.DataFrame, augment=False, filter_blocks=False) -> pd.DataFrame:
        if augment:
            chances = events[events["frame"].notna() & events["spadl_type"].isin(config.SHOT + ["pass"])].copy()
            chances.loc[chances["spadl_type"].isin(config.SHOT), "action_type"] = "shot"
            chances.loc[chances["spadl_type"] == "pass", "action_type"] = "pass"

            chances["start_x"] = chances.apply(lambda a: self.traces.at[a["frame"], f"{a['object_id']}_x"], axis=1)
            chances["start_y"] = chances.apply(lambda a: self.traces.at[a["frame"], f"{a['object_id']}_y"], axis=1)
            chances["start_z"] = chances["frame"].apply(lambda t: self.traces.at[t, "ball_z"])

            chances["dist_x"] = chances["start_x"]
            chances["dist_y"] = chances["start_y"] - config.FIELD_SIZE[1] / 2
            home_chances = chances[chances["object_id"].str.startswith("home")]
            chances.loc[home_chances.index, "dist_x"] = config.FIELD_SIZE[0] - home_chances["start_x"]
            chances["goal_dist"] = chances[["dist_x", "dist_y"]].apply(np.linalg.norm, axis=1).round(2)

            chances = chances[(chances["action_type"] == "shot") | (chances["goal_dist"] < 40)].copy()
            shot_features = XGModel.calc_shot_features(chances)
            chances["xg_unblocked"] = self.xg_model.pred(shot_features)
            chances["blocker_id"] = chances.apply(utils.find_blocker, axis=1, args=(self.traces, self.keepers))

            augmented_shots = (
                (chances["xg_unblocked"] > 0.05) & (chances["blocker_id"].notna()) & (chances["start_z"] < 1)
            )
            chances = chances[(chances["action_type"] == "shot") | augmented_shots].copy()
            chances["blocked"] = (chances["next_type"] == "shot_block") | (~chances["action_type"].isin(config.SHOT))

        else:
            chances = events[events["frame"].notna() & events["spadl_type"].isin(config.SHOT)].copy()
            chances["action_type"] = "shot"
            chances["blocked"] = chances["next_type"] == "shot_block"

        if filter_blocks:
            return chances[chances["blocked"]].astype({"frame": int}).copy()
        else:
            return chances.astype({"frame": int})

    def filter_valid_actions(self, events: pd.DataFrame) -> pd.DataFrame:
        passes = self.filter_passes(events)
        shots = self.filter_chances(events, augment=False)

        pass_shot_indices = passes.index.union(shots.index)
        other_events = events.loc[events["frame"].notna() & ~events.index.isin(pass_shot_indices)].copy()
        other_events["action_type"] = None
        other_events["blocked"] = False

        return pd.concat([passes, shots, other_events.astype({"frame": int})]).sort_values("frame")
