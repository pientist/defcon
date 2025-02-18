import argparse
import os
import re
import sys
from typing import List

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


class FeatureEngineer:
    def __init__(
        self,
        events: pd.DataFrame,
        traces: pd.DataFrame,
        lineup: pd.DataFrame,
        xg_model: XGModel = None,
        action_type="all",
        include_keepers=True,
        include_goals=False,
        fps: int = 25,
    ):
        self.events = events.copy()
        self.traces = traces.copy()
        self.lineup = lineup.copy()

        self.xg_model = xg_model
        self.action_type = action_type
        self.fps = fps

        object_id_map = self.events.set_index("player_id")["object_id"].drop_duplicates()
        self.lineup["object_id"] = self.lineup["player_id"].map(object_id_map)
        self.lineup = self.lineup[config.LINEUP_HEADER].sort_values(["contestant_name", "shirt_number"])
        self.lineup = self.lineup.dropna(subset=["object_id"])
        self.keepers = self.lineup.loc[lineup["advanced_position"] == "goal_keeper", "object_id"].dropna().values

        if "episode_id" not in self.traces.columns:
            self.traces = tp.label_frames_and_episodes(self.traces, self.fps).set_index("frame")

        if "start_x" not in self.events.columns:
            self.events["start_x"] = self.events.apply(
                lambda e: traces.at[e["frame"], f"{e['object_id']}_x"] if e["frame"] == e["frame"] else np.nan, axis=1
            )
            self.events["start_y"] = self.events.apply(
                lambda e: traces.at[e["frame"], f"{e['object_id']}_y"] if e["frame"] == e["frame"] else np.nan, axis=1
            )
            self.events["start_z"] = self.events["frame"].apply(
                lambda t: self.traces.at[t, "ball_z"] if t == t else np.nan
            )

        if "end_x" not in self.events.columns:
            self.events["end_x"] = self.events["receive_frame"].apply(
                lambda t: traces.at[t, "ball_x"] if t == t else np.nan
            )
            self.events["end_y"] = self.events["receive_frame"].apply(
                lambda t: traces.at[t, "ball_y"] if t == t else np.nan
            )

        if action_type == "predefined":
            self.actions = events.copy()
        elif action_type == "pass":
            self.actions = self.filter_passes()
        elif action_type == "shot":
            self.actions = self.filter_chances(augment=True)
        elif action_type == "pass_shot":
            passes = self.filter_passes()
            shots = self.filter_chances(augment=False)
            self.actions = pd.concat([passes, shots]).sort_index()
        elif action_type == "failure":
            include_goals = True
            failed_passes = self.filter_passes(failure_only=True)
            blocked_shots = self.filter_chances(block_only=True)
            self.actions = pd.concat([failed_passes, blocked_shots]).sort_index()
        elif action_type == "all":
            self.actions = self.filter_valid_actions()

        self.include_keepers = include_keepers
        self.include_goals = include_goals
        self.max_players = 20 + int(include_keepers) * 2 + int(include_goals) * 2

        if self.include_goals:
            features = ["x", "y", "vx", "vy", "speed", "accel"]
            self.traces[[f"{team}_goal_{x}" for team in ["home", "away"] for x in features]] = 0.0
            self.traces["home_goal_x"] = config.FIELD_SIZE[0]
            self.traces["home_goal_y"] = config.FIELD_SIZE[1] / 2
            self.traces["away_goal_y"] = config.FIELD_SIZE[1] / 2

        self.features = None
        self.features_receiving = None
        self.labels = None

        self.augmented_features = None
        self.augmented_labels = None

    # To make the home team always play from left to right (not needed for the current dataset)
    def rotate_pitch_per_phase(self):
        for phase in self.traces["phase"].unique():
            phase_traces = self.traces[self.traces["phase"] == phase].dropna(axis=1, how="all")

            x_cols = [c for c in phase_traces.columns if c.endswith("_x")]
            y_cols = [c for c in phase_traces.columns if c.endswith("_y")]
            vx_cols = [c for c in phase_traces.columns if c.endswith("_vx")]
            vy_cols = [c for c in phase_traces.columns if c.endswith("_vy")]

            if not tp.is_home_on_left(phase_traces, halfline_x=config.FIELD_SIZE[0] / 2):
                self.traces.loc[phase_traces.index, x_cols] = config.FIELD_SIZE[0] - phase_traces[x_cols]
                self.traces.loc[phase_traces.index, y_cols] = config.FIELD_SIZE[1] - phase_traces[y_cols]
                self.traces.loc[phase_traces.index, vx_cols] = -phase_traces[vx_cols]
                self.traces.loc[phase_traces.index, vy_cols] = -phase_traces[vy_cols]

        self.events["x"] = self.traces.loc[self.events["frame"], "ball_x"].values
        self.events["y"] = self.traces.loc[self.events["frame"], "ball_y"].values

    def filter_passes(self, failure_only=False) -> pd.DataFrame:
        passes: pd.DataFrame = self.events[
            self.events["spadl_type"].isin(config.PASS)
            & self.events[["frame", "receive_frame"]].notna().all(axis=1)
            & (
                ~(self.events["frame"].diff() < self.fps * 0.5)
                | (self.events["object_id"] == self.events["object_id"].shift(1))
            )
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

        if failure_only:
            return passes[~passes["outcome"]].astype({"frame": int}).copy()
        else:
            return passes.astype({"frame": int})

    def filter_chances(self, augment=False, block_only=False) -> pd.DataFrame:
        if augment:
            chances = self.events[
                self.events["frame"].notna() & self.events["spadl_type"].isin(config.SHOT + ["pass"])
            ].copy()
            chances.loc[chances["spadl_type"].isin(config.SHOT), "action_type"] = "shot"
            chances.loc[chances["spadl_type"] == "pass", "action_type"] = "pass"

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
            chances = self.events[self.events["frame"].notna() & self.events["spadl_type"].isin(config.SHOT)].copy()
            chances["action_type"] = "shot"
            chances["blocked"] = chances["next_type"] == "shot_block"

        if block_only:
            return chances[chances["blocked"]].astype({"frame": int}).copy()
        else:
            return chances.astype({"frame": int})

    def filter_valid_actions(self) -> pd.DataFrame:
        passes = self.filter_passes()
        shots = self.filter_chances(augment=False)

        clearances = self.events[self.events["spadl_type"] == "clearance"].dropna(subset=["frame"]).copy()
        clearances["action_type"] = "clearance"
        clearances["blocked"] = False

        tackles = self.events[self.events["spadl_type"] == "tackle"].dropna(subset=["frame"]).copy()
        tackles["action_type"] = "tackle"
        tackles["blocked"] = False

        actions = pd.concat([passes, shots, clearances, tackles]).astype({"frame": int}).sort_index()
        return actions.dropna(subset=["next_type"]).copy()

        # kick_indices = passes.index.union(shots.index)
        # other_events = self.events.loc[self.events["frame"].notna() & ~self.events.index.isin(kick_indices)].copy()
        # other_events["action_type"] = None
        # other_events["blocked"] = False

        # return pd.concat([passes, shots, other_events.astype({"frame": int})]).sort_index()

    def label_intended_receivers(self, max_angle=45, eps=1e-6):
        self.events["team"] = self.events["object_id"].apply(lambda x: x[:4])
        self.actions["intent_id"] = pd.Series(index=self.actions.index, dtype="object")

        for i in self.actions.index:
            event_frame = self.actions.at[i, "frame"]
            possessor = self.actions.at[i, "object_id"]
            snapshot: pd.Series = self.traces.loc[event_frame]

            receive_frame = self.actions.at[i, "receive_frame"]
            receiver = self.actions.at[i, "receiver_id"]

            if self.actions.at[i, "action_type"] == "tackle":
                tackler = self.actions.at[i, "object_id"]
                possessor = self.events.loc[self.events["team"] != tackler[:4], "object_id"].loc[: i - 1].iloc[-1]
                self.actions.at[i, "intent_id"] = possessor

            elif self.actions.at[i, "action_type"] not in ["pass", "shot"]:  # Mostly for clearances
                self.actions.at[i, "intent_id"] = possessor

            elif self.actions.at[i, "action_type"] == "shot" or self.action_type == "shot":
                self.actions.at[i, "intent_id"] = f"{possessor[:4]}_goal"

                if self.actions.at[i, "action_type"] == "shot":  # For real shots
                    if self.actions.at[i, "outcome"]:
                        self.actions.at[i, "receiver_id"] = f"{possessor[:4]}_goal"

                    elif self.actions.at[i, "next_type"] in config.SET_PIECE_OOP:
                        self.actions.at[i, "receiver_id"] = "out"

                    else:
                        # elif self.actions.at[i, "next_type"] in config.INCOMING + ["clearance"]:
                        self.actions.at[i, "receiver_id"] = self.actions.at[i, "next_player_id"]

                else:  # For passes that would be blocked if they were shots
                    self.actions.at[i, "receiver_id"] = self.actions.at[i, "blocker_id"]

            elif self.actions.at[i, "outcome"]:  # For successful passes
                self.actions.at[i, "intent_id"] = self.actions.at[i, "receiver_id"]

            elif receive_frame == receive_frame and receiver == receiver:  # For failed passes
                receive_snapshot: pd.Series = self.traces.loc[int(receive_frame)]

                teammates = [c[:-2] for c in snapshot.dropna().index if re.match(rf"{possessor[:4]}_\d+_x", c)]
                teammates.remove(possessor)

                start_x = snapshot[f"{possessor}_x"]
                start_y = snapshot[f"{possessor}_y"]
                end_x = receive_snapshot[f"{receiver}_x"] if receiver != "out" else receive_snapshot["ball_x"]
                end_y = receive_snapshot[f"{receiver}_y"] if receiver != "out" else receive_snapshot["ball_y"]
                player_x = receive_snapshot[[f"{p}_x" for p in teammates]].values
                player_y = receive_snapshot[[f"{p}_y" for p in teammates]].values

                angles = utils.calc_angle(start_x, start_y, end_x, end_y, player_x, player_y) + eps
                dists = utils.calc_dist(player_x, player_y, end_x, end_y)[-1] + eps

                max_radian = max_angle / 180 * np.pi
                if np.min(angles) < max_radian:
                    scores = (np.min(dists) / dists) * (np.min(angles) / angles)
                    scores = np.where(angles < max_radian, scores, 0)
                    self.actions.at[i, "intent_id"] = teammates[np.argmax(scores)]

    def label_returns(self, lookahead_len: int = 10):
        # self.events["xg"] = 0.0
        # shot_features = self.xg_model.calc_shot_features(self.events, shots_only=True)
        # self.events.loc[shot_features.index, "xg"] = self.xg_model.pred(shot_features)

        self.events["goal"] = self.events["expected_goal"].notna() & self.events["outcome"]
        self.events["scores"] = 0.0
        self.events["scores_xg"] = 0.0
        self.events["concedes"] = 0.0
        self.events["concedes_xg"] = 0.0

        for period in self.events["period_id"].unique():
            period_events = self.events[self.events["period_id"] == period]
            labels = period_events[["team", "goal", "expected_goal"]].copy()

            for i in range(lookahead_len):
                shifted_teams = labels["team"].shift(-i)
                shifted_goals = labels[["goal", "expected_goal"]].shift(-i).fillna(0)
                # shifted_returns = labels.shift(-i).fillna(0).infer_objects(copy=False)
                labels[f"sg+{i}"] = shifted_goals["goal"] * (shifted_teams == labels["team"]).astype(int)
                labels[f"cg+{i}"] = shifted_goals["goal"] * (shifted_teams != labels["team"]).astype(int)
                labels[f"sxg+{i}"] = shifted_goals["expected_goal"] * (shifted_teams == labels["team"]).astype(int)
                labels[f"cxg+{i}"] = shifted_goals["expected_goal"] * (shifted_teams != labels["team"]).astype(int)

            scoring_cols = [c for c in labels.columns if c.startswith("sg+")]
            scoring_xg_cols = [c for c in labels.columns if c.startswith("sxg+")]
            conceding_cols = [c for c in labels.columns if c.startswith("cg+")]
            conceding_xg_cols = [c for c in labels.columns if c.startswith("cxg+")]

            self.events.loc[labels.index, "scores"] = labels[scoring_cols].sum(axis=1).clip(0, 1).astype(int)
            self.events.loc[labels.index, "scores_xg"] = 1 - (1 - labels[scoring_xg_cols]).prod(axis=1)
            self.events.loc[labels.index, "concedes"] = labels[conceding_cols].sum(axis=1).clip(0, 1).astype(int)
            self.events.loc[labels.index, "concedes_xg"] = 1 - (1 - labels[conceding_xg_cols]).prod(axis=1)

        self.actions["scores"] = self.events.loc[self.actions.index, "scores"]
        self.actions["scores_xg"] = self.events.loc[self.actions.index, "scores_xg"]
        self.actions["concedes"] = self.events.loc[self.actions.index, "concedes"]
        self.actions["concedes_xg"] = self.events.loc[self.actions.index, "concedes_xg"]

    def generate_label_tensors(self, lookahead_len: int = 10) -> torch.Tensor:
        self.label_intended_receivers()
        self.label_returns(lookahead_len)

        labels_list = []

        for i in self.actions.index:
            frame = self.actions.at[i, "frame"]
            snapshot = self.traces.loc[frame].dropna()
            if frame != frame:
                continue

            if self.include_goals:
                home_players = [c[:-2] for c in snapshot.index if re.match(r"home_.*_x", c)]
                away_players = [c[:-2] for c in snapshot.index if re.match(r"away_.*_x", c)]
            else:
                home_players = [c[:-2] for c in snapshot.index if re.match(r"home_\d+_x", c)]
                away_players = [c[:-2] for c in snapshot.index if re.match(r"away_\d+_x", c)]

            intent: str = self.actions.at[i, "intent_id"]
            if intent != intent or self.action_type == "predefined":
                intent_index = -1
            elif intent.startswith("home"):
                intent_index = home_players.index(intent)
            else:  # if intent.startswith("away"):
                intent_index = away_players.index(intent)

            if self.actions.at[i, "action_type"] == "pass" and self.action_type != "shot":
                try:
                    receiver: str = self.actions.at[i, "receiver_id"]
                    receive_frame: float = self.actions.at[i, "receive_frame"]
                    duration = round((receive_frame - frame) / self.fps, 2) if receive_frame == receive_frame else 0.0

                    if receiver == "out":
                        receiver_index = -1
                    elif self.actions.at[i, "object_id"].startswith("home"):
                        receiver_index = (home_players + away_players).index(receiver)
                    elif self.actions.at[i, "object_id"].startswith("away"):
                        receiver_index = (away_players + home_players).index(receiver)

                except ValueError:
                    continue

                end_x = self.actions.at[i, "end_x"]
                end_y = self.actions.at[i, "end_y"]

                if self.actions.at[i, "object_id"].startswith("away"):
                    end_x = config.FIELD_SIZE[0] - end_x
                    end_y = config.FIELD_SIZE[1] - end_y

                end_x = end_x if end_x == end_x else -100.0
                end_y = end_y if end_y == end_y else -100.0

                # For shot feature generation, selected passes are regarded as potentially blocked shots
                is_real = 0 if self.action_type == "shot" else 1

            else:
                duration = 0.0
                end_x = -100.0
                end_y = -100.0

                if self.actions.at[i, "action_type"] == "shot" or self.action_type == "shot":
                    receiver: str = self.actions.at[i, "receiver_id"]
                    is_real = int(self.actions.at[i, "action_type"] == "shot")
                    if receiver != receiver:
                        continue
                    if receiver == "out" or receiver.endswith("goal"):
                        receiver_index = -1
                    elif self.actions.at[i, "object_id"].startswith("home"):
                        receiver_index = (home_players + away_players).index(receiver)
                    elif self.actions.at[i, "object_id"].startswith("away"):
                        receiver_index = (away_players + home_players).index(receiver)

                else:
                    is_real = 1
                    receiver_index = -1

            labels_list.append(
                [
                    i,
                    int(self.actions.at[i, "action_type"] == "pass"),
                    int(self.actions.at[i, "action_type"] == "takeon"),
                    int(self.actions.at[i, "action_type"] == "shot"),
                    len(home_players) + len(away_players),
                    intent_index,
                    receiver_index,
                    duration,
                    end_x,
                    end_y,
                    is_real,  # whether this label is for a real or an augmented event
                    int(self.actions.at[i, "blocked"]),
                    int(self.actions.at[i, "outcome"]),
                    self.actions.at[i, "scores"],
                    self.actions.at[i, "scores_xg"],
                    self.actions.at[i, "concedes"],
                    self.actions.at[i, "concedes_xg"],
                ]
            )

        return torch.FloatTensor(labels_list)

    def adjust_receiving_tags(self):  # To accurately estimate expected returns after actions
        for period in self.actions["period_id"].unique():
            period_events = self.events[self.events["period_id"] == period]
            period_actions = self.actions[self.actions["period_id"] == period]

            for i in period_actions.index:
                event_frame = self.events.at[i, "frame"]
                next_player_id = self.events.at[i, "next_player_id"]
                next_type = self.actions.at[i, "next_type"]

                if next_type in config.SET_PIECE_OOP:
                    self.actions.at[i, "receiver_id"] = next_player_id
                    self.actions.at[i, "receive_frame"] = self.events.at[i + 1, "frame"]

                elif next_type in config.DEFENSIVE_TOUCH:
                    next_action_indices = period_events.loc[i + 2 :, "frame"].dropna().index
                    if len(next_action_indices) > 0:
                        self.actions.at[i, "receiver_id"] = self.events.at[next_action_indices[0], "object_id"]
                        self.actions.at[i, "receive_frame"] = self.events.at[next_action_indices[0], "frame"]

                elif self.actions.at[i, "spadl_type"] == "tackle":
                    if next_player_id == next_player_id:
                        next_frame = self.events.at[i + 1, "frame"]
                        self.actions.at[i, "receiver_id"] = next_player_id
                        self.actions.at[i, "receive_frame"] = next_frame if next_frame == next_frame else event_frame

    def generate_event_features(self, snapshot: pd.DataFrame, sequential=False, eps=1e-6) -> np.ndarray:
        seq_len = len(snapshot) if sequential else 1

        possessor = snapshot["possessor_id"].iloc[-1]
        possessor_aware = len(possessor) > 4
        # If possessor in ["home", "away"], do not calculate possessor features (currently not implemented)

        home_cols = [c for c in snapshot.dropna(axis=1).columns if c.startswith("home")]
        away_cols = [c for c in snapshot.dropna(axis=1).columns if c.startswith("away")]
        if not self.include_goals:
            home_cols = [c for c in home_cols if not c.startswith("home_goal")]
            away_cols = [c for c in away_cols if not c.startswith("away_goal")]

        player_cols = home_cols + away_cols if possessor.startswith("home") else away_cols + home_cols
        players = [c[:-2] for c in player_cols if c.endswith("_x")]

        is_teammate = np.tile([int(p[:4] == possessor[:4]) for p in players], (seq_len, 1))
        is_keeper = np.tile([int(p in self.keepers) for p in players], (seq_len, 1))
        is_goal = np.tile([int("goal" in p) for p in players], (seq_len, 1))

        if not sequential:
            snapshot = snapshot[-1:].copy()

        player_x = snapshot[player_cols[0::6]].values
        player_y = snapshot[player_cols[1::6]].values
        player_vx = snapshot[player_cols[2::6]].values
        player_vy = snapshot[player_cols[3::6]].values
        player_speeds = snapshot[player_cols[4::6]].values
        player_accels = snapshot[player_cols[5::6]].values

        if possessor_aware:  # Calculate possessor features only when there is a valid possessor label
            if possessor.endswith("out"):
                poss_x = snapshot["ball_x"].values
                poss_y = snapshot["ball_y"].values
            poss_x = snapshot[f"{possessor}_x"].values[:, np.newaxis]
            poss_y = snapshot[f"{possessor}_y"].values[:, np.newaxis]
            poss_vx = snapshot[f"{possessor}_vx"].values[:, np.newaxis]
            poss_vy = snapshot[f"{possessor}_vy"].values[:, np.newaxis]

        # Make the attacking team play from left to right for a given snapshot
        if possessor[:4] == "away":
            player_x = config.FIELD_SIZE[0] - player_x
            player_y = config.FIELD_SIZE[1] - player_y
            player_vx = -player_vx
            player_vy = -player_vy

            if possessor_aware:
                poss_x = config.FIELD_SIZE[0] - poss_x
                poss_y = config.FIELD_SIZE[1] - poss_y
                poss_vx = -poss_vx
                poss_vy = -poss_vy

        goal_x = config.FIELD_SIZE[0]
        goal_y = config.FIELD_SIZE[1] / 2
        goal_dx, goal_dy, goal_dists = utils.calc_dist(player_x, player_y, goal_x, goal_y)

        if "ball_z" in snapshot.columns:
            ball_z = np.ones((seq_len, len(players))) * snapshot["ball_z"].iloc[-1]
        else:
            ball_z = np.zeros((seq_len, len(players)))

        event_features = [
            # Binary features
            is_teammate,
            is_keeper,
            is_goal,
            # Possessor-independent features
            player_x,
            player_y,
            player_vx,
            player_vy,
            player_speeds,
            player_accels,
            goal_dists,
            goal_dx / (goal_dists + eps),  # Cosine between each player-goal line and the x-axis
            goal_dy / (goal_dists + eps),  # Sine between each player-goal line and the x-axis
            ball_z,
        ]

        if possessor_aware:  # Attach possessor features
            is_possessor = np.tile((np.array(players) == possessor).astype(int), (seq_len, 1))
            poss_dx, poss_dy, poss_dists = utils.calc_dist(player_x, player_y, poss_x, poss_y)
            poss_vangles = utils.calc_angle(player_vx, player_vy, poss_vx, poss_vy, eps=eps)

            event_features.extend(
                [
                    is_possessor,
                    poss_dists,
                    poss_dx / (poss_dists + eps),  # Cosine between each player-possessor line and the x-axis
                    poss_dy / (poss_dists + eps),  # Sine between each player-possessor line and the x-axis
                    np.cos(poss_vangles),  # Cosine between each player's velocity and the possessor's velocity
                    np.sin(poss_vangles),  # Sine between each player's velocity and the possessor's velocity
                ]
            )

        return np.stack(event_features, axis=-1)  # [T, N, x]

    def generate_feature_graphs(self, at_receiving=False, flip_tackle_poss=False, verbose=True) -> List[Data]:
        if "ball_accel" not in self.traces.columns:
            self.traces = tp.calc_physical_features(self.traces, self.fps)

        if at_receiving:
            self.adjust_receiving_tags()

        feature_tensors: List[torch.Tensor] = []

        if not self.include_keepers:
            keeper_cols = [c for c in self.traces.columns if "_".join(c.split("_")[:2]) in self.keepers]
            traces = self.traces.copy().drop(keeper_cols, axis=1)
        else:
            traces = self.traces.copy()

        for period in self.events["period_id"].unique():
            period_traces: pd.DataFrame = traces[traces["period_id"] == period]
            period_actions: pd.DataFrame = self.actions[self.actions["period_id"] == period]
            action_indices = np.intersect1d(period_actions.index, self.labels[:, 0].long().numpy())

            tqdm_desc = f"Generating feature tensors for P{period}"
            # bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"
            iterator = tqdm(action_indices, desc=tqdm_desc) if verbose else action_indices

            for i in iterator:
                if at_receiving:
                    frame = period_actions.at[i, "receive_frame"]
                    possessor = period_actions.at[i, "receiver_id"]
                elif flip_tackle_poss and self.actions.at[i, "spadl_type"] == "tackle":
                    frame = period_actions.at[i, "frame"]
                    tackler = period_actions.at[i, "object_id"]
                    possessor = self.events.loc[self.events["team"] != tackler[:4], "object_id"].loc[: i - 1].iloc[-1]
                else:
                    frame = period_actions.at[i, "frame"]
                    possessor = period_actions.at[i, "object_id"]

                if frame == frame and possessor == possessor:
                    snapshot = period_traces.loc[frame - 1 : frame].dropna(axis=1, how="all").copy()
                    snapshot["possessor_id"] = possessor

                    # snapshot["spadl_type"] = period_actions.at[i, "spadl_type"]
                    event_features = self.generate_event_features(snapshot)
                    missing_players = self.max_players - event_features.shape[1]
                    padding_features = -torch.ones((missing_players, event_features.shape[2]))
                    event_features = torch.cat([torch.FloatTensor(event_features[0]), padding_features], 0)
                    feature_tensors.append(event_features)

                else:
                    padding_features = -torch.ones((self.max_players, event_features.shape[-1]))
                    feature_tensors.append(padding_features)

        node_attr = torch.stack(feature_tensors, axis=0)  # [B, N, x]
        distances = torch.cdist(node_attr[..., 3:5], node_attr[..., 3:5], p=2)  # [B, N, N]
        teammates = (node_attr[..., 0].unsqueeze(-1) == node_attr[..., 0].unsqueeze(-2)).float()  # [B, N, N]

        feature_graphs: List[Data] = []

        for i in range(node_attr.shape[0]):
            if node_attr[i, 0, 0] == -1:
                feature_graphs.append(None)

            else:
                node_mask = node_attr[i, :, 0] != -1
                node_attr_i = node_attr[i][node_mask]

                distances_i = distances[i][node_mask][:, node_mask]
                teammates_i = teammates[i][node_mask][:, node_mask]
                edge_index, _ = dense_to_sparse(torch.ones_like(distances_i))

                distances_i = distances_i[edge_index[0], edge_index[1]]
                teammates_i = teammates_i[edge_index[0], edge_index[1]]
                edge_attr_i = torch.stack([distances_i, teammates_i], dim=-1)  # [N * N, 2]

                graph = Data(x=node_attr_i, edge_index=edge_index.clone(), edge_attr=edge_attr_i)
                feature_graphs.append(graph)

        return feature_graphs

    def augment_blocked_actions(self, max_block_dist=5, max_block_angle=15):
        augmented_features = []
        augmented_labels = []

        action_indices = self.labels[:, 0].numpy().astype(int)
        tqdm_desc = "Augmenting features and labels"
        # bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"

        for i, action_index in enumerate(tqdm(action_indices, desc=tqdm_desc)):
            augmented_features.append(self.features[i])
            augmented_labels.append(self.labels[i])

            if self.actions.at[action_index, "spadl_type"] in config.SET_PIECE:
                continue

            frame = self.actions.at[action_index, "frame"]
            possessor = self.actions.at[action_index, "object_id"]
            real_intent = self.actions.at[action_index, "intent_id"]
            snapshot: pd.Series = self.traces.loc[frame]

            teammates = [c[:-2] for c in snapshot.dropna().index if re.match(rf"{possessor[:4]}_.*_x", c)]
            team_x = snapshot[[f"{p}_x" for p in teammates]].values
            team_y = snapshot[[f"{p}_y" for p in teammates]].values

            team_dist_x = (team_x - snapshot[f"{possessor}_x"]).astype(float)
            team_dist_y = (team_y - snapshot[f"{possessor}_y"]).astype(float)
            team_dists = np.sqrt(team_dist_x**2 + team_dist_y**2)

            oppo_team = "away" if possessor[:4] == "home" else "home"
            opponents = [c[:-2] for c in snapshot.dropna().index if re.match(rf"{oppo_team}_.*_x", c)]
            oppo_x = snapshot[[f"{p}_x" for p in opponents]].values
            oppo_y = snapshot[[f"{p}_y" for p in opponents]].values

            oppo_dist_x = (oppo_x - snapshot[f"{possessor}_x"]).astype(float)
            oppo_dist_y = (oppo_y - snapshot[f"{possessor}_y"]).astype(float)
            oppo_dists = np.sqrt(oppo_dist_x**2 + oppo_dist_y**2)
            blockers = np.array(opponents)[np.where(oppo_dists < max_block_dist)[0]][:3]

            if self.include_goals:
                goal_index = teammates.index(f"{possessor[:4]}_goal")

            for blocker in blockers:
                poss_x = snapshot[f"{possessor}_x"]
                poss_y = snapshot[f"{possessor}_y"]
                block_x = snapshot[f"{blocker}_x"]
                block_y = snapshot[f"{blocker}_y"]

                team_angles = utils.calc_angle(poss_x, poss_y, block_x, block_y, team_x, team_y)
                oppo_angles = utils.calc_angle(poss_x, poss_y, block_x, block_y, oppo_x, oppo_y)
                blocked_teammates = np.where(team_angles < max_block_angle / 180 * np.pi)[0].tolist()
                blocked_opponents = np.where(oppo_angles < max_block_angle / 180 * np.pi)[0].tolist()
                blocked_opponents = [k for k in blocked_opponents if k != opponents.index(blocker)]

                if not blocked_opponents:
                    continue

                close_teammates = np.where(team_dists < oppo_dists[blocked_opponents].max() - 10)[0].tolist()
                if self.include_goals and team_dists[goal_index] < 40:
                    close_teammates.append(goal_index)  # Assume that the shot was prevented if goal distance was < 30
                blocked_intent_indices = list(set(blocked_teammates) & set(close_teammates))
                blocked_intents = [p for p in np.array(teammates)[blocked_intent_indices] if p != real_intent]

                for blocked_intent in blocked_intents:
                    augmented_labels_i = self.labels[i].clone()

                    if blocked_intent.endswith("goal"):  # An augmented shot that would be blocked
                        augmented_labels_i[1] = 0
                        augmented_labels_i[3] = 1
                    else:  # An augmented pass that would be blocked
                        augmented_labels_i[1] = 1
                        augmented_labels_i[3] = 0

                    augmented_labels_i[5] = teammates.index(blocked_intent)
                    augmented_labels_i[6] = (teammates + opponents).index(blocker)
                    augmented_labels_i[-7] = 0  # Indicating that this is an augmented event and not a real one
                    augmented_labels_i[-6] = 1  # Indicating that this is a blocked event
                    augmented_labels_i[-5] = 0  # Indicating that this is a failed event

                    augmented_features.append(self.features[i].clone())
                    augmented_labels.append(augmented_labels_i)

        self.augmented_features = augmented_features
        self.augmented_labels = torch.stack(augmented_labels, dim=0)

    def save(self, game_id, feature_dir="data/ajax/pyg/features", label_dir="data/ajax/pyg/labels"):
        if self.features is not None:
            os.makedirs(feature_dir, exist_ok=True)
            torch.save(self.features, f"{feature_dir}/{game_id}.pt")

        if self.labels is not None:
            os.makedirs(label_dir, exist_ok=True)
            torch.save(self.labels, f"{label_dir}/{game_id}.pt")

    def save_augmented(self, game_id, feature_dir="data/ajax/pyg/aug_features", label_dir="data/ajax/pyg/aug_labels"):
        if self.augmented_features is not None:
            os.makedirs(feature_dir, exist_ok=True)
            torch.save(self.augmented_features, f"{feature_dir}/{game_id}.pt")

        if self.augmented_labels is not None:
            os.makedirs(label_dir, exist_ok=True)
            torch.save(self.augmented_labels, f"{label_dir}/{game_id}.pt")


parser = argparse.ArgumentParser()
parser.add_argument("--action_type", type=str, required=True, choices=["all", "pass", "shot", "failure"])
args, _ = parser.parse_known_args()


if __name__ == "__main__":
    include_goals = args.action_type in ["all", "failure"]
    augment = args.action_type == "failure"

    feature_dir = f"data/ajax/pyg/{args.action_type}_features"
    label_dir = f"data/ajax/pyg/{args.action_type}_labels"

    if augment:
        augmented_feature_dir = f"data/ajax/pyg/{args.action_type}_aug_features"
        augmented_label_dir = f"data/ajax/pyg/{args.action_type}_aug_labels"

    event_files = np.sort(os.listdir("data/ajax/event_synced"))
    game_ids = np.sort([f.split(".")[0] for f in event_files if f.endswith(".csv")])
    lineups = pd.read_parquet("data/ajax/lineup/line_up.parquet")

    xg_model = XGModel(unblocked=True)
    xg_model.train()

    for i, game_id in enumerate(game_ids):
        game_lineup = lineups.loc[lineups["stats_perform_match_id"] == game_id]
        print(f"\n[{i}] {game_lineup['game'].iloc[0]}")

        events = pd.read_csv(f"data/ajax/event_synced/{game_id}.csv", header=0, parse_dates=["utc_timestamp"])
        traces = pd.read_parquet(f"data/ajax/tracking_processed/{game_id}.parquet")

        eng = FeatureEngineer(events, traces, game_lineup, xg_model, args.action_type, include_goals=include_goals)
        eng.labels = eng.generate_label_tensors()
        eng.features = eng.generate_feature_graphs()

        action_indices = eng.labels[:, 0].numpy().astype(int)
        assert np.all(np.sort(action_indices) == action_indices)

        eng.save(game_id, feature_dir, label_dir)
        print(f"Successfully saved for {eng.labels.shape[0]} events.")

        if augment:
            eng.augment_blocked_actions()
            eng.save_augmented(game_id, augmented_feature_dir, augmented_label_dir)
            print(f"Successfully saved for {eng.augmented_labels.shape[0]} augmented events.")
