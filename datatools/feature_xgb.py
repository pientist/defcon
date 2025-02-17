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
        task="task",
        include_keepers=True,
        fps: int = 25,
    ):
        self.events = events.copy()
        self.traces = traces.copy()
        self.lineup = lineup.copy()

        self.xg_model = xg_model
        self.action_type = action_type
        self.task = task
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
        elif action_type == "all":
            self.actions = self.filter_valid_actions()

        self.include_keepers = include_keepers
        features = ["x", "y", "vx", "vy", "speed", "accel"]
        self.traces[[f"{team}_goal_{x}" for team in ["home", "away"] for x in features]] = 0.0
        self.traces["home_goal_x"] = config.FIELD_SIZE[0]
        self.traces["home_goal_y"] = config.FIELD_SIZE[1] / 2
        self.traces["away_goal_y"] = config.FIELD_SIZE[1] / 2

        self.features = {}
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

            home_players = [c[:-2] for c in snapshot.index if re.match(r"home_.*_x", c)]
            away_players = [c[:-2] for c in snapshot.index if re.match(r"away_.*_x", c)]

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

    def count_opponents_in_corridor(self, target_x, target_y, poss_x, poss_y, opponent_x, opponent_y, width=10):
        """Counts the number of opponents inside a corridor defined by target_x, target_y and poss_x, poss_y."""

        def compute_perpendicular_offset(x1, y1, x2, y2, width):
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length == 0:
                return None
            perp_x, perp_y = -dy / length, dx / length
            offset_x = (width / 2) * perp_x
            offset_y = (width / 2) * perp_y
            return (
                (x1 - offset_x, y1 - offset_y),
                (x1 + offset_x, y1 + offset_y),
                (x2 - offset_x, y2 - offset_y),
                (x2 + offset_x, y2 + offset_y),
            )

        def is_point_in_corridor(px, py, x1, y1, x2, y2, width):
            offsets = compute_perpendicular_offset(x1, y1, x2, y2, width)
            if offsets is None:
                return np.zeros_like(px, dtype=bool)
            p1, p2, p3, p4 = offsets
            A = np.array(p1)
            B = np.array(p2)
            D = np.array(p3)
            P = np.column_stack((px, py))
            AB = B - A
            AD = D - A
            AP = P - A
            within_AB = (0 <= np.dot(AP, AB)[:, None]) & (np.dot(AP, AB)[:, None] <= np.dot(AB, AB))
            within_AD = (0 <= np.dot(AP, AD)[:, None]) & (np.dot(AP, AD)[:, None] <= np.dot(AD, AD))
            return np.logical_and(within_AB, within_AD).flatten()

        return np.sum(is_point_in_corridor(opponent_x, opponent_y, target_x, target_y, poss_x, poss_y, width))

    def closest_opponent_distance(self, target_x, target_y, poss_x, poss_y, opponent_x, opponent_y):
        """Finds the closest opponent to the line between target and poss and returns the distance."""
        dx, dy = poss_x - target_x, poss_y - target_y
        length_sq = dx**2 + dy**2
        if length_sq == 0:
            return np.min(np.sqrt((opponent_x - target_x) ** 2 + (opponent_y - target_y) ** 2))

        t = ((opponent_x - target_x) * dx + (opponent_y - target_y) * dy) / length_sq
        t = np.clip(t, 0, 1)
        closest_x = target_x + t * dx
        closest_y = target_y + t * dy
        distances = np.sqrt((opponent_x - closest_x) ** 2 + (opponent_y - closest_y) ** 2)
        return np.min(distances)

    def count_opponents_in_triangle(self, target_x, target_y, goal_x, goal_y, opponent_x, opponent_y):
        """
        Counts the number of opponents inside the triangle formed by target,
        (goal_x, goal_y - 7.32/2), (goal_x, goal_y + 7.32/2).
        """
        goal_left = np.array([goal_x, goal_y - 3.66])
        goal_right = np.array([goal_x, goal_y + 3.66])
        target = np.array([target_x, target_y])

        def is_inside_triangle(p):
            v0 = goal_right - target
            v1 = goal_left - target
            v2 = p - target

            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)

            denom = dot00 * dot11 - dot01 * dot01
            if denom == 0:
                return False

            u = (dot11 * dot02 - dot01 * dot12) / denom
            v = (dot00 * dot12 - dot01 * dot02) / denom

            return (u >= 0) & (v >= 0) & (u + v <= 1)

        opponent_positions = np.column_stack((opponent_x, opponent_y))
        inside_mask = np.apply_along_axis(is_inside_triangle, 1, opponent_positions)

        return np.sum(inside_mask)

    def calculate_event_features(self, snapshot: pd.DataFrame, action_type: str, eps=1e-6) -> np.ndarray:

        target = snapshot["target_id"].iloc[-1]
        possessor = snapshot["possessor_id"].iloc[-1]
        possessor_aware = len(possessor) > 4
        # If possessor in ["home", "away"], do not calculate possessor features (currently not implemented)

        home_cols = [c for c in snapshot.dropna(axis=1).columns if c.startswith("home")]
        away_cols = [c for c in snapshot.dropna(axis=1).columns if c.startswith("away")]

        opponents_cols = away_cols if target.startswith("home") else home_cols

        target_x = snapshot[f"{target}_x"].values
        target_y = snapshot[f"{target}_y"].values
        target_vx = snapshot[f"{target}_vx"].values
        target_vy = snapshot[f"{target}_vy"].values
        target_speed = snapshot[f"{target}_speed"].values
        target_accel = snapshot[f"{target}_accel"].values

        opponent_x = snapshot[opponents_cols[0::6]].values.flatten()
        opponent_y = snapshot[opponents_cols[1::6]].values.flatten()
        opponent_vx = snapshot[opponents_cols[2::6]].values.flatten()
        opponent_vy = snapshot[opponents_cols[3::6]].values.flatten()

        _, _, opp_dist = utils.calc_dist(target_x, target_y, opponent_x, opponent_y)

        nearby_opponents = np.sum(opp_dist <= 3)
        nearest_opponent = np.min(opp_dist)

        if possessor_aware:  # Calculate possessor features only when there is a valid possessor label
            if possessor.endswith("out"):
                poss_x = snapshot["ball_x"].values
                poss_y = snapshot["ball_y"].values
            poss_x = snapshot[f"{possessor}_x"].values
            poss_y = snapshot[f"{possessor}_y"].values
            poss_vx = snapshot[f"{possessor}_vx"].values
            poss_vy = snapshot[f"{possessor}_vy"].values
            poss_speed = snapshot[f"{possessor}_speed"].values

        # Make the attacking team play from left to right for a given snapshot
        if target[:4] == "away":
            target_x = config.FIELD_SIZE[0] - target_x
            target_y = config.FIELD_SIZE[1] - target_y
            target_vx = -target_vx
            target_vy = -target_vy
            opponent_x = config.FIELD_SIZE[0] - opponent_x
            opponent_y = config.FIELD_SIZE[1] - opponent_y
            opponent_vx = -opponent_vx
            opponent_vy = -opponent_vy

            if possessor_aware:
                poss_x = config.FIELD_SIZE[0] - poss_x
                poss_y = config.FIELD_SIZE[1] - poss_y
                poss_vx = -poss_vx
                poss_vy = -poss_vy

        goal_x = config.FIELD_SIZE[0]
        goal_y = config.FIELD_SIZE[1] / 2
        goal_dx, goal_dy, goal_dists = utils.calc_dist(target_x, target_y, goal_x, goal_y)

        _, _, opp_goal_dist = utils.calc_dist(opponent_x, opponent_y, goal_x, goal_y)
        closer_opps = sum(opp_goal_dist <= goal_dists[0])

        if "ball_z" in snapshot.columns:
            ball_z = snapshot["ball_z"].iloc[-1]
        else:
            ball_z = 0

        event_features = [
            # Possessor-independent features
            target_x[0],
            target_y[0],
            min(target_y[0], (config.FIELD_SIZE[1] - target_y[0])),
            goal_dists[0],
            target_vx[0],
            target_vy[0],
            target_speed[0],
            target_accel[0],
            goal_dx[0] / (goal_dists[0] + eps),  # Cosine between each player-goal line and the x-axis
            goal_dy[0] / (goal_dists[0] + eps),  # Sine between each player-goal line and the x-axis
            ball_z,
            nearest_opponent,
            nearby_opponents,
            closer_opps,
        ]

        if action_type == "shot":
            blockers = self.count_opponents_in_triangle(
                target_x[0], target_y[0], goal_x[0], goal_y[0], opponent_x, opponent_y
            )
            event_features.append(blockers)

        if possessor_aware:  # Attach possessor features
            poss_dx, poss_dy, poss_dists = utils.calc_dist(target_x, target_y, poss_x, poss_y)
            _, _, poss_goal_dist = utils.calc_dist(poss_x, poss_y, goal_x, goal_y)
            _, _, poss_opp_dist = utils.calc_dist(poss_x, poss_y, opponent_x, opponent_y)
            poss_vangles = utils.calc_angle(target_vx, target_vy, poss_vx, poss_vy, eps=eps)
            closer_opps_poss = sum(opp_goal_dist.flatten() <= poss_goal_dist[0])
            nearest_opponent_poss = min(poss_opp_dist)
            opps_in_path = self.count_opponents_in_corridor(
                target_x[0], target_y[0], poss_x[0], poss_y[0], opponent_x, opponent_y, width=10
            )
            nearest_opponent_passingline = self.closest_opponent_distance(
                target_x[0], target_y[0], poss_x[0], poss_y[0], opponent_x, opponent_y
            )
            event_features.extend(
                [
                    poss_dists[0],
                    poss_x[0],
                    poss_y[0],
                    min(poss_y[0], (config.FIELD_SIZE[1] - poss_y[0])),
                    poss_goal_dist[0],
                    poss_vx[0],
                    poss_vy[0],
                    poss_speed[0],
                    poss_dx[0] / (poss_dists[0] + eps),  # Cosine between each player-possessor line and the x-axis
                    poss_dy[0] / (poss_dists[0] + eps),  # Sine between each player-possessor line and the x-axis
                    np.cos(poss_vangles)[0],  # Cosine between each player's velocity and the possessor's velocity
                    np.sin(poss_vangles)[0],  # Sine between each player's velocity and the possessor's velocity
                    nearest_opponent_poss,
                    closer_opps_poss,
                    opps_in_path,
                    nearest_opponent_passingline,
                ]
            )
        return np.stack(event_features, axis=-1)  # [T, N, x]

    def generate_feature_1D(self, task: str, at_receiving=False, flip_tackle_poss=False, verbose=True) -> torch.Tensor:
        """
        Generates a 1D feature tensor per action for different tasks.

        Arguments:
        - `task`: Determines which entity's features are extracted.
            Options: ["return", "pass_success", "shot_blocking", "pass_return"]
        - `at_receiving`: If True, extract features at ball reception instead of action moment.
        """

        if "ball_accel" not in self.traces.columns:
            self.traces = tp.calc_physical_features(self.traces, self.fps)

        if at_receiving:
            self.adjust_receiving_tags()

        feature_tensors = []

        for period in self.events["period_id"].unique():
            period_traces = self.traces[self.traces["period_id"] == period]
            period_actions = self.actions[self.actions["period_id"] == period]
            action_indices = np.intersect1d(period_actions.index, self.labels[:, 0].long().numpy())

            tqdm_desc = f"Generating 1D features for task '{task}', P{period}"
            iterator = tqdm(action_indices, desc=tqdm_desc) if verbose else action_indices

            for i in iterator:
                frame = period_actions.at[i, "frame"]
                possessor = period_actions.at[i, "object_id"]
                receiver = period_actions.at[i, "intent_id"] if "intent_id" in period_actions else None

                # Select the correct target for feature extraction based on the task
                if task == "return" or task == "shot_blocking":
                    feature_target = possessor  # Extract features about the possessor
                elif task == "pass_success_return":
                    feature_target = receiver  # Extract features about the intended receiver

                if frame == frame and feature_target == feature_target:  # Skip NaN cases
                    snapshot = period_traces.loc[frame:frame].dropna(axis=1, how="all").copy()
                    snapshot["target_id"] = feature_target
                    snapshot["possessor_id"] = " "
                    if task == "pass_success_return":
                        snapshot["possessor_id"] = possessor

                    # Extract features using possessor-aware calculation
                    event_features = self.calculate_event_features(snapshot, task)
                    feature_vector = torch.FloatTensor(event_features)

                    feature_tensors.append(feature_vector)

                else:
                    feature_tensors.append(torch.full((event_features.shape[-1],), -1.0))

        return torch.stack(feature_tensors)  # [B, feature_dim]

    def save(self, game_id, feature_dir="data/ajax/handcrafted/features", label_dir="data/ajax/handcrafted/labels"):
        if self.features is not None:
            os.makedirs(feature_dir + "/pt", exist_ok=True)
            torch.save(self.features, f"{feature_dir}/pt/{game_id}.pt")
            os.makedirs(feature_dir + "/np", exist_ok=True)
            np.save(f"{feature_dir}/np/{game_id}.npy", self.features.numpy())

        if self.labels is not None:
            os.makedirs(label_dir + "/pt", exist_ok=True)
            torch.save(self.labels, f"{label_dir}/pt/{game_id}.pt")
            os.makedirs(label_dir + "/np", exist_ok=True)
            np.save(f"{label_dir}/np/{game_id}.npy", self.labels.numpy())


tasks = ["return", "pass_success_return", "shot_blocking"]

if __name__ == "__main__":
    task = "return"
    if task == "return":
        action_type = "all"
    elif task == "pass_success_return":
        action_type = "pass"
    elif task == "shot_blocking":
        action_type = "shot"
    feature_dir = f"data/ajax/handcrafted/{action_type}_features"
    label_dir = f"data/ajax/handcrafted/{action_type}_labels"

    # tracking_files = np.sort(os.listdir("data/ajax/tracking_processed"))
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

        eng = FeatureEngineer(events, traces, game_lineup, xg_model, action_type)
        eng.labels = eng.generate_label_tensors()
        eng.features = eng.generate_feature_1D(task=task)

        action_indices = eng.labels[:, 0].numpy().astype(int)
        assert np.all(np.sort(action_indices) == action_indices)

        eng.save(game_id, feature_dir, label_dir)
        print(f"Successfully saved for {eng.labels.shape[0]} events.")
