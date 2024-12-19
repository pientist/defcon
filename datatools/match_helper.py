import os
import sys
from typing import Any, Dict, List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from datatools.xg_model import XGModel
from utils import sparsify_edges


class MatchHelper:
    def __init__(self, traces: pd.DataFrame, events: pd.DataFrame, pitch_size: tuple = (104, 68), exclude_reds=False):
        self.traces = traces
        self.events = pd.merge(traces["episode"].reset_index(), events)
        self.pitch_size = pitch_size
        self.exclude_reds = exclude_reds  # whether to exclude short-handed periods (usually due to red cards)
        self.max_players = 22

        self.rotate_pitch_per_phase()
        self.passes = self.filter_passes()
        self.runs = None

        self.features = None
        self.labels = None

    @staticmethod
    def is_home_on_left(traces: pd.DataFrame, halfline_x=52):
        home_x_cols = [c for c in traces.columns if c[0] == "H" and c.endswith("_x")]
        away_x_cols = [c for c in traces.columns if c[0] == "A" and c.endswith("_x")]

        home_gk = (traces[home_x_cols].mean() - halfline_x).abs().idxmax()[:3]
        away_gk = (traces[away_x_cols].mean() - halfline_x).abs().idxmax()[:3]

        home_gk_x = traces[f"{home_gk}_x"].mean()
        away_gk_x = traces[f"{away_gk}_x"].mean()

        return home_gk_x < away_gk_x

    # To make the home team always play from left to right
    def rotate_pitch_per_phase(self):
        for phase in self.traces["phase"].unique():
            phase_traces = self.traces[self.traces["phase"] == phase].dropna(axis=1, how="all")

            x_cols = [c for c in phase_traces.columns if c.endswith("_x")]
            y_cols = [c for c in phase_traces.columns if c.endswith("_y")]
            vx_cols = [c for c in phase_traces.columns if c.endswith("_vx")]
            vy_cols = [c for c in phase_traces.columns if c.endswith("_vy")]

            if not MatchHelper.is_home_on_left(phase_traces, halfline_x=self.pitch_size[0] / 2):
                self.traces.loc[phase_traces.index, x_cols] = self.pitch_size[0] - phase_traces[x_cols]
                self.traces.loc[phase_traces.index, y_cols] = self.pitch_size[1] - phase_traces[y_cols]
                self.traces.loc[phase_traces.index, vx_cols] = -phase_traces[vx_cols]
                self.traces.loc[phase_traces.index, vy_cols] = -phase_traces[vy_cols]

        self.events["x"] = self.traces.loc[self.events["frame"], "ball_x"].values
        self.events["y"] = self.traces.loc[self.events["frame"], "ball_y"].values

    def filter_passes(self):
        self.events[["home_away", "player_code"]] = self.events[["home_away", "player_code"]].fillna("O")
        self.events["event_types"] = self.events["event_types"].fillna("")

        passes = self.events[
            self.events["event_types"].str.contains("pass") | self.events["event_types"].str.contains("cross")
        ].copy()
        passes["is_pass"] = 1
        passes["success"] = 0

        for s in self.events["session"].unique():
            session_passes = passes[passes["session"] == s]
            last_idx = self.events[self.events["session"] == s].index[-1]

            for i in session_passes.index:
                if i == last_idx:
                    passes.drop(i, inplace=True)
                    continue

                frame = self.events.at[i, "frame"]
                cur_types = self.events.at[i, "event_types"]
                cur_player = self.events.at[i, "player_code"]
                next_player = self.events.at[i + 1, "player_code"]

                if cur_player == next_player:
                    passes.drop(i, inplace=True)
                    continue

                passes.at[i, "event_types"] = self.events.at[i, "event_types"]
                self.traces.at[frame, "event_types"] = self.events.at[i, "event_types"]

        return passes

    def label_blocks(self, max_diff=10):
        self.passes["is_blocked"] = 0
        passes = self.passes[self.passes["is_pass"] == 1]

        for i in passes.index:
            event_frame = self.events.at[i, "frame"]
            receive_frame = self.events.at[i + 1, "frame"]

            passer = self.events.at[i, "player_code"]
            receiver = self.events.at[i + 1, "player_code"]

            is_blocked = passer[0] != receiver[0] and receive_frame - event_frame <= max_diff
            self.passes.at[i, "is_blocked"] = int(is_blocked)

    def label_receivers(self, max_angle=45, eps=1e-6):
        # self.actions["intent"] = np.nan
        # self.actions["receiver"] = np.nan
        max_radian = np.pi * max_angle / 180

        tqdm_desc = "Labeling intented receiver per event"
        bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"

        for i in tqdm(self.passes.index, desc=tqdm_desc, bar_format=bar_format):
            event_frame = self.events.at[i, "frame"]
            event_snapshot = self.traces.loc[event_frame]
            event_player = event_snapshot["player_code"]
            self.passes.at[i, "receiver"] = self.events.at[i + 1, "player_code"]

            if self.passes.at[i, "is_pass"] == 0:
                self.passes.at[i, "intent"] = event_player

            else:
                receive_frame = min(self.events.at[i + 1, "frame"], event_frame + 50)
                receive_snapshot = self.traces.loc[receive_frame]

                players = [c[:-2] for c in event_snapshot.dropna().index if c.endswith("_x")]
                teammates = [p for p in players if p[0] == event_player[0] and p != event_player]

                start_x = event_snapshot["ball_x"]
                start_y = event_snapshot["ball_y"]
                end_x = receive_snapshot["ball_x"]
                end_y = receive_snapshot["ball_y"]
                player_x = receive_snapshot[[f"{p}_x" for p in teammates]].values
                player_y = receive_snapshot[[f"{p}_y" for p in teammates]].values

                angles = MatchHelper.calc_angle(start_x, start_y, end_x, end_y, player_x, player_y) + eps
                dists = MatchHelper.calc_dist(player_x, player_y, end_x, end_y)[-1] + eps

                if np.min(angles) < max_radian:
                    scores = (np.min(dists) / dists) * (np.min(angles) / angles)
                    scores = np.where(angles < max_radian, scores, 0)
                    self.passes.at[i, "intent"] = teammates[np.argmax(scores)]

    def label_destinations(self):
        if "intent" not in self.passes.columns:
            self.label_receivers()

        self.passes["intent_x"] = np.nan
        self.passes["intent_y"] = np.nan
        self.passes["end_x"] = np.nan
        self.passes["end_y"] = np.nan
        valid_events = self.events[self.events["event_types"] != ""]

        for i in self.passes.index:
            start_x = self.passes.at[i, "x"]
            start_y = self.passes.at[i, "y"]
            end_x = self.passes.at[i, "end_x"] = self.events.at[i + 1, "x"]
            end_y = self.passes.at[i, "end_y"] = self.events.at[i + 1, "y"]
            self.passes.at[i, "move_dist"] = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

            intent = self.passes.at[i, "intent"]
            receive_frame = self.events.at[i + 1, "frame"]
            receive_types = valid_events.loc[i + 1 :, "event_types"].iloc[0]

            # If there is no intented player or a foul occurs, skip this pass.
            if intent == intent and "freeKick" not in receive_types:
                self.passes.at[i, "intent_x"] = self.traces.at[receive_frame, f"{intent}_x"]
                self.passes.at[i, "intent_y"] = self.traces.at[receive_frame, f"{intent}_y"]

    def label_returns(self, xg_model: XGModel, lookahead_len: int = 10):
        et = self.events["event_types"]

        # self.events["shot"] = (et.str.contains("shot") | et.str.contains("goal ")).astype(int)
        self.events["goal"] = et.str.contains("goal ").astype(int)
        shot_features = xg_model.calc_shot_features(self.events)
        shot_features = xg_model.pred(shot_features)
        self.events["xg"] = 0.0
        self.events.loc[shot_features.index, "xg"] = shot_features["xg"]

        self.events["scores"] = 0.0
        self.events["scores_xg"] = 0.0
        self.events["concedes"] = 0.0
        self.events["concedes_xg"] = 0.0

        for s in self.events["session"].unique():
            session_events = self.events[(self.events["session"] == s) & (self.events["home_away"].isin(["H", "A"]))]
            labels = session_events[["home_away", "goal", "xg"]].copy()

            for i in range(lookahead_len):
                shifted = labels.shift(-i).fillna(0)
                labels[f"sg+{i}"] = shifted["goal"] * (shifted["home_away"] == labels["home_away"]).astype(int)
                labels[f"cg+{i}"] = shifted["goal"] * (shifted["home_away"] != labels["home_away"]).astype(int)
                labels[f"sxg+{i}"] = shifted["xg"] * (shifted["home_away"] == labels["home_away"]).astype(int)
                labels[f"cxg+{i}"] = shifted["xg"] * (shifted["home_away"] != labels["home_away"]).astype(int)

            sg_cols = [c for c in labels.columns if c.startswith("sg+")]
            cg_cols = [c for c in labels.columns if c.startswith("cg+")]
            sxg_cols = [c for c in labels.columns if c.startswith("sxg+")]
            cxg_cols = [c for c in labels.columns if c.startswith("cxg+")]

            self.events.loc[labels.index, "scores"] = labels[sg_cols].sum(axis=1).clip(0, 1).astype(int)
            self.events.loc[labels.index, "scores_xg"] = 1 - (1 - labels[sxg_cols]).prod(axis=1)
            self.events.loc[labels.index, "concedes"] = labels[cg_cols].sum(axis=1).clip(0, 1).astype(int)
            self.events.loc[labels.index, "concedes_xg"] = 1 - (1 - labels[cxg_cols]).prod(axis=1)

        self.passes["scores"] = self.events.loc[self.passes.index, "scores"]
        self.passes["scores_xg"] = self.events.loc[self.passes.index, "scores_xg"]
        self.passes["concedes"] = self.events.loc[self.passes.index, "concedes"]
        self.passes["concedes_xg"] = self.events.loc[self.passes.index, "concedes_xg"]

    def construct_labels(self, xg_model: XGModel = None, lookahead_len: int = 10):
        self.label_blocks()
        self.label_receivers()
        self.label_destinations()
        self.label_returns(xg_model=xg_model, lookahead_len=lookahead_len)

        labels_list = []
        for phase in self.passes["phase"].unique():
            phase_traces = self.traces[self.traces["phase"] == phase]

            home_players = [c[:-2] for c in phase_traces.dropna(axis=1).columns if c[0] == "H" and c.endswith("_x")]
            away_players = [c[:-2] for c in phase_traces.dropna(axis=1).columns if c[0] == "A" and c.endswith("_x")]

            if self.exclude_reds and len(home_players) + len(away_players) < self.max_players:
                continue

            phase_actions = self.passes[self.passes["phase"] == phase]
            for i in phase_actions.index:
                try:
                    intent = phase_actions.at[i, "intent"]
                    if intent != intent:
                        intent_idx = -1
                    elif intent[0] == "H":
                        intent_idx = home_players.index(intent)
                    else:
                        intent_idx = away_players.index(intent)

                    receiver = phase_actions.at[i, "receiver"]
                    if receiver == "O":
                        receiver_idx = -1
                    elif phase_actions.at[i, "home_away"] == "H":
                        receiver_idx = (home_players + away_players).index(receiver)
                    else:
                        receiver_idx = (away_players + home_players).index(receiver)

                except ValueError:
                    continue

                intent_x = phase_actions.at[i, "intent_x"]
                intent_y = phase_actions.at[i, "intent_y"]
                end_x = phase_actions.at[i, "end_x"]
                end_y = phase_actions.at[i, "end_y"]
                if phase_actions.at[i, "player_code"][0] == "A":
                    intent_x = self.pitch_size[0] - intent_x
                    intent_y = self.pitch_size[1] - intent_y
                    end_x = self.pitch_size[0] - end_x
                    end_y = self.pitch_size[1] - end_y

                labels_list.append(
                    [
                        i,
                        len(home_players) + len(away_players),
                        intent_idx,
                        receiver_idx,
                        intent_x,
                        intent_y,
                        end_x,
                        end_y,
                        phase_actions.at[i, "move_dist"],
                        phase_actions.at[i, "is_pass"],
                        phase_actions.at[i, "is_blocked"],
                        phase_actions.at[i, "success"],
                        phase_actions.at[i, "scores"],
                        phase_actions.at[i, "scores_xg"],
                        phase_actions.at[i, "concedes"],
                        phase_actions.at[i, "concedes_xg"],
                    ]
                )

        self.labels = torch.FloatTensor(labels_list)

    @staticmethod
    def calc_dist(x: np.ndarray, y: np.ndarray, origin_x: np.ndarray, origin_y: np.ndarray):
        dist_x = (x - origin_x).astype(float)
        dist_y = (y - origin_y).astype(float)
        return dist_x, dist_y, np.sqrt(dist_x**2 + dist_y**2)

    @staticmethod
    def calc_angle(
        ax: np.ndarray,
        ay: np.ndarray,
        bx: np.ndarray,
        by: np.ndarray,
        cx: np.ndarray = None,
        cy: np.ndarray = None,
        eps: float = 1e-6,
    ):
        if cx is None or cy is None:
            # Calculate angles between the vectors a and b
            a_len = np.sqrt(ax**2 + ay**2) + eps
            b_len = np.sqrt(bx**2 + by**2) + eps
            cos = np.clip((ax * bx + ay * by) / (a_len * b_len), -1, 1)

        else:
            # Calculate angles between the lines AB and AC
            ab_x = (bx - ax).astype(float)
            ab_y = (by - ay).astype(float)
            ab_len = np.sqrt(ab_x**2 + ab_y**2) + eps

            ac_x = (cx - ax).astype(float)
            ac_y = (cy - ay).astype(float)
            ac_len = np.sqrt(ac_x**2 + ac_y**2) + eps

            cos = np.clip((ab_x * ac_x + ab_y * ac_y) / (ab_len * ac_len), -1, 1)

        return np.arccos(cos)

    @staticmethod
    def construct_event_features(
        traces: pd.DataFrame,
        possessor: str = None,
        home_cols: list = None,
        away_cols: list = None,
        pitch_size: tuple = (104, 68),
        eps: float = 1e-6,
    ):
        if home_cols is None or away_cols is None:
            home_cols = [c for c in traces.dropna(axis=1).columns if c[0] == "H"]
            away_cols = [c for c in traces.dropna(axis=1).columns if c[0] == "A"]

        passer = traces["player_code"].iloc[-1] if possessor is None else possessor
        passer_aware = len(passer) == 3  # if passer in ["H", "A"], do not calculate passer features

        player_cols = home_cols + away_cols if passer[0] == "H" else away_cols + home_cols
        players = [c[:3] for c in player_cols if c.endswith("_x")]

        is_teammate = np.tile([int(p[0] == passer[0]) for p in players], (len(traces), 1))

        if passer_aware:
            if traces[traces["player_code"] == passer].empty:
                is_one_touch = np.zeros((len(traces), len(players)))
                is_aerial_duel = np.zeros((len(traces), len(players)))

            else:
                event_frame = traces[traces["player_code"] == passer].index[-1]
                event_types = traces.at[event_frame, "event_types"]

                if len(traces) == 1 or traces.at[event_frame - 1, "player_code"] != passer:
                    is_one_touch = np.ones((len(traces), len(players)))
                else:
                    is_one_touch = np.zeros((len(traces), len(players)))

                if event_types == event_types and "aerial" in event_types:
                    is_aerial_duel = np.ones((len(traces), len(players)))
                else:
                    is_aerial_duel = np.zeros((len(traces), len(players)))

        else:  # if passer in ["H", "A"], hypothesize a non-aerial one-touch pass
            is_one_touch = np.ones((len(traces), len(players)))
            is_aerial_duel = np.zeros((len(traces), len(players)))

        player_x = traces[player_cols[0::6]].values
        player_y = traces[player_cols[1::6]].values
        player_vx = traces[player_cols[2::6]].values
        player_vy = traces[player_cols[3::6]].values
        player_speeds = traces[player_cols[4::6]].values
        player_accels = traces[player_cols[5::6]].values

        if passer_aware:  # Calculate passer features only when there is a valid passer label
            passer_x = traces[f"{passer}_x"].values[:, np.newaxis]
            passer_y = traces[f"{passer}_y"].values[:, np.newaxis]
            passer_vx = traces[f"{passer}_vx"].values[:, np.newaxis]
            passer_vy = traces[f"{passer}_vy"].values[:, np.newaxis]

        # Make the attacking team play from left to right for a given pass scene
        if passer[0] == "A":
            player_x = pitch_size[0] - player_x
            player_y = pitch_size[1] - player_y
            player_vx = -player_vx
            player_vy = -player_vy

            if passer_aware:
                passer_x = pitch_size[0] - passer_x
                passer_y = pitch_size[1] - passer_y
                passer_vx = -passer_vx
                passer_vy = -passer_vy

        goal_x = pitch_size[0]
        goal_y = pitch_size[1] / 2

        goal_dx, goal_dy, goal_dists = MatchHelper.calc_dist(player_x, player_y, goal_x, goal_y)

        if passer_aware:
            is_passer = np.tile((np.array(players) == passer).astype(int), (len(traces), 1))
            passer_dx, passer_dy, passer_dists = MatchHelper.calc_dist(player_x, player_y, passer_x, passer_y)
            # pos_angles = MatchHelper.calc_angle(passer_x, passer_y, goal_x, goal_y, player_x, player_y)
            passer_vangles = MatchHelper.calc_angle(player_vx, player_vy, passer_vx, passer_vy, eps=eps)

        event_features = [
            # Binary features
            is_teammate,
            is_one_touch,
            is_aerial_duel,
            # Passer-independent features
            player_x,
            player_y,
            player_vx,
            player_vy,
            player_speeds,
            player_accels,
            goal_dists,
            goal_dx / (goal_dists + eps),  # Cosine between each player-goal line and the x-axis
            goal_dy / (goal_dists + eps),  # Sine between each player-goal line and the x-axis
        ]

        if passer_aware:  # Attach passer features
            event_features.extend(
                [
                    is_passer,
                    passer_dists,
                    passer_dx / (passer_dists + eps),  # Cosine between each player-passer line and the x-axis
                    passer_dy / (passer_dists + eps),  # Sine between each player-passer line and the x-axis
                    np.cos(passer_vangles),  # Cosine between each player's velocity and the passer's velocity
                    np.sin(passer_vangles),  # Sine between each player's velocity and the passer's velocity
                ]
            )

        return np.stack(event_features, axis=-1)  # [T, N, x]

    def construct_graph_features(self):
        features_list = []

        tqdm_desc = "Calculating features per phase"
        bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"

        for phase in tqdm(self.passes["phase"].unique(), desc=tqdm_desc, bar_format=bar_format):
            phase_actions = self.passes[self.passes["phase"] == phase]
            phase_traces = self.traces[self.traces["phase"] == phase]

            home_cols = [c for c in phase_traces.dropna(axis=1).columns if c[0] == "H"]
            away_cols = [c for c in phase_traces.dropna(axis=1).columns if c[0] == "A"]

            if self.exclude_reds and len(home_cols) + len(away_cols) < self.max_players * 6:
                continue

            for i in np.intersect1d(phase_actions.index, self.labels[:, 0].long().numpy()):
                event_frame = phase_actions.at[i, "frame"]
                event_snapshot = phase_traces.loc[event_frame:event_frame].dropna(axis=1, how="all").copy()
                event_features = MatchHelper.construct_event_features(event_snapshot, None, home_cols, away_cols)
                padding_features = -torch.ones((self.max_players - event_features.shape[1], event_features.shape[2]))
                event_features = torch.cat([torch.FloatTensor(event_features[0]), padding_features], 0)
                features_list.append(event_features)

        node_attr = torch.stack(features_list, axis=0)  # [B, N, x]
        distances = torch.cdist(node_attr[..., 3:5], node_attr[..., 3:5], p=2)  # [B, N, N]
        teammates = (node_attr[..., 0].unsqueeze(-1) == node_attr[..., 0].unsqueeze(-2)).float()  # [B, N, N]

        self.features = []
        for i in range(node_attr.shape[0]):
            node_mask = node_attr[i][:, 0] != -1
            node_attr_i = node_attr[i][node_mask]

            distances_i = distances[i][node_mask][:, node_mask]
            teammates_i = teammates[i][node_mask][:, node_mask]
            edge_index, _ = dense_to_sparse(torch.ones_like(distances_i))

            distances_i = distances_i[edge_index[0], edge_index[1]]
            teammates_i = teammates_i[edge_index[0], edge_index[1]]
            edge_attr_i = torch.stack([distances_i, teammates_i], dim=-1)  # [N * N, 2]

            graph = Data(x=node_attr_i, edge_index=edge_index.clone(), edge_attr=edge_attr_i)
            self.features.append(graph)

    def construct_sequential_features(self, scene_len=30):
        features_list = []

        tqdm_desc = "Calculating features per phase"
        bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"

        for phase in tqdm(self.passes["phase"].unique(), desc=tqdm_desc, bar_format=bar_format):
            phase_actions = self.passes[self.passes["phase"] == phase]
            phase_traces = self.traces[self.traces["phase"] == phase]

            home_cols = [c for c in phase_traces.dropna(axis=1).columns if c[0] == "H"]
            away_cols = [c for c in phase_traces.dropna(axis=1).columns if c[0] == "A"]
            if len(home_cols) + len(away_cols) < self.max_players * 6:
                continue

            for i in phase_actions.index:
                event_frame = phase_actions.at[i, "frame"]
                scene_start = event_frame - scene_len + 1
                event_traces = phase_traces.loc[scene_start:event_frame].dropna(axis=1, how="all").copy()

                event_features = MatchHelper.construct_event_features(event_traces, None, home_cols, away_cols)
                if len(event_features) < scene_len:
                    pad_width = ((scene_len - len(event_features), 0), (0, 0), (0, 0))
                    event_features = np.pad(event_features, pad_width, "edge")

                features_list.append(event_features)

        self.features = torch.FloatTensor(np.stack(features_list, axis=0))

    def filter_features_and_labels(self, args: Dict[str, Any], replace=False) -> Tuple[List[Data], torch.Tensor]:
        condition = torch.ones(self.labels.shape[0]).bool()
        features = []
        labels = []

        if args["target"].startswith("success_"):
            condition &= self.labels[:, 11] == 1
        elif args["target"].startswith("failure_"):
            condition &= self.labels[:, 11] == 0

        if args["pass_only"]:
            condition &= self.labels[:, 9] == 1
        else:
            condition &= (self.labels[:, 8] < 5) | (self.labels[:, 9] == 1)

        intended_only = "intent" in args["target"] or args["target"].startswith("failure_")
        intent_aware = args["target"].startswith("intent_") or args["target"].startswith("failure_")

        if intended_only or intent_aware:
            condition &= self.labels[:, 2] != -1

        if not args["residual"]:
            condition &= self.labels[:, 3] != -1

        for i in condition.nonzero()[:, 0].numpy():
            graph: Data = self.features[i].clone()
            try:
                passer_index = torch.nonzero(graph.x[:, 12] == 1).item()
            except RuntimeError:
                continue

            if not args["passer_aware"] and args["xy_only"]:
                graph.x = graph.x[..., :7]
            elif not args["passer_aware"] and not args["xy_only"]:
                graph.x = graph.x[..., :12]
            elif args["passer_aware"] and args["xy_only"]:
                graph.x = torch.cat([graph.x[..., :7], graph.x[..., 12:13]], -1)

            if "one_touch_aware" in args.keys() and not args["one_touch_aware"]:
                graph.x = torch.cat([graph.x[..., :1], graph.x[..., 3:]], -1)

            if args["target"] in ["intent_success", "failure_receiver"]:
                assert graph.x.shape[1] + 1 == args["node_in_dim"]
            else:
                assert graph.x.shape[1] == args["node_in_dim"]

            if args["sparsify"] == "distance":
                assert args["passer_aware"]
                graph = sparsify_edges(graph, "distance", passer_index, args["max_edge_dist"])
            elif args["sparsify"] == "delaunay":
                graph = sparsify_edges(graph, "delaunay")

            features.append(graph)
            labels.append(self.labels[i])

        if replace:
            self.features = features
            self.labels = torch.stack(labels, axis=0)
            return self.features, self.labels
        else:
            return features, torch.stack(labels, axis=0)

    def save(self, match_id, feature_dir="data/train_features", label_dir="data/train_labels"):
        if self.features is not None:
            os.makedirs(feature_dir, exist_ok=True)
            torch.save(self.features, f"{feature_dir}/{match_id}.pt")

        if self.labels is not None:
            os.makedirs(label_dir, exist_ok=True)
            torch.save(self.labels, f"{label_dir}/{match_id}.pt")


if __name__ == "__main__":
    league = "2022 K League 1"
    split = "train"
    feature_dir = f"data/{split}_features"
    label_dir = f"data/{split}_labels"
    match_ids = np.sort([int(f.split(".")[0]) for f in os.listdir(f"data/{league}/traces") if f.endswith(".csv")])

    xg_model = XGModel()
    xg_model.train()

    for i, match_id in enumerate(match_ids):
        print(f"\n[{i}] {match_id}")
        traces = pd.read_csv(f"data/{league}/traces/{match_id}.csv", header=0, index_col=0, low_memory=False)
        events = pd.read_csv(f"data/{league}/events/{match_id}.csv", header=0)

        helper = MatchHelper(traces, events)
        helper.construct_labels(xg_model=xg_model)
        helper.construct_graph_features()
        assert len(helper.features) == len(helper.labels)

        helper.save(match_id, feature_dir=feature_dir, label_dir=label_dir)
        print(f"Successfully saved for {helper.labels.shape[0]} events.")
