import os
import sys
from collections import defaultdict

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from datatools import config
from datatools.feature import FeatureEngineer
from datatools.trace_snapshot import TraceSnapshot
from datatools.utils import sparsify_edges
from inference import (
    find_active_players,
    inference,
    inference_posterior,
    inference_success,
)
from models.utils import load_model


class DEFCON:
    def __init__(
        self,
        eng: FeatureEngineer,
        scoring_model_id="scoring/01",
        pass_intent_model_id="intent/01",
        pass_success_model_id="intent_success/01",
        pass_scoring_model_id="intent_scoring/01",
        shot_blocking_model_id="shot_blocking/01",
        posterior_model_id="failure_receiver/01",
        likelihood_model_id="oppo_agn_intent/01",
        device="cuda",
    ):
        self.eng = eng
        self.device = device

        self.scoring_model = load_model(scoring_model_id, device)
        self.pass_intent_model = load_model(pass_intent_model_id, device)
        self.pass_success_model = load_model(pass_success_model_id, device)
        self.pass_scoring_model = load_model(pass_scoring_model_id, device)
        self.shot_blocking_model = load_model(shot_blocking_model_id, device)
        self.posterior_model = load_model(posterior_model_id, device)
        self.likelihood_model = load_model(likelihood_model_id, device)

        self.shot_eng = None  # To generate features for shot_blocking_model

        self.xreturns = None
        self.intent_probs = None
        self.success_probs = None
        self.goal_if_success = None
        self.goal_if_failure = None
        self.posteriors = None
        self.receive_probs = None
        self.likelihoods = None

        self.advantages = None
        self.team_credits = None
        self.player_credits = None

        self.player_scores = None

    def estimate_components(self):
        self.xreturns = inference(self.eng, self.scoring_model, self.device)[0]

        actions = self.eng.actions.copy()
        shot_features = self.eng.xg_model.calc_shot_features(actions)
        xg_unblocked = self.eng.xg_model.pred(shot_features)  # xG if the shot were unblocked

        freekicks = actions[actions["spadl_type"].str.startswith("freekick")]
        corners = actions[actions["spadl_type"].str.startswith("corner")]
        before_corners = actions[actions["next_type"].str.startswith("corner")]

        self.xreturns.loc[freekicks.index, "value_before"] = xg_unblocked.loc[freekicks.index].values
        self.xreturns.loc[corners.index, "value_before"] = 0.01
        self.xreturns.loc[before_corners.index, "value_after"] = 0.01

        event_teams = actions["object_id"].apply(lambda x: x[:4])
        receive_teams = actions["receiver_id"].fillna(actions["object_id"]).apply(lambda x: x[:4])
        self.xreturns["value_before"] *= np.where(actions["action_type"] == "tackle", -1, 1)
        self.xreturns["value_after"] *= np.where(event_teams != receive_teams, -1, 1)

        is_goal_scoring = (actions["action_type"] == "shot") & (actions["outcome"])
        self.xreturns.loc[is_goal_scoring, "value_after"] = 1.0

        self.xreturns["diff"] = self.xreturns["value_after"] - self.xreturns["value_before"]

        self.intent_probs = inference(self.eng, self.pass_intent_model, self.device)[0]
        self.success_probs = inference_success(self.eng, self.pass_success_model, self.device)
        self.goal_if_success, self.goal_if_failure = inference(self.eng, self.pass_scoring_model, self.device)
        self.posteriors = inference_posterior(self.eng, self.posterior_model, self.device)

        self.likelihoods = inference(self.eng, self.likelihood_model, self.device)[0]
        self.likelihoods["home_goal"] = np.where(self.likelihoods["home_goal"].notna(), 1, np.nan)
        self.likelihoods["away_goal"] = np.where(self.likelihoods["away_goal"].notna(), 1, np.nan)

        if self.shot_blocking_model is not None:
            potential_shots = actions[(xg_unblocked > 0.01) & (actions["spadl_type"] != "tackle")].copy()
            self.shot_eng = FeatureEngineer(
                potential_shots,
                self.eng.traces,
                self.eng.lineup,
                self.eng.xg_model,
                action_type="predefined",
            )
            self.shot_eng.labels = self.shot_eng.generate_label_tensors()
            self.shot_eng.features = self.shot_eng.generate_feature_graphs(verbose=False)
            block_probs: pd.Series = inference(self.shot_eng, self.shot_blocking_model, self.device)[0]

            self.success_probs["home_goal"] = np.nan
            self.success_probs["away_goal"] = np.nan
            self.goal_if_success["home_goal"] = np.nan
            self.goal_if_success["away_goal"] = np.nan
            self.goal_if_failure["home_goal"] = np.nan
            self.goal_if_failure["away_goal"] = np.nan

            for i in self.success_probs.index:
                if actions.at[i, "object_id"][:4] == "home":
                    target_goal = "away_goal" if actions.at[i, "action_type"] == "tackle" else "home_goal"
                else:
                    target_goal = "home_goal" if actions.at[i, "action_type"] == "tackle" else "away_goal"

                self.success_probs.at[i, target_goal] = 1 - block_probs.at[i] if i in block_probs.index else 0.0
                self.goal_if_success.at[i, target_goal] = xg_unblocked.at[i] if i not in corners.index else 0.0
                self.goal_if_failure.at[i, target_goal] = 0.0

    def find_unique_options(self, action_index: int, intent: str = None, max_dist=50, eps=0.4) -> pd.DataFrame:
        options = self.success_probs.loc[action_index].drop(["home_goal", "away_goal"]).dropna().index
        if intent is None:
            intent = self.eng.actions.at[action_index, "intent_id"]

        # Take distance and angle to the possessor as input features for DBSCAN
        data_index = torch.nonzero(self.eng.labels[:, 0] == action_index).item()
        graph = self.eng.features[data_index]
        polar_features = graph.x[(graph.x[:, 0] == 1) & (graph.x[:, 2] == 0), -5:-2].numpy().round(4)
        polar_features[:, 0] = polar_features[:, 0] / max_dist * 2

        # Cluster options and choose only one option per cluster
        dbscan = DBSCAN(eps=eps, min_samples=1)
        clusters = pd.DataFrame(dbscan.fit_predict(polar_features), index=options, columns=["cluster"])

        event_credits = self.team_credits.loc[action_index].dropna()
        unique_options = event_credits[options].groupby(clusters["cluster"]).idxmax()
        if not intent.endswith("_goal"):
            unique_options.at[clusters.at[intent, "cluster"]] = intent

        clusters["mask"] = clusters.index.isin(unique_options.values).astype(int)
        return clusters

    def compute_team_credits(self, mask_likelihood=0.03):
        actions = self.eng.actions
        actions["intent_id"] = np.where(actions["intent_id"].notna(), actions["intent_id"], actions["object_id"])

        indices = self.intent_probs.index
        state_values = self.xreturns.loc[indices, ["value_before"]].values
        self.advantages = self.goal_if_success - state_values

        adv_prevented = self.advantages.copy()
        adv_intended = [self.advantages.at[i, actions.at[i, "intent_id"]] for i in indices]
        adv_intended = pd.Series(adv_intended, index=indices)

        for i in adv_prevented.index:
            adv_i = adv_prevented.loc[i]
            if actions.at[i, "action_type"] == "tackle" or self.eng.events.at[i, "goal"]:
                adv_prevented.loc[i] = adv_i.mask(adv_i.notna(), 0)
            else:
                adv_prevented.loc[i] = adv_i.mask(adv_i <= max(adv_intended.at[i], 0), 0)

        # team_credits = (advantages - adv_intended.values[:, np.newaxis]).clip(0) * self.success_probs
        self.team_credits: pd.DataFrame = (1 - self.success_probs) * adv_prevented
        self.team_credits = self.team_credits.astype(float).mask(self.likelihoods < mask_likelihood, 0)

        self.team_credits["player_id"] = actions.loc[self.team_credits.index, "object_id"]
        self.team_credits["receiver_id"] = actions.loc[self.team_credits.index, "receiver_id"]
        self.team_credits["outcome"] = actions.loc[self.team_credits.index, "outcome"]
        self.team_credits["intent_id"] = actions.loc[self.team_credits.index, "intent_id"]

        for i in tqdm(self.team_credits.index, desc="team_credit"):
            intent = actions.at[i, "intent_id"]
            option_clusters = self.find_unique_options(i, intent)
            self.team_credits.loc[i, option_clusters.index] *= option_clusters["mask"].values

            if actions.at[i, "spadl_type"] == "shot" and not actions.at[i, "blocked"]:
                self.team_credits.at[i, intent] = -self.goal_if_success.at[i, intent]
                # self.team_credits.at[i, intent] = -max(self.advantages.at[i, intent], 0)
            elif actions.at[i, "action_type"] == "tackle":
                self.team_credits.at[i, intent] = self.xreturns.at[i, "diff"]
            else:
                # elif actions.at[i, "action_type"] != "clearance":
                self.team_credits.at[i, intent] = -self.xreturns.at[i, "diff"]

    def compute_player_credits(self):
        # Reshape team_credits to have the same indices with posteriors
        cols_to_drop = ["player_id", "intent_id", "receiver_id", "outcome"]
        reshaped_team_credits = []

        for i in self.team_credits.index:
            reshaped_i = self.team_credits.loc[i].drop(index=cols_to_drop).dropna().rename("team_credit")
            reshaped_i = reshaped_i.reset_index().rename(columns={"index": "option"})

            reshaped_i["index"] = i
            reshaped_i["intended"] = reshaped_i["option"] == self.eng.actions.at[i, "intent_id"]
            reshaped_i["outcome"] = self.eng.actions.at[i, "outcome"]

            reshaped_i.loc[~reshaped_i["intended"] & (reshaped_i["team_credit"] > 0), "defense_type"] = "prevent"

            if self.eng.actions.at[i, "action_type"] == "tackle":
                reshaped_i.loc[reshaped_i["intended"], "defense_type"] = "tackle"
                reshaped_i.loc[reshaped_i["intended"], "interceptor"] = self.eng.actions.at[i, "object_id"]

            elif self.eng.actions.at[i, "action_type"] == "shot":
                next_type = self.eng.actions.at[i, "next_type"]
                if next_type in config.SET_PIECE_OOP or self.eng.actions.at[i, "outcome"]:
                    reshaped_i.loc[reshaped_i["intended"], "defense_type"] = "concede"
                else:
                    reshaped_i.loc[reshaped_i["intended"], "interceptor"] = self.eng.actions.at[i, "next_player_id"]
                    if next_type in ["shot_block", "keeper_save"]:
                        reshaped_i.loc[reshaped_i["intended"], "defense_type"] = next_type
                    else:
                        reshaped_i.loc[reshaped_i["intended"], "defense_type"] = "interception"

            elif self.eng.actions.at[i, "action_type"] == "pass" and self.eng.actions.at[i, "outcome"]:
                reshaped_i.loc[reshaped_i["intended"], "defense_type"] = "concede"

            elif self.eng.actions.at[i, "next_type"] in config.SET_PIECE_OOP:
                reshaped_i.loc[reshaped_i["intended"], "defense_type"] = "induce_out"

            elif self.eng.actions.at[i, "next_type"] not in config.SET_PIECE_OOP:
                reshaped_i.loc[reshaped_i["intended"], "defense_type"] = "interception"
                if self.eng.actions.at[i, "next_type"] in config.DEFENSIVE_TOUCH:
                    reshaped_i.loc[reshaped_i["intended"], "interceptor"] = self.eng.actions.at[i, "next_player_id"]
                else:
                    reshaped_i.loc[reshaped_i["intended"], "interceptor"] = self.eng.actions.at[i, "receiver_id"]

            reshaped_team_credits.append(reshaped_i)

        team_credits = pd.concat(reshaped_team_credits, ignore_index=True)

        # Initialize player credits
        posteriors = self.posteriors.reset_index().copy()
        players = [c for c in posteriors.columns if c[:4] in ["home", "away"]]
        self.player_credits = posteriors[players].astype(float) * 0

        # Credits for conceding/preventing passes
        pass_indices = team_credits["interceptor"].isna() & (team_credits["team_credit"] != 0)
        pass_credits = team_credits.loc[pass_indices, ["team_credit"]].values.astype(float)
        self.player_credits.loc[pass_indices] = posteriors.loc[pass_indices, players] * pass_credits

        # Credits for interceptions and tackles
        intercept_credits = team_credits[team_credits["interceptor"].notna()]
        for i in intercept_credits.index:
            interceptor = team_credits.at[i, "interceptor"]
            self.player_credits.at[i, interceptor] = team_credits.at[i, "team_credit"]

        self.player_credits["index"] = self.posteriors["index"].values
        self.player_credits["option"] = self.posteriors.index
        self.player_credits["defense_type"] = team_credits["defense_type"]
        self.player_credits = self.player_credits[["index", "option", "defense_type"] + players]

    def compute_playing_times(self) -> pd.DataFrame:
        traces = self.eng.traces
        seconds = dict()

        for p in self.eng.lineup["object_id"]:
            inplay_traces = traces[traces[f"{p}_x"].notna() & (traces["ball_state"] == "alive")]
            player_seconds = inplay_traces.groupby("ball_owning_home_away")["timestamp"].count() / self.eng.fps
            if p[:4] == "home":
                seconds[p] = [player_seconds["home"], player_seconds["away"]]
            else:
                seconds[p] = [player_seconds["away"], player_seconds["home"]]

        seconds = pd.DataFrame(seconds, index=["attack_time", "defend_time"]).T
        seconds.index.name = "object_id"

        return seconds

    def evaluate(self, mask_likelihood=0.03):
        if self.success_probs is None:
            self.estimate_components()

        if self.team_credits is None:
            self.compute_team_credits(mask_likelihood)

        if self.player_credits is None:
            self.compute_player_credits()

        player_scores = self.player_credits.drop(["index", "option"], axis=1).groupby("defense_type").sum().T
        player_scores["defcon"] = player_scores.sum(axis=1)
        player_scores = player_scores.reset_index().rename(columns={"index": "object_id"})

        for t in config.DEFENSE:
            if t not in player_scores.columns:
                player_scores[t] = 0.0

        playing_times = self.compute_playing_times().reset_index()
        self.player_scores = self.eng.lineup.merge(player_scores).merge(playing_times)
        self.player_scores["defcon_normal"] = self.player_scores["defcon"] / playing_times["defend_time"] * 2000
        self.player_scores = self.player_scores[config.DEFCON_HEADER]

    def visualize(
        self,
        action_index,
        hypo_intent=None,
        size=None,
        color=None,
        annot=None,
        show_edges=False,
    ) -> pd.DataFrame:
        frame = self.eng.actions.at[action_index, "frame"]
        frame_data = self.eng.traces.loc[frame:frame].dropna(axis=1).copy()

        players = find_active_players(self.eng, action_index, include_goals=True)
        team = players[0][0][:4]

        possessor = self.eng.actions.at[action_index, "object_id"]
        action_type = self.eng.actions.at[action_index, "action_type"]
        intent = self.eng.actions.at[action_index, "intent_id"]
        receiver = self.eng.actions.at[action_index, "receiver_id"]

        self.intent_probs["home_goal"] = 0.0
        self.intent_probs["away_goal"] = 0.0

        values = pd.DataFrame(index=players[0])
        values["pass_intent"] = self.intent_probs.loc[action_index, players[0]].dropna().astype(float)
        values["pass_success"] = self.success_probs.loc[action_index, players[0]].dropna().astype(float)
        values["goal_if_success"] = self.goal_if_success.loc[action_index, players[0]].dropna().astype(float)
        values["goal_if_failure"] = self.goal_if_failure.loc[action_index, players[0]].dropna().astype(float)
        values["oppo_agn_intent"] = self.likelihoods.loc[action_index, players[0]].dropna().astype(float)

        if "advantage" in [size, color, annot] or "team_credit" in [size, color, annot]:
            values["advantage"] = self.advantages.loc[action_index, players[0]].dropna().astype(float)
            values["team_credit"] = self.team_credits.loc[action_index, players[0]].dropna().astype(float)

        if hypo_intent is None:
            next_player = self.eng.actions.at[action_index, "next_player_id"]
            next_type = self.eng.actions.at[action_index, "next_type"]
            highlights = dict()

            if intent == intent and not intent.endswith("_goal"):
                highlights["black"] = [intent]

            if receiver == receiver:
                if action_type == "shot" and (receiver.endswith("_goal") or next_type in config.SET_PIECE_OOP):
                    arrows = [(possessor, f"{possessor[:4]}_goal")]
                elif next_type in config.DEFENSIVE_TOUCH:
                    highlights["gold"] = [next_player]
                    arrows = [(possessor, next_player)]
                elif next_type not in config.SET_PIECE_OOP:
                    highlights["gold"] = [receiver]
                    arrows = [(possessor, receiver)]
                else:
                    arrows = []
        else:
            hypo_intent = f"{team}_goal" if hypo_intent == -1 else f"{team}_{hypo_intent}"
            highlights = dict() if hypo_intent.endswith("_goal") else {"black": [hypo_intent]}
            arrows = [(intent, hypo_intent)] if action_type == "tackle" else [(possessor, hypo_intent)]

        if "posterior" in [size, color, annot] or "player_credit" in [size, color, annot]:
            assert hypo_intent is not None
            opponents = [p for p in players[1] if not p.endswith("_goal")]
            if "posterior" in [size, color, annot]:
                posteriors_i = self.posteriors[self.posteriors["index"] == action_index]
                posteriors_i = posteriors_i.loc[hypo_intent, opponents].dropna().astype(float)
            if "player_credit" in [size, color, annot]:
                player_credits_i = self.player_credits[self.player_credits["index"] == action_index]
                player_credits_i = player_credits_i.set_index("option").loc[hypo_intent, opponents].astype(float)

        snapshot_args = {"traces": frame_data, "highlights": highlights, "arrows": arrows}
        # snapshot_args = {"traces": frame_data}
        for k, col_name in {"player_sizes": size, "player_colors": color, "player_annots": annot}.items():
            if col_name is not None:
                if col_name == "posterior":
                    snapshot_args[k] = posteriors_i
                elif col_name == "player_credit":
                    snapshot_args[k] = player_credits_i
                else:
                    snapshot_args[k] = values[col_name]

        if show_edges:
            data_index = torch.argwhere(self.eng.labels[:, 0] == action_index).item()
            graph: Data = self.eng.features[data_index]
            graph = sparsify_edges(graph, "delaunay")
            edge_index = graph.edge_index.cpu().detach().numpy()

            src = [(players[0] + players[1])[i] for i in edge_index[0]]
            dst = [(players[0] + players[1])[i] for i in edge_index[1]]
            snapshot_args["edges"] = np.array([src, dst]).T

        snapshot = TraceSnapshot(**snapshot_args)

        style_args = pd.DataFrame(
            [
                [400, 2000, 0, 0.5],
                [500, 2000, 0, 0.5],
                [0, 1600, 0.3, 1],
                [400, 2000, 0, 0.5],
                [500, 20500, -0.01, 0.01],
                [500, 20500, -0.005, 0.005],
            ],
            index=["pass_intent", "oppo_agn_intent", "pass_success", "posterior", "team_credit", "player_credit"],
            columns=["min_size", "max_size", "min_color", "max_color"],
        )
        min_sizes = defaultdict(lambda: 500, style_args["min_size"].to_dict())
        max_sizes = defaultdict(lambda: 20500, style_args["max_size"].to_dict())

        min_colors = {"pass_intent": 0, "oppo_agn_intent": 0, "pass_success": 0.3, "posterior": 0}
        max_colors = {"pass_intent": 0.5, "oppo_agn_intent": 0.5, "pass_success": 1, "posterior": 0.5}
        min_colors = defaultdict(lambda: 0.01, style_args["min_color"].to_dict())
        max_colors = defaultdict(lambda: 0.05, style_args["max_color"].to_dict())

        snapshot.plot(
            smin=min_sizes[size],
            smax=max_sizes[size],
            cmin=min_colors[color],
            cmax=max_colors[color],
            annot_type=annot,
        )

        return values.round(4)
