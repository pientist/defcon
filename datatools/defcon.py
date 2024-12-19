import os
import sys
from typing import Dict, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

from datatools.match_helper import MatchHelper
from datatools.trace_snapshot import TraceSnapshot
from inference import inference, inference_given_intents
from models.gat import GAT
from utils import load_model, sparsify_edges


class DEFCON:
    def __init__(
        self,
        match: MatchHelper,
        intent_model_id="intent/01",
        success_model_id="intent_success/10",
        intercept_model_id="failure_receiver/01",
        scoring_model_id="intent_scoring/10",
        device="cuda",
    ):
        self.match = match
        self.device = device

        self.intent_model = load_model(intent_model_id, device)
        self.success_model = load_model(success_model_id, device)
        self.intercept_model = load_model(intercept_model_id, device)
        self.scoring_model = load_model(scoring_model_id, device)

        self.intent_probs = None
        self.success_probs = None
        self.intercept_probs = None
        self.receive_probs = None
        self.success_xreturns = None
        self.failure_xreturns = None

    def estimate_components(self):
        self.intent_probs = inference(self.match, self.intent_model, device=self.device)[0]
        self.success_xreturns, self.failure_xreturns = inference(self.match, self.scoring_model, device=self.device)

        args = [self.match, self.success_model, self.intercept_model, self.device]
        self.success_probs, self.intercept_probs, self.receive_probs = inference_given_intents(*args)

    def estimate_probs(self, data_index: int, plot=True) -> pd.DataFrame:
        graph: Data = self.match.features[data_index].to(self.device)
        passer_index = torch.nonzero(graph.x[:, -6] == 1).item()
        if graph.x.shape[1] == 18:
            graph.x = torch.cat([graph.x[..., :1], graph.x[..., 3:]], -1)

        with torch.no_grad():
            if self.intent_model.args["sparsify"]:
                max_edge_dist = self.intent_model.args["max_edge_dist"]
                input_graph = sparsify_edges(graph, passer_index, max_edge_dist)
                receiver_probs = nn.Softmax(dim=0)(self.intent_model(input_graph)).cpu().detach().numpy()
            else:
                receiver_probs = nn.Softmax(dim=0)(self.intent_model(graph)).cpu().detach().numpy()

            if self.scoring_model.args["sparsify"]:
                max_edge_dist = self.scoring_model.args["max_edge_dist"]
                input_graph = sparsify_edges(graph, passer_index, max_edge_dist)
                scoring_probs = self.scoring_model(input_graph).cpu().detach().numpy()
            else:
                scoring_probs = self.scoring_model(graph).cpu().detach().numpy()

        action_index = int(self.match.labels[data_index, 0].item())
        frame = self.match.passes.at[action_index, "frame"]
        traces = self.match.traces.loc[frame:frame].dropna(axis=1).copy()

        passer = self.match.passes.at[action_index, "player_code"]
        receiver = self.match.passes.at[action_index, "receiver"]
        home_players = [c[:-2] for c in traces.columns if c[0] == "H" and c[-2:] == "_x"]
        away_players = [c[:-2] for c in traces.columns if c[0] == "A" and c[-2:] == "_x"]

        if passer[0] == "H":
            players = home_players + away_players
            receiver_probs = pd.Series(receiver_probs[:-1], index=players).rename("receiver_prob").round(6)
            scoring_probs = pd.Series(scoring_probs, index=players).rename("scoring_prob").round(6)
            attacker_scoring_probs = scoring_probs[home_players]
        else:
            players = away_players + home_players
            receiver_probs = pd.Series(receiver_probs[:-1], index=players).rename("receiver_prob").round(6)
            scoring_probs = pd.Series(scoring_probs, index=players).rename("scoring_prob").round(6)
            attacker_scoring_probs = scoring_probs[away_players]

        if plot:
            snapshot = TraceSnapshot(traces, receiver_probs, attacker_scoring_probs, highlights=[receiver])
            snapshot.plot()

        return DEFCON.compute_team_credits(receiver_probs, scoring_probs, passer, receiver)

    @staticmethod
    def compute_team_credits(
        receiver_probs: pd.DataFrame,
        scoring_probs: pd.DataFrame,
        passer: str,
        receiver: str,
    ) -> pd.DataFrame:
        state_value = (scoring_probs * receiver_probs).sum()
        advantages = (scoring_probs - state_value).rename("advantage")

        df = pd.concat([receiver_probs, advantages], axis=1).copy()
        df["def_credit"] = np.nan

        attackers = [p for p in receiver_probs.index if p[0] == passer[0]]
        for p in attackers:
            if p == receiver:
                df.at[p, "def_credit"] = -advantages[receiver]
            elif df.at[p, "advantage"] > max(advantages[receiver], 0):
                df.at[p, "def_credit"] = (1 - df.at[p, "receiver_prob"]) * df.at[p, "advantage"]

        return df
