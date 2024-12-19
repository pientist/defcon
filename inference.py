from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from datatools.match_helper import MatchHelper
from models.gat import GAT
from utils import load_model


def find_active_players(match: MatchHelper, action_index: int) -> dict:
    frame = match.passes.at[action_index, "frame"]
    instance_data = match.traces.loc[frame:frame].dropna(axis=1).copy()

    passer = match.passes.at[action_index, "player_code"]
    home_players = [c[:-2] for c in instance_data.columns if c[0] == "H" and c[-2:] == "_x"]
    away_players = [c[:-2] for c in instance_data.columns if c[0] == "A" and c[-2:] == "_x"]
    players = [home_players, away_players] if passer[0] == "H" else [away_players, home_players]

    return players


def inference(
    match: MatchHelper,
    model: GAT,
    device="cuda",
    replace_features=False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # For intent, receiver and {intent/receiver}_{scoring/conceding}

    graphs, labels = match.filter_features_and_labels(model.args, replace_features)

    if model.args["target"] in ["intent_scoring", "intent_conceding"]:
        probs_if_success = []
        probs_if_failure = []
    else:
        probs = []

    with torch.no_grad():
        graphs = Batch.from_data_list(graphs).to(device)
        graphs.x = graphs.x[:, : model.args["node_in_dim"]]

        if "intent" in model.args["target"]:  # Select components corresponding to teammates
            batch = graphs.batch[graphs.x[:, 0] == 1]
            out = model(graphs)[graphs.x[:, 0] == 1]
        elif "receiver" in model.args["target"]:
            batch = graphs.batch
            if model.args["residual"]:
                batch = torch.cat([batch, torch.unique(graphs.batch)])
            out = model(graphs)

        for i in tqdm(range(graphs.num_graphs), desc=model.args["target"].split("_")[-1]):
            action_index = int(labels[i, 0].item())
            active_players = find_active_players(match, action_index)

            if model.args["target"] in ["intent", "receiver"]:
                probs_i = nn.Softmax(dim=0)(out[batch == i]).cpu().detach().numpy()
            elif model.args["target"].split("_")[1] in ["scoring", "conceding"]:
                probs_i = (out[batch == i]).cpu().detach().numpy()

            if model.args["target"].startswith("intent"):
                player_indices = active_players[0]
            elif model.args["target"].startswith("receiver"):
                player_indices = active_players[0] + active_players[1]
                if model.args["residual"]:
                    player_indices.append("out")

            if model.args["target"] in ["intent_scoring", "intent_conceding"]:
                probs_i_if_success = dict(zip(player_indices, probs_i[:, 1].tolist()))
                probs_i_if_failure = dict(zip(player_indices, probs_i[:, 0].tolist()))
                probs_if_success.append(dict(**probs_i_if_success, **{"index": action_index}))
                probs_if_failure.append(dict(**probs_i_if_failure, **{"index": action_index}))
            else:
                probs_i = dict(zip(player_indices, probs_i.tolist()))
                probs.append(dict(**probs_i, **{"index": action_index}))

    valid_traces = match.traces.dropna(axis=1, how="all")
    players = [c[:3] for c in valid_traces.columns if c[0] in ["H", "A"] and c.endswith("_x")]

    if model.args["target"] in ["intent_scoring", "intent_conceding"]:
        probs_if_success = pd.DataFrame(probs_if_success).set_index("index")[players]
        probs_if_failure = pd.DataFrame(probs_if_failure).set_index("index")[players]
        return probs_if_success, probs_if_failure
    else:
        return pd.DataFrame(probs).set_index("index")[players], None


def inference_given_intents(
    match: MatchHelper,
    success_model: GAT,
    intercept_model: GAT = None,
    device="cuda",
    replace_features=False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # For intent_{success/scoring/conceding} and failure_receiver

    graphs, labels = match.filter_features_and_labels(success_model.args, replace_features)

    if intercept_model is not None:
        assert intercept_model.args["sparsify"] == success_model.args["sparsify"]
    # if scoring_model_id is not None:
    #     scoring_model = load_model(scoring_model_id, device)
    #     assert scoring_model.args["sparsify"] == success_model.args["sparsify"]

    success_probs = []
    intercept_probs = []
    receive_probs = []
    # success_xreturns = []
    # failure_xreturns = []

    for data_index in tqdm(range(len(graphs)), desc="success"):
        graph_i = graphs[data_index].to(device)

        action_index = int(labels[data_index, 0].item())
        active_players = find_active_players(match, action_index)
        n_teammates = len(active_players[0])

        intended_graphs = []
        for intent_index in range(n_teammates):
            intent_onehot = torch.zeros(graph_i.x.shape[0]).to(device)
            intent_onehot[intent_index] = 1
            intended_nodes = torch.cat([graph_i.x, intent_onehot.unsqueeze(1)], -1)
            intended_graph = Data(x=intended_nodes, edge_index=graph_i.edge_index, edge_attr=graph_i.edge_attr)
            intended_graphs.append(intended_graph)
        intended_graphs = Batch.from_data_list(intended_graphs).to(device)

        if success_model is not None:
            with torch.no_grad():
                success_probs_i = success_model(intended_graphs).cpu().detach().numpy()  # [11,]
                success_probs_i = dict(zip(active_players[0], success_probs_i))
                success_probs.append(dict(**success_probs_i, **{"index": action_index}))

        # if scoring_model_id is not None:
        #     with torch.no_grad():
        #         scoring_probs_i = scoring_model(intended_graphs).cpu().detach().numpy()  # [11, 2]

        #     success_xreturns_i = dict(zip(active_players[0], scoring_probs_i[:, 1]))
        #     failure_xreturns_i = dict(zip(active_players[0], scoring_probs_i[:, 0]))
        #     success_xreturns.append(dict(**success_xreturns_i, **{"index": action_index}))
        #     failure_xreturns.append(dict(**failure_xreturns_i, **{"index": action_index}))

        if intercept_model is not None:
            with torch.no_grad():
                logits = intercept_model(intended_graphs)  # [11 * 22 + 11,]

            receive_logits = logits[:-n_teammates].reshape(11, -1)
            ballout_logits = logits[-n_teammates:].unsqueeze(1)
            logits = torch.cat([receive_logits, ballout_logits], 1)  # [11, 23]
            intercept_probs_i = nn.Softmax(dim=1)(logits[:, n_teammates:]).cpu().detach().numpy()  # [11, 12]

            intercept_probs_i = pd.DataFrame(
                intercept_probs_i,
                index=active_players[0],
                columns=active_players[1] + ["out"],
            )
            intercept_probs_i["index"] = action_index
            intercept_probs.append(intercept_probs_i)

        if success_model is not None and intercept_model is not None:
            receive_probs_i = intercept_probs_i.drop("index", axis=1).mul(1 - pd.Series(success_probs_i), axis=0)
            for p in active_players[0]:
                receive_probs_i[p] = 0.0
                receive_probs_i.at[p, p] = success_probs_i[p]

            receive_probs_i["index"] = action_index
            receive_probs.append(receive_probs_i)

    valid_traces = match.traces.dropna(axis=1, how="all")
    players = [c[:3] for c in valid_traces.columns if c[0] in ["H", "A"] and c.endswith("_x")]

    if success_probs:
        success_probs = pd.DataFrame(success_probs).set_index("index")[players]
    if intercept_probs:
        intercept_probs = pd.concat(intercept_probs)[["index"] + players]
    if receive_probs:
        receive_probs = pd.concat(receive_probs)[["index"] + players]
    # if success_xreturns and failure_xreturns:
    #     success_xreturns = pd.DataFrame(success_xreturns).set_index("index")[players]
    #     failure_xreturns = pd.DataFrame(failure_xreturns).set_index("index")[players]

    return success_probs, intercept_probs, receive_probs  # success_xreturns, failure_xreturns
