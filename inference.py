import re
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from datatools.feature import FeatureEngineer
from datatools.utils import filter_features_and_labels
from models.gat import GAT


def find_active_players(eng: FeatureEngineer, action_index: int, include_goals=False) -> dict:
    frame = eng.actions.at[action_index, "frame"]
    action_type = eng.actions.at[action_index, "action_type"]
    snapshot = eng.traces.loc[frame:frame].dropna(axis=1).copy()

    if include_goals:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_.*_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_.*_x", c)]
    else:
        home_players = [c[:-2] for c in snapshot.columns if re.match(r"home_\d+_x", c)]
        away_players = [c[:-2] for c in snapshot.columns if re.match(r"away_\d+_x", c)]

    possessor = eng.actions.at[action_index, "object_id"]
    if possessor[:4] == "home":
        players = [home_players, away_players] if action_type != "tackle" else [away_players, home_players]
    else:
        players = [away_players, home_players] if action_type != "tackle" else [home_players, away_players]

    return players


def inference(eng: FeatureEngineer, model: GAT, device="cuda") -> Tuple[pd.DataFrame, pd.DataFrame]:
    # For intent, receiver, scoring/conceding, {intent/receiver}_{scoring/conceding}, shot_blocking

    graphs, labels = filter_features_and_labels(eng.features, eng.labels, model.args)
    if model.args["target"] in ["scoring", "conceding"]:
        features_after, _ = filter_features_and_labels(eng.features_receiving, eng.labels, model.args)

    if model.args["target"] in ["intent_scoring", "intent_conceding"]:
        probs_if_success = []
        probs_if_failure = []
    else:
        probs = []

    with torch.no_grad():
        graphs = Batch.from_data_list(graphs).to(device)
        graphs.x = graphs.x[:, : model.args["node_in_dim"]]

        if model.args["target"] == "shot_blocking":
            out = nn.Sigmoid()(model(graphs)).cpu().detach().numpy()  # [B,]
            action_indices = labels[:, 0].cpu().detach().numpy().astype(int)
            return pd.Series(out, index=action_indices), None

        elif model.args["target"] in ["scoring", "conceding"]:
            out_before = nn.Sigmoid()(model(graphs)).cpu().detach().numpy()  # [B,]

            graphs_after = [data for data in features_after if data is not None]
            graphs_after = Batch.from_data_list(graphs_after).to(device)
            graphs_after.x = graphs_after.x[:, : model.args["node_in_dim"]]

            model_out_after = nn.Sigmoid()(model(graphs_after)).cpu().detach().numpy()  # [B,]
            model_out_index = 0
            out_after = []

            for data in features_after:
                if data is None:
                    out_after.append(0)
                else:
                    out_after.append(model_out_after[model_out_index])
                    model_out_index += 1

            probs = np.stack([out_before, np.array(out_after)]).T
            action_indices = labels[:, 0].cpu().detach().numpy().astype(int)
            return pd.DataFrame(probs, index=action_indices, columns=["value_before", "value_after"]), None

        elif "intent" in model.args["target"]:  # Select components corresponding to teammates
            batch = graphs.batch[graphs.x[:, 0] == 1]
            out = model(graphs)[graphs.x[:, 0] == 1]  # [N',]

        elif "receiver" in model.args["target"]:
            batch = graphs.batch
            if model.args["residual"]:
                batch = torch.cat([batch, torch.unique(graphs.batch)])
            out = model(graphs)  # [N,] or [N + B,]

    include_goals = model.args["target"] == "oppo_agn_intent"
    tqdm_desc = "weight" if model.args["target"] == "oppo_agn_intent" else model.args["target"].split("_")[-1]

    for i in tqdm(range(graphs.num_graphs), desc=tqdm_desc):
        action_index = int(labels[i, 0].item())
        active_players = find_active_players(eng, action_index, include_goals=include_goals)

        if model.args["target"] in ["intent", "oppo_agn_intent", "receiver"]:  # node_selection
            probs_i = nn.Softmax(dim=0)(out[batch == i]).cpu().detach().numpy()
        elif model.args["target"].split("_")[1] in ["scoring", "conceding"]:  # node_binary or graph_binary
            probs_i = nn.Sigmoid()(out[batch == i]).cpu().detach().numpy()

        if "intent" in model.args["target"]:
            player_indices = active_players[0]
        elif "receiver" in model.args["target"]:
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

    valid_traces = eng.traces.dropna(axis=1, how="all")
    if include_goals:
        home_players = [c[:-2] for c in valid_traces.columns if re.match(r"home_.*_x", c)]
        away_players = [c[:-2] for c in valid_traces.columns if re.match(r"away_.*_x", c)]
    else:
        home_players = [c[:-2] for c in valid_traces.columns if re.match(r"home_\d+_x", c)]
        away_players = [c[:-2] for c in valid_traces.columns if re.match(r"away_\d+_x", c)]
    players = home_players + away_players

    if model.args["target"] in ["intent_scoring", "intent_conceding"]:
        probs_if_success = pd.DataFrame(probs_if_success).set_index("index")[players]
        probs_if_failure = pd.DataFrame(probs_if_failure).set_index("index")[players]
        return probs_if_success, probs_if_failure
    else:
        return pd.DataFrame(probs).set_index("index")[players], None


def inference_success(eng: FeatureEngineer, model: GAT, device="cuda") -> pd.DataFrame:
    graphs, labels = filter_features_and_labels(eng.features, eng.labels, model.args)
    success_probs = []

    for data_index in tqdm(range(len(graphs)), desc="success"):
        graph_i = graphs[data_index].to(device)

        action_index = int(labels[data_index, 0].item())
        active_players = find_active_players(eng, action_index)
        n_teammates = len(active_players[0])

        intended_graphs = []
        for intent_index in range(n_teammates):
            intent_onehot = torch.zeros(graph_i.x.shape[0]).to(device)
            intent_onehot[intent_index] = 1
            intended_nodes = torch.cat([graph_i.x, intent_onehot.unsqueeze(1)], -1)
            intended_graph = Data(x=intended_nodes, edge_index=graph_i.edge_index, edge_attr=graph_i.edge_attr)
            intended_graphs.append(intended_graph)
        intended_graphs = Batch.from_data_list(intended_graphs).to(device)

        if model is not None:
            with torch.no_grad():
                out = model(intended_graphs)
                success_probs_i = nn.Sigmoid()(out).cpu().detach().numpy()  # [11,]
                success_probs_i = dict(zip(active_players[0], success_probs_i))
                success_probs.append(dict(**success_probs_i, **{"index": action_index}))

    valid_traces = eng.traces.dropna(axis=1, how="all")
    home_players = [c[:-2] for c in valid_traces.columns if re.match(r"home_\d+_x", c)]
    away_players = [c[:-2] for c in valid_traces.columns if re.match(r"away_\d+_x", c)]
    return pd.DataFrame(success_probs).set_index("index")[home_players + away_players]


def inference_intercept(eng: FeatureEngineer, model: GAT = None, device="cuda") -> pd.DataFrame:
    graphs, labels = filter_features_and_labels(eng.features, eng.labels, model.args)
    include_goals = (graphs[0].x[:, 2] == 1).any().item()
    intercept_probs = []

    for data_index in tqdm(range(len(graphs)), desc="intercept"):
        graph_i = graphs[data_index].to(device)

        action_index = int(labels[data_index, 0].item())
        active_players = find_active_players(eng, action_index, include_goals=include_goals)
        n_teammates = len(active_players[0])

        intended_graphs = []
        for intent_index in range(n_teammates):
            intent_onehot = torch.zeros(graph_i.x.shape[0]).to(device)
            intent_onehot[intent_index] = 1
            intended_nodes = torch.cat([graph_i.x, intent_onehot.unsqueeze(1)], -1)
            intended_graph = Data(x=intended_nodes, edge_index=graph_i.edge_index, edge_attr=graph_i.edge_attr)
            intended_graphs.append(intended_graph)
        intended_graphs = Batch.from_data_list(intended_graphs).to(device)

        with torch.no_grad():
            logits = model(intended_graphs)  # [12 * 22 + 12,]

        receive_logits = logits[:-n_teammates].reshape(n_teammates, -1)  # [12, 22]
        ballout_logits = logits[-n_teammates:].unsqueeze(1)  # [12, 1]
        logits = torch.cat([receive_logits, ballout_logits], 1)  # [12, 23]
        intercept_probs_i = nn.Softmax(dim=1)(logits[:, n_teammates:]).cpu().detach().numpy()  # [12, 12]

        intercept_probs_i = pd.DataFrame(
            intercept_probs_i, index=active_players[0], columns=active_players[1] + ["out"]
        )
        intercept_probs_i["index"] = action_index
        intercept_probs_i.index.name = "option"
        intercept_probs.append(intercept_probs_i)

    valid_traces = eng.traces.dropna(axis=1, how="all")
    home_players = [c[:-2] for c in valid_traces.columns if re.match(r"home_\d+_x", c)]
    away_players = [c[:-2] for c in valid_traces.columns if re.match(r"away_\d+_x", c)]
    return pd.concat(intercept_probs)[["index"] + home_players + away_players + ["out"]]
