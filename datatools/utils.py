from fnmatch import fnmatch
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon
from torch_geometric.data import Batch, Data

from datatools import config


def calc_dist(x: np.ndarray, y: np.ndarray, origin_x: np.ndarray, origin_y: np.ndarray):
    dist_x = (x - origin_x).astype(float)
    dist_y = (y - origin_y).astype(float)
    return dist_x, dist_y, np.sqrt(dist_x**2 + dist_y**2)


def calc_angle(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray = None,
    cy: np.ndarray = None,
    eps: float = 1e-6,
) -> np.ndarray:
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


# To make the attacking team always play from right to left (not needed for the current dataset)
def rotate_events_for_xg(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()

    for i in events["period_id"].unique():
        shots = events[(events["period_id"] == i) & events["spadl_type"].isin(config.SHOT)]

        if not shots.empty:
            home_shot_x = shots.loc[shots["object_id"].str.startswith("home"), "start_x"].mean()
            if home_shot_x > config.FIELD_SIZE[0] / 2:
                home_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("home"))]
                events.loc[home_events.index, "start_x"] = config.FIELD_SIZE[0] - home_events["start_x"]
                events.loc[home_events.index, "start_y"] = config.FIELD_SIZE[1] - home_events["start_y"]

            away_shot_x = shots.loc[shots["object_id"].str.startswith("away"), "start_y"].mean()
            if away_shot_x > config.FIELD_SIZE[0] / 2:
                away_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("away"))]
                events.loc[away_events.index, "start_x"] = config.FIELD_SIZE[0] - away_events["start_x"]
                events.loc[away_events.index, "start_y"] = config.FIELD_SIZE[1] - away_events["start_y"]

        else:
            gk_events = events[(events["period_id"] == i) & (events["advanced_position"] == "goal_keeper")]

            home_gk_x = gk_events.loc[gk_events["object_id"].str.startswith("home"), "start_x"].mean()
            if home_gk_x < config.FIELD_SIZE[0] / 2:
                home_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("home"))]
                events.loc[home_events.index, "start_x"] = config.FIELD_SIZE[0] - home_events["start_x"]
                events.loc[home_events.index, "start_y"] = config.FIELD_SIZE[1] - home_events["start_y"]

            away_gk_x = gk_events.loc[gk_events["object_id"].str.startswith("away"), "start_x"].mean()
            if away_gk_x < config.FIELD_SIZE[0] / 2:
                away_events = events[(events["period_id"] == i) & (events["object_id"].str.startswith("away"))]
                events.loc[away_events.index, "start_x"] = config.FIELD_SIZE[0] - away_events["start_x"]
                events.loc[away_events.index, "start_y"] = config.FIELD_SIZE[1] - away_events["start_y"]

    return events


def find_blocker(event: pd.Series, traces: pd.DataFrame, keepers: np.ndarray) -> int:
    snapshot: pd.Series = traces.loc[event["frame"]]
    event_xy = event[["start_x", "start_y"]].values.tolist()

    goal_x = config.FIELD_SIZE[0] if event["object_id"].startswith("home") else 0
    goal_xy_lower = [goal_x, config.FIELD_SIZE[1] / 2 - 4]
    goal_xy_upper = [goal_x, config.FIELD_SIZE[1] / 2 + 4]
    goal_side_vertices = np.array([event_xy, goal_xy_lower, goal_xy_upper])
    goal_side = Polygon(goal_side_vertices).buffer(1)  # .intersection(Point(event_xy).buffer(10))

    oppo_team = "away" if event["object_id"].startswith("home") else "home"
    oppo_x_cols = [c for c in snapshot.index if fnmatch(c, f"{oppo_team}_*_x") and c[:-2] not in keepers]
    oppo_y_cols = [c for c in snapshot.index if fnmatch(c, f"{oppo_team}_*_y") and c[:-2] not in keepers]
    player_xy = np.stack([snapshot[oppo_x_cols].values, snapshot[oppo_y_cols].values]).T
    player_xy = pd.DataFrame(player_xy, index=[c[:-2] for c in oppo_x_cols], columns=["x", "y"])

    can_block = player_xy.apply(lambda p: goal_side.contains(Point(p["x"], p["y"])), axis=1)
    potential_blockers = player_xy.loc[can_block[can_block].index]

    if potential_blockers.empty:
        return np.nan
    else:
        potential_blockers["dist_x"] = potential_blockers["x"] - event_xy[0]
        potential_blockers["dist_y"] = potential_blockers["y"] - event_xy[1]
        blocker_dists = potential_blockers[["dist_x", "dist_y"]].apply(np.linalg.norm, axis=1)
        return blocker_dists.idxmin()


def drop_nodes(graph: Data, labels: torch.Tensor, node_mask: torch.BoolTensor) -> Tuple[Data, torch.Tensor]:
    node_mask_indices = torch.where(node_mask)[0]
    index_map = -torch.ones((graph.num_nodes,)).long()
    index_map[node_mask_indices] = torch.arange(len(node_mask_indices))
    index_map = torch.cat([index_map, torch.LongTensor([-1])])  # To map -1 to -1

    node_attr = graph.x[node_mask]
    edge_mask = node_mask[graph.edge_index[0]] & node_mask[graph.edge_index[1]]
    edge_index = index_map[graph.edge_index[:, edge_mask]]
    edge_attr = graph.edge_attr[edge_mask]
    masked_graph = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)

    masked_labels = labels.clone()
    masked_labels[4] = node_mask.long().sum()  # number of players
    masked_labels[5] = index_map[masked_labels[5].long()]  # intent index
    masked_labels[6] = index_map[masked_labels[6].long()]  # receiver index

    return masked_graph, masked_labels


def drop_opponent_nodes(graph: Data, labels: torch.Tensor) -> Tuple[Data, torch.Tensor]:
    node_mask = graph.x[:, 0] == 1
    return drop_nodes(graph, labels, node_mask)


def drop_goal_nodes(graph: Data, labels: torch.Tensor) -> Tuple[Data, torch.Tensor]:
    node_mask = graph.x[:, 2] == 0
    return drop_nodes(graph, labels, node_mask)


def drop_non_blocker_nodes(
    graph: Data,
    labels: torch.Tensor,
    poss_flag_index=-6,
    buffer_x=5,
) -> Tuple[Data, torch.Tensor]:
    poss_or_oppo = (graph.x[:, poss_flag_index] == 1) | (graph.x[:, 0] == 0)
    poss_x = graph.x[graph.x[:, poss_flag_index] == 1, 3].item()
    node_mask = poss_or_oppo & (graph.x[:, 3] > poss_x - buffer_x)
    return drop_nodes(graph, labels, node_mask)


def sparsify_edges(graph: Data, how="distance", possessor_index: int = None, max_dist=10) -> Data:
    if how == "distance":
        edge_index = graph.edge_index
        if possessor_index is not None:
            passer_edges = (edge_index[0] == possessor_index) | (edge_index[1] == possessor_index)
        close_edges = graph.edge_attr[:, 0] <= max_dist

        graph.edge_index = edge_index[:, passer_edges | close_edges]
        graph.edge_attr = graph.edge_attr[passer_edges | close_edges]

    elif how == "delaunay":
        # xy = graph.x[:, 1:3] if graph.x.shape[1] < 18 else graph.x[:, 3:5]
        xy = graph.x[:, 3:5]
        tri_pts = Delaunay(xy.cpu().detach().numpy()).simplices
        tri_edges = np.concatenate((tri_pts[:, :2], tri_pts[:, 1:], tri_pts[:, ::2]), axis=0)
        tri_edges = np.unique(tri_edges, axis=0).tolist()

        for [i, j] in tri_edges:
            if [j, i] not in tri_edges:
                tri_edges.append([j, i])

        complete_edges = graph.edge_index.cpu().detach().numpy().T
        complete_edge_dict = {tuple(e): i for i, e in enumerate(complete_edges)}
        tri_edge_index = np.sort([complete_edge_dict[tuple(e)] for e in tri_edges]).tolist()

        graph.edge_index = graph.edge_index[:, tri_edge_index]
        graph.edge_attr = graph.edge_attr[tri_edge_index]

    return graph


def filter_features_and_labels(
    features: List[Data],
    labels: torch.Tensor,
    args: Dict[str, Any],
) -> Tuple[List[Data], torch.Tensor]:
    condition = torch.ones(labels.shape[0]).bool()
    filtered_features = []
    filtered_labels = []

    for i in condition.nonzero()[:, 0].numpy():
        graph: Data = features[i]
        graph_labels: torch.Tensor = labels[i]

        if graph is None:
            filtered_features.append(graph)
            continue
        else:
            graph = graph.clone()

        try:
            possessor_index = torch.nonzero(graph.x[:, -6] == 1).item()
        except RuntimeError:
            continue

        if not args["keeper_aware"]:
            graph.x[:, 1] = 0

        if not args["ball_z_aware"]:
            graph.x[:, -7] = 0

        if not args["poss_vel_aware"]:
            graph.x[graph.x[:, -6] == 1, 5:9] = 0
            graph.x[:, -2:] = 0

        if not args["possessor_aware"] and args["xy_only"]:
            graph.x = graph.x[..., :7]
        elif not args["possessor_aware"] and not args["xy_only"]:
            graph.x = graph.x[..., :-6]
        elif args["possessor_aware"] and args["xy_only"]:
            graph.x = torch.cat([graph.x[..., :7], graph.x[..., -6:-5]], -1)

        if args["target"].startswith("oppo_agn"):
            graph, graph_labels = drop_opponent_nodes(graph, graph_labels)
            possessor_index = torch.nonzero(graph.x[:, -6] == 1).item()

        if args["target"] not in ["failure_receiver", "oppo_agn_intent"]:
            graph, graph_labels = drop_goal_nodes(graph, graph_labels)
            possessor_index = torch.nonzero(graph.x[:, -6] == 1).item()

        if "filter_blockers" in args and args["filter_blockers"]:
            graph, graph_labels = drop_non_blocker_nodes(graph, graph_labels)
            possessor_index = torch.nonzero(graph.x[:, -6] == 1).item()

        if args["target"] in ["intent_success", "failed_pass_receiver", "failure_receiver"]:
            assert graph.x.shape[1] + 1 == args["node_in_dim"]
        else:
            assert graph.x.shape[1] == args["node_in_dim"]

        if args["sparsify"] == "distance":
            assert args["possessor_aware"]
            graph = sparsify_edges(graph, "distance", possessor_index, args["max_edge_dist"])
        elif args["sparsify"] == "delaunay" and graph.x.shape[0] > 3:
            graph = sparsify_edges(graph, "delaunay")

        filtered_features.append(graph)
        filtered_labels.append(graph_labels)

    return filtered_features, torch.stack(filtered_labels, axis=0)
