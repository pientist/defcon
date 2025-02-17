import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from datatools.utils import drop_non_blocker_nodes, drop_opponent_nodes, sparsify_edges


class ActionDataset(Dataset):
    def __init__(
        self,
        game_ids,
        feature_dir="data/ajax/pyg/pass_features",
        label_dir="data/ajax/pyg/pass_labels",
        action_type=None,  # pass/takeon/shot or {successful/failed}_{pass/takeon/shot}
        intended_only=False,
        inplay_only=False,
        min_duration=0,
        xy_only=False,
        possessor_aware=True,
        keeper_aware=True,
        ball_z_aware=True,
        poss_vel_aware=True,
        intent_given=False,
        drop_opponents=False,
        drop_non_blockers=False,
        sparsify="delaunay",
        max_edge_dist=10,
    ):
        features = [x for id in tqdm(game_ids) for x in torch.load(f"{feature_dir}/{id}.pt", weights_only=False)]
        labels = torch.cat([torch.load(f"{label_dir}/{id}.pt", weights_only=False) for id in game_ids])

        condition: torch.Tensor = torch.ones(labels.shape[0]).bool()

        if action_type.split("_")[0] == "successful":  # Only include successful actions
            condition &= labels[:, -5] == 1
        elif action_type.split("_")[0] in ["failed", "failure"]:  # Only include failed actions
            condition &= labels[:, -5] == 0
        elif action_type.split("_")[0] == "blocked":  # Only include blocked actions
            condition &= labels[:, -6] == 0

        if action_type.endswith("pass_shot"):
            condition &= (labels[:, 1] == 1) | (labels[:, 3] == 1)

        # if action_type.split("_")[-1] == "pass":
        #     condition &= labels[:, 1] == 1
        # elif action_type.split("_")[-1] == "takeon":
        #     condition &= labels[:, 2] == 1
        # elif action_type.split("_")[-1] == "shot":
        #     condition &= labels[:, 3] == 1

        if intended_only or intent_given:  # Only include actions with valid intended receivers
            condition &= labels[:, 5] != -1

        if inplay_only:  # Only include actions with valid receivers
            condition &= labels[:, 6] != -1

        if min_duration > 0:  # Only include actions with enough duration
            condition &= labels[:, 7] >= min_duration

        self.features = []
        self.labels = []

        for i in tqdm(condition.nonzero()[:, 0].numpy()):
            graph: Data = features[i]
            graph_labels: torch.Tensor = labels[i]

            try:
                possessor_index = torch.nonzero(graph.x[:, -6] == 1).item()
            except RuntimeError:
                continue

            if not keeper_aware:  # Do not distinguish between goalkeepers and outfield players
                graph.x[:, 1] = 0

            if not ball_z_aware:  # Set the ball height for every action as 0
                graph.x[:, -7] = 0

            if not poss_vel_aware:  # Ignore the features related to the ball possessor's velocity
                graph.x[graph.x[:, -6] == 1, 5:9] = 0
                graph.x[:, -2:] = 0

            if not possessor_aware and xy_only:  # 8 features
                graph.x = torch.cat([graph.x[..., :7], graph.x[..., -7:-6]], -1)
            elif not possessor_aware and not xy_only:  # 13 features
                graph.x = graph.x[..., :-7]
            elif possessor_aware and xy_only:  # 9 features
                graph.x = torch.cat([graph.x[..., :7], graph.x[..., -7:-5]], -1)
                poss_flag_index = -1
            else:
                poss_flag_index = -6

            if drop_opponents:
                graph, graph_labels = drop_opponent_nodes(graph, graph_labels)
                possessor_index = torch.nonzero(graph.x[:, poss_flag_index] == 1).item()

            if drop_non_blockers:
                assert possessor_aware
                graph, graph_labels = drop_non_blocker_nodes(graph, graph_labels, poss_flag_index)
                possessor_index = torch.nonzero(graph.x[:, poss_flag_index] == 1).item()

            if sparsify == "distance":
                assert possessor_aware
                graph = sparsify_edges(graph, "distance", possessor_index, max_edge_dist)
            elif sparsify == "delaunay" and graph.x.shape[0] > 3:
                graph = sparsify_edges(graph, "delaunay")

            if intent_given:
                intent_onehot = torch.zeros(graph.x.shape[0])
                intent_onehot[labels[i, 5].long()] = 1
                graph.x = torch.cat([graph.x, intent_onehot.unsqueeze(1)], -1)

            self.features.append(graph)
            self.labels.append(graph_labels)

        self.labels = torch.stack(self.labels, axis=0)

    def balance_real_and_augmented(self):
        real_indices = torch.nonzero(self.labels[:, -7] == 1).flatten()
        augmented_indices = torch.nonzero(self.labels[:, -7] == 0).flatten()

        if len(real_indices) < len(augmented_indices):
            sampled_indices = torch.randperm(len(augmented_indices))[: len(real_indices)]
            augmented_indices = augmented_indices[sampled_indices]

        indices = torch.cat([real_indices, augmented_indices])
        self.features = [self.features[i] for i in indices]
        self.labels = self.labels[indices]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]
