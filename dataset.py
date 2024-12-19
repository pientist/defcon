import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from utils import sparsify_edges


class ActionDataset(Dataset):
    def __init__(
        self,
        match_ids,
        split="train",
        outcome="all",
        pass_only=False,
        intended_only=False,
        inplay_only=False,
        xy_only=False,
        passer_aware=True,
        one_touch_aware=True,
        intent_aware=False,
        sparsify="delaunay",
        max_edge_dist=10,
    ):
        features = []
        labels = []

        for match_id in tqdm(match_ids, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
            match_features = torch.load(f"data/{split}_features/{match_id}.pt", weights_only=False)
            match_labels = torch.load(f"data/{split}_labels/{match_id}.pt", weights_only=False)
            condition: torch.Tensor = torch.ones(match_labels.shape[0]).bool()

            if outcome == "success":  # Only include successful actions
                condition &= match_labels[:, 11] == 1
            elif outcome == "failure":  # Only include failed actions
                condition &= match_labels[:, 11] == 0

            if pass_only:
                condition &= match_labels[:, 9] == 1
            else:  # Include passes and short ball carries (with distance less than 5 meters)
                condition &= (match_labels[:, 8] < 5) | (match_labels[:, 9] == 1)

            if intended_only or intent_aware:  # Only include actions with valid intended receivers
                condition &= match_labels[:, 2] != -1

            if inplay_only:  # Only include ball moves with valid receivers
                condition &= match_labels[:, 3] != -1

            for i in condition.nonzero()[:, 0].numpy():
                graph: Data = match_features[i]
                try:
                    passer_index = torch.nonzero(graph.x[:, 12] == 1).item()
                except RuntimeError:
                    continue

                if not passer_aware and xy_only:
                    graph.x = graph.x[..., :7]
                elif not passer_aware and not xy_only:
                    graph.x = graph.x[..., :12]
                elif passer_aware and xy_only:
                    graph.x = torch.cat([graph.x[..., :7], graph.x[..., 12:13]], -1)

                if not one_touch_aware:
                    graph.x = torch.cat([graph.x[..., :1], graph.x[..., 3:]], -1)

                if intent_aware:
                    intent_onehot = torch.zeros(graph.x.shape[0])
                    intent_onehot[match_labels[i, 2].long()] = 1
                    graph.x = torch.cat([graph.x, intent_onehot.unsqueeze(1)], -1)

                if sparsify == "distance":
                    assert passer_aware
                    graph = sparsify_edges(graph, "distance", passer_index, max_edge_dist)
                elif sparsify == "delaunay":
                    graph = sparsify_edges(graph, "delaunay")

                features.append(graph)
                labels.append(match_labels[i])

        self.features = features
        self.labels = torch.stack(labels, axis=0)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]
