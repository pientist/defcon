import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from datatools import config


class XGModel:
    def __init__(self, unblocked=False):
        self.unblocked = unblocked
        self.features = ["x", "y", "distance", "angle", "freekick", "header"]
        self.fit = None

    def train(self, data_path="data/shots_train.csv", verbose=False):
        formula = "goal ~ " + " + ".join(self.features)

        data_train = pd.read_csv(data_path, header=0)
        data_train = data_train[data_train["sub_event_type"] != "Penalty"].copy()
        data_train["y"] = data_train["y"].abs()

        if self.unblocked:
            data_train = data_train[data_train["blocked"] == 0].copy()

        self.fit = smf.glm(formula=formula, data=data_train, family=sm.families.Binomial()).fit()
        if verbose:
            print(self.fit.summary())

    @staticmethod
    def calc_shot_features(events: pd.DataFrame) -> pd.DataFrame:
        events = events.copy()

        # Make sure that the home team always play from left to right
        # The home team's events are rotated to align the target goal to the left side of the pitch
        home_events = events[events["object_id"].str.startswith("home")]
        events.loc[home_events.index, "start_x"] = config.FIELD_SIZE[0] - home_events["start_x"]
        events.loc[home_events.index, "start_y"] = config.FIELD_SIZE[1] - home_events["start_y"]

        events["x"] = x = (events["start_x"] / config.FIELD_SIZE[0] * 104).round(2)
        events["y"] = y = (events["start_y"] / config.FIELD_SIZE[1] * 68 - 34).abs()
        events["distance"] = events[["x", "y"]].apply(np.linalg.norm, axis=1)

        goal_width = 7.32
        angles = np.arctan((goal_width * x) / (x**2 + y**2 - (goal_width / 2) ** 2)) * 180 / np.pi
        events["angle"] = np.where(angles >= 0, angles, angles + 180)

        events["freekick"] = events["spadl_type"].str.contains("freekick")
        events["header"] = events["start_z"] > 1
        # shots["goal"] = shots["outcome"].astype(int)

        return events[["x", "y", "distance", "angle", "freekick", "header"]]

    def pred(self, shot_features: pd.DataFrame) -> pd.Series:
        sum = self.fit.params.values[0]
        for i, c in enumerate(self.features):
            sum += self.fit.params.values[i + 1] * shot_features[c]
        return 1 / (1 + np.exp(-sum))
