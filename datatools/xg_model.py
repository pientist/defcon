import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


class XGModel:
    def __init__(self):
        self.fit = None

    def train(self, data_path="data/xg_train.csv", verbose=False):
        formula = "goal ~ x + y + distance + angle + freekick + header"
        data_train = pd.read_csv(data_path, header=0)
        self.fit = smf.glm(formula=formula, data=data_train, family=sm.families.Binomial()).fit()
        if verbose:
            print(self.fit.summary())

    @staticmethod
    def calc_shot_features(events: pd.DataFrame, pitch_size=(104, 68)) -> pd.DataFrame:
        et = events["event_types"]
        shots = events[(et.str.contains("shot")) | (et.str.contains("goal "))].copy()

        for i in events["session"].unique():
            home_mean_x = events.loc[(events["session"] == i) & (events["home_away"] == "H"), "x"].mean()
            away_mean_x = events.loc[(events["session"] == i) & (events["home_away"] == "A"), "x"].mean()

            if home_mean_x < away_mean_x:  # If the home team plays from left to right
                shots_to_rotate = shots[(shots["session"] == i) & (shots["home_away"] == "H")]
            else:  # If the home team plays from right to left
                shots_to_rotate = shots[(shots["session"] == i) & (shots["home_away"] == "A")]

            shots.loc[shots_to_rotate.index, "x"] = pitch_size[0] - shots_to_rotate["x"]
            shots.loc[shots_to_rotate.index, "y"] = pitch_size[1] - shots_to_rotate["y"]

        shots["x"] = x = shots["x"] / pitch_size[0] * 104
        shots["y"] = y = shots["y"] / pitch_size[1] * 68 - 34
        shots["distance"] = shots[["x", "y"]].apply(np.linalg.norm, axis=1)

        x = shots["x"]
        y = shots["y"]
        goal_width = pitch_size[1] * 7.32 / 68
        angles = np.arctan((goal_width * x) / (x**2 + y**2 - (goal_width / 2) ** 2)) * 180 / np.pi
        shots["angle"] = np.where(angles >= 0, angles, angles + 180)

        shots["freekick"] = shots["event_types"].str.contains("freeKick").astype(int)
        shots["header"] = shots["event_types"].str.contains("header").astype(int)
        shots["goal"] = shots["event_types"].str.contains("goal").astype(int)

        return shots[["x", "y", "distance", "angle", "freekick", "header", "goal"]]

    def pred(self, shot_features: pd.DataFrame) -> pd.DataFrame:
        sum = self.fit.params.values[0]
        for i, c in enumerate(shot_features.columns[:6]):
            sum += self.fit.params.values[i + 1] * shot_features[c]

        shot_features["xg"] = 1 / (1 + np.exp(-sum))
        return shot_features
