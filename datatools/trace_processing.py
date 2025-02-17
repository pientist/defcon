import os
import sys
from fnmatch import fnmatch

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm

from datatools import config


def is_home_on_left(period_traces: pd.DataFrame):
    home_x_cols = [c for c in period_traces.columns if fnmatch(c, "home_*_x")]
    away_x_cols = [c for c in period_traces.columns if fnmatch(c, "away_*_x")]

    home_gk = (period_traces[home_x_cols].mean() - config.FIELD_SIZE[0] / 2).abs().idxmax()[:-2]
    away_gk = (period_traces[away_x_cols].mean() - config.FIELD_SIZE[0] / 2).abs().idxmax()[:-2]

    home_gk_x = period_traces[f"{home_gk}_x"].mean()
    away_gk_x = period_traces[f"{away_gk}_x"].mean()

    return home_gk_x < away_gk_x


def label_frames_and_episodes(traces: pd.DataFrame, fps=25) -> pd.DataFrame:
    traces = traces.sort_values(["period_id", "timestamp"], ignore_index=True)

    traces["frame"] = (traces["timestamp"] * fps).round().astype(int)
    max_frame_p1 = traces.loc[traces["period_id"] == 1, "frame"].max()
    traces.loc[traces["period_id"] == 2, "frame"] += max_frame_p1 + 1

    traces["episode_id"] = 0
    n_prev_episodes = 0

    for i in traces["period_id"].unique():
        period_traces = traces[(traces["period_id"] == i) & (traces["ball_state"] == "alive")]
        frame_diffs = np.diff(period_traces["frame"].values, prepend=-5)
        period_episode_ids = (frame_diffs >= 5).astype(int).cumsum() + n_prev_episodes
        traces.loc[period_traces.index, "episode_id"] = period_episode_ids
        n_prev_episodes = period_episode_ids.max()

    return traces.set_index("frame")


def summarize_playing_times(traces: pd.DataFrame) -> pd.DataFrame:
    players = [c[:-2] for c in traces.columns if c[:4] in ["home", "away"] and c.endswith("_x")]
    play_records = dict()

    for p in players:
        player_x = traces[f"{p}_x"].dropna()
        if not player_x.empty:
            play_records[p] = {"in_frame": player_x.index[0], "out_frame": player_x.index[-1]}

    return pd.DataFrame(play_records).T


def calc_physical_features(traces: pd.DataFrame, fps=25) -> pd.DataFrame:
    from scipy.signal import savgol_filter

    if "episode_id" not in traces:
        traces = label_frames_and_episodes(traces)

    home_players = [c[:-2] for c in traces.dropna(axis=1, how="all").columns if fnmatch(c, "home_*_x")]
    away_players = [c[:-2] for c in traces.dropna(axis=1, how="all").columns if fnmatch(c, "away_*_x")]
    objects = home_players + away_players + ["ball"]
    physical_features = ["x", "y", "vx", "vy", "speed", "accel"]

    tqdm_desc = "Calculating physical features"
    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"

    for p in tqdm(objects, desc=tqdm_desc, bar_format=bar_format):
        new_features = pd.DataFrame(np.nan, index=traces.index, columns=[f"{p}_{f}" for f in physical_features[2:]])
        traces = pd.concat([traces, new_features], axis=1)

        for i in traces["period_id"].unique():
            x: pd.Series = traces.loc[traces["period_id"] == i, f"{p}_x"].dropna()
            y: pd.Series = traces.loc[traces["period_id"] == i, f"{p}_y"].dropna()
            if x.empty:
                continue

            vx = savgol_filter(np.diff(x.values) * fps, window_length=15, polyorder=2)
            vy = savgol_filter(np.diff(y.values) * fps, window_length=15, polyorder=2)
            ax = savgol_filter(np.diff(vx) * fps, window_length=9, polyorder=2)
            ay = savgol_filter(np.diff(vy) * fps, window_length=9, polyorder=2)

            traces.loc[x.index[1:], f"{p}_vx"] = vx
            traces.loc[x.index[1:], f"{p}_vy"] = vy
            traces.loc[x.index[1:], f"{p}_speed"] = np.sqrt(vx**2 + vy**2)
            traces.loc[x.index[1:-1], f"{p}_accel"] = np.sqrt(ax**2 + ay**2)

            traces.at[x.index[0], f"{p}_vx"] = traces.at[x.index[1], f"{p}_vx"]
            traces.at[x.index[0], f"{p}_vy"] = traces.at[x.index[1], f"{p}_vy"]
            traces.at[x.index[0], f"{p}_speed"] = traces.at[x.index[1], f"{p}_speed"]
            traces.loc[[x.index[0], x.index[-1]], f"{p}_accel"] = 0

    state_cols = ["period_id", "timestamp", "episode_id", "ball_state", "ball_owning_home_away"]
    feature_cols = [f"{p}_{f}" for p in objects for f in physical_features] + ["ball_z"]

    return traces[state_cols + feature_cols].copy()


if __name__ == "__main__":
    events = pd.read_parquet("data/ajax/event/event_new.parquet")
    events["utc_timestamp"] = pd.to_datetime(events["utc_timestamp"])
    events = events.sort_values(["stats_perform_match_id", "utc_timestamp"], ignore_index=True)
    game_ids = np.sort(events["stats_perform_match_id"].unique())[1:]

    # trace_files = np.sort(os.listdir("data/ajax/tracking"))
    # game_ids = np.sort([f.split("_")[0] for f in trace_files if f.endswith(".parquet")])
    os.makedirs("data/ajax/tracking_processed", exist_ok=True)

    for i, game_id in enumerate(game_ids):
        if not os.path.exists(f"data/ajax/tracking/{game_id}.parquet"):
            continue

        print(f"\n[{i}] {game_id}")
        traces = pd.read_parquet(f"data/ajax/tracking/{game_id}.parquet")

        traces[["timestamp", "ball_x", "ball_y"]] = traces[["timestamp", "ball_x", "ball_y"]].round(2)
        traces["ball_z"] = (traces["ball_z"].astype(float) / 100).round(2)  # centimeters to meters

        traces = label_frames_and_episodes(traces)
        traces = calc_physical_features(traces)

        traces.to_parquet(f"data/ajax/tracking_processed/{game_id}.parquet")
