import os
import sys
from typing import Dict, List, Tuple

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import axes

import datatools.matplotsoccer as mps


class TraceSnapshot:
    def __init__(
        self,
        traces: pd.DataFrame,
        player_sizes: pd.Series = None,
        player_colors: pd.Series = None,
        player_annots: pd.Series = None,
        highlights: pd.DataFrame | Dict[str, List[str]] = None,
        edges: np.ndarray = None,
        arrows: List[Tuple[str, str]] = None,
        show_velocities=True,
        unnormalize=False,
        pitch_size=(104, 68),
    ):
        self.traces = traces.copy()
        self.sizes = player_sizes
        self.colors = player_colors
        self.annots = player_annots
        self.highlights = highlights
        self.edges = edges
        self.arrows = arrows
        self.show_velocities = show_velocities
        self.pitch_size = pitch_size

        if unnormalize:
            x_cols = [c for c in self.traces.columns if c.endswith("_x")]
            y_cols = [c for c in self.traces.columns if c.endswith("_y")]
            self.traces[x_cols] = (self.traces[x_cols] + 1) * pitch_size[0] / 2
            self.traces[y_cols] = (self.traces[y_cols] + 1) * pitch_size[1] / 2

    def plot_team_players(
        self,
        ax: axes.Axes,
        xy: pd.DataFrame,
        sizes: np.ndarray,
        colors: np.ndarray,
        anonymize=True,
        annotate=None,
    ):
        n_features = len(np.unique([c.split("_")[-1] for c in xy.columns]))  # ["x", "y", "vx", "vy"] for each player
        x = xy[xy.columns[0::n_features]].values[-1]
        y = xy[xy.columns[1::n_features]].values[-1]

        players = [c[:-2] for c in xy.columns[0::n_features]]
        player_dict = dict(zip(players, np.arange(len(players)) + 1))

        if self.highlights is None:
            ax.scatter(x, y, s=sizes, c=colors, zorder=2)

        else:
            if isinstance(self.highlights, pd.DataFrame):
                self.highlights = self.highlights[self.highlights["player_code"].str.startswith(players[0][0])]
                edgecolors = np.where(np.isin(players, self.highlights["player_code"]), "gold", "none")

                for i in self.highlights.index:
                    p = self.highlights.at[i, "player_code"]
                    start_frame = self.highlights.at[i, "start_frame"]
                    end_frame = self.highlights.at[i, "end_frame"]
                    player_x = xy.loc[start_frame:end_frame, f"{p}_x"].values
                    player_y = xy.loc[start_frame:end_frame, f"{p}_y"].values
                    ax.plot(player_x, player_y, c="gold", linewidth=5, zorder=1)

                    if annotate is not None:
                        text = self.highlights.at[i, annotate]

                        dx = player_x[-1] - player_x[-4]
                        dy = player_y[-1] - player_y[-4]
                        norm = np.sqrt(dx**2 + dy**2)
                        text_xy = [player_x[-1] + dx / norm * 3, player_y[-1] + dy / norm * 3]

                        assert isinstance(colors, str)
                        text_color = colors.replace("tab:", "dark")

                        ax.annotate(text, xy=text_xy, ha="center", va="center", color=text_color, fontsize=24, zorder=3)

            else:
                # highlights = [p for p in self.highlights if p[0] == players[0][0]]
                edgecolors = np.repeat("none", len(players))

                for color, players_to_highlight in self.highlights.items():
                    teammates_to_highlight = [p for p in players_to_highlight if p[0] == players[0][0]]
                    edgecolors[np.isin(players, teammates_to_highlight)] = color

                    for p in teammates_to_highlight:
                        player_x = xy[f"{p}_x"].values
                        player_y = xy[f"{p}_y"].values
                        ax.plot(player_x, player_y, c=color, linewidth=5, zorder=1)

            ax.scatter(x, y, s=sizes, c=colors, edgecolors=edgecolors, linewidths=3, zorder=2)

        if self.show_velocities:
            assert n_features >= 4
            vx = xy[xy.columns[2::n_features]].values[-1]
            vy = xy[xy.columns[3::n_features]].values[-1]
            plt.quiver(
                x,
                y,
                vx * 1.5,
                vy * 1.5,
                angles="xy",
                scale_units="xy",
                scale=1,
                color=colors,
                width=0.003,
                headwidth=5,
            )

        for i, p in enumerate(players):
            player_xy = xy[[f"{p}_x", f"{p}_y"]].values
            player_num = player_dict[p] if anonymize else int(p[1:])
            player_color = colors if isinstance(colors, str) else colors[i]

            ax.plot(player_xy[:, 0], player_xy[:, 1], c=player_color, ls="--", zorder=0)
            ax.annotate(
                player_num,
                xy=player_xy[-1],
                ha="center",
                va="center",
                color="w",
                fontsize=18,
                fontweight="bold",
                zorder=3,
            )

        return ax

    def plot(
        self,
        rotate_pitch=False,
        half=None,
        focus_xy: Tuple[float, float] = None,
        switch_teams=False,
        anonymize=False,
        annotate=None,
        min_size=400,
        max_size=2000,
        min_color=0,
        max_color=0.035,
        min_annot=0,
    ):
        if focus_xy is None:
            figsize = (15.6, 10.8) if half is None else (10.4, 14.4)  # (10.4, 12)
        else:
            figsize = (12, 8)  # (8, 8)

        if self.colors is not None:
            figsize = (figsize[0] * 1.2, figsize[1])

        fig, ax = plt.subplots(figsize=figsize)
        mps.field("green", self.pitch_size[0], self.pitch_size[1], fig, ax, show=False)

        if half == "left":
            ax.set_xlim(-4, self.pitch_size[0] / 2)
        elif half == "right":
            ax.set_xlim(self.pitch_size[0] / 2, self.pitch_size[0] + 4)

        if focus_xy is not None:
            ax.set_xlim(focus_xy[0] - 30, focus_xy[0] + 30)
            ax.set_ylim(focus_xy[1] - 20, focus_xy[1] + 20)

        traces = self.traces.dropna(axis=1, how="all")
        xy_cols = [c for c in traces.columns if c.split("_")[-1] in ["x", "y", "vx", "vy"]]

        if rotate_pitch:
            traces[xy_cols[0::4]] = self.pitch_size[0] - traces[xy_cols[0::4]]
            traces[xy_cols[1::4]] = self.pitch_size[1] - traces[xy_cols[1::4]]

        home_xy = traces[[c for c in xy_cols if c[0] == "H"]]
        away_xy = traces[[c for c in xy_cols if c[0] == "A"]]

        home_players = [c[:-2] for c in xy_cols[0::4] if c[0] == "H"]
        away_players = [c[:-2] for c in xy_cols[0::4] if c[0] == "A"]

        if self.sizes is not None and home_players[0] in self.sizes.index:
            home_sizes = self.sizes[home_players].values * (max_size - min_size) + min_size
        else:
            home_sizes = 800

        if self.sizes is not None and away_players[0] in self.sizes.index:
            away_sizes = self.sizes[away_players].values * (max_size - min_size) + min_size
        else:
            away_sizes = 800

        home_colors = "tab:blue" if switch_teams else "tab:red"
        away_colors = "tab:red" if switch_teams else "tab:blue"
        annot_args = {"ha": "center", "va": "center", "color": "k", "fontsize": 16, "fontweight": "bold", "zorder": 5}

        if self.annots is not None:
            for p, value in self.annots[self.annots >= min_annot].items():
                text = f"{value * 100:.1f}"
                text_xy = traces[[f"{p}_x", f"{p}_y"]].values[-1]
                text_xy[1] -= 2
                ax.annotate(text, xy=text_xy, **annot_args)

        if self.colors is not None:
            if home_players[0] in self.colors.index:  # Use [0.65, 1] of jet_r as the colormap
                cmap = plt.get_cmap("jet_r") if switch_teams else plt.get_cmap("jet")
                scores = np.clip((self.colors[home_players].values - min_color) * 0.3 / max_color + 0.7, 0, 1)
                cmap_segment = mpl.colors.LinearSegmentedColormap.from_list("cmap", cmap(np.linspace(0.7, 1, 256)))
                home_colors = cmap(scores)

            elif away_players[0] in self.colors.index:  # Use [0.65, 1] of jet as the colormap
                cmap = plt.get_cmap("jet") if switch_teams else plt.get_cmap("jet_r")
                scores = np.clip((self.colors[away_players].values - min_color) * 0.35 / max_color + 0.65, 0, 1)
                cmap_segment = mpl.colors.LinearSegmentedColormap.from_list("cmap", cmap(np.linspace(0.65, 1, 256)))
                away_colors = cmap(scores)

            norm = mpl.colors.Normalize(vmin=min_color, vmax=max_color)
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_segment)
            sm.set_array([])

            cbar = fig.colorbar(sm, ax=ax)
            # cbar.set_label("Goal probability if receiving a pass", fontsize=20)
            cbar.ax.tick_params(labelsize=20)

        if self.edges is not None:
            for src, dst in self.edges:
                edge_x = traces[[f"{src}_x", f"{dst}_x"]].values[-1]
                edge_y = traces[[f"{src}_y", f"{dst}_y"]].values[-1]
                if src[0] == "H" and dst[0] == "H":
                    edgecolor = "tab:blue" if switch_teams else "tab:red"
                elif src[0] == "A" and dst[0] == "A":
                    edgecolor = "tab:red" if switch_teams else "tab:blue"
                else:
                    edgecolor = "dimgray"
                plt.plot(edge_x, edge_y, color=edgecolor, linewidth=2, alpha=0.5)

        if self.arrows is not None:
            for src, dst in self.arrows:
                x = traces[f"{src}_x"].values[-1]
                y = traces[f"{src}_y"].values[-1]
                dx = traces[f"{dst}_x"].values[-1] - x
                dy = traces[f"{dst}_y"].values[-1] - y
                ax.arrow(x, y, dx, dy, color="k", width=0.5, length_includes_head=True, zorder=4)

        self.plot_team_players(ax, home_xy, home_sizes, home_colors, anonymize, annotate)
        self.plot_team_players(ax, away_xy, away_sizes, away_colors, anonymize, annotate)

        if "ball_x" in traces.columns:
            ball_x = traces["ball_x"].values
            ball_y = traces["ball_y"].values
            ax.scatter(ball_x[-1], ball_y[-1], s=300, c="w", edgecolors="k", marker="o", zorder=5)
            ax.plot(ball_x[-30:], ball_y[-30:], "k", zorder=3)

        plt.show()