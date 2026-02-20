from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, PowerNorm
from matplotlib.lines import Line2D


# Color-blind friendly palette with restrained tones suitable for paper figures.
PALETTE = {
    "ink": "#1b263b",
    "slate": "#415a77",
    "muted": "#6b7280",
    "grid": "#d8dee9",
    "panel": "#f7f8fa",
    "paper": "#ffffff",
    "rl": "#1f77b4",
    "greedy": "#d08c00",
    "random": "#7f7f7f",
    "green": "#2a9d8f",
    "red": "#c94c4c",
    "obstacle": "#222831",
}

ROBOT_COLORS = [
    "#1f77b4", "#e15759", "#2ca02c", "#f28e2b", "#9467bd",
    "#17becf", "#8c564b", "#bcbd22", "#ff9da7", "#4e79a7",
]

TRAJ_CMAP = LinearSegmentedColormap.from_list(
    "traj_coverage",
    ["#ffffff", "#edf4ff", "#c7dcf7", "#8db7e1", "#4e79a7"],
)

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "visit_heat",
    ["#fff7ec", "#fee8c8", "#fdbb84", "#e34a33", "#7f0000"],
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_plot_defaults() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 420,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.facecolor": PALETTE["paper"],
            "figure.facecolor": PALETTE["paper"],
            "axes.edgecolor": "#aab4c3",
            "axes.linewidth": 0.8,
            "axes.titlesize": 13.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.8,
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def _style_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b6c0cc")
    ax.spines["bottom"].set_color("#b6c0cc")


def _panel_label(ax: plt.Axes, label: str) -> None:
    # Panel tags were used in an earlier draft. Intentionally disabled for the final exported figures.
    return


def _metric_box(ax: plt.Axes, text: str, loc: tuple[float, float] = (0.98, 0.04), ha: str = "right") -> None:
    ax.text(
        loc[0],
        loc[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va="bottom",
        fontsize=9.3,
        color=PALETTE["ink"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cfd8e3", "alpha": 0.96},
    )


def _endpoint_label(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9.2,
        color=color,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "linewidth": 0.8, "alpha": 0.96},
    )


def _world_background(obstacles: np.ndarray, base_values: np.ndarray | None = None, *, cmap, norm=None):
    h, w = obstacles.shape
    if base_values is None:
        image = np.zeros((h, w), dtype=float)
        mask = np.ma.masked_where(obstacles, image)
    else:
        mask = np.ma.masked_where(obstacles, base_values)
    obstacle_layer = np.ma.masked_where(~obstacles, obstacles)
    return mask, obstacle_layer


def plot_training(history: Mapping[str, np.ndarray], out_dir: Path) -> None:
    ensure_dir(out_dir)
    set_plot_defaults()

    episodes = history["episode"]
    smooth_n = 35

    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    ax.plot(episodes, history["reward"], color="#b8c8dd", linewidth=1.0, alpha=0.65)
    ax.plot(episodes, history["reward_ma"], color=PALETTE["rl"], linewidth=2.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Team reward")
    ax.set_title("Training reward")
    _style_axes(ax)
    _metric_box(ax, f"Moving average: {smooth_n} episodes")
    _panel_label(ax, "A")
    fig.tight_layout()
    fig.savefig(out_dir / "training_reward_curve.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    ax.plot(episodes, 100 * history["coverage"], color="#bfe8d6", linewidth=1.0, alpha=0.72)
    ax.plot(episodes, 100 * history["coverage_ma"], color=PALETTE["green"], linewidth=2.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Final coverage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Coverage during training")
    _style_axes(ax)
    _metric_box(ax, f"Last-50 mean: {100*np.mean(history['coverage'][-50:]):.1f}%")
    _panel_label(ax, "A")
    fig.tight_layout()
    fig.savefig(out_dir / "training_coverage_curve.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.45))
    ax.plot(episodes, history["collisions"], color="#f3c1c1", linewidth=1.0, alpha=0.72)
    ax.plot(episodes, history["collisions_ma"], color=PALETTE["red"], linewidth=2.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Collisions per episode")
    ax.set_title("Collision rate during training")
    _style_axes(ax)
    _metric_box(ax, f"Last-50 mean: {np.mean(history['collisions'][-50:]):.2f}")
    _panel_label(ax, "A")
    fig.tight_layout()
    fig.savefig(out_dir / "training_collision_curve.png", bbox_inches="tight")
    plt.close(fig)


def plot_policy_comparison(rollouts: Mapping[str, dict], out_dir: Path) -> None:
    ensure_dir(out_dir)
    set_plot_defaults()

    labels = {
        "learned": "RL swarm",
        "greedy": "Greedy Coverage Algorithm",
        "random": "Random walk",
    }
    colors = {
        "learned": PALETTE["rl"],
        "greedy": PALETTE["greedy"],
        "random": PALETTE["random"],
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for key in ["learned", "greedy", "random"]:
        r = rollouts[key]
        x = np.arange(len(r["coverage"]))
        y = 100 * r["coverage"]
        ax.plot(x, y, label=labels[key], color=colors[key], linewidth=2.6 if key == "learned" else 2.2)
        ax.scatter([x[-1]], [y[-1]], s=28, color=colors[key], zorder=4)
        _endpoint_label(ax, x[-1], y[-1], f"{labels[key]}  {y[-1]:.1f}%", colors[key])

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Covered free space (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(rollouts["learned"]["coverage"]) - 1 + 18)
    ax.set_title("Coverage attained during evaluation")
    _style_axes(ax)
    _panel_label(ax, "A")
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99))
    fig.tight_layout()
    fig.savefig(out_dir / "policy_comparison_coverage.png", bbox_inches="tight")
    plt.close(fig)

    names = [labels[k] for k in ["learned", "greedy", "random"]]
    keys = ["learned", "greedy", "random"]
    coverage = [100 * rollouts[k]["final_coverage"] for k in keys]
    collisions = [rollouts[k]["total_collisions"] for k in keys]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    y_pos = np.arange(len(keys))
    bars = ax.barh(y_pos, coverage, color=[colors[k] for k in keys], height=0.58)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Final coverage (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Controller-wise final coverage")
    _style_axes(ax, grid_axis="x")
    for bar, value in zip(bars, coverage):
        ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", ha="left", fontsize=9.3)
    margin = coverage[0] - coverage[1]
    _metric_box(ax, f"RL margin over greedy: {margin:.2f} points", loc=(0.98, 0.02))
    _panel_label(ax, "A")
    fig.tight_layout()
    fig.savefig(out_dir / "final_coverage_bars.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bars = ax.barh(y_pos, collisions, color=[colors[k] for k in keys], height=0.58)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Total collisions")
    ax.set_title("Controller-wise collision count")
    _style_axes(ax, grid_axis="x")
    xmax = max(collisions) * 1.10 if max(collisions) > 0 else 1.0
    ax.set_xlim(0, xmax)
    for bar, value in zip(bars, collisions):
        x_text = value + max(0.03 * xmax, 0.75)
        ax.text(x_text, bar.get_y() + bar.get_height() / 2, f"{value}", va="center", ha="left", fontsize=9.3)
    _panel_label(ax, "A")
    fig.tight_layout()
    fig.savefig(out_dir / "controller_collision_bars.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for key in ["learned", "greedy", "random"]:
        r = rollouts[key]
        x = np.arange(len(r["spacing"]))
        ax.plot(x, r["spacing"], label=labels[key], color=colors[key], linewidth=2.4 if key == "learned" else 2.0)
    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Mean nearest-neighbour spacing (cells)")
    ax.set_title("Swarm dispersion over time")
    _style_axes(ax)
    ax.legend(loc="lower right")
    _panel_label(ax, "A")
    fig.tight_layout()
    fig.savefig(out_dir / "robot_spacing_over_time.png", bbox_inches="tight")
    plt.close(fig)


def draw_world(ax: plt.Axes, obstacles: np.ndarray, visit_counts: np.ndarray | None = None) -> None:
    if visit_counts is None:
        base_values = np.zeros_like(obstacles, dtype=float)
        masked, obstacle_layer = _world_background(obstacles, base_values, cmap=TRAJ_CMAP)
        ax.imshow(masked, cmap=ListedColormap(["#fbfdff"]), origin="upper", interpolation="nearest")
    else:
        vmax = float(np.max(visit_counts)) if np.max(visit_counts) > 0 else 1.0
        masked, obstacle_layer = _world_background(obstacles, visit_counts.astype(float), cmap=TRAJ_CMAP)
        ax.imshow(masked, cmap=TRAJ_CMAP, norm=PowerNorm(gamma=0.58, vmin=0, vmax=vmax), origin="upper", interpolation="nearest")

    ax.imshow(obstacle_layer, cmap=ListedColormap([PALETTE["obstacle"]]), origin="upper", interpolation="nearest")
    h, w = obstacles.shape
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#bcc7d4")
        spine.set_linewidth(0.9)


def plot_trajectories(rollout: dict, out_dir: Path) -> None:
    ensure_dir(out_dir)
    set_plot_defaults()

    positions = rollout["positions"]
    visit_counts = rollout["visit_counts"]
    obstacles = rollout["obstacles"]
    final_cov = 100 * float(rollout["final_coverage"])
    total_collisions = int(rollout["total_collisions"])

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    draw_world(ax, obstacles, visit_counts)

    for robot_id in range(positions.shape[1]):
        path = positions[:, robot_id, :]
        color = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
        ax.plot(path[:, 1], path[:, 0], color=color, linewidth=1.55, alpha=0.96, zorder=3)
        ax.scatter(path[0, 1], path[0, 0], s=42, facecolors="white", edgecolors=color, linewidth=1.15, zorder=5)
        ax.scatter(path[-1, 1], path[-1, 0], s=40, color=color, marker="D", edgecolors="white", linewidth=0.55, zorder=6)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=PALETTE["ink"], markersize=6.5, linewidth=0, label="Start"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["ink"], markeredgecolor="white", markersize=6.5, linewidth=0, label="End"),
        Line2D([0, 1], [0, 0], color=PALETTE["ink"], linewidth=1.6, label="Robot trajectory"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.08), columnspacing=1.0, handletextpad=0.5)
    ax.set_title("RL swarm rollout: robot trajectories on the 50×50 map")
    _metric_box(ax, f"Coverage: {final_cov:.2f}%\nCollisions: {total_collisions}\nRobots: {positions.shape[1]}")
    _panel_label(ax, "A")
    fig.tight_layout()
    fig.savefig(out_dir / "learned_trajectory_map.png", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(rollout: dict, out_dir: Path) -> None:
    ensure_dir(out_dir)
    set_plot_defaults()

    visit_counts = rollout["visit_counts"].astype(float)
    obstacles = rollout["obstacles"]
    masked = np.ma.masked_where(obstacles, visit_counts)
    vmax = float(np.max(visit_counts)) if np.max(visit_counts) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7.05, 6.8))
    im = ax.imshow(masked, cmap=HEAT_CMAP, norm=PowerNorm(gamma=0.56, vmin=0, vmax=vmax), origin="upper", interpolation="nearest")
    obstacle_layer = np.ma.masked_where(~obstacles, obstacles)
    ax.imshow(obstacle_layer, cmap=ListedColormap([PALETTE["obstacle"]]), origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#bcc7d4")
        spine.set_linewidth(0.9)
    ax.set_title("Spatial visitation density of the learned swarm policy")
    cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.028)
    cbar.set_label("Visit count", rotation=90)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("#bcc7d4")
    _metric_box(ax, "Obstacle cells shown in dark slate", loc=(0.98, 0.02))
    _panel_label(ax, "A")
    fig.tight_layout()
    fig.savefig(out_dir / "learned_coverage_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_snapshot(rollout: dict, step: int, out_dir: Path) -> None:
    ensure_dir(out_dir)
    set_plot_defaults()

    step = int(min(max(step, 0), len(rollout["positions"]) - 1))
    positions_all = rollout["positions"]
    positions = positions_all[step]
    obstacles = rollout["obstacles"]
    visit_counts = np.zeros_like(rollout["visit_counts"], dtype=float)

    for frame in positions_all[: step + 1]:
        for r, c in frame:
            visit_counts[r, c] += 1

    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    draw_world(ax, obstacles, visit_counts)

    trail_len = min(18, step + 1)
    for robot_id, pos in enumerate(positions):
        color = ROBOT_COLORS[robot_id % len(ROBOT_COLORS)]
        trail = positions_all[max(0, step - trail_len + 1): step + 1, robot_id]
        if len(trail) > 1:
            ax.plot(trail[:, 1], trail[:, 0], color=color, linewidth=2.1, alpha=0.9, zorder=5)
        ax.scatter(pos[1], pos[0], s=150, color=color, edgecolor="white", linewidth=1.6, zorder=7)
        ax.text(
            pos[1],
            pos[0],
            str(robot_id + 1),
            ha="center",
            va="center",
            color="white",
            fontsize=7.2,
            fontweight="bold",
            zorder=8,
        )

    covered = np.count_nonzero((visit_counts > 0) & (~obstacles))
    free = np.count_nonzero(~obstacles)
    coverage_pct = 100.0 * covered / max(free, 1)
    _metric_box(
        ax,
        f"Step: {step}\nCoverage so far: {coverage_pct:.2f}%\nRobots: {len(positions)}",
        loc=(0.98, 0.02),
    )

    legend_handles = [
        Line2D([0, 1], [0, 0], color=PALETTE["ink"], linewidth=2.0, label="Recent path"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["ink"], markeredgecolor="white", markersize=8, label="Robot"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.985, 1.01), ncol=2, columnspacing=1.2, handletextpad=0.5)
    ax.set_title(f"RL swarm rollout snapshot")
    fig.tight_layout()
    fig.savefig(out_dir / f"snapshot_step_{step:03d}.png", bbox_inches="tight")
    plt.close(fig)


def save_rollout_frames(rollout: dict, out_dir: Path) -> None:
    steps = [0, min(45, len(rollout["positions"]) - 1), min(95, len(rollout["positions"]) - 1), len(rollout["positions"]) - 1]
    for step in steps:
        plot_snapshot(rollout, step, out_dir)


def make_all_figures(history: Mapping[str, np.ndarray], rollouts: Mapping[str, dict], out_dir: Path) -> None:
    plot_training(history, out_dir)
    plot_policy_comparison(rollouts, out_dir)
    plot_trajectories(rollouts["learned"], out_dir)
    plot_heatmap(rollouts["learned"], out_dir)
    save_rollout_frames(rollouts["learned"], out_dir)
