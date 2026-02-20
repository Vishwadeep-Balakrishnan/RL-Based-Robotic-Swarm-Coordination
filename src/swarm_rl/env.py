
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


DIRECTIONS = np.array(
    [
        [-1, 0],   # north
        [-1, 1],   # north-east
        [0, 1],    # east
        [1, 1],    # south-east
        [1, 0],    # south
        [1, -1],   # south-west
        [0, -1],   # west
        [-1, -1],  # north-west
        [0, 0],    # hold
    ],
    dtype=np.int16,
)

ACTION_NAMES = (
    "N",
    "NE",
    "E",
    "SE",
    "S",
    "SW",
    "W",
    "NW",
    "HOLD",
)


@dataclass(frozen=True)
class SwarmConfig:
    """Simulation parameters for the coverage task."""

    height: int = 50
    width: int = 50
    robot_count: int = 10
    max_steps: int = 150
    sensor_radius: int = 6
    target_coverage: float = 0.90
    seed: int = 17


class SwarmCoverageEnv:
    """A small multi-robot coverage environment.

    The world is a 2-D grid with static obstacles. Robots receive local,
    deliberately compact observations. The shared Q-agent learns a single
    policy reused by all robots, which is a useful trick for making swarm
    policies scale without making the code heavy.
    """

    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config or SwarmConfig()
        self.rng = np.random.default_rng(self.config.seed)

        self.obstacles = self._build_map()
        self.free_mask = ~self.obstacles
        self.free_cells = np.argwhere(self.free_mask)
        self.free_total = int(self.free_mask.sum())

        rows, cols = np.indices(self.free_mask.shape)
        self.cluster_cells = np.argwhere(
            self.free_mask
            & (rows > self.config.height - 13)
            & (cols < 14)
        )

        self.positions = np.zeros((self.config.robot_count, 2), dtype=np.int16)
        self.coverage = np.zeros((self.config.height, self.config.width), dtype=np.int16)
        self.covered_count = 0
        self.t = 0

    def _build_map(self) -> np.ndarray:
        h, w = self.config.height, self.config.width
        grid = np.zeros((h, w), dtype=bool)

        # Border walls.
        grid[0, :] = True
        grid[-1, :] = True
        grid[:, 0] = True
        grid[:, -1] = True

        # Warehouse-like shelf islands scaled for a 50 x 50 world. The gaps form
        # several competing corridors, which is exactly where coordinated robot
        # splitting should matter.
        shelves = [
            (6, 17, 8, 14),
            (6, 17, 22, 28),
            (6, 17, 36, 42),
            (31, 42, 8, 14),
            (31, 42, 22, 28),
            (31, 42, 36, 42),
            (20, 27, 16, 20),
            (20, 27, 30, 34),
        ]
        for r0, r1, c0, c1 in shelves:
            grid[r0:r1, c0:c1] = True

        # Thin partial walls create bottlenecks without closing the map.
        grid[23:26, 4:13] = True
        grid[23:26, 37:46] = True
        grid[3:10, 24:27] = True
        grid[40:47, 24:27] = True

        # Small pillars break symmetry and make random walking visibly poor.
        pillars = [(12, 4), (20, 7), (37, 4), (12, 46), (20, 43), (37, 46), (25, 25)]
        for r, c in pillars:
            grid[max(1, r - 1): min(h - 1, r + 2), max(1, c - 1): min(w - 1, c + 2)] = True

        return grid

    def reset(self, starts: Optional[np.ndarray] = None, cluster: bool = True) -> tuple[np.ndarray, dict]:
        self.t = 0
        self.coverage.fill(0)

        if starts is None:
            self.positions = self._sample_starts(cluster=cluster)
        else:
            starts = np.asarray(starts, dtype=np.int16)
            if starts.shape != (self.config.robot_count, 2):
                raise ValueError("starts must have shape (robot_count, 2)")
            self.positions = starts.copy()

        self.covered_count = 0
        for r, c in self.positions:
            if self.coverage[r, c] == 0:
                self.covered_count += 1
            self.coverage[r, c] += 1

        return self.states(), self.info(last_collisions=0, step_new_cells=self.covered_count)

    def _sample_starts(self, cluster: bool) -> np.ndarray:
        if cluster:
            candidates = self.cluster_cells.copy()
        else:
            candidates = self.free_cells.copy()

        self.rng.shuffle(candidates)
        starts = []
        taken = set()
        for cell in candidates:
            p = (int(cell[0]), int(cell[1]))
            if p not in taken:
                starts.append(p)
                taken.add(p)
            if len(starts) == self.config.robot_count:
                break

        if len(starts) < self.config.robot_count:
            raise RuntimeError("not enough free cells to place robots")

        return np.array(starts, dtype=np.int16)

    def states(self) -> np.ndarray:
        occupied = {tuple(map(int, p)) for p in self.positions}
        return np.array([self.encode_state(i, occupied) for i in range(self.config.robot_count)], dtype=np.int16)

    def encode_state(self, i: int, occupied: set[tuple[int, int]] | None = None) -> int:
        pos = self.positions[i]
        sector = self.coverage_direction(pos)

        crowd = self._crowd_bucket(pos, i)
        visits = int(self.coverage[pos[0], pos[1]])
        visit_bucket = 0 if visits == 0 else (1 if visits == 1 else 2)

        blocked = self._blocked_neighbors(pos, ignore_robot=i, occupied=occupied)
        hazard_bucket = 0 if blocked <= 1 else (1 if blocked <= 3 else 2)

        edge_distance = min(pos[0], pos[1], self.config.height - 1 - pos[0], self.config.width - 1 - pos[1])
        edge_bucket = 0 if edge_distance <= 2 else (1 if edge_distance <= 6 else 2)

        return (((sector * 3 + crowd) * 3 + visit_bucket) * 3 + hazard_bucket) * 3 + edge_bucket

    @property
    def state_count(self) -> int:
        return 9 * 3 * 3 * 3 * 3

    @property
    def action_count(self) -> int:
        return len(DIRECTIONS)

    def _crowd_bucket(self, pos: np.ndarray, robot_index: int) -> int:
        neighbors = 0
        for j, other in enumerate(self.positions):
            if j == robot_index:
                continue
            if max(abs(int(other[0]) - int(pos[0])), abs(int(other[1]) - int(pos[1]))) <= 2:
                neighbors += 1
        return 0 if neighbors == 0 else (1 if neighbors == 1 else 2)

    def _blocked_neighbors(
        self,
        pos: np.ndarray,
        ignore_robot: int,
        occupied: set[tuple[int, int]] | None = None,
    ) -> int:
        blocked = 0
        r0, c0 = int(pos[0]), int(pos[1])
        own_cell = (r0, c0)
        if occupied is None:
            occupied = {tuple(map(int, p)) for p in self.positions}

        for dr in (-1, 0, 1):
            rr = r0 + dr
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                cc = c0 + dc
                cell = (rr, cc)
                if (
                    rr < 0
                    or rr >= self.config.height
                    or cc < 0
                    or cc >= self.config.width
                    or self.obstacles[rr, cc]
                    or (cell in occupied and cell != own_cell)
                ):
                    blocked += 1
        return blocked

    def coverage_direction(self, pos: Iterable[int]) -> int:
        """Return the best local direction for finding under-visited space."""
        r0, c0 = int(pos[0]), int(pos[1])
        radius = self.config.sensor_radius

        best_action = 8
        best_score = (10_000, 10_000)

        for action, direction in enumerate(DIRECTIONS[:8]):
            dr, dc = int(direction[0]), int(direction[1])
            for dist in range(1, radius + 1):
                rr = r0 + dist * dr
                cc = c0 + dist * dc
                if (
                    rr < 0
                    or rr >= self.config.height
                    or cc < 0
                    or cc >= self.config.width
                    or self.obstacles[rr, cc]
                ):
                    break

                visits = int(self.coverage[rr, cc])
                score = (visits, dist)
                if score < best_score:
                    best_score = score
                    best_action = action

        return best_action

    def is_free(self, pos: Iterable[int]) -> bool:
        r, c = int(pos[0]), int(pos[1])
        if r < 0 or r >= self.config.height or c < 0 or c >= self.config.width:
            return False
        return not bool(self.obstacles[r, c])

    def step(self, actions: Iterable[int]) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        actions = np.asarray(actions, dtype=np.int16)
        if actions.shape != (self.config.robot_count,):
            raise ValueError("one action is required for each robot")

        old_positions = self.positions.copy()
        proposed = old_positions + DIRECTIONS[actions]

        valid = np.array([self.is_free(p) for p in proposed], dtype=bool)
        collision = ~valid

        # Same target cell.
        targets = {}
        for i, p in enumerate(proposed):
            key = tuple(map(int, p))
            targets.setdefault(key, []).append(i)
        for robots in targets.values():
            if len(robots) > 1:
                collision[robots] = True

        # Edge swaps are real collisions for disk-like robots.
        for i in range(self.config.robot_count):
            for j in range(i + 1, self.config.robot_count):
                if np.array_equal(proposed[i], old_positions[j]) and np.array_equal(proposed[j], old_positions[i]):
                    collision[i] = True
                    collision[j] = True

        next_positions = old_positions.copy()
        for i in range(self.config.robot_count):
            if not collision[i]:
                next_positions[i] = proposed[i]

        self.positions = next_positions
        self.t += 1

        rewards = np.full(self.config.robot_count, -0.015, dtype=np.float32)
        new_cell_count = 0

        for i, pos in enumerate(self.positions):
            action = int(actions[i])
            if collision[i]:
                rewards[i] -= 1.00
            else:
                visits_before = int(self.coverage[pos[0], pos[1]])
                if visits_before == 0:
                    rewards[i] += 2.20
                    new_cell_count += 1
                else:
                    rewards[i] += 0.08 / float(visits_before + 1)

                if action == 8:
                    rewards[i] -= 0.08
                else:
                    wanted_sector = self.coverage_direction(old_positions[i])
                    if action == wanted_sector:
                        rewards[i] += 0.28

            close_neighbors = 0
            nearest = 99
            for j, other in enumerate(self.positions):
                if i == j:
                    continue
                d = max(abs(int(other[0]) - int(pos[0])), abs(int(other[1]) - int(pos[1])))
                nearest = min(nearest, d)
                if d <= 1:
                    close_neighbors += 1

            rewards[i] -= 0.08 * close_neighbors
            if nearest >= 4:
                rewards[i] += 0.03

        if new_cell_count:
            rewards += 0.04 * new_cell_count

        for r, c in self.positions:
            self.coverage[r, c] += 1
        self.covered_count += new_cell_count

        done = self.coverage_fraction >= self.config.target_coverage or self.t >= self.config.max_steps
        return self.states(), rewards, done, self.info(last_collisions=int(collision.sum()), step_new_cells=int(new_cell_count))

    @property
    def coverage_fraction(self) -> float:
        return float(self.covered_count / self.free_total)

    def info(self, last_collisions: int, step_new_cells: int) -> dict:
        return {
            "step": self.t,
            "coverage": self.coverage_fraction,
            "collisions": last_collisions,
            "new_cells": step_new_cells,
            "positions": self.positions.copy(),
        }
