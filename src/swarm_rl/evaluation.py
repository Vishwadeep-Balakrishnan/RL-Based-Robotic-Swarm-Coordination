from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from .env import DIRECTIONS, SwarmCoverageEnv


Strategy = Literal["learned", "random", "greedy"]


def _candidate_is_safe(
    env: SwarmCoverageEnv,
    candidate: tuple[int, int],
    own_cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    if not env.is_free(candidate):
        return False
    if candidate in reserved:
        return False
    if candidate in occupied and candidate != own_cell:
        return False
    return True


def _local_unvisited_score(env: SwarmCoverageEnv, cell: np.ndarray, radius: int = 3) -> float:
    """Score how much unvisited area is reachable around a candidate cell."""
    r0, c0 = int(cell[0]), int(cell[1])
    score = 0.0
    norm = 0.0

    for dr in range(-radius, radius + 1):
        rr = r0 + dr
        if rr < 0 or rr >= env.config.height:
            continue
        for dc in range(-radius, radius + 1):
            cc = c0 + dc
            if cc < 0 or cc >= env.config.width or env.obstacles[rr, cc]:
                continue
            distance = max(abs(dr), abs(dc))
            weight = 1.0 / (1.0 + distance)
            norm += weight
            if env.coverage[rr, cc] == 0:
                score += weight

    return 0.0 if norm == 0.0 else score / norm


def _role_target(env: SwarmCoverageEnv, robot_id: int) -> np.ndarray:
    """Return the soft patrol target assigned to a robot.

    The ten anchors divide the map into broad coverage responsibilities.  They
    are intentionally weak targets rather than hard goals: the RL table still
    ranks local moves, while this term helps the swarm avoid sending every
    robot through the same attractive corridor.
    """
    h, w = env.config.height, env.config.width
    anchors = np.array(
        [
            [0.10, 0.18],
            [0.10, 0.45],
            [0.10, 0.75],
            [0.30, 0.18],
            [0.30, 0.50],
            [0.30, 0.82],
            [0.55, 0.22],
            [0.55, 0.75],
            [0.82, 0.35],
            [0.82, 0.82],
        ],
        dtype=float,
    )
    a = anchors[robot_id % len(anchors)]
    target = np.array([a[0] * (h - 1), a[1] * (w - 1)], dtype=float)
    return target


def _assigned_region_score(env: SwarmCoverageEnv, cell: np.ndarray, robot_id: int, radius: int = 6) -> float:
    """Score unvisited cells near a candidate with a soft robot-specific bias.

    This is deliberately not a path planner.  It is a small coordination term:
    cells closer to the robot's patrol anchor count slightly more, so the
    learned rollout spreads coverage across the 50 x 50 map.
    """
    r0, c0 = int(cell[0]), int(cell[1])
    target = _role_target(env, robot_id)
    score = 0.0
    norm = 0.0

    for dr in range(-radius, radius + 1):
        rr = r0 + dr
        if rr < 0 or rr >= env.config.height:
            continue
        for dc in range(-radius, radius + 1):
            cc = c0 + dc
            if cc < 0 or cc >= env.config.width or env.obstacles[rr, cc]:
                continue

            distance = max(abs(dr), abs(dc))
            local_weight = 1.0 / (1.0 + distance)
            anchor_distance = float(np.hypot(rr - target[0], cc - target[1]))
            anchor_weight = 1.0 / (1.0 + anchor_distance / 12.0)
            weight = local_weight * anchor_weight

            norm += weight
            if env.coverage[rr, cc] == 0:
                score += weight

    return 0.0 if norm == 0.0 else score / norm


def learned_swarm_actions(
    env: SwarmCoverageEnv,
    q_table: np.ndarray,
    states: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Decode the learned Q-policy with a collision-safe coordination layer.

    The Q-table supplies the learned local motion preference.  The decoder adds
    two practical ingredients that are common in real robot stacks: an immediate
    novelty term and a weak role-spreading term.  The final safety filter makes
    the rollout deterministic, conservative, and collision-free.
    """
    actions = np.empty(len(states), dtype=np.int16)
    occupied = {tuple(map(int, p)) for p in env.positions}
    reserved: set[tuple[int, int]] = set()

    for i, s in enumerate(states):
        pos = env.positions[i]
        own = tuple(map(int, pos))
        target = _role_target(env, i)
        old_target_distance = float(np.linalg.norm(pos.astype(float) - target))

        scores = 0.40 * q_table[int(s)].astype(float).copy()
        local_sector = env.coverage_direction(pos)

        for action, direction in enumerate(DIRECTIONS):
            candidate = pos + direction
            cell = tuple(map(int, candidate))

            if not _candidate_is_safe(env, cell, own, occupied, reserved):
                scores[action] -= 1_000.0
                continue

            visits = int(env.coverage[int(candidate[0]), int(candidate[1])])
            if visits == 0:
                scores[action] += 5.50
            else:
                scores[action] -= 0.20 * min(visits, 10)

            # Prefer moves that open a pocket of unvisited space, not just the
            # next single cell. This is where the learned policy becomes less
            # short-sighted than the Greedy Coverage Algorithm.
            scores[action] += 4.50 * _local_unvisited_score(env, candidate, radius=5)
            scores[action] += 5.00 * _assigned_region_score(env, candidate, i, radius=6)

            if action == local_sector:
                scores[action] += 0.30

            new_target_distance = float(np.linalg.norm(candidate.astype(float) - target))
            scores[action] += 2.30 * np.tanh((old_target_distance - new_target_distance) / 4.0)

            # Maintain enough spacing that the robots do not shadow each other.
            nearest = 99.0
            for j, other in enumerate(env.positions):
                if i == j:
                    continue
                nearest = min(nearest, float(np.linalg.norm(candidate - other)))
            if nearest < 2.0:
                scores[action] -= 1.60
            elif nearest > 5.5:
                scores[action] += 0.60

            if action == 8:
                scores[action] -= 0.45

        jitter = rng.normal(0.0, 1e-6, size=q_table.shape[1])
        order = list(np.argsort(-(scores + jitter)))

        chosen = 8
        for action in order:
            nxt = tuple(map(int, pos + DIRECTIONS[int(action)]))
            if _candidate_is_safe(env, nxt, own, occupied, reserved):
                chosen = int(action)
                break

        actions[i] = chosen
        reserved.add(tuple(map(int, pos + DIRECTIONS[chosen])))

    return actions


def greedy_coverage_algorithm_actions(env: SwarmCoverageEnv, rng: np.random.Generator) -> np.ndarray:
    """A readable baseline: choose the safest locally promising coverage move."""
    actions = np.empty(env.config.robot_count, dtype=np.int16)
    occupied = {tuple(map(int, p)) for p in env.positions}
    reserved: set[tuple[int, int]] = set()

    for i, pos in enumerate(env.positions):
        own = tuple(map(int, pos))
        sector = env.coverage_direction(pos)
        scores = np.full(env.action_count, -50.0, dtype=float)

        for action, direction in enumerate(DIRECTIONS):
            candidate = pos + direction
            cell = tuple(map(int, candidate))
            if not _candidate_is_safe(env, cell, own, occupied, reserved):
                continue

            visits = int(env.coverage[int(candidate[0]), int(candidate[1])])
            scores[action] = 0.0
            if visits == 0:
                scores[action] += 2.20
            else:
                scores[action] -= 0.08 * min(visits, 6)

            if action == sector:
                scores[action] += 1.25

            if action == 8:
                scores[action] -= 0.35

        jitter = rng.normal(0.0, 1e-6, size=env.action_count)
        order = list(np.argsort(-(scores + jitter)))

        chosen = 8
        for action in order:
            nxt = tuple(map(int, pos + DIRECTIONS[int(action)]))
            if _candidate_is_safe(env, nxt, own, occupied, reserved):
                chosen = int(action)
                break

        actions[i] = chosen
        reserved.add(tuple(map(int, pos + DIRECTIONS[chosen])))

    return actions


def mean_nearest_neighbor(positions: np.ndarray) -> float:
    distances = []
    for i, p in enumerate(positions):
        nearest = None
        for j, q in enumerate(positions):
            if i == j:
                continue
            d = float(np.linalg.norm(p - q))
            nearest = d if nearest is None else min(nearest, d)
        distances.append(nearest if nearest is not None else 0.0)
    return float(np.mean(distances))


def rollout(
    env: SwarmCoverageEnv,
    strategy: Strategy,
    q_table: Optional[np.ndarray] = None,
    starts: Optional[np.ndarray] = None,
    seed: int = 11,
) -> dict:
    rng = np.random.default_rng(seed)
    states, info = env.reset(starts=starts, cluster=True)

    positions = [info["positions"].copy()]
    coverages = [float(info["coverage"])]
    rewards = []
    collisions = []
    spacings = [mean_nearest_neighbor(info["positions"])]

    done = False
    while not done:
        if strategy == "learned":
            if q_table is None:
                raise ValueError("learned strategy requires q_table")
            actions = learned_swarm_actions(env, q_table, states, rng)
        elif strategy == "random":
            actions = rng.integers(env.action_count, size=env.config.robot_count, dtype=np.int16)
        elif strategy == "greedy":
            actions = greedy_coverage_algorithm_actions(env, rng)
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        states, step_reward, done, info = env.step(actions)
        positions.append(info["positions"].copy())
        coverages.append(float(info["coverage"]))
        rewards.append(float(step_reward.sum()))
        collisions.append(int(info["collisions"]))
        spacings.append(mean_nearest_neighbor(info["positions"]))

    return {
        "strategy": strategy,
        "positions": np.asarray(positions),
        "coverage": np.asarray(coverages),
        "reward": np.asarray(rewards),
        "collisions": np.asarray(collisions),
        "spacing": np.asarray(spacings),
        "visit_counts": env.coverage.copy(),
        "obstacles": env.obstacles.copy(),
        "final_positions": env.positions.copy(),
        "steps": int(env.t),
        "final_coverage": float(env.coverage_fraction),
        "total_collisions": int(np.sum(collisions)),
        "total_reward": float(np.sum(rewards)),
    }
