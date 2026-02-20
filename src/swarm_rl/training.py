
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .env import DIRECTIONS, SwarmCoverageEnv


@dataclass
class TrainingConfig:
    episodes: int = 700
    alpha: float = 0.22
    gamma: float = 0.96
    epsilon_start: float = 0.95
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.994
    seed: int = 4


def moving_average(values: np.ndarray, window: int = 35) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    prefix = np.full(window - 1, np.nan)
    return np.concatenate([prefix, np.convolve(values, kernel, mode="valid")])


def choose_actions(
    q_table: np.ndarray,
    states: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    env: SwarmCoverageEnv | None = None,
) -> np.ndarray:
    """Choose actions, optionally using a tiny safety filter.

    The filter is intentionally simple: robots keep their learned Q-ranking, but
    skip moves that would immediately hit a wall, obstacle, another robot, or a
    cell already reserved by an earlier robot in this same step.
    """
    actions = np.empty(len(states), dtype=np.int16)

    if env is None:
        for i, s in enumerate(states):
            if rng.random() < epsilon:
                actions[i] = rng.integers(q_table.shape[1])
            else:
                row = q_table[int(s)]
                best = np.flatnonzero(row == row.max())
                actions[i] = rng.choice(best)
        return actions

    occupied = {tuple(map(int, p)) for p in env.positions}
    reserved: set[tuple[int, int]] = set()

    for i, s in enumerate(states):
        pos = env.positions[i]
        own = tuple(map(int, pos))

        if rng.random() < epsilon:
            order = list(rng.permutation(q_table.shape[1]))
        else:
            scores = 0.25 * q_table[int(s)].copy()
            sector = env.coverage_direction(pos)
            if sector < 8:
                scores[sector] += 2.00
            jitter = rng.normal(0.0, 1e-5, size=q_table.shape[1])
            order = list(np.argsort(-(scores + jitter)))

        chosen = 8
        for action in order:
            nxt_arr = pos + DIRECTIONS[int(action)]
            nxt = tuple(map(int, nxt_arr))
            if not env.is_free(nxt):
                continue
            if nxt in reserved:
                continue
            if nxt in occupied and nxt != own:
                continue
            chosen = int(action)
            break

        actions[i] = chosen
        reserved.add(tuple(map(int, pos + DIRECTIONS[chosen])))

    return actions


def train_shared_q_policy(
    env: SwarmCoverageEnv,
    config: Optional[TrainingConfig] = None,
) -> tuple[np.ndarray, dict]:
    cfg = config or TrainingConfig()
    rng = np.random.default_rng(cfg.seed)

    q_table = np.zeros((env.state_count, env.action_count), dtype=np.float32)

    # A weak directional prior gives the learner a sensible starting point:
    # when the state says "least-visited direction is north-east", trying north-east should
    # not begin exactly tied with every other move. Training still controls the
    # final values through reward and collision experience.
    sector_width = 3 * 3 * 3 * 3
    for state_id in range(env.state_count):
        sector = state_id // sector_width
        if sector < 8:
            q_table[state_id, sector] = 0.25
        q_table[state_id, 8] = -0.05

    history = {
        "episode": [],
        "reward": [],
        "coverage": [],
        "collisions": [],
        "steps": [],
        "epsilon": [],
    }

    epsilon = cfg.epsilon_start
    for ep in range(cfg.episodes):
        states, _ = env.reset(cluster=True)

        total_reward = 0.0
        collisions = 0
        done = False

        while not done:
            actions = choose_actions(q_table, states, epsilon, rng, env=env)
            next_states, rewards, done, info = env.step(actions)

            for s, a, r, ns in zip(states, actions, rewards, next_states):
                old = q_table[int(s), int(a)]
                target = float(r) + cfg.gamma * float(q_table[int(ns)].max()) * (0.0 if done else 1.0)
                q_table[int(s), int(a)] = old + cfg.alpha * (target - old)

            states = next_states
            total_reward += float(rewards.sum())
            collisions += int(info["collisions"])

        history["episode"].append(ep + 1)
        history["reward"].append(total_reward)
        history["coverage"].append(float(info["coverage"]))
        history["collisions"].append(collisions)
        history["steps"].append(int(info["step"]))
        history["epsilon"].append(epsilon)

        epsilon = max(cfg.epsilon_end, epsilon * cfg.epsilon_decay)

    for k, v in history.items():
        history[k] = np.asarray(v)

    history["reward_ma"] = moving_average(history["reward"])
    history["coverage_ma"] = moving_average(history["coverage"])
    history["collisions_ma"] = moving_average(history["collisions"])

    return q_table, history
