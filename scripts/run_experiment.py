#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swarm_rl.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the swarm RL coverage controller.")
    parser.add_argument("--episodes", type=int, default=180, help="number of Q-learning episodes")
    parser.add_argument("--seed", type=int, default=17, help="environment seed")
    args = parser.parse_args()

    summary = run_experiment(PROJECT_ROOT, episodes=args.episodes, seed=args.seed)
    print("Experiment complete.")
    print(f"RL final coverage: {summary['evaluation']['learned']['final_coverage_percent']}%")
    print(f"Figures saved in: {PROJECT_ROOT / 'results' / 'figures'}")


if __name__ == "__main__":
    main()
