"""CLI principal del proyecto: entrenamiento y evaluación PPO."""

from __future__ import annotations

import argparse

from config import CONFIG
from train_ppo import train_ppo_agent
from evaluate_and_plot import evaluate_and_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline PPO cobertura ELM")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["train", "evaluate", "all"],
        help="train: entrena | evaluate: evalúa | all: entrena y evalúa",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        train_ppo_agent(CONFIG)
    elif args.mode == "evaluate":
        evaluate_and_plot(CONFIG)
    elif args.mode == "all":
        train_ppo_agent(CONFIG)
        evaluate_and_plot(CONFIG)


if __name__ == "__main__":
    main()