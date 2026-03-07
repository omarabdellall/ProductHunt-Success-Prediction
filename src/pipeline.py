from __future__ import annotations

import argparse

from collect import collect_posts
from evaluate import evaluate_models
from features import build_features
from preprocess import preprocess
from train import train_models


STAGES = ("collect", "preprocess", "features", "train", "evaluate")


def run_stage(stage: str) -> None:
    if stage == "collect":
        collect_posts()
    elif stage == "preprocess":
        preprocess()
    elif stage == "features":
        build_features()
    elif stage == "train":
        train_models()
    elif stage == "evaluate":
        evaluate_models()
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def run_pipeline() -> None:
    for stage in STAGES:
        print(f"\n=== Running stage: {stage} ===")
        run_stage(stage)
    print("\nPipeline completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Product Hunt prediction pipeline.")
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default=None,
        help="Run only one stage; omit to run all stages.",
    )
    args = parser.parse_args()

    if args.stage:
        run_stage(args.stage)
        print(f"Stage '{args.stage}' completed successfully.")
    else:
        run_pipeline()


if __name__ == "__main__":
    main()
