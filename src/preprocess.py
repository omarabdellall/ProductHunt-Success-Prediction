from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import ensure_directories, load_settings


REQUIRED_COLUMNS = {
    "post_id",
    "name",
    "tagline",
    "description",
    "votes_count",
    "comments_count",
    "created_at",
    "topics",
    "topic_count",
    "maker_count",
    "media_count",
}


def _validate_columns(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {sorted(missing)}")


def _build_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame["launch_date"] = frame["created_at"].dt.date
    p80_by_day = frame.groupby("launch_date")["votes_count"].transform(lambda series: series.quantile(0.8))
    frame["is_top20"] = (frame["votes_count"] >= p80_by_day).astype(int)
    return frame


def _save_plot(path: Path, title: str) -> None:
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _run_eda(frame: pd.DataFrame, results_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(frame["votes_count"], bins=60, kde=False)
    _save_plot(results_dir / "eda_upvote_distribution.png", "Upvote Distribution")

    plt.figure(figsize=(6, 4))
    sns.countplot(x="is_top20", data=frame)
    _save_plot(results_dir / "eda_label_balance.png", "Top-20% Label Balance")

    heatmap_frame = (
        frame.groupby(["day_of_week", "hour"])["votes_count"].mean().reset_index().pivot(
            index="day_of_week", columns="hour", values="votes_count"
        )
    )
    plt.figure(figsize=(14, 5))
    sns.heatmap(heatmap_frame, cmap="viridis")
    _save_plot(results_dir / "eda_upvotes_hour_day_heatmap.png", "Mean Upvotes by Day/Hour")

    expanded_topics = (
        frame.assign(topic=frame["topics"].str.split("|"))
        .explode("topic")
        .query("topic != ''")
        .groupby("topic")
        .size()
        .sort_values(ascending=False)
        .head(20)
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(x=expanded_topics.values, y=expanded_topics.index, orient="h")
    _save_plot(results_dir / "eda_top_topics.png", "Top Topics Frequency")

    plt.figure(figsize=(10, 6))
    sns.histplot(frame["tagline_length"], bins=50, color="steelblue", alpha=0.7, label="Tagline")
    sns.histplot(frame["description_length"], bins=50, color="darkorange", alpha=0.5, label="Description")
    plt.legend()
    _save_plot(results_dir / "eda_text_length_distributions.png", "Tagline/Description Length Distributions")


def preprocess() -> pd.DataFrame:
    settings = load_settings()
    ensure_directories(settings)

    frame = pd.read_csv(settings.raw_data_path)
    _validate_columns(frame)

    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    frame["description"] = frame["description"].fillna("")
    frame["tagline"] = frame["tagline"].fillna("")

    frame["hour"] = frame["created_at"].dt.hour
    frame["day_of_week"] = frame["created_at"].dt.dayofweek
    frame["month"] = frame["created_at"].dt.month
    frame["year"] = frame["created_at"].dt.year
    frame["tagline_length"] = frame["tagline"].str.len()
    frame["description_length"] = frame["description"].str.len()

    frame = _build_labels(frame)
    _run_eda(frame, settings.results_dir)

    frame.to_csv(settings.processed_data_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Product Hunt data and generate EDA.")
    parser.parse_args()
    frame = preprocess()
    print(f"Saved {len(frame)} rows to data/processed_posts.csv")


if __name__ == "__main__":
    main()
