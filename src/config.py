from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    features_dir: Path
    results_dir: Path
    raw_data_path: Path
    processed_data_path: Path
    models_path: Path
    metrics_path: Path
    product_hunt_token: str
    graphql_endpoint: str
    posted_after: str
    posted_before: str
    page_size: int
    max_posts: int
    random_state: int


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    required = (
        "PH_TOKEN",
        "POSTED_AFTER",
        "POSTED_BEFORE",
        "PAGE_SIZE",
        "MAX_POSTS",
        "RANDOM_STATE",
    )
    missing = [key for key in required if not os.getenv(key, "").strip()]
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"Missing required .env keys: {missing_keys}")

    token = os.environ["PH_TOKEN"].strip()
    posted_after = os.environ["POSTED_AFTER"].strip()
    posted_before = os.environ["POSTED_BEFORE"].strip()
    page_size = int(os.environ["PAGE_SIZE"])
    max_posts = int(os.environ["MAX_POSTS"])
    random_state = int(os.environ["RANDOM_STATE"])

    data_dir = project_root / "data"
    features_dir = data_dir / "features"
    results_dir = project_root / "results"

    return Settings(
        project_root=project_root,
        data_dir=data_dir,
        features_dir=features_dir,
        results_dir=results_dir,
        raw_data_path=data_dir / "raw_posts.csv",
        processed_data_path=data_dir / "processed_posts.csv",
        models_path=results_dir / "models.joblib",
        metrics_path=results_dir / "metrics.json",
        product_hunt_token=token,
        graphql_endpoint="https://api.producthunt.com/v2/api/graphql",
        posted_after=posted_after,
        posted_before=posted_before,
        page_size=page_size,
        max_posts=max_posts,
        random_state=random_state,
    )


def ensure_directories(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.features_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
