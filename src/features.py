from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from config import ensure_directories, load_settings


METADATA_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "maker_count",
    "media_count",
    "tagline_length",
    "description_length",
    "topic_count",
]


def _save_sparse_matrix(path: Path, matrix: sparse.csr_matrix) -> None:
    sparse.save_npz(path, matrix)


def _save_dense_matrix(path: Path, matrix: np.ndarray) -> None:
    np.save(path, matrix)


def build_features() -> dict[str, Any]:
    settings = load_settings()
    ensure_directories(settings)

    frame = pd.read_csv(settings.processed_data_path)
    text = (frame["tagline"].astype(str) + " " + frame["description"].astype(str)).str.strip()

    metadata_matrix = frame[METADATA_COLUMNS].to_numpy(dtype=float)
    scaler = StandardScaler()
    metadata_scaled = scaler.fit_transform(metadata_matrix)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf_vectorizer.fit_transform(text)

    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    sbert_matrix = sbert_model.encode(text.tolist(), batch_size=64, show_progress_bar=True)
    sbert_matrix = np.asarray(sbert_matrix, dtype=np.float32)

    metadata_sparse = sparse.csr_matrix(metadata_scaled)
    metadata_tfidf = sparse.hstack([metadata_sparse, tfidf_matrix], format="csr")
    metadata_sbert = np.hstack([metadata_scaled.astype(np.float32), sbert_matrix])

    labels_class = frame["is_top20"].to_numpy(dtype=int)
    labels_reg = frame["votes_count"].to_numpy(dtype=float)

    _save_dense_matrix(settings.features_dir / "metadata_only.npy", metadata_scaled.astype(np.float32))
    _save_sparse_matrix(settings.features_dir / "metadata_tfidf.npz", metadata_tfidf)
    _save_dense_matrix(settings.features_dir / "metadata_sbert.npy", metadata_sbert.astype(np.float32))
    _save_dense_matrix(settings.features_dir / "y_class.npy", labels_class)
    _save_dense_matrix(settings.features_dir / "y_reg.npy", labels_reg)

    feature_info = {
        "metadata_columns": METADATA_COLUMNS,
        "tfidf_feature_count": int(len(tfidf_vectorizer.get_feature_names_out())),
        "tfidf_feature_preview": tfidf_vectorizer.get_feature_names_out().tolist()[:20],
        "metadata_only_shape": list(metadata_scaled.shape),
        "metadata_tfidf_shape": list(metadata_tfidf.shape),
        "metadata_sbert_shape": list(metadata_sbert.shape),
    }

    with (settings.features_dir / "feature_info.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_info, handle, indent=2)

    joblib.dump(scaler, settings.features_dir / "metadata_scaler.joblib")
    joblib.dump(tfidf_vectorizer, settings.features_dir / "tfidf_vectorizer.joblib")

    return feature_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature matrices for Product Hunt modeling.")
    parser.parse_args()
    info = build_features()
    print("Saved feature matrices.")
    print(
        json.dumps(
            {
                "metadata_only_shape": info["metadata_only_shape"],
                "metadata_tfidf_shape": info["metadata_tfidf_shape"],
                "metadata_sbert_shape": info["metadata_sbert_shape"],
                "tfidf_feature_count": info["tfidf_feature_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
