from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from config import ensure_directories, load_settings


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _plot_classification_metrics(metrics_df: pd.DataFrame, results_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=metrics_df, x="model", y="roc_auc")
    plt.xticks(rotation=35, ha="right")
    plt.title("Classification ROC-AUC by Model")
    plt.tight_layout()
    plt.savefig(results_dir / "classification_roc_auc.png", dpi=160)
    plt.close()


def _plot_regression_metrics(metrics_df: pd.DataFrame, results_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=metrics_df, x="model", y="r2")
    plt.xticks(rotation=35, ha="right")
    plt.title("Regression R-squared by Model")
    plt.tight_layout()
    plt.savefig(results_dir / "regression_r2.png", dpi=160)
    plt.close()


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, results_dir: Path) -> None:
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.ax_.set_title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(results_dir / f"confusion_matrix_{model_name}.png", dpi=160)
    plt.close()


def _build_metadata_sbert_feature_labels(feature_info: dict[str, Any]) -> list[str]:
    metadata_columns = feature_info["metadata_columns"]
    metadata_count = len(metadata_columns)
    total_count = int(feature_info["metadata_sbert_shape"][1])
    sbert_count = total_count - metadata_count
    if sbert_count < 0:
        raise ValueError("Invalid feature_info: metadata_sbert_shape is smaller than metadata column count")
    sbert_labels = [f"sbert_dim_{idx}" for idx in range(sbert_count)]
    return [*metadata_columns, *sbert_labels]


def _plot_named_feature_importance(
    model: Any, feature_labels: list[str], output_path: Path, title: str
) -> None:
    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(importances) != len(feature_labels):
        raise ValueError("Feature importance length does not match feature label count")

    top_k = min(20, len(importances))
    top_indices = np.argsort(importances)[::-1][:top_k]
    top_values = importances[top_indices]
    top_labels = [feature_labels[idx] for idx in top_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_values, y=top_labels, orient="h")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _daily_mean_baseline(processed: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, float]:
    frame = processed.copy()
    frame["launch_date"] = pd.to_datetime(frame["created_at"], utc=True).dt.date
    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx]

    daily_means = train.groupby("launch_date")["votes_count"].mean()
    global_mean = float(train["votes_count"].mean())
    baseline_pred = test["launch_date"].map(daily_means).fillna(global_mean).to_numpy(dtype=float)
    y_true = test["votes_count"].to_numpy(dtype=float)

    return {
        "r2": float(r2_score(y_true, baseline_pred)),
        "mae": float(mean_absolute_error(y_true, baseline_pred)),
    }


def evaluate_models() -> dict[str, Any]:
    settings = load_settings()
    ensure_directories(settings)

    payload = joblib.load(settings.models_path)
    processed = pd.read_csv(settings.processed_data_path)
    with (settings.features_dir / "feature_info.json").open("r", encoding="utf-8") as handle:
        feature_info = json.load(handle)
    metadata_sbert_labels = _build_metadata_sbert_feature_labels(feature_info)

    class_rows: list[dict[str, Any]] = []
    reg_rows: list[dict[str, Any]] = []

    for model_name, result in payload["classification"]["predictions"].items():
        y_true = np.asarray(result["y_true"])
        y_pred = np.asarray(result["y_pred"])
        y_score = np.asarray(result["y_score"])
        class_rows.append(
            {
                "model": model_name,
                "roc_auc": float(roc_auc_score(y_true, y_score)),
                "f1": float(f1_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred)),
                "recall": float(recall_score(y_true, y_pred)),
            }
        )

    for model_name, result in payload["regression"]["predictions"].items():
        y_true = np.asarray(result["y_true"], dtype=float)
        y_pred = np.asarray(result["y_pred"], dtype=float)
        reg_rows.append(
            {
                "model": model_name,
                "r2": float(r2_score(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
            }
        )

    daily_baseline = _daily_mean_baseline(
        processed=processed,
        train_idx=np.asarray(payload["train_indices"]),
        test_idx=np.asarray(payload["test_indices"]),
    )
    reg_rows.append({"model": "daily_mean_baseline", **daily_baseline})

    class_df = pd.DataFrame(class_rows).sort_values("roc_auc", ascending=False)
    reg_df = pd.DataFrame(reg_rows).sort_values("r2", ascending=False)

    class_df.to_csv(settings.results_dir / "classification_metrics.csv", index=False)
    reg_df.to_csv(settings.results_dir / "regression_metrics.csv", index=False)
    _plot_classification_metrics(class_df, settings.results_dir)
    _plot_regression_metrics(reg_df, settings.results_dir)

    best_class = class_df.iloc[0]["model"]
    best_class_result = payload["classification"]["predictions"][best_class]
    _plot_confusion_matrix(
        np.asarray(best_class_result["y_true"]),
        np.asarray(best_class_result["y_pred"]),
        best_class,
        settings.results_dir,
    )

    xgb_cls = payload["classification"]["models"]["xgb_classifier_metadata_sbert"]
    xgb_reg = payload["regression"]["models"]["xgb_regressor_metadata_sbert"]
    _plot_named_feature_importance(
        xgb_cls,
        metadata_sbert_labels,
        settings.results_dir / "feature_importance_xgb_classifier.png",
        "Top Feature Importances (XGBoost Classifier)",
    )
    _plot_named_feature_importance(
        xgb_reg,
        metadata_sbert_labels,
        settings.results_dir / "feature_importance_xgb_regressor.png",
        "Top Feature Importances (XGBoost Regressor)",
    )

    metrics = {
        "classification": class_df.to_dict(orient="records"),
        "regression": reg_df.to_dict(orient="records"),
        "targets": {"roc_auc": 0.70, "r2": 0.30},
    }
    _save_json(settings.metrics_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Product Hunt prediction models.")
    parser.parse_args()
    metrics = evaluate_models()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
