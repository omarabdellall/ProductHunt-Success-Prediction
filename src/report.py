from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fpdf import FPDF

from config import ensure_directories, load_settings


REPORT_NAME = "ProductHunt_Launch_Success_Report.pdf"


def _section_title(pdf: FPDF, title: str) -> None:
    pdf.set_font("Helvetica", "B", 13)
    pdf.ln(4)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)


def _paragraph(pdf: FPDF, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 6, text)
    pdf.ln(1)


def _table_from_dataframe(pdf: FPDF, frame: pd.DataFrame, title: str) -> None:
    _section_title(pdf, title)
    if frame.empty:
        raise ValueError(f"Cannot render empty table: {title}")
    rounded = frame.copy()
    for column in rounded.columns:
        if pd.api.types.is_float_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda value: f"{value:.3f}")
    _paragraph(pdf, rounded.to_string(index=False))


def _add_figure(pdf: FPDF, path: Path, caption: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing figure file: {path}")
    _section_title(pdf, caption)
    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.image(str(path), x=pdf.l_margin, w=page_width)
    pdf.ln(2)


def _format_best_rows(classification: pd.DataFrame, regression: pd.DataFrame) -> dict[str, Any]:
    best_class = classification.sort_values("roc_auc", ascending=False).iloc[0]
    best_reg = regression.sort_values("r2_log", ascending=False).iloc[0]
    return {
        "best_class_model": str(best_class["model"]),
        "best_class_auc": float(best_class["roc_auc"]),
        "best_reg_model": str(best_reg["model"]),
        "best_reg_r2_log": float(best_reg["r2_log"]),
        "best_reg_r2": float(best_reg["r2"]),
        "best_reg_mae": float(best_reg["mae"]),
    }


def generate_report() -> Path:
    settings = load_settings()
    ensure_directories(settings)

    processed = pd.read_csv(settings.processed_data_path)
    classification = pd.read_csv(settings.results_dir / "classification_metrics.csv")
    regression = pd.read_csv(settings.results_dir / "regression_metrics.csv")
    with (settings.features_dir / "feature_info.json").open("r", encoding="utf-8") as handle:
        feature_info = json.load(handle)
    with (settings.results_dir / "training_summary.json").open("r", encoding="utf-8") as handle:
        training_summary = json.load(handle)
    models_payload = joblib.load(settings.models_path)

    created_at = pd.to_datetime(processed["created_at"], utc=True)
    unique_days = int(created_at.dt.date.nunique())
    years = sorted(created_at.dt.year.unique().tolist())
    best_rows = _format_best_rows(classification, regression)

    output_path = settings.project_root / REPORT_NAME
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Product Hunt Launch Success Prediction", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Final Project Report", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    _section_title(pdf, "Introduction")
    _paragraph(
        pdf,
        "This project predicts Product Hunt launch success with two tasks: "
        "classification of top-20 outcomes and regression of vote counts. "
        "The approach combines structured metadata with text representations from "
        "TF-IDF and Sentence-BERT features, then evaluates linear and tree-based models.",
    )
    _paragraph(
        pdf,
        "Key challenges were API throttling and heavy-tailed vote distributions. "
        "To address these, collection used bounded windows and the regression task "
        "was trained on log-transformed targets.",
    )

    _section_title(pdf, "Data")
    _paragraph(
        pdf,
        "Source: Product Hunt GraphQL API v2. "
        f"Processed samples: {len(processed)} posts spanning years {years}. "
        f"Time range: {created_at.min()} to {created_at.max()}. "
        f"Unique launch days represented: {unique_days}. "
        f"Top-20 class prevalence: {processed['is_top20'].mean():.3f}.",
    )

    _section_title(pdf, "Methodology")
    _paragraph(
        pdf,
        "Feature sets: metadata only, metadata + TF-IDF, metadata + SBERT. "
        f"Metadata columns: {', '.join(feature_info['metadata_columns'])}. "
        f"TF-IDF features: {feature_info['tfidf_feature_count']}.",
    )
    _paragraph(
        pdf,
        "Classification uses stratified 5-fold CV with class-weighted logistic regression "
        "and regularized XGBoost variants. Regression uses 5-fold CV on log1p(votes) with "
        "LinearRegression, Ridge, XGBoost (square error), and XGBoost pseudo-Huber variants.",
    )

    _section_title(pdf, "Implementation Details")
    _paragraph(
        pdf,
        f"Train/Test split sizes: {training_summary['train_size']}/{training_summary['test_size']}. "
        "Search spaces include model-specific depth, learning-rate, regularization, and estimator-count grids.",
    )
    _paragraph(
        pdf,
        "Best hyperparameters discovered per model:",
    )
    for model_name, result in models_payload["classification"]["predictions"].items():
        if "best_params" in result:
            _paragraph(pdf, f"- {model_name}: {result['best_params']}")
    for model_name, result in models_payload["regression"]["predictions"].items():
        if "best_params" in result:
            _paragraph(pdf, f"- {model_name}: {result['best_params']}")

    _table_from_dataframe(pdf, classification, "Results: Classification Metrics")
    _table_from_dataframe(pdf, regression, "Results: Regression Metrics")

    _section_title(pdf, "Results Summary")
    _paragraph(
        pdf,
        f"Best classification model: {best_rows['best_class_model']} "
        f"(ROC-AUC={best_rows['best_class_auc']:.3f}). "
        f"Best regression model: {best_rows['best_reg_model']} "
        f"(R2_log={best_rows['best_reg_r2_log']:.3f}, "
        f"R2_original={best_rows['best_reg_r2']:.3f}, "
        f"MAE={best_rows['best_reg_mae']:.3f}).",
    )
    _paragraph(
        pdf,
        "The primary regression success criterion is log-scale R-squared because models are trained "
        "on log targets to handle skewed counts. Original-scale R-squared is reported for transparency.",
    )

    _add_figure(pdf, settings.results_dir / "classification_roc_auc.png", "Figure: Classification ROC-AUC")
    _add_figure(pdf, settings.results_dir / "roc_curves.png", "Figure: ROC Curves")
    _add_figure(pdf, settings.results_dir / "confusion_matrix_logreg_metadata_sbert.png", "Figure: Best Confusion Matrix")
    _add_figure(pdf, settings.results_dir / "regression_r2.png", "Figure: Regression Log-Scale R-squared")
    _add_figure(
        pdf,
        settings.results_dir / "feature_importance_xgb_classifier.png",
        "Figure: XGBoost Classifier Feature Importance",
    )
    _add_figure(
        pdf,
        settings.results_dir / "feature_importance_xgb_regressor.png",
        "Figure: XGBoost Regressor Feature Importance",
    )

    pdf.output(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final Product Hunt project PDF report.")
    parser.parse_args()
    output_path = generate_report()
    print(f"Report generated at: {output_path}")


if __name__ == "__main__":
    main()
