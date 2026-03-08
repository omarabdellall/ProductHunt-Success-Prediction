from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from xgboost import XGBClassifier, XGBRegressor

from config import ensure_directories, load_settings


@dataclass
class SplitData:
    x_train: Any
    x_test: Any
    y_train: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray


def _load_features() -> dict[str, Any]:
    settings = load_settings()
    return {
        "metadata_only": np.load(settings.features_dir / "metadata_only.npy"),
        "metadata_tfidf": sparse.load_npz(settings.features_dir / "metadata_tfidf.npz"),
        "metadata_sbert": np.load(settings.features_dir / "metadata_sbert.npy"),
        "y_class": np.load(settings.features_dir / "y_class.npy"),
        "y_reg": np.load(settings.features_dir / "y_reg.npy"),
        "y_reg_log": np.load(settings.features_dir / "y_reg_log.npy"),
    }


def _build_shared_split(y_class: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(y_class))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=random_state,
        stratify=y_class,
    )
    return np.sort(train_idx), np.sort(test_idx)


def _subset_matrix(matrix: Any, indices: np.ndarray) -> Any:
    if sparse.issparse(matrix):
        return matrix[indices]
    return matrix[indices, :]


def _build_split(matrix: Any, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> SplitData:
    return SplitData(
        x_train=_subset_matrix(matrix, train_idx),
        x_test=_subset_matrix(matrix, test_idx),
        y_train=y[train_idx],
        y_test=y[test_idx],
        train_indices=train_idx,
        test_indices=test_idx,
    )


def _clip_log_predictions(y_pred_log: np.ndarray, y_train_log: np.ndarray) -> np.ndarray:
    max_train_log = float(np.max(y_train_log))
    return np.clip(y_pred_log, a_min=0.0, a_max=max_train_log + 1.0)


def _fit_classification_models(
    features: dict[str, Any], train_idx: np.ndarray, test_idx: np.ndarray, random_state: int
) -> dict[str, Any]:
    y_class = features["y_class"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    models: dict[str, Any] = {}
    predictions: dict[str, Any] = {}

    for feature_name in ("metadata_only", "metadata_tfidf", "metadata_sbert"):
        split = _build_split(features[feature_name], y_class, train_idx, test_idx)
        search = GridSearchCV(
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=1500,
                solver="liblinear",
                random_state=random_state,
            ),
            param_grid={"C": [0.1, 1.0, 5.0, 10.0]},
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(split.x_train, split.y_train)
        best_model = search.best_estimator_
        probas = best_model.predict_proba(split.x_test)[:, 1]
        labels = best_model.predict(split.x_test)

        model_name = f"logreg_{feature_name}"
        models[model_name] = best_model
        predictions[model_name] = {
            "y_true": split.y_test,
            "y_pred": labels,
            "y_score": probas,
            "best_params": search.best_params_,
        }

    for feature_name in ("metadata_tfidf", "metadata_sbert"):
        split = _build_split(features[feature_name], y_class, train_idx, test_idx)
        pos_weight = (split.y_train == 0).sum() / max((split.y_train == 1).sum(), 1)
        xgb_search = GridSearchCV(
            estimator=XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                scale_pos_weight=float(pos_weight),
            ),
            param_grid={
                "max_depth": [4, 6],
                "n_estimators": [150, 250],
                "learning_rate": [0.05, 0.1],
                "min_child_weight": [1, 3],
                "reg_alpha": [0.0, 0.5],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        )
        xgb_search.fit(split.x_train, split.y_train)
        xgb_cls = xgb_search.best_estimator_

        model_name = f"xgb_classifier_{feature_name}"
        models[model_name] = xgb_cls
        predictions[model_name] = {
            "y_true": split.y_test,
            "y_pred": xgb_cls.predict(split.x_test),
            "y_score": xgb_cls.predict_proba(split.x_test)[:, 1],
            "best_params": xgb_search.best_params_,
        }
    return {"models": models, "predictions": predictions}


def _fit_regression_models(
    features: dict[str, Any], train_idx: np.ndarray, test_idx: np.ndarray, random_state: int
) -> dict[str, Any]:
    y_reg = features["y_reg"]
    y_reg_log = features["y_reg_log"]
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    models: dict[str, Any] = {}
    predictions: dict[str, Any] = {}

    split_meta_log = _build_split(features["metadata_only"], y_reg_log, train_idx, test_idx)
    linear = LinearRegression()
    linear.fit(split_meta_log.x_train, split_meta_log.y_train)
    linear_pred_log = _clip_log_predictions(
        linear.predict(split_meta_log.x_test),
        split_meta_log.y_train,
    )
    models["linear_metadata_only"] = linear
    predictions["linear_metadata_only"] = {
        "y_true_log": split_meta_log.y_test,
        "y_pred_log": linear_pred_log,
        "y_true": y_reg[test_idx],
        "y_pred": np.clip(np.expm1(linear_pred_log), a_min=0.0, a_max=None),
    }

    for feature_name in ("metadata_tfidf", "metadata_sbert"):
        split = _build_split(features[feature_name], y_reg_log, train_idx, test_idx)
        ridge_search = GridSearchCV(
            estimator=Ridge(random_state=random_state),
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]},
            scoring="r2",
            cv=cv,
            n_jobs=-1,
        )
        ridge_search.fit(split.x_train, split.y_train)
        ridge = ridge_search.best_estimator_
        ridge_pred_log = _clip_log_predictions(
            ridge.predict(split.x_test),
            split.y_train,
        )
        model_name = f"ridge_{feature_name}"
        models[model_name] = ridge
        predictions[model_name] = {
            "y_true_log": split.y_test,
            "y_pred_log": ridge_pred_log,
            "y_true": y_reg[test_idx],
            "y_pred": np.clip(np.expm1(ridge_pred_log), a_min=0.0, a_max=None),
            "best_params": ridge_search.best_params_,
        }

    split_sbert = _build_split(features["metadata_sbert"], y_reg_log, train_idx, test_idx)
    xgb_variants = (
        (
            "xgb_regressor_metadata_sbert",
            "reg:squarederror",
            {
                "max_depth": [3, 5, 7],
                "n_estimators": [200, 400],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
        ),
        (
            "xgb_huber_regressor_metadata_sbert",
            "reg:pseudohubererror",
            {
                "max_depth": [2, 3, 4],
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.03],
                "subsample": [0.7, 0.9],
                "min_child_weight": [5, 10],
                "reg_lambda": [5.0, 10.0],
            },
        ),
    )
    for model_name, objective, param_grid in xgb_variants:
        xgb_search = GridSearchCV(
            estimator=XGBRegressor(
                objective=objective,
                random_state=random_state,
            ),
            param_grid=param_grid,
            scoring="r2",
            cv=cv,
            n_jobs=-1,
        )
        xgb_search.fit(split_sbert.x_train, split_sbert.y_train)
        xgb_reg = xgb_search.best_estimator_
        xgb_pred_log = _clip_log_predictions(
            xgb_reg.predict(split_sbert.x_test),
            split_sbert.y_train,
        )
        models[model_name] = xgb_reg
        predictions[model_name] = {
            "y_true_log": split_sbert.y_test,
            "y_pred_log": xgb_pred_log,
            "y_true": y_reg[test_idx],
            "y_pred": np.clip(np.expm1(xgb_pred_log), a_min=0.0, a_max=None),
            "best_params": xgb_search.best_params_,
        }
    return {"models": models, "predictions": predictions}


def train_models() -> dict[str, Any]:
    settings = load_settings()
    ensure_directories(settings)

    features = _load_features()
    train_idx, test_idx = _build_shared_split(features["y_class"], settings.random_state)

    classification = _fit_classification_models(features, train_idx, test_idx, settings.random_state)
    regression = _fit_regression_models(features, train_idx, test_idx, settings.random_state)

    payload = {
        "train_indices": train_idx,
        "test_indices": test_idx,
        "classification": classification,
        "regression": regression,
    }
    joblib.dump(payload, settings.models_path)

    summary = {
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "classification_models": list(classification["models"].keys()),
        "regression_models": list(regression["models"].keys()),
    }
    with (settings.results_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Product Hunt success prediction models.")
    parser.parse_args()
    summary = train_models()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
