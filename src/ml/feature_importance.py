"""ML feature importance — extracts and ranks feature importances from trained models.

Two strategies are supported:

1. **Model-native importance** — uses ``feature_importances_`` from tree-based
   estimators (Random Forest, XGBoost).
2. **Permutation importance** — uses
   :func:`sklearn.inspection.permutation_importance`, which works for any
   estimator and scoring function.

Both strategies return a ranked list of ``FeatureImportanceEntry`` named tuples
so callers can iterate, sort, or serialise without further processing.

No GenAI is used in this module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.ml.task_detector import TaskType
from src.ml.trainer import TrainResult

__all__ = [
    "FeatureImportanceEntry",
    "ImportanceResult",
    "compute_feature_importance",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureImportanceEntry:
    """A single (feature, importance_score) pair.

    Attributes
    ----------
    feature:
        Feature name.
    importance:
        Importance score (higher = more important).  For permutation
        importance this is the mean decrease in score when the feature is
        shuffled.
    std:
        Standard deviation of permutation importance across repeats (``0.0``
        for model-native importance which has no repeats).
    """

    feature: str
    importance: float
    std: float = 0.0


@dataclass
class ImportanceResult:
    """Container returned by :func:`compute_feature_importance`.

    Attributes
    ----------
    method:
        ``"model_native"`` or ``"permutation"``.
    model_name:
        Human-readable model name.
    ranked:
        Feature importances sorted descending by importance.
    raw_feature_names:
        Original feature column names (before any OHE expansion).
    """

    method: str
    model_name: str
    ranked: list[FeatureImportanceEntry]
    raw_feature_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "method": self.method,
            "model_name": self.model_name,
            "ranked": [
                {"feature": e.feature, "importance": e.importance, "std": e.std}
                for e in self.ranked
            ],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_feature_names_from_pipeline(pipeline: Any, raw_feature_names: list[str]) -> list[str]:
    """Extract transformed feature names from the pipeline's ColumnTransformer.

    Falls back to *raw_feature_names* if names cannot be determined.
    """
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            return raw_feature_names
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return raw_feature_names


def _get_terminal_estimator(pipeline: Any) -> Optional[Any]:
    """Return the last step of a Pipeline (the estimator), or None."""
    try:
        # The last step has the fitted model
        last_step = list(pipeline.steps)[-1][1]
        return last_step
    except Exception:
        return None


def _has_feature_importances(estimator: Any) -> bool:
    return hasattr(estimator, "feature_importances_")


def _model_native_importance(
    pipeline: Any,
    transformed_feature_names: list[str],
    model_name: str,
    raw_feature_names: list[str],
) -> ImportanceResult:
    """Extract model-native ``feature_importances_``."""
    estimator = _get_terminal_estimator(pipeline)
    importances: np.ndarray = estimator.feature_importances_

    # Pad / truncate in case transformer produced a different number of features
    n = min(len(importances), len(transformed_feature_names))
    names = transformed_feature_names[:n]
    scores = importances[:n]

    ranked = sorted(
        [FeatureImportanceEntry(feature=n_, importance=float(s)) for n_, s in zip(names, scores)],
        key=lambda e: e.importance,
        reverse=True,
    )
    return ImportanceResult(
        method="model_native",
        model_name=model_name,
        ranked=ranked,
        raw_feature_names=raw_feature_names,
    )


def _permutation_importance(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    transformed_feature_names: list[str],
    model_name: str,
    raw_feature_names: list[str],
    task_type: TaskType,
    n_repeats: int = 10,
) -> ImportanceResult:
    """Compute permutation importance on X (un-transformed; the pipeline handles it)."""
    scoring = "f1_weighted" if task_type == TaskType.CLASSIFICATION else "r2"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = permutation_importance(
                pipeline,
                X,
                y,
                n_repeats=n_repeats,
                random_state=42,
                scoring=scoring,
            )
            importances_mean = result.importances_mean
            importances_std = result.importances_std
        except Exception:
            # Graceful fallback: return uniform importances
            n = len(X.columns)
            importances_mean = np.ones(n) / n
            importances_std = np.zeros(n)

    # Pipeline permutes raw columns → use raw feature names
    col_names = list(X.columns)
    ranked = sorted(
        [
            FeatureImportanceEntry(
                feature=col_names[i],
                importance=float(importances_mean[i]),
                std=float(importances_std[i]),
            )
            for i in range(len(col_names))
        ],
        key=lambda e: e.importance,
        reverse=True,
    )
    return ImportanceResult(
        method="permutation",
        model_name=model_name,
        ranked=ranked,
        raw_feature_names=raw_feature_names,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_feature_importance(
    result: TrainResult,
    method: str = "auto",
    n_repeats: int = 10,
) -> ImportanceResult:
    """Compute feature importance for a trained model.

    Parameters
    ----------
    result:
        A :class:`~src.ml.trainer.TrainResult` from :func:`~src.ml.trainer.train_models`.
    method:
        ``"auto"`` (use model-native if available, else permutation),
        ``"model_native"``, or ``"permutation"``.
    n_repeats:
        Number of permutation repeats (only used when method is
        ``"permutation"`` or ``"auto"`` with fallback).

    Returns
    -------
    ImportanceResult
    """
    if result.task_type == TaskType.CLUSTERING:
        # For clustering, return uniform importance (no supervised signal)
        n = len(result.feature_names)
        uniform = 1.0 / n if n > 0 else 0.0
        ranked = [
            FeatureImportanceEntry(feature=f, importance=uniform)
            for f in result.feature_names
        ]
        return ImportanceResult(
            method="uniform",
            model_name=result.model_name,
            ranked=ranked,
            raw_feature_names=result.feature_names,
        )

    pipeline = result.pipeline
    raw_feature_names = result.feature_names
    transformed_names = _get_feature_names_from_pipeline(pipeline, raw_feature_names)
    estimator = _get_terminal_estimator(pipeline)

    # Decide method
    use_native = _has_feature_importances(estimator)
    if method == "model_native":
        if not use_native:
            raise ValueError(
                f"Model '{result.model_name}' does not expose feature_importances_. "
                "Use method='permutation' or 'auto'."
            )
        return _model_native_importance(
            pipeline, transformed_names, result.model_name, raw_feature_names
        )

    if method == "permutation":
        use_native = False

    if method == "auto" and use_native:
        return _model_native_importance(
            pipeline, transformed_names, result.model_name, raw_feature_names
        )

    # Permutation importance — needs X and y
    X = result.X_test if len(result.X_test) > 0 else result.X_train
    y = result.y_test if result.y_test is not None and len(result.y_test) > 0 else result.y_train
    if y is None:
        ranked = [
            FeatureImportanceEntry(feature=f, importance=0.0)
            for f in raw_feature_names
        ]
        return ImportanceResult(
            method="permutation",
            model_name=result.model_name,
            ranked=ranked,
            raw_feature_names=raw_feature_names,
        )

    return _permutation_importance(
        pipeline,
        X,
        y,
        transformed_names,
        result.model_name,
        raw_feature_names,
        result.task_type,
        n_repeats=n_repeats,
    )
