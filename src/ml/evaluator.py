"""ML evaluator — computes metrics for trained models.

All metrics are computed deterministically from the fitted pipeline and the
held-out test split produced by :mod:`src.ml.trainer`.  No GenAI is used here.

The evaluator returns a :class:`ModelMetrics` dataclass.  The values are plain
Python floats / numpy arrays — safe to serialise and inject into GenAI prompts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

from src.ml.task_detector import TaskType
from src.ml.trainer import TrainResult

__all__ = [
    "ModelMetrics",
    "evaluate_model",
]


# ---------------------------------------------------------------------------
# ModelMetrics dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelMetrics:
    """Evaluation results for a single trained model.

    Attributes shared by all task types
    ------------------------------------
    task_type:
        The :class:`~src.ml.task_detector.TaskType` of this evaluation.
    model_name:
        Human-readable model name.

    Classification-specific (``None`` for other task types)
    --------------------------------------------------------
    accuracy:      Overall accuracy on the test set.
    precision:     Weighted-average precision.
    recall:        Weighted-average recall.
    f1_weighted:   Weighted-average F1.
    f1_macro:      Macro-average F1.
    roc_auc:       ROC-AUC (OvR, weighted; ``None`` if not applicable).
    confusion_matrix: 2-D numpy array.

    Regression-specific (``None`` for other task types)
    ----------------------------------------------------
    mae:           Mean Absolute Error.
    mse:           Mean Squared Error.
    rmse:          Root Mean Squared Error.
    r2:            R² score.
    adj_r2:        Adjusted R².

    Clustering-specific (``None`` for other task types)
    ----------------------------------------------------
    inertia:            KMeans inertia (within-cluster sum of squares).
    silhouette:         Silhouette score (``None`` if < 2 clusters or 1 sample).
    davies_bouldin:     Davies–Bouldin index.
    n_clusters:         Number of clusters used.
    """

    task_type: TaskType
    model_name: str

    # Classification
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_weighted: Optional[float] = None
    f1_macro: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None

    # Regression
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    adj_r2: Optional[float] = None

    # Clustering
    inertia: Optional[float] = None
    silhouette: Optional[float] = None
    davies_bouldin: Optional[float] = None
    n_clusters: Optional[int] = None

    # Cross-validation
    cv_mean: float = float("nan")
    cv_std: float = float("nan")

    # Extra metadata (e.g. class labels for confusion matrix)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable dict of all non-None metrics."""
        out: dict[str, Any] = {
            "task_type": self.task_type.value,
            "model_name": self.model_name,
            "cv_mean": self.cv_mean,
            "cv_std": self.cv_std,
        }
        for attr in (
            "accuracy", "precision", "recall", "f1_weighted", "f1_macro", "roc_auc",
            "mae", "mse", "rmse", "r2", "adj_r2",
            "inertia", "silhouette", "davies_bouldin", "n_clusters",
        ):
            val = getattr(self, attr)
            if val is not None:
                out[attr] = (
                    val.tolist() if isinstance(val, np.ndarray) else val
                )
        if self.confusion_matrix is not None:
            out["confusion_matrix"] = self.confusion_matrix.tolist()
        out.update(self.extra)
        return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _adjusted_r2(r2: float, n: int, p: int) -> float:
    """Compute adjusted R²."""
    if n <= p + 1:
        return float("nan")
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def _safe_roc_auc(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray],
    n_classes: int,
) -> Optional[float]:
    """Compute ROC-AUC safely, returning ``None`` when it cannot be computed."""
    if y_proba is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if n_classes == 2:
                # Binary: use probability of positive class
                p = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                return float(roc_auc_score(y_true, p))
            else:
                return float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_model(result: TrainResult) -> ModelMetrics:
    """Compute evaluation metrics for a trained model.

    Parameters
    ----------
    result:
        A :class:`~src.ml.trainer.TrainResult` as returned by
        :func:`~src.ml.trainer.train_models`.

    Returns
    -------
    ModelMetrics
    """
    task = result.task_type
    metrics = ModelMetrics(
        task_type=task,
        model_name=result.model_name,
        cv_mean=result.cv_mean,
        cv_std=result.cv_std,
    )

    # ---- Classification ----
    if task == TaskType.CLASSIFICATION:
        X_test = result.X_test
        y_test = result.y_test

        if X_test is None or y_test is None or len(X_test) == 0:
            return metrics  # nothing to evaluate

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = result.pipeline.predict(X_test)

        n_classes = int(y_test.nunique())
        avg = "binary" if n_classes == 2 else "weighted"

        metrics.accuracy = float(accuracy_score(y_test, y_pred))
        metrics.precision = float(
            precision_score(y_test, y_pred, average=avg, zero_division=0)
        )
        metrics.recall = float(
            recall_score(y_test, y_pred, average=avg, zero_division=0)
        )
        metrics.f1_weighted = float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        metrics.f1_macro = float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        )
        metrics.confusion_matrix = confusion_matrix(y_test, y_pred)

        # ROC-AUC: requires predict_proba
        y_proba: Optional[np.ndarray] = None
        if hasattr(result.pipeline, "predict_proba"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_proba = result.pipeline.predict_proba(X_test)
            except Exception:
                y_proba = None
        metrics.roc_auc = _safe_roc_auc(y_test.to_numpy(), y_proba, n_classes)

        # Store class labels for confusion matrix display
        if result.label_encoder is not None:
            metrics.extra["class_labels"] = result.label_encoder.classes_.tolist()

    # ---- Regression ----
    elif task == TaskType.REGRESSION:
        X_test = result.X_test
        y_test = result.y_test

        if X_test is None or y_test is None or len(X_test) == 0:
            return metrics

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = result.pipeline.predict(X_test)

        n = len(y_test)
        p = len(result.feature_names)
        r2 = float(r2_score(y_test, y_pred))

        metrics.mae = float(mean_absolute_error(y_test, y_pred))
        metrics.mse = float(mean_squared_error(y_test, y_pred))
        metrics.rmse = float(np.sqrt(metrics.mse))
        metrics.r2 = r2
        metrics.adj_r2 = _adjusted_r2(r2, n, p)

    # ---- Clustering ----
    elif task == TaskType.CLUSTERING:
        X = result.X_train
        cluster_labels = result.cluster_labels

        if cluster_labels is None or len(X) == 0:
            return metrics

        # Extract KMeans from the pipeline to get inertia
        kmeans_step = result.pipeline.named_steps.get("kmeans")
        if kmeans_step is not None:
            metrics.inertia = float(kmeans_step.inertia_)

        metrics.n_clusters = result.n_clusters

        # Silhouette & Davies-Bouldin require >= 2 clusters and >= 2 samples
        n_unique_labels = len(set(cluster_labels))
        if n_unique_labels >= 2 and len(X) >= 2:
            preprocessor = result.pipeline.named_steps.get("preprocessor")
            if preprocessor is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        X_scaled = preprocessor.transform(X)
                        metrics.silhouette = float(
                            silhouette_score(X_scaled, cluster_labels)
                        )
                        metrics.davies_bouldin = float(
                            davies_bouldin_score(X_scaled, cluster_labels)
                        )
                    except Exception:
                        pass  # graceful degradation

    return metrics
