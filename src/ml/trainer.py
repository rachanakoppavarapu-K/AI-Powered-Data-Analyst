"""ML trainer — builds sklearn Pipelines and trains models for all task types.

Design
------
* All preprocessing (imputation, encoding, scaling) is performed *inside*
  a :class:`sklearn.pipeline.Pipeline` so the test-set transformation uses
  parameters fitted exclusively on training data (no leakage).
* The caller receives a :class:`TrainResult` that bundles the fitted pipeline,
  split indices, feature names, and metadata — everything the evaluator and
  feature-importance modules need.
* No GenAI: this module is strictly deterministic.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGBOOST_AVAILABLE = False

from src.ml.task_detector import TaskType

__all__ = [
    "TrainResult",
    "train_models",
    "TrainerError",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TrainerError(ValueError):
    """Raised when training cannot proceed."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Container for everything produced by :func:`train_models`.

    Attributes
    ----------
    task_type:
        The :class:`~src.ml.task_detector.TaskType` used for training.
    model_name:
        Human-readable model identifier (e.g. ``"Random Forest"``).
    pipeline:
        Fitted :class:`~sklearn.pipeline.Pipeline` (preprocessor + estimator).
    X_train:
        Training feature matrix (original, un-transformed).
    X_test:
        Test feature matrix (original, un-transformed).
    y_train:
        Training target series (``None`` for clustering).
    y_test:
        Test target series (``None`` for clustering).
    feature_names:
        List of feature column names fed into the pipeline.
    label_encoder:
        Fitted :class:`~sklearn.preprocessing.LabelEncoder` for the target
        (classification only; ``None`` otherwise).
    cv_scores:
        Array of cross-validation scores (empty for clustering).
    cv_mean:
        Mean CV score.
    cv_std:
        Std-dev of CV scores.
    n_clusters:
        Number of clusters (clustering only; ``None`` otherwise).
    cluster_labels:
        Array of cluster labels for the full dataset (clustering only).
    extra:
        Arbitrary extra metadata dict (e.g. elbow scores).
    """

    task_type: TaskType
    model_name: str
    pipeline: Any  # sklearn Pipeline or KMeans
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]
    feature_names: list[str]
    label_encoder: Optional[LabelEncoder]
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    n_clusters: Optional[int]
    cluster_labels: Optional[np.ndarray]
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_preprocessor(
    X: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str]]:
    """Build a :class:`~sklearn.compose.ColumnTransformer` for *X*.

    Returns the transformer and the list of numeric + categorical column names
    (in the same order expected by the transformer).
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, num_cols + cat_cols


def _encode_target(y: pd.Series) -> tuple[pd.Series, Optional[LabelEncoder]]:
    """Label-encode a classification target if it is non-numeric."""
    if pd.api.types.is_object_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(y):
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=y.name)
        return y_encoded, le
    return y, None


def _cv_score_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
    n_splits: int = 5,
) -> np.ndarray:
    """Return cross-validation scores for a fitted-style pipeline definition."""
    n = len(X)
    # Reduce splits if the dataset is small
    n_splits = min(n_splits, n // 2) if n >= 4 else 2
    n_splits = max(n_splits, 2)

    if task_type == TaskType.CLASSIFICATION:
        scoring = "f1_weighted"
        unique_classes = y.nunique()
        if unique_classes < 2:
            return np.array([])
        actual_splits = min(n_splits, int(y.value_counts().min()))
        actual_splits = max(actual_splits, 2)
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    else:
        scoring = "r2"
        actual_splits = n_splits
        cv = actual_splits  # type: ignore[assignment]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        except Exception:
            scores = np.array([])
    return scores


# ---------------------------------------------------------------------------
# Classification trainers
# ---------------------------------------------------------------------------


def _train_classification(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    label_encoder: Optional[LabelEncoder],
) -> list[TrainResult]:
    """Train Logistic Regression, Random Forest, and XGBoost classifiers."""
    preprocessor, _ = _build_preprocessor(X_train)

    n_classes = y_train.nunique()
    is_multiclass = n_classes > 2

    estimators: list[tuple[str, Any]] = [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
            ),
        ),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=100, random_state=42),
        ),
    ]

    if _XGBOOST_AVAILABLE:
        estimators.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    eval_metric="logloss",
                    verbosity=0,
                    use_label_encoder=False,
                ),
            )
        )

    results: list[TrainResult] = []
    for model_name, estimator in estimators:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor(X_train)[0]),
                ("classifier", estimator),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)

        cv_scores = _cv_score_pipeline(
            Pipeline(
                steps=[
                    ("preprocessor", _build_preprocessor(X_train)[0]),
                    ("classifier", type(estimator)(**estimator.get_params())),
                ]
            ),
            X_train,
            y_train,
            TaskType.CLASSIFICATION,
        )

        results.append(
            TrainResult(
                task_type=TaskType.CLASSIFICATION,
                model_name=model_name,
                pipeline=pipeline,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_names=feature_names,
                label_encoder=label_encoder,
                cv_scores=cv_scores,
                cv_mean=float(cv_scores.mean()) if len(cv_scores) else float("nan"),
                cv_std=float(cv_scores.std()) if len(cv_scores) else float("nan"),
                n_clusters=None,
                cluster_labels=None,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Regression trainers
# ---------------------------------------------------------------------------


def _train_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
) -> list[TrainResult]:
    """Train Linear Regression, Random Forest Regressor, and XGBoost Regressor."""
    estimators: list[tuple[str, Any]] = [
        ("Linear Regression", LinearRegression()),
        (
            "Random Forest Regressor",
            RandomForestRegressor(n_estimators=100, random_state=42),
        ),
    ]

    if _XGBOOST_AVAILABLE:
        estimators.append(
            (
                "XGBoost Regressor",
                XGBRegressor(
                    n_estimators=100,
                    random_state=42,
                    verbosity=0,
                ),
            )
        )

    results: list[TrainResult] = []
    for model_name, estimator in estimators:
        preprocessor, _ = _build_preprocessor(X_train)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", estimator),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)

        cv_scores = _cv_score_pipeline(
            Pipeline(
                steps=[
                    ("preprocessor", _build_preprocessor(X_train)[0]),
                    ("regressor", type(estimator)(**estimator.get_params())),
                ]
            ),
            X_train,
            y_train,
            TaskType.REGRESSION,
        )

        results.append(
            TrainResult(
                task_type=TaskType.REGRESSION,
                model_name=model_name,
                pipeline=pipeline,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_names=feature_names,
                label_encoder=None,
                cv_scores=cv_scores,
                cv_mean=float(cv_scores.mean()) if len(cv_scores) else float("nan"),
                cv_std=float(cv_scores.std()) if len(cv_scores) else float("nan"),
                n_clusters=None,
                cluster_labels=None,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Clustering trainer
# ---------------------------------------------------------------------------


def _elbow_k(X_scaled: np.ndarray, k_range: range) -> tuple[int, dict[int, float]]:
    """Select k via the elbow method (largest second-derivative of inertia)."""
    from sklearn.cluster import KMeans

    inertias: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X_scaled)
        inertias[k] = float(km.inertia_)

    if len(inertias) < 3:
        return min(k_range), inertias

    ks = list(inertias.keys())
    vals = [inertias[k] for k in ks]

    # Second derivative (elbow point)
    best_k = ks[0]
    best_dd = -float("inf")
    for i in range(1, len(vals) - 1):
        dd = vals[i - 1] - 2 * vals[i] + vals[i + 1]
        if dd > best_dd:
            best_dd = dd
            best_k = ks[i]

    return best_k, inertias


def _train_clustering(
    X: pd.DataFrame,
    feature_names: list[str],
    n_clusters: Optional[int] = None,
) -> TrainResult:
    """Train K-Means with automatic k selection (elbow method) if k not given."""
    from sklearn.cluster import KMeans

    preprocessor, _ = _build_preprocessor(X)

    # Scale for clustering
    pipeline_pre = Pipeline(steps=[("preprocessor", preprocessor)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_scaled = pipeline_pre.fit_transform(X)

    n = len(X)
    elbow_scores: dict[int, float] = {}

    if n_clusters is None:
        max_k = min(10, n - 1)
        if max_k < 2:
            n_clusters = 2
        else:
            k_range = range(2, max_k + 1)
            n_clusters, elbow_scores = _elbow_k(X_scaled, k_range)

    n_clusters = max(2, min(n_clusters, n - 1))

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_labels = km.fit_predict(X_scaled)

    # Build pipeline that contains the preprocessor + KMeans
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(X)[0]),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X)

    return TrainResult(
        task_type=TaskType.CLUSTERING,
        model_name="K-Means",
        pipeline=pipeline,
        X_train=X,
        X_test=pd.DataFrame(columns=X.columns),
        y_train=None,
        y_test=None,
        feature_names=feature_names,
        label_encoder=None,
        cv_scores=np.array([]),
        cv_mean=float("nan"),
        cv_std=float("nan"),
        n_clusters=n_clusters,
        cluster_labels=cluster_labels,
        extra={"elbow_scores": elbow_scores},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_models(
    df: pd.DataFrame,
    target_column: Optional[str],
    task_type: TaskType,
    *,
    test_size: float = 0.2,
    n_clusters: Optional[int] = None,
) -> list[TrainResult]:
    """Train models for the given task type and return all results.

    Parameters
    ----------
    df:
        Full dataset (features + optional target column).  Not modified.
    target_column:
        Name of the target column.  Ignored for clustering.
    task_type:
        One of :class:`~src.ml.task_detector.TaskType`.
    test_size:
        Fraction of data held out for testing (supervised tasks only).
    n_clusters:
        Override the number of clusters for K-Means (``None`` = auto).

    Returns
    -------
    list[TrainResult]
        One :class:`TrainResult` per trained model.

    Raises
    ------
    TrainerError
        If the dataset is unsuitable (too few rows, missing target, etc.).
    """
    df = df.copy()

    # --- Clustering ---
    if task_type == TaskType.CLUSTERING:
        feature_cols = [c for c in df.columns if c != target_column] if target_column else df.columns.tolist()
        if not feature_cols:
            raise TrainerError("No feature columns available for clustering.")
        X = df[feature_cols]
        return [_train_clustering(X, feature_cols, n_clusters=n_clusters)]

    # --- Supervised ---
    if not target_column or target_column not in df.columns:
        raise TrainerError(f"Target column '{target_column}' not found in DataFrame.")

    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise TrainerError("No feature columns available after removing the target.")

    X = df[feature_cols]
    y = df[target_column]

    # Drop rows where target is NaN
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) < 4:
        raise TrainerError(
            f"Dataset has only {len(X)} complete rows; too few to train a model."
        )

    if task_type == TaskType.CLASSIFICATION:
        y, label_encoder = _encode_target(y)
        if y.nunique() < 2:
            raise TrainerError("Target column has fewer than 2 unique classes.")

        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
        return _train_classification(X_train, X_test, y_train, y_test, feature_cols, label_encoder)

    else:  # REGRESSION
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
        )
        return _train_regression(X_train, X_test, y_train, y_test, feature_cols)
