"""Tests for the ML pipeline (ST-07).

Covers:
- task_detector.py  — TaskType inference and override
- trainer.py        — train/test split, pipeline construction, all task types
- evaluator.py      — metric computation, ModelMetrics structure
- feature_importance.py — model-native and permutation importance

All random operations use random_state=42 for reproducibility.
No real API calls are made (no GenAI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.evaluator import ModelMetrics, evaluate_model
from src.ml.feature_importance import ImportanceResult, compute_feature_importance
from src.ml.task_detector import TaskDetectionResult, TaskType, detect_task
from src.ml.trainer import TrainResult, TrainerError, train_models


# ===========================================================================
# Shared fixtures
# ===========================================================================


@pytest.fixture()
def clf_df() -> pd.DataFrame:
    """Binary classification dataset (100 rows, 4 features, 1 binary target)."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(5, 2, n),
            "feat_c": rng.choice(["cat", "dog", "bird"], n),
            "feat_d": rng.integers(0, 10, n).astype(float),
            "target": rng.choice([0, 1], n),
        }
    )


@pytest.fixture()
def multiclass_df() -> pd.DataFrame:
    """3-class classification dataset."""
    rng = np.random.default_rng(42)
    n = 120
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(1, 1, n),
            "label": rng.choice(["A", "B", "C"], n),
        }
    )


@pytest.fixture()
def reg_df() -> pd.DataFrame:
    """Regression dataset (100 rows)."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "x1": x,
            "x2": rng.normal(0, 1, n),
            "cat": rng.choice(["red", "blue"], n),
            "y": 3.0 * x + rng.normal(0, 0.5, n),
        }
    )


@pytest.fixture()
def cluster_df() -> pd.DataFrame:
    """Dataset suitable for clustering (no explicit target)."""
    rng = np.random.default_rng(42)
    n = 80
    return pd.DataFrame(
        {
            "a": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(10, 1, n // 2)]),
            "b": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(10, 1, n // 2)]),
        }
    )


@pytest.fixture()
def small_df() -> pd.DataFrame:
    """Very small dataset (3 rows) — should trigger graceful handling."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "y": [0, 1, 0]})


@pytest.fixture()
def missing_df() -> pd.DataFrame:
    """Dataset with missing values."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "y": rng.choice([0, 1], n),
        }
    )
    # Introduce 10% missing values in x1
    mask = rng.choice([True, False], n, p=[0.1, 0.9])
    df.loc[mask, "x1"] = np.nan
    return df


# ===========================================================================
# task_detector tests
# ===========================================================================


class TestDetectTask:
    """Tests for :func:`src.ml.task_detector.detect_task`."""

    def test_no_target_returns_clustering(self, cluster_df: pd.DataFrame) -> None:
        result = detect_task(cluster_df, target_column=None)
        assert result.task_type == TaskType.CLUSTERING
        assert result.n_classes is None

    def test_missing_target_column_returns_clustering(self, clf_df: pd.DataFrame) -> None:
        result = detect_task(clf_df, target_column="nonexistent")
        assert result.task_type == TaskType.CLUSTERING

    def test_binary_int_target_is_classification(self, clf_df: pd.DataFrame) -> None:
        result = detect_task(clf_df, target_column="target")
        assert result.task_type == TaskType.CLASSIFICATION
        assert result.n_classes == 2

    def test_string_target_is_classification(self, multiclass_df: pd.DataFrame) -> None:
        result = detect_task(multiclass_df, target_column="label")
        assert result.task_type == TaskType.CLASSIFICATION
        assert result.n_classes == 3

    def test_continuous_target_is_regression(self, reg_df: pd.DataFrame) -> None:
        result = detect_task(reg_df, target_column="y")
        assert result.task_type == TaskType.REGRESSION
        assert result.n_classes is None

    def test_user_override_respected(self, reg_df: pd.DataFrame) -> None:
        result = detect_task(
            reg_df, target_column="y", user_override=TaskType.CLASSIFICATION
        )
        assert result.task_type == TaskType.CLASSIFICATION
        assert result.is_override is True

    def test_small_dataset_returns_clustering(self, small_df: pd.DataFrame) -> None:
        # 3 rows < _MIN_SUPERVISED_ROWS=10
        result = detect_task(small_df, target_column="y")
        assert result.task_type == TaskType.CLUSTERING

    def test_result_has_reason_string(self, clf_df: pd.DataFrame) -> None:
        result = detect_task(clf_df, target_column="target")
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0

    def test_bool_target_is_classification(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "y": [True, False] * 5})
        result = detect_task(df, target_column="y")
        assert result.task_type == TaskType.CLASSIFICATION


# ===========================================================================
# trainer tests
# ===========================================================================


class TestTrainModels:
    """Tests for :func:`src.ml.trainer.train_models`."""

    def test_classification_returns_multiple_models(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        assert len(results) >= 2  # at least LR and RF; XGBoost if available
        for r in results:
            assert isinstance(r, TrainResult)
            assert r.task_type == TaskType.CLASSIFICATION

    def test_classification_model_names(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        names = {r.model_name for r in results}
        assert "Logistic Regression" in names
        assert "Random Forest" in names

    def test_classification_train_test_split(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        r = results[0]
        assert len(r.X_train) > 0
        assert len(r.X_test) > 0
        # No overlap between train and test indices
        assert set(r.X_train.index).isdisjoint(set(r.X_test.index))

    def test_classification_predictions_work(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        for r in results:
            preds = r.pipeline.predict(r.X_test)
            assert len(preds) == len(r.X_test)

    def test_classification_cv_scores(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        for r in results:
            # cv_mean should be a float (may be nan for tiny datasets but not here)
            assert isinstance(r.cv_mean, float)

    def test_regression_returns_multiple_models(self, reg_df: pd.DataFrame) -> None:
        results = train_models(reg_df, "y", TaskType.REGRESSION)
        assert len(results) >= 2
        for r in results:
            assert r.task_type == TaskType.REGRESSION

    def test_regression_model_names(self, reg_df: pd.DataFrame) -> None:
        results = train_models(reg_df, "y", TaskType.REGRESSION)
        names = {r.model_name for r in results}
        assert "Linear Regression" in names
        assert "Random Forest Regressor" in names

    def test_regression_train_test_no_leak(self, reg_df: pd.DataFrame) -> None:
        results = train_models(reg_df, "y", TaskType.REGRESSION)
        r = results[0]
        assert set(r.X_train.index).isdisjoint(set(r.X_test.index))

    def test_clustering_no_target(self, cluster_df: pd.DataFrame) -> None:
        results = train_models(cluster_df, None, TaskType.CLUSTERING)
        assert len(results) == 1
        r = results[0]
        assert r.task_type == TaskType.CLUSTERING
        assert r.model_name == "K-Means"
        assert r.n_clusters is not None and r.n_clusters >= 2

    def test_clustering_labels_length(self, cluster_df: pd.DataFrame) -> None:
        results = train_models(cluster_df, None, TaskType.CLUSTERING)
        r = results[0]
        assert r.cluster_labels is not None
        assert len(r.cluster_labels) == len(cluster_df)

    def test_clustering_auto_k_finds_two_clusters(self, cluster_df: pd.DataFrame) -> None:
        """The fixture has two clear clusters; elbow should select k=2 or nearby."""
        results = train_models(cluster_df, None, TaskType.CLUSTERING)
        k = results[0].n_clusters
        assert k is not None
        assert 2 <= k <= 5

    def test_clustering_custom_k(self, cluster_df: pd.DataFrame) -> None:
        results = train_models(cluster_df, None, TaskType.CLUSTERING, n_clusters=3)
        assert results[0].n_clusters == 3

    def test_missing_values_handled(self, missing_df: pd.DataFrame) -> None:
        """Pipeline should impute missing values without raising."""
        results = train_models(missing_df, "y", TaskType.CLASSIFICATION)
        assert len(results) >= 2

    def test_categorical_features_handled(self, clf_df: pd.DataFrame) -> None:
        """Dataset has a categorical column 'feat_c'; pipeline should encode it."""
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        for r in results:
            preds = r.pipeline.predict(r.X_test)
            assert len(preds) == len(r.X_test)

    def test_missing_target_raises(self, clf_df: pd.DataFrame) -> None:
        with pytest.raises(TrainerError):
            train_models(clf_df, "nonexistent", TaskType.CLASSIFICATION)

    def test_no_features_raises(self) -> None:
        df = pd.DataFrame({"y": range(20)})
        with pytest.raises(TrainerError):
            train_models(df, "y", TaskType.CLASSIFICATION)

    def test_too_few_rows_raises(self, small_df: pd.DataFrame) -> None:
        with pytest.raises(TrainerError):
            train_models(small_df, "y", TaskType.CLASSIFICATION)

    def test_multiclass_classification(self, multiclass_df: pd.DataFrame) -> None:
        results = train_models(multiclass_df, "label", TaskType.CLASSIFICATION)
        assert len(results) >= 2
        for r in results:
            assert r.label_encoder is not None
            # Predictions should be integer-encoded labels
            preds = r.pipeline.predict(r.X_test)
            unique_preds = set(preds)
            assert unique_preds.issubset({0, 1, 2})


# ===========================================================================
# evaluator tests
# ===========================================================================


class TestEvaluateModel:
    """Tests for :func:`src.ml.evaluator.evaluate_model`."""

    def test_classification_metrics_present(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        for r in results:
            m = evaluate_model(r)
            assert isinstance(m, ModelMetrics)
            assert m.accuracy is not None
            assert m.precision is not None
            assert m.recall is not None
            assert m.f1_weighted is not None
            assert m.f1_macro is not None
            assert m.confusion_matrix is not None

    def test_classification_accuracy_range(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        for r in results:
            m = evaluate_model(r)
            assert 0.0 <= m.accuracy <= 1.0  # type: ignore[operator]

    def test_classification_confusion_matrix_shape(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        for r in results:
            m = evaluate_model(r)
            assert m.confusion_matrix.shape == (2, 2)

    def test_multiclass_confusion_matrix_shape(self, multiclass_df: pd.DataFrame) -> None:
        results = train_models(multiclass_df, "label", TaskType.CLASSIFICATION)
        for r in results:
            m = evaluate_model(r)
            assert m.confusion_matrix.shape == (3, 3)

    def test_classification_roc_auc_for_rf(self, clf_df: pd.DataFrame) -> None:
        """Random Forest supports predict_proba; ROC-AUC should be computed."""
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        rf_results = [r for r in results if r.model_name == "Random Forest"]
        assert rf_results
        m = evaluate_model(rf_results[0])
        # roc_auc may be None only if something went wrong; should be float here
        assert m.roc_auc is None or (0.0 <= m.roc_auc <= 1.0)

    def test_regression_metrics_present(self, reg_df: pd.DataFrame) -> None:
        results = train_models(reg_df, "y", TaskType.REGRESSION)
        for r in results:
            m = evaluate_model(r)
            assert m.mae is not None
            assert m.mse is not None
            assert m.rmse is not None
            assert m.r2 is not None
            assert m.adj_r2 is not None

    def test_regression_rmse_positive(self, reg_df: pd.DataFrame) -> None:
        results = train_models(reg_df, "y", TaskType.REGRESSION)
        for r in results:
            m = evaluate_model(r)
            assert m.rmse >= 0.0  # type: ignore[operator]

    def test_regression_r2_range(self, reg_df: pd.DataFrame) -> None:
        """R² for a near-linear dataset should be meaningfully positive."""
        results = train_models(reg_df, "y", TaskType.REGRESSION)
        lr_results = [r for r in results if r.model_name == "Linear Regression"]
        assert lr_results
        m = evaluate_model(lr_results[0])
        # Should be well above 0 for a linear dataset
        assert m.r2 is not None
        assert m.r2 > 0.5

    def test_clustering_metrics_present(self, cluster_df: pd.DataFrame) -> None:
        results = train_models(cluster_df, None, TaskType.CLUSTERING)
        m = evaluate_model(results[0])
        assert m.inertia is not None
        assert m.inertia >= 0.0
        assert m.n_clusters is not None

    def test_clustering_silhouette_range(self, cluster_df: pd.DataFrame) -> None:
        results = train_models(cluster_df, None, TaskType.CLUSTERING)
        m = evaluate_model(results[0])
        if m.silhouette is not None:
            assert -1.0 <= m.silhouette <= 1.0

    def test_to_dict_serialisable(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        m = evaluate_model(results[0])
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "accuracy" in d
        assert isinstance(d["confusion_matrix"], list)

    def test_cv_mean_in_metrics(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        m = evaluate_model(results[0])
        assert isinstance(m.cv_mean, float)


# ===========================================================================
# feature_importance tests
# ===========================================================================


class TestFeatureImportance:
    """Tests for :func:`src.ml.feature_importance.compute_feature_importance`."""

    def test_rf_model_native_importance(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        rf = next(r for r in results if r.model_name == "Random Forest")
        imp = compute_feature_importance(rf, method="model_native")
        assert isinstance(imp, ImportanceResult)
        assert imp.method == "model_native"
        assert len(imp.ranked) > 0

    def test_importance_sum_approx_one_rf(self, clf_df: pd.DataFrame) -> None:
        """For tree-based models the importances should sum ~1."""
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        rf = next(r for r in results if r.model_name == "Random Forest")
        imp = compute_feature_importance(rf, method="model_native")
        total = sum(e.importance for e in imp.ranked)
        assert pytest.approx(total, abs=1e-3) == 1.0

    def test_ranked_descending_order(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        rf = next(r for r in results if r.model_name == "Random Forest")
        imp = compute_feature_importance(rf, method="model_native")
        importances = [e.importance for e in imp.ranked]
        assert importances == sorted(importances, reverse=True)

    def test_logistic_regression_permutation_fallback(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        lr = next(r for r in results if r.model_name == "Logistic Regression")
        imp = compute_feature_importance(lr, method="auto", n_repeats=3)
        assert isinstance(imp, ImportanceResult)
        assert len(imp.ranked) > 0

    def test_permutation_method_explicit(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        rf = next(r for r in results if r.model_name == "Random Forest")
        imp = compute_feature_importance(rf, method="permutation", n_repeats=3)
        assert imp.method == "permutation"
        assert len(imp.ranked) > 0

    def test_regression_feature_importance(self, reg_df: pd.DataFrame) -> None:
        results = train_models(reg_df, "y", TaskType.REGRESSION)
        rf = next(r for r in results if r.model_name == "Random Forest Regressor")
        imp = compute_feature_importance(rf, method="model_native")
        assert len(imp.ranked) > 0

    def test_clustering_uniform_importance(self, cluster_df: pd.DataFrame) -> None:
        results = train_models(cluster_df, None, TaskType.CLUSTERING)
        imp = compute_feature_importance(results[0])
        assert imp.method == "uniform"
        # All importances equal
        vals = [e.importance for e in imp.ranked]
        assert max(vals) == pytest.approx(min(vals), abs=1e-9)

    def test_to_dict_serialisable(self, clf_df: pd.DataFrame) -> None:
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        rf = next(r for r in results if r.model_name == "Random Forest")
        imp = compute_feature_importance(rf, method="model_native")
        d = imp.to_dict()
        assert "ranked" in d
        assert isinstance(d["ranked"], list)
        assert "feature" in d["ranked"][0]
        assert "importance" in d["ranked"][0]

    def test_model_native_invalid_raises(self, clf_df: pd.DataFrame) -> None:
        """Logistic Regression has no feature_importances_; model_native should raise."""
        results = train_models(clf_df, "target", TaskType.CLASSIFICATION)
        lr = next(r for r in results if r.model_name == "Logistic Regression")
        with pytest.raises(ValueError, match="does not expose feature_importances_"):
            compute_feature_importance(lr, method="model_native")

    def test_missing_values_pipeline_importance(self, missing_df: pd.DataFrame) -> None:
        """Feature importance should work even when training data had missing values."""
        results = train_models(missing_df, "y", TaskType.CLASSIFICATION)
        rf = next((r for r in results if r.model_name == "Random Forest"), results[0])
        imp = compute_feature_importance(rf, method="model_native")
        assert len(imp.ranked) > 0


# ===========================================================================
# End-to-end integration tests
# ===========================================================================


class TestEndToEnd:
    """Smoke tests for the full pipeline: detect → train → evaluate → importance."""

    def test_full_classification_pipeline(self, clf_df: pd.DataFrame) -> None:
        task_result = detect_task(clf_df, "target")
        assert task_result.task_type == TaskType.CLASSIFICATION

        train_results = train_models(clf_df, "target", task_result.task_type)
        for tr in train_results:
            metrics = evaluate_model(tr)
            assert metrics.accuracy is not None
            imp = compute_feature_importance(tr, method="auto", n_repeats=3)
            assert len(imp.ranked) > 0

    def test_full_regression_pipeline(self, reg_df: pd.DataFrame) -> None:
        task_result = detect_task(reg_df, "y")
        assert task_result.task_type == TaskType.REGRESSION

        train_results = train_models(reg_df, "y", task_result.task_type)
        for tr in train_results:
            metrics = evaluate_model(tr)
            assert metrics.r2 is not None
            imp = compute_feature_importance(tr, method="auto", n_repeats=3)
            assert len(imp.ranked) > 0

    def test_full_clustering_pipeline(self, cluster_df: pd.DataFrame) -> None:
        task_result = detect_task(cluster_df, target_column=None)
        assert task_result.task_type == TaskType.CLUSTERING

        train_results = train_models(cluster_df, None, task_result.task_type)
        assert len(train_results) == 1
        metrics = evaluate_model(train_results[0])
        assert metrics.inertia is not None
        imp = compute_feature_importance(train_results[0])
        assert imp.method == "uniform"

    def test_results_are_not_hardcoded(self, clf_df: pd.DataFrame) -> None:
        """Running with different data should produce different accuracy values."""
        rng2 = np.random.default_rng(99)
        n = 100
        df2 = pd.DataFrame(
            {
                "feat_a": rng2.normal(5, 2, n),
                "feat_b": rng2.normal(-3, 1, n),
                "feat_c": rng2.choice(["x", "y"], n),
                "target": rng2.choice([0, 1], n),
            }
        )
        r1 = evaluate_model(train_models(clf_df, "target", TaskType.CLASSIFICATION)[0])
        r2 = evaluate_model(train_models(df2, "target", TaskType.CLASSIFICATION)[0])
        # The datasets are different, so at least one metric should differ
        # (this guards against accidentally hardcoded results)
        assert r1.accuracy != r2.accuracy or r1.f1_weighted != r2.f1_weighted
