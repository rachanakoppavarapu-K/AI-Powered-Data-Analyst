"""Tests for src/eda/eda_engine.py — ST-05."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.eda.eda_engine import EDAEngine, run_eda

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    """DataFrame with both numeric and categorical columns."""
    return pd.DataFrame(
        {
            "age": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0],
            "income": [30000.0, 45000.0, 60000.0, 75000.0, 90000.0, 105000.0],
            "score": [5.0, 7.0, 6.0, 8.0, 9.0, 7.0],
            "city": ["NY", "LA", "NY", "SF", "LA", "NY"],
            "gender": ["M", "F", "M", "F", "M", "F"],
        }
    )


@pytest.fixture()
def numeric_only_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "c": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )


@pytest.fixture()
def df_with_nulls() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.0, np.nan, 3.0, np.nan, 5.0],
            "category": ["a", None, "b", "a", None],
        }
    )


@pytest.fixture()
def classification_df() -> pd.DataFrame:
    """DataFrame suitable for classification analysis with a target column."""
    return pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "label": ["cat", "dog", "cat", "cat", "dog", "dog"],
        }
    )


@pytest.fixture()
def regression_df() -> pd.DataFrame:
    """DataFrame suitable for regression analysis with a continuous target."""
    rng = np.arange(20, dtype=float)
    return pd.DataFrame(
        {
            "x1": rng,
            "x2": rng * 2.0,
            "target": rng * 3.0 + 1.0,
        }
    )


# ---------------------------------------------------------------------------
# EDAEngine.analyze — top-level structure
# ---------------------------------------------------------------------------


class TestAnalyzeStructure:
    def test_returns_dict(self, mixed_df: pd.DataFrame) -> None:
        result = run_eda(mixed_df)
        assert isinstance(result, dict)

    def test_top_level_keys_without_target(self, mixed_df: pd.DataFrame) -> None:
        result = run_eda(mixed_df)
        assert "univariate" in result
        assert "bivariate" in result
        assert "groupby" in result
        assert "target_analysis" not in result

    def test_target_analysis_key_present_when_target_given(
        self, mixed_df: pd.DataFrame
    ) -> None:
        result = run_eda(mixed_df, target_column="city")
        assert "target_analysis" in result

    def test_target_analysis_key_absent_for_invalid_target(
        self, mixed_df: pd.DataFrame
    ) -> None:
        result = run_eda(mixed_df, target_column="nonexistent_col")
        assert "target_analysis" not in result

    def test_original_df_not_modified(self, mixed_df: pd.DataFrame) -> None:
        original_shape = mixed_df.shape
        original_cols = list(mixed_df.columns)
        run_eda(mixed_df, target_column="city")
        assert mixed_df.shape == original_shape
        assert list(mixed_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Univariate — numeric columns
# ---------------------------------------------------------------------------


class TestUnivariateNumeric:
    def test_numeric_columns_present(self, mixed_df: pd.DataFrame) -> None:
        result = run_eda(mixed_df)
        numeric = result["univariate"]["numeric"]
        assert "age" in numeric
        assert "income" in numeric
        assert "score" in numeric

    def test_categorical_not_in_numeric(self, mixed_df: pd.DataFrame) -> None:
        numeric = run_eda(mixed_df)["univariate"]["numeric"]
        assert "city" not in numeric
        assert "gender" not in numeric

    def test_summary_stats_keys(self, mixed_df: pd.DataFrame) -> None:
        stats = run_eda(mixed_df)["univariate"]["numeric"]["age"]
        for key in ("count", "mean", "median", "std", "min", "max", "q1", "q3", "iqr"):
            assert key in stats, f"Missing key: {key}"

    def test_mean_value_correct(self, mixed_df: pd.DataFrame) -> None:
        stats = run_eda(mixed_df)["univariate"]["numeric"]["age"]
        assert stats["mean"] == pytest.approx(
            float(np.mean([25, 30, 35, 40, 45, 50])), rel=1e-6
        )

    def test_min_max_correct(self, mixed_df: pd.DataFrame) -> None:
        stats = run_eda(mixed_df)["univariate"]["numeric"]["age"]
        assert stats["min"] == pytest.approx(25.0)
        assert stats["max"] == pytest.approx(50.0)

    def test_histogram_present(self, mixed_df: pd.DataFrame) -> None:
        stats = run_eda(mixed_df)["univariate"]["numeric"]["age"]
        assert "histogram" in stats
        hist = stats["histogram"]
        assert "counts" in hist
        assert "bin_edges" in hist

    def test_histogram_counts_sum_to_n_rows(self, mixed_df: pd.DataFrame) -> None:
        stats = run_eda(mixed_df)["univariate"]["numeric"]["age"]
        assert sum(stats["histogram"]["counts"]) == 6

    def test_null_count_tracked(self, df_with_nulls: pd.DataFrame) -> None:
        stats = run_eda(df_with_nulls)["univariate"]["numeric"]["x"]
        assert stats["null_count"] == 2

    def test_all_null_column_safe(self) -> None:
        df = pd.DataFrame({"all_null": [np.nan, np.nan, np.nan]})
        result = run_eda(df)
        stats = result["univariate"]["numeric"]["all_null"]
        assert stats["count"] == 0
        assert stats["mean"] is None

    def test_skewness_kurtosis_are_floats_or_none(self, mixed_df: pd.DataFrame) -> None:
        stats = run_eda(mixed_df)["univariate"]["numeric"]["income"]
        for key in ("skewness", "kurtosis"):
            val = stats[key]
            assert val is None or isinstance(val, float)

    def test_no_nan_or_inf_in_results(self, mixed_df: pd.DataFrame) -> None:
        """All float values must be finite Python floats or None."""
        numeric = run_eda(mixed_df)["univariate"]["numeric"]
        for col_stats in numeric.values():
            for val in col_stats.values():
                if isinstance(val, float):
                    assert math.isfinite(val), f"Non-finite value: {val}"


# ---------------------------------------------------------------------------
# Univariate — categorical columns
# ---------------------------------------------------------------------------


class TestUnivariateCategorical:
    def test_categorical_columns_present(self, mixed_df: pd.DataFrame) -> None:
        categorical = run_eda(mixed_df)["univariate"]["categorical"]
        assert "city" in categorical
        assert "gender" in categorical

    def test_numeric_not_in_categorical(self, mixed_df: pd.DataFrame) -> None:
        categorical = run_eda(mixed_df)["univariate"]["categorical"]
        assert "age" not in categorical

    def test_summary_keys(self, mixed_df: pd.DataFrame) -> None:
        stats = run_eda(mixed_df)["univariate"]["categorical"]["city"]
        for key in ("count", "null_count", "cardinality", "mode", "top_values"):
            assert key in stats, f"Missing key: {key}"

    def test_mode_is_most_frequent(self, mixed_df: pd.DataFrame) -> None:
        stats = run_eda(mixed_df)["univariate"]["categorical"]["city"]
        # "NY" appears 3 times — should be mode
        assert stats["mode"] == "NY"

    def test_top_values_contains_all_categories(self, mixed_df: pd.DataFrame) -> None:
        stats = run_eda(mixed_df)["univariate"]["categorical"]["city"]
        assert "NY" in stats["top_values"]
        assert "LA" in stats["top_values"]
        assert "SF" in stats["top_values"]

    def test_top_values_pct_sum_to_100(self, mixed_df: pd.DataFrame) -> None:
        stats = run_eda(mixed_df)["univariate"]["categorical"]["city"]
        total_pct = sum(v for v in stats["top_values_pct"].values() if v is not None)
        assert total_pct == pytest.approx(100.0, abs=1e-6)

    def test_null_count_tracked(self, df_with_nulls: pd.DataFrame) -> None:
        stats = run_eda(df_with_nulls)["univariate"]["categorical"]["category"]
        assert stats["null_count"] == 2

    def test_all_null_categorical_safe(self) -> None:
        df = pd.DataFrame({"cat": [None, None, None]})
        result = run_eda(df)
        stats = result["univariate"]["categorical"]["cat"]
        assert stats["count"] == 0
        assert stats["mode"] is None

    def test_top_n_categories_respected(self) -> None:
        # 25 distinct categories — engine should cap at top_n_categories
        df = pd.DataFrame({"col": [str(i) for i in range(25)]})
        result = EDAEngine(top_n_categories=5).analyze(df)
        top_vals = result["univariate"]["categorical"]["col"]["top_values"]
        assert len(top_vals) <= 5


# ---------------------------------------------------------------------------
# Bivariate — correlation
# ---------------------------------------------------------------------------


class TestBivariateCorrelation:
    def test_correlation_keys_present(self, numeric_only_df: pd.DataFrame) -> None:
        result = run_eda(numeric_only_df)
        corr = result["bivariate"]["correlation"]
        assert "pearson" in corr
        assert "spearman" in corr

    def test_pearson_diagonal_is_one(self, numeric_only_df: pd.DataFrame) -> None:
        pearson = run_eda(numeric_only_df)["bivariate"]["correlation"]["pearson"]
        for col in ("a", "b", "c"):
            assert pearson[col][col] == pytest.approx(1.0)

    def test_pearson_perfect_correlation(self, numeric_only_df: pd.DataFrame) -> None:
        """b = 10*a — Pearson correlation should be 1.0."""
        pearson = run_eda(numeric_only_df)["bivariate"]["correlation"]["pearson"]
        assert pearson["a"]["b"] == pytest.approx(1.0, abs=1e-6)

    def test_spearman_symmetric(self, numeric_only_df: pd.DataFrame) -> None:
        spearman = run_eda(numeric_only_df)["bivariate"]["correlation"]["spearman"]
        assert spearman["a"]["b"] == pytest.approx(spearman["b"]["a"], abs=1e-10)

    def test_single_numeric_col_returns_empty_corr(self) -> None:
        df = pd.DataFrame({"only": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        corr = run_eda(df)["bivariate"]["correlation"]
        assert corr["pearson"] == {}
        assert corr["spearman"] == {}

    def test_no_nan_in_correlation(self, numeric_only_df: pd.DataFrame) -> None:
        pearson = run_eda(numeric_only_df)["bivariate"]["correlation"]["pearson"]
        for row in pearson.values():
            for val in row.values():
                assert val is None or (isinstance(val, float) and math.isfinite(val))


# ---------------------------------------------------------------------------
# Bivariate — scatter data
# ---------------------------------------------------------------------------


class TestBivariateScatter:
    def test_scatter_keys_are_pairs(self, numeric_only_df: pd.DataFrame) -> None:
        scatter = run_eda(numeric_only_df)["bivariate"]["scatter_data"]
        expected_keys = {"a_vs_b", "a_vs_c", "b_vs_c"}
        assert set(scatter.keys()) == expected_keys

    def test_scatter_pair_has_x_and_y(self, numeric_only_df: pd.DataFrame) -> None:
        pair = run_eda(numeric_only_df)["bivariate"]["scatter_data"]["a_vs_b"]
        assert "x" in pair
        assert "y" in pair
        assert "x_col" in pair
        assert "y_col" in pair

    def test_scatter_x_y_same_length(self, numeric_only_df: pd.DataFrame) -> None:
        scatter = run_eda(numeric_only_df)["bivariate"]["scatter_data"]
        for pair in scatter.values():
            assert len(pair["x"]) == len(pair["y"])

    def test_scatter_sample_size_respected(self) -> None:
        df = pd.DataFrame({"p": list(range(200)), "q": list(range(200, 400))})
        scatter = EDAEngine(scatter_sample_size=50).analyze(df)["bivariate"]["scatter_data"]
        pair = scatter["p_vs_q"]
        assert len(pair["x"]) <= 50

    def test_no_scatter_for_single_numeric_col(self) -> None:
        df = pd.DataFrame({"only": [1.0, 2.0, 3.0], "cat": ["a", "b", "c"]})
        scatter = run_eda(df)["bivariate"]["scatter_data"]
        assert scatter == {}


# ---------------------------------------------------------------------------
# Group-by analysis
# ---------------------------------------------------------------------------


class TestGroupBy:
    def test_groupby_keyed_by_categorical_col(self, mixed_df: pd.DataFrame) -> None:
        groupby = run_eda(mixed_df)["groupby"]
        assert "city" in groupby
        assert "gender" in groupby

    def test_numeric_col_in_groupby_results(self, mixed_df: pd.DataFrame) -> None:
        city_groupby = run_eda(mixed_df)["groupby"]["city"]
        assert "age" in city_groupby
        assert "income" in city_groupby

    def test_aggregations_present(self, mixed_df: pd.DataFrame) -> None:
        age_by_city = run_eda(mixed_df)["groupby"]["city"]["age"]
        for agg in ("mean", "sum", "count", "median"):
            assert agg in age_by_city, f"Missing aggregation: {agg}"

    def test_mean_values_correct(self, mixed_df: pd.DataFrame) -> None:
        age_by_city = run_eda(mixed_df)["groupby"]["city"]["age"]
        # NY rows: 25, 35, 50 → mean = 110/3 ≈ 36.67
        assert age_by_city["mean"]["NY"] == pytest.approx(110.0 / 3, rel=1e-6)

    def test_count_values_correct(self, mixed_df: pd.DataFrame) -> None:
        age_by_city = run_eda(mixed_df)["groupby"]["city"]["age"]
        assert age_by_city["count"]["NY"] == 3.0
        assert age_by_city["count"]["LA"] == 2.0
        assert age_by_city["count"]["SF"] == 1.0

    def test_no_numeric_cols_returns_empty(self) -> None:
        df = pd.DataFrame({"cat1": ["a", "b"], "cat2": ["x", "y"]})
        assert run_eda(df)["groupby"] == {}

    def test_no_categorical_cols_returns_empty(self, numeric_only_df: pd.DataFrame) -> None:
        assert run_eda(numeric_only_df)["groupby"] == {}

    def test_high_cardinality_category_skipped(self) -> None:
        """Columns with > 50 unique values should be skipped."""
        df = pd.DataFrame(
            {
                "id": [str(i) for i in range(100)],
                "value": list(range(100)),
            }
        )
        groupby = run_eda(df)["groupby"]
        assert "id" not in groupby

    def test_null_values_in_numeric_handled(self, df_with_nulls: pd.DataFrame) -> None:
        """Null rows in numeric col should be dropped before aggregation."""
        df = pd.DataFrame(
            {
                "group": ["a", "a", "b", "b", "b"],
                "val": [1.0, np.nan, 3.0, 4.0, np.nan],
            }
        )
        groupby = run_eda(df)["groupby"]
        # "a" has only one non-null value (1.0)
        assert groupby["group"]["val"]["count"]["a"] == 1.0


# ---------------------------------------------------------------------------
# Target-variable analysis — classification
# ---------------------------------------------------------------------------


class TestTargetClassification:
    def test_task_type_in_result(self, classification_df: pd.DataFrame) -> None:
        ta = run_eda(
            classification_df, target_column="label", task_type="classification"
        )["target_analysis"]
        assert ta["task_type"] == "classification"

    def test_n_classes_correct(self, classification_df: pd.DataFrame) -> None:
        ta = run_eda(
            classification_df, target_column="label", task_type="classification"
        )["target_analysis"]
        assert ta["n_classes"] == 2

    def test_class_distribution_present(self, classification_df: pd.DataFrame) -> None:
        ta = run_eda(
            classification_df, target_column="label", task_type="classification"
        )["target_analysis"]
        assert "class_distribution" in ta
        assert "cat" in ta["class_distribution"]
        assert "dog" in ta["class_distribution"]

    def test_class_distribution_counts_correct(
        self, classification_df: pd.DataFrame
    ) -> None:
        ta = run_eda(
            classification_df, target_column="label", task_type="classification"
        )["target_analysis"]
        # 3 cats, 3 dogs
        assert ta["class_distribution"]["cat"] == 3
        assert ta["class_distribution"]["dog"] == 3

    def test_class_distribution_pct_sums_100(
        self, classification_df: pd.DataFrame
    ) -> None:
        ta = run_eda(
            classification_df, target_column="label", task_type="classification"
        )["target_analysis"]
        total = sum(
            v for v in ta["class_distribution_pct"].values() if v is not None
        )
        assert total == pytest.approx(100.0, abs=1e-6)

    def test_imbalance_ratio_balanced_is_one(
        self, classification_df: pd.DataFrame
    ) -> None:
        ta = run_eda(
            classification_df, target_column="label", task_type="classification"
        )["target_analysis"]
        assert ta["imbalance_ratio"] == pytest.approx(1.0)

    def test_imbalance_ratio_imbalanced(self) -> None:
        df = pd.DataFrame({"label": ["a"] * 9 + ["b"] * 1, "val": list(range(10))})
        ta = run_eda(df, target_column="label", task_type="classification")[
            "target_analysis"
        ]
        assert ta["imbalance_ratio"] == pytest.approx(9.0)

    def test_most_least_common_class(self) -> None:
        df = pd.DataFrame(
            {"label": ["a", "a", "a", "b", "b", "c"], "val": list(range(6))}
        )
        ta = run_eda(df, target_column="label", task_type="classification")[
            "target_analysis"
        ]
        assert ta["most_common_class"] == "a"
        assert ta["least_common_class"] == "c"

    def test_null_count_in_target_analysis(self) -> None:
        df = pd.DataFrame(
            {"label": ["a", None, "b", None, "a"], "val": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )
        ta = run_eda(df, target_column="label", task_type="classification")[
            "target_analysis"
        ]
        assert ta["null_count"] == 2

    def test_auto_detection_string_target(self, classification_df: pd.DataFrame) -> None:
        """String target should be auto-detected as classification."""
        ta = run_eda(classification_df, target_column="label")["target_analysis"]
        assert ta["task_type"] == "classification"


# ---------------------------------------------------------------------------
# Target-variable analysis — regression
# ---------------------------------------------------------------------------


class TestTargetRegression:
    def test_task_type_regression(self, regression_df: pd.DataFrame) -> None:
        ta = run_eda(
            regression_df, target_column="target", task_type="regression"
        )["target_analysis"]
        assert ta["task_type"] == "regression"

    def test_distribution_present(self, regression_df: pd.DataFrame) -> None:
        ta = run_eda(
            regression_df, target_column="target", task_type="regression"
        )["target_analysis"]
        assert "distribution" in ta
        dist = ta["distribution"]
        for key in ("count", "mean", "median", "std", "min", "max"):
            assert key in dist

    def test_distribution_mean_correct(self, regression_df: pd.DataFrame) -> None:
        ta = run_eda(
            regression_df, target_column="target", task_type="regression"
        )["target_analysis"]
        expected_mean = float(np.mean(np.arange(20) * 3.0 + 1.0))
        assert ta["distribution"]["mean"] == pytest.approx(expected_mean, rel=1e-6)

    def test_histogram_present(self, regression_df: pd.DataFrame) -> None:
        dist = run_eda(
            regression_df, target_column="target", task_type="regression"
        )["target_analysis"]["distribution"]
        assert "histogram" in dist
        assert sum(dist["histogram"]["counts"]) == 20

    def test_feature_means_present(self, regression_df: pd.DataFrame) -> None:
        ta = run_eda(
            regression_df, target_column="target", task_type="regression"
        )["target_analysis"]
        assert "feature_means" in ta
        assert "x1" in ta["feature_means"]
        assert "x2" in ta["feature_means"]

    def test_target_not_in_feature_means(self, regression_df: pd.DataFrame) -> None:
        ta = run_eda(
            regression_df, target_column="target", task_type="regression"
        )["target_analysis"]
        assert "target" not in ta["feature_means"]

    def test_auto_detection_high_cardinality_numeric(
        self, regression_df: pd.DataFrame
    ) -> None:
        """Numeric target with >20 unique values → auto-detected as regression."""
        ta = run_eda(regression_df, target_column="target")["target_analysis"]
        assert ta["task_type"] == "regression"

    def test_no_nan_in_distribution(self, regression_df: pd.DataFrame) -> None:
        dist = run_eda(
            regression_df, target_column="target", task_type="regression"
        )["target_analysis"]["distribution"]
        for val in dist.values():
            if isinstance(val, float):
                assert math.isfinite(val)


# ---------------------------------------------------------------------------
# run_eda convenience wrapper
# ---------------------------------------------------------------------------


class TestRunEdaWrapper:
    def test_equivalent_to_engine_analyze(self, mixed_df: pd.DataFrame) -> None:
        engine_result = EDAEngine().analyze(mixed_df)
        wrapper_result = run_eda(mixed_df)
        # Both should have the same top-level keys
        assert set(engine_result.keys()) == set(wrapper_result.keys())

    def test_custom_n_bins_passed_through(self) -> None:
        df = pd.DataFrame({"v": list(range(100))})
        result = run_eda(df, n_bins=5)
        hist = result["univariate"]["numeric"]["v"]["histogram"]
        assert len(hist["counts"]) == 5

    def test_custom_top_n_categories_passed_through(self) -> None:
        df = pd.DataFrame({"cat": [str(i) for i in range(30)]})
        result = run_eda(df, top_n_categories=3)
        top_vals = result["univariate"]["categorical"]["cat"]["top_values"]
        assert len(top_vals) <= 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_row_df(self) -> None:
        df = pd.DataFrame({"a": [1.0], "b": ["x"]})
        result = run_eda(df)
        assert result["univariate"]["numeric"]["a"]["count"] == 1

    def test_single_column_numeric_df(self) -> None:
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        result = run_eda(df)
        assert "v" in result["univariate"]["numeric"]
        assert result["bivariate"]["correlation"]["pearson"] == {}

    def test_all_same_values(self) -> None:
        """Constant column — std = 0, skewness/kurtosis may be None or 0."""
        df = pd.DataFrame({"const": [7.0] * 10})
        result = run_eda(df)
        stats = result["univariate"]["numeric"]["const"]
        assert stats["std"] == pytest.approx(0.0)
        assert stats["mean"] == pytest.approx(7.0)

    def test_large_scatter_sample_size_no_error(self) -> None:
        df = pd.DataFrame({"p": list(range(10)), "q": list(range(10, 20))})
        result = EDAEngine(scatter_sample_size=1000).analyze(df)
        pair = result["bivariate"]["scatter_data"]["p_vs_q"]
        assert len(pair["x"]) == 10  # Only 10 rows available

    def test_empty_groupby_when_only_high_cardinality_cats(self) -> None:
        df = pd.DataFrame(
            {
                "unique_id": [str(i) for i in range(100)],
                "val": list(range(100)),
            }
        )
        assert run_eda(df)["groupby"] == {}
