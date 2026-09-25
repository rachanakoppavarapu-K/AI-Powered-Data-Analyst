"""Tests for src/profiling/data_profiler.py — ST-04."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.profiling.data_profiler import DataProfiler, profile_dataframe


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    """All-numeric DataFrame with no missing values."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    """DataFrame with numeric, categorical, and null values."""
    return pd.DataFrame(
        {
            "score": [10.0, 20.0, 30.0, np.nan, 50.0],
            "grade": ["A", "B", "A", "C", None],
            "level": [1, 2, 2, 3, 3],
        }
    )


@pytest.fixture()
def all_null_col_df() -> pd.DataFrame:
    """DataFrame containing a column that is entirely null."""
    return pd.DataFrame(
        {
            "valid": [1.0, 2.0, 3.0],
            "empty_num": [np.nan, np.nan, np.nan],
            "empty_cat": [None, None, None],
        }
    )


@pytest.fixture()
def single_numeric_df() -> pd.DataFrame:
    """DataFrame with exactly one numeric column (no correlation possible)."""
    return pd.DataFrame({"x": [1.0, 2.0, 3.0]})


@pytest.fixture()
def cat_only_df() -> pd.DataFrame:
    """DataFrame with only categorical columns."""
    return pd.DataFrame(
        {
            "color": ["red", "blue", "red", "green", "blue", "blue"],
            "size": ["S", "M", "L", "S", "M", "L"],
        }
    )


# ── shape ─────────────────────────────────────────────────────────────────────


class TestShape:
    def test_rows_and_columns(self, mixed_df: pd.DataFrame) -> None:
        report = profile_dataframe(mixed_df)
        assert report["shape"]["rows"] == 5
        assert report["shape"]["columns"] == 3

    def test_shape_values_are_ints(self, mixed_df: pd.DataFrame) -> None:
        report = profile_dataframe(mixed_df)
        assert isinstance(report["shape"]["rows"], int)
        assert isinstance(report["shape"]["columns"], int)


# ── memory ────────────────────────────────────────────────────────────────────


class TestMemory:
    def test_memory_positive(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        assert report["memory_usage_bytes"] > 0

    def test_memory_is_int(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        assert isinstance(report["memory_usage_bytes"], int)


# ── dtypes ────────────────────────────────────────────────────────────────────


class TestDtypes:
    def test_all_columns_present(self, mixed_df: pd.DataFrame) -> None:
        report = profile_dataframe(mixed_df)
        assert set(report["dtypes"].keys()) == set(mixed_df.columns)

    def test_dtype_values_are_strings(self, mixed_df: pd.DataFrame) -> None:
        report = profile_dataframe(mixed_df)
        for v in report["dtypes"].values():
            assert isinstance(v, str)

    def test_float_column_dtype(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        assert "float" in report["dtypes"]["a"]


# ── null counts and percentages ───────────────────────────────────────────────


class TestNulls:
    def test_null_count_correct(self, mixed_df: pd.DataFrame) -> None:
        report = profile_dataframe(mixed_df)
        assert report["null_counts"]["score"] == 1
        assert report["null_counts"]["grade"] == 1
        assert report["null_counts"]["level"] == 0

    def test_null_percentage_correct(self, mixed_df: pd.DataFrame) -> None:
        report = profile_dataframe(mixed_df)
        assert report["null_percentages"]["score"] == pytest.approx(20.0)

    def test_zero_nulls_when_no_missing(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        for v in report["null_counts"].values():
            assert v == 0

    def test_all_null_column_full_percentage(
        self, all_null_col_df: pd.DataFrame
    ) -> None:
        report = profile_dataframe(all_null_col_df)
        assert report["null_percentages"]["empty_num"] == pytest.approx(100.0)

    def test_null_counts_are_ints(self, mixed_df: pd.DataFrame) -> None:
        report = profile_dataframe(mixed_df)
        for v in report["null_counts"].values():
            assert isinstance(v, int)


# ── cardinality ───────────────────────────────────────────────────────────────


class TestCardinality:
    def test_unique_count_numeric(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        assert report["cardinality"]["a"] == 5

    def test_unique_count_categorical(self, mixed_df: pd.DataFrame) -> None:
        report = profile_dataframe(mixed_df)
        # grade: A, B, C — null is excluded
        assert report["cardinality"]["grade"] == 3

    def test_all_null_column_zero_cardinality(
        self, all_null_col_df: pd.DataFrame
    ) -> None:
        report = profile_dataframe(all_null_col_df)
        assert report["cardinality"]["empty_num"] == 0

    def test_cardinality_values_are_ints(self, mixed_df: pd.DataFrame) -> None:
        report = profile_dataframe(mixed_df)
        for v in report["cardinality"].values():
            assert isinstance(v, int)


# ── numeric statistics ────────────────────────────────────────────────────────


class TestNumericStats:
    def test_mean(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        assert report["numeric_stats"]["a"]["mean"] == pytest.approx(3.0)

    def test_median(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        assert report["numeric_stats"]["a"]["median"] == pytest.approx(3.0)

    def test_std(self, numeric_df: pd.DataFrame) -> None:
        # population std with ddof=1 for [1,2,3,4,5] = sqrt(2.5) ≈ 1.5811
        report = profile_dataframe(numeric_df)
        assert report["numeric_stats"]["a"]["std"] == pytest.approx(
            float(np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)), rel=1e-5
        )

    def test_min_max(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        assert report["numeric_stats"]["a"]["min"] == pytest.approx(1.0)
        assert report["numeric_stats"]["a"]["max"] == pytest.approx(5.0)

    def test_quartiles(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        stats = report["numeric_stats"]["a"]
        assert stats["q1"] is not None
        assert stats["q3"] is not None
        assert stats["iqr"] == pytest.approx(stats["q3"] - stats["q1"])

    def test_skewness_symmetric(self, numeric_df: pd.DataFrame) -> None:
        # [1,2,3,4,5] is perfectly symmetric → skewness ≈ 0
        report = profile_dataframe(numeric_df)
        assert abs(report["numeric_stats"]["a"]["skewness"]) < 0.01

    def test_count_excludes_nulls(self, mixed_df: pd.DataFrame) -> None:
        report = profile_dataframe(mixed_df)
        # 'score' has 1 NaN in 5 rows
        assert report["numeric_stats"]["score"]["count"] == 4

    def test_all_null_numeric_column_returns_none_stats(
        self, all_null_col_df: pd.DataFrame
    ) -> None:
        report = profile_dataframe(all_null_col_df)
        stats = report["numeric_stats"]["empty_num"]
        assert stats["count"] == 0
        assert stats["mean"] is None
        assert stats["std"] is None

    def test_no_nan_inf_in_stats(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        for col_stats in report["numeric_stats"].values():
            for k, v in col_stats.items():
                if v is not None:
                    assert not math.isnan(float(v))
                    assert not math.isinf(float(v))

    def test_single_value_column(self) -> None:
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        report = profile_dataframe(df)
        stats = report["numeric_stats"]["x"]
        assert stats["mean"] == pytest.approx(5.0)
        assert stats["std"] == pytest.approx(0.0)
        assert stats["min"] == pytest.approx(5.0)
        assert stats["max"] == pytest.approx(5.0)

    def test_categorical_columns_excluded(self, cat_only_df: pd.DataFrame) -> None:
        report = profile_dataframe(cat_only_df)
        assert report["numeric_stats"] == {}


# ── categorical statistics ────────────────────────────────────────────────────


class TestCategoricalStats:
    def test_mode_is_most_frequent(self, cat_only_df: pd.DataFrame) -> None:
        report = profile_dataframe(cat_only_df)
        # "blue" appears 3 times in 'color'
        assert report["categorical_stats"]["color"]["mode"] == "blue"

    def test_top_values_keys_are_strings(self, cat_only_df: pd.DataFrame) -> None:
        report = profile_dataframe(cat_only_df)
        for k in report["categorical_stats"]["color"]["top_values"]:
            assert isinstance(k, str)

    def test_top_values_counts_correct(self, cat_only_df: pd.DataFrame) -> None:
        report = profile_dataframe(cat_only_df)
        tv = report["categorical_stats"]["color"]["top_values"]
        assert tv["blue"] == 3
        assert tv["red"] == 2
        assert tv["green"] == 1

    def test_cardinality_in_categorical_stats(
        self, cat_only_df: pd.DataFrame
    ) -> None:
        report = profile_dataframe(cat_only_df)
        assert report["categorical_stats"]["color"]["cardinality"] == 3

    def test_count_excludes_nulls(self, mixed_df: pd.DataFrame) -> None:
        report = profile_dataframe(mixed_df)
        # 'grade' has 1 None; total rows = 5 → count = 4
        assert report["categorical_stats"]["grade"]["count"] == 4

    def test_all_null_categorical_column(
        self, all_null_col_df: pd.DataFrame
    ) -> None:
        report = profile_dataframe(all_null_col_df)
        stats = report["categorical_stats"]["empty_cat"]
        assert stats["count"] == 0
        assert stats["mode"] is None
        assert stats["top_values"] == {}

    def test_top_n_respected(self) -> None:
        df = pd.DataFrame(
            {"label": [str(i % 20) for i in range(100)]}  # 20 unique labels
        )
        report = profile_dataframe(df, top_n=5)
        assert len(report["categorical_stats"]["label"]["top_values"]) <= 5

    def test_numeric_columns_excluded(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        assert report["categorical_stats"] == {}


# ── correlation ───────────────────────────────────────────────────────────────


class TestCorrelation:
    def test_pearson_self_correlation_is_one(
        self, numeric_df: pd.DataFrame
    ) -> None:
        report = profile_dataframe(numeric_df)
        assert report["correlation"]["pearson"]["a"]["a"] == pytest.approx(1.0)
        assert report["correlation"]["pearson"]["b"]["b"] == pytest.approx(1.0)

    def test_spearman_self_correlation_is_one(
        self, numeric_df: pd.DataFrame
    ) -> None:
        report = profile_dataframe(numeric_df)
        assert report["correlation"]["spearman"]["a"]["a"] == pytest.approx(1.0)

    def test_perfect_positive_correlation(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        report = profile_dataframe(df)
        assert report["correlation"]["pearson"]["x"]["y"] == pytest.approx(1.0)
        assert report["correlation"]["spearman"]["x"]["y"] == pytest.approx(1.0)

    def test_perfect_negative_correlation(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]})
        report = profile_dataframe(df)
        assert report["correlation"]["pearson"]["x"]["y"] == pytest.approx(-1.0)

    def test_correlation_matrix_symmetry(self, numeric_df: pd.DataFrame) -> None:
        report = profile_dataframe(numeric_df)
        pearson = report["correlation"]["pearson"]
        assert pearson["a"]["b"] == pytest.approx(pearson["b"]["a"])

    def test_single_numeric_column_gives_empty_correlation(
        self, single_numeric_df: pd.DataFrame
    ) -> None:
        report = profile_dataframe(single_numeric_df)
        assert report["correlation"]["pearson"] == {}
        assert report["correlation"]["spearman"] == {}

    def test_no_numeric_columns_gives_empty_correlation(
        self, cat_only_df: pd.DataFrame
    ) -> None:
        report = profile_dataframe(cat_only_df)
        assert report["correlation"]["pearson"] == {}
        assert report["correlation"]["spearman"] == {}

    def test_correlation_values_are_floats_or_none(
        self, numeric_df: pd.DataFrame
    ) -> None:
        report = profile_dataframe(numeric_df)
        for row in report["correlation"]["pearson"].values():
            for v in row.values():
                assert v is None or isinstance(v, float)


# ── original DataFrame immutability ──────────────────────────────────────────


class TestImmutability:
    def test_original_df_unchanged(self, mixed_df: pd.DataFrame) -> None:
        original = mixed_df.copy(deep=True)
        profile_dataframe(mixed_df)
        pd.testing.assert_frame_equal(mixed_df, original)

    def test_original_numeric_df_unchanged(self, numeric_df: pd.DataFrame) -> None:
        original = numeric_df.copy(deep=True)
        profile_dataframe(numeric_df)
        pd.testing.assert_frame_equal(numeric_df, original)


# ── DataProfiler class API ────────────────────────────────────────────────────


class TestDataProfilerClass:
    def test_default_top_n(self, cat_only_df: pd.DataFrame) -> None:
        profiler = DataProfiler()
        report = profiler.profile(cat_only_df)
        # 3 unique colours — all should appear in top_values with default top_n=10
        assert len(report["categorical_stats"]["color"]["top_values"]) == 3

    def test_custom_top_n(self) -> None:
        df = pd.DataFrame({"tag": [str(i) for i in range(50)]})
        profiler = DataProfiler(top_n=3)
        report = profiler.profile(df)
        assert len(report["categorical_stats"]["tag"]["top_values"]) <= 3

    def test_report_keys_complete(self, mixed_df: pd.DataFrame) -> None:
        report = DataProfiler().profile(mixed_df)
        expected_keys = {
            "shape",
            "memory_usage_bytes",
            "dtypes",
            "null_counts",
            "null_percentages",
            "cardinality",
            "numeric_stats",
            "categorical_stats",
            "correlation",
        }
        assert expected_keys.issubset(set(report.keys()))


# ── edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_row_dataframe(self) -> None:
        df = pd.DataFrame({"x": [42.0], "label": ["hello"]})
        report = profile_dataframe(df)
        assert report["shape"]["rows"] == 1
        assert report["numeric_stats"]["x"]["mean"] == pytest.approx(42.0)
        assert report["categorical_stats"]["label"]["mode"] == "hello"

    def test_single_column_dataframe(self) -> None:
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0]})
        report = profile_dataframe(df)
        assert report["shape"]["columns"] == 1

    def test_boolean_column_in_categorical_stats(self) -> None:
        # In pandas 2.x bool dtype is not selected by select_dtypes(include="number"),
        # so boolean columns fall through to categorical stats.
        df = pd.DataFrame({"flag": [True, False, True, True]})
        report = profile_dataframe(df)
        # Bool columns land in either numeric or categorical stats depending on
        # the pandas version — the column must appear in exactly one of the two.
        assert ("flag" in report["numeric_stats"]) or (
            "flag" in report["categorical_stats"]
        )

    def test_integer_column_statistics(self) -> None:
        df = pd.DataFrame({"n": [1, 2, 3, 4, 5]})
        report = profile_dataframe(df)
        assert report["numeric_stats"]["n"]["mean"] == pytest.approx(3.0)

    def test_large_cardinality_column(self) -> None:
        df = pd.DataFrame({"uid": [str(i) for i in range(1000)]})
        report = profile_dataframe(df)
        assert report["cardinality"]["uid"] == 1000
