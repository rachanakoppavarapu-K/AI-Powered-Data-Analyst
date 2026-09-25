"""Tests for src/preprocessing/preprocessor.py — ST-03."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.preprocessor import (
    PreprocessingError,
    detect_outliers_iqr,
    detect_outliers_zscore,
    encode_categoricals,
    extract_datetime_features,
    handle_missing_values,
    handle_outliers,
    remove_duplicates,
    run_preprocessing,
    scale_numerics,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    """Simple all-numeric DataFrame without missing values."""
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "c": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )


@pytest.fixture()
def df_with_missing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0, np.nan, 5.0],
            "cat": ["a", None, "b", "a", None],
        }
    )


@pytest.fixture()
def df_with_dupes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1, 2, 2, 3, 3],
            "y": ["a", "b", "b", "c", "c"],
        }
    )


@pytest.fixture()
def df_with_outliers() -> pd.DataFrame:
    """DataFrame where the last value in col 'v' is a clear outlier."""
    return pd.DataFrame({"v": [10.0, 11.0, 10.5, 11.5, 10.2, 200.0]})


@pytest.fixture()
def df_categorical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red", "blue"],
            "size": ["S", "M", "L", "M", "S"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture()
def df_datetime() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "created_date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "score": [10, 20, 30, 40],
        }
    )


# ── handle_missing_values ─────────────────────────────────────────────────────


class TestHandleMissingValues:
    def test_mean_fills_numeric(self, df_with_missing: pd.DataFrame) -> None:
        result, summary = handle_missing_values(df_with_missing, strategy="mean")
        assert result["num"].isnull().sum() == 0
        assert result["num"].iloc[1] == pytest.approx((1.0 + 3.0 + 5.0) / 3)

    def test_median_fills_numeric(self, df_with_missing: pd.DataFrame) -> None:
        result, summary = handle_missing_values(df_with_missing, strategy="median")
        assert result["num"].isnull().sum() == 0
        assert result["num"].iloc[1] == pytest.approx(3.0)

    def test_mode_fills_categorical(self, df_with_missing: pd.DataFrame) -> None:
        result, summary = handle_missing_values(df_with_missing, strategy="mode")
        assert result["cat"].isnull().sum() == 0
        # Mode of ["a", None, "b", "a", None] → "a"
        assert result["cat"].iloc[1] == "a"

    def test_drop_removes_rows(self, df_with_missing: pd.DataFrame) -> None:
        result, summary = handle_missing_values(df_with_missing, strategy="drop")
        assert result.isnull().sum().sum() == 0
        assert len(result) < len(df_with_missing)
        assert summary["rows_dropped"] == 3  # rows 1, 2 (cat), 4

    def test_constant_fill(self, df_with_missing: pd.DataFrame) -> None:
        result, summary = handle_missing_values(
            df_with_missing, strategy="constant", fill_value=-1
        )
        assert result["num"].isnull().sum() == 0
        assert result["num"].iloc[1] == -1.0

    def test_column_subset(self, df_with_missing: pd.DataFrame) -> None:
        result, summary = handle_missing_values(
            df_with_missing, strategy="mean", columns=["num"]
        )
        assert result["num"].isnull().sum() == 0
        # 'cat' column still has nulls because we only processed 'num'
        assert result["cat"].isnull().sum() > 0

    def test_invalid_strategy_raises(self, df_with_missing: pd.DataFrame) -> None:
        with pytest.raises(PreprocessingError, match="Unknown imputation strategy"):
            handle_missing_values(df_with_missing, strategy="interpolate")

    def test_original_not_modified(self, df_with_missing: pd.DataFrame) -> None:
        original_nulls = df_with_missing.isnull().sum().sum()
        handle_missing_values(df_with_missing, strategy="mean")
        assert df_with_missing.isnull().sum().sum() == original_nulls

    def test_summary_keys_mean(self, df_with_missing: pd.DataFrame) -> None:
        _, summary = handle_missing_values(df_with_missing, strategy="mean")
        assert "strategy" in summary
        assert "missing_before" in summary
        assert summary["strategy"] == "mean"

    def test_summary_keys_drop(self, df_with_missing: pd.DataFrame) -> None:
        _, summary = handle_missing_values(df_with_missing, strategy="drop")
        assert "rows_dropped" in summary

    def test_no_missing_data_noop(self, numeric_df: pd.DataFrame) -> None:
        result, summary = handle_missing_values(numeric_df, strategy="mean")
        assert result.equals(numeric_df)
        assert summary["cells_imputed"] == 0


# ── remove_duplicates ─────────────────────────────────────────────────────────


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self, df_with_dupes: pd.DataFrame) -> None:
        result, summary = remove_duplicates(df_with_dupes)
        assert len(result) == 3  # rows: (1,a), (2,b), (3,c)
        assert summary["duplicates_removed"] == 2

    def test_keep_last(self, df_with_dupes: pd.DataFrame) -> None:
        result, summary = remove_duplicates(df_with_dupes, keep="last")
        assert len(result) == 3

    def test_subset(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
        result, summary = remove_duplicates(df, subset=["a"])
        assert len(result) == 2
        assert summary["duplicates_removed"] == 1

    def test_original_not_modified(self, df_with_dupes: pd.DataFrame) -> None:
        original_len = len(df_with_dupes)
        remove_duplicates(df_with_dupes)
        assert len(df_with_dupes) == original_len

    def test_no_dupes_noop(self, numeric_df: pd.DataFrame) -> None:
        result, summary = remove_duplicates(numeric_df)
        assert len(result) == len(numeric_df)
        assert summary["duplicates_removed"] == 0

    def test_summary_structure(self, df_with_dupes: pd.DataFrame) -> None:
        _, summary = remove_duplicates(df_with_dupes)
        for key in ("rows_before", "rows_after", "duplicates_removed"):
            assert key in summary


# ── detect_outliers_iqr ───────────────────────────────────────────────────────


class TestDetectOutliersIQR:
    def test_identifies_outlier(self, df_with_outliers: pd.DataFrame) -> None:
        masks = detect_outliers_iqr(df_with_outliers, columns=["v"])
        assert masks["v"].iloc[-1] is np.bool_(True)  # 200.0 is an outlier

    def test_normal_values_not_flagged(self, numeric_df: pd.DataFrame) -> None:
        masks = detect_outliers_iqr(numeric_df, columns=["a"])
        assert masks["a"].sum() == 0

    def test_returns_dict_of_series(self, df_with_outliers: pd.DataFrame) -> None:
        masks = detect_outliers_iqr(df_with_outliers)
        assert isinstance(masks, dict)
        for col, mask in masks.items():
            assert isinstance(mask, pd.Series)

    def test_missing_column_raises(self, numeric_df: pd.DataFrame) -> None:
        with pytest.raises(PreprocessingError, match="not found"):
            detect_outliers_iqr(numeric_df, columns=["nonexistent"])


# ── detect_outliers_zscore ────────────────────────────────────────────────────


class TestDetectOutliersZScore:
    def test_identifies_extreme_outlier(self) -> None:
        # 10000.0 is unambiguously many standard deviations from the mean
        df = pd.DataFrame({"x": [1.0, 2.0, 1.5, 1.8, 2.2, 1.9, 1.7, 1.6, 2.1, 10000.0]})
        masks = detect_outliers_zscore(df, columns=["x"], threshold=3.0)
        assert bool(masks["x"].iloc[-1]) is True

    def test_tight_threshold_flags_more(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        masks_3 = detect_outliers_zscore(df, threshold=3.0)
        masks_1 = detect_outliers_zscore(df, threshold=1.0)
        assert masks_1["x"].sum() >= masks_3["x"].sum()

    def test_missing_column_raises(self, numeric_df: pd.DataFrame) -> None:
        with pytest.raises(PreprocessingError, match="not found"):
            detect_outliers_zscore(numeric_df, columns=["nonexistent"])


# ── handle_outliers ───────────────────────────────────────────────────────────


class TestHandleOutliers:
    def test_cap_clips_value(self, df_with_outliers: pd.DataFrame) -> None:
        result, summary = handle_outliers(
            df_with_outliers, method="iqr", action="cap", columns=["v"]
        )
        assert result["v"].max() < 200.0
        assert summary["action"] == "cap"

    def test_drop_removes_row(self, df_with_outliers: pd.DataFrame) -> None:
        result, summary = handle_outliers(
            df_with_outliers, method="iqr", action="drop", columns=["v"]
        )
        assert len(result) < len(df_with_outliers)

    def test_flag_adds_column(self, df_with_outliers: pd.DataFrame) -> None:
        result, summary = handle_outliers(
            df_with_outliers, method="iqr", action="flag", columns=["v"]
        )
        assert "is_outlier" in result.columns
        assert result["is_outlier"].sum() >= 1

    def test_zscore_cap(self) -> None:
        # 10000.0 is unambiguously a z-score outlier in this dataset
        df = pd.DataFrame({"x": [1.0, 2.0, 1.5, 1.8, 2.2, 1.9, 1.7, 1.6, 2.1, 10000.0]})
        result, summary = handle_outliers(df, method="zscore", action="cap", columns=["x"])
        assert result["x"].max() < 10000.0

    def test_original_not_modified(self, df_with_outliers: pd.DataFrame) -> None:
        original_max = df_with_outliers["v"].max()
        handle_outliers(df_with_outliers, method="iqr", action="cap", columns=["v"])
        assert df_with_outliers["v"].max() == original_max

    def test_invalid_method_raises(self, df_with_outliers: pd.DataFrame) -> None:
        with pytest.raises(PreprocessingError, match="Unknown outlier method"):
            handle_outliers(df_with_outliers, method="mad")

    def test_invalid_action_raises(self, df_with_outliers: pd.DataFrame) -> None:
        with pytest.raises(PreprocessingError, match="Unknown outlier action"):
            handle_outliers(df_with_outliers, action="ignore")

    def test_summary_keys(self, df_with_outliers: pd.DataFrame) -> None:
        _, summary = handle_outliers(df_with_outliers, method="iqr", action="cap")
        for key in ("method", "action", "columns", "outlier_counts_per_column"):
            assert key in summary


# ── encode_categoricals ───────────────────────────────────────────────────────


class TestEncodeCategoricals:
    def test_label_encodes_strings(self, df_categorical: pd.DataFrame) -> None:
        result, summary, encoders = encode_categoricals(df_categorical, method="label")
        assert result["color"].dtype in (np.int64, np.int32, int, object)
        # Values should be integers 0, 1, 2
        assert set(result["color"].unique()).issubset({0, 1, 2})

    def test_label_returns_encoders(self, df_categorical: pd.DataFrame) -> None:
        _, _, encoders = encode_categoricals(df_categorical, method="label")
        assert "color" in encoders
        assert "size" in encoders

    def test_onehot_expands_columns(self, df_categorical: pd.DataFrame) -> None:
        result, summary, encoders = encode_categoricals(df_categorical, method="onehot")
        assert "color" not in result.columns
        # Expect new one-hot columns
        assert any(c.startswith("color_") for c in result.columns)

    def test_onehot_drop_first(self, df_categorical: pd.DataFrame) -> None:
        result_full, _, _ = encode_categoricals(df_categorical, method="onehot", drop_first=False)
        result_drop, _, _ = encode_categoricals(df_categorical, method="onehot", drop_first=True)
        color_full = [c for c in result_full.columns if c.startswith("color_")]
        color_drop = [c for c in result_drop.columns if c.startswith("color_")]
        assert len(color_drop) == len(color_full) - 1

    def test_column_subset(self, df_categorical: pd.DataFrame) -> None:
        result, summary, _ = encode_categoricals(
            df_categorical, method="label", columns=["color"]
        )
        # 'size' column should remain as-is (string)
        assert result["size"].dtype == object

    def test_original_not_modified(self, df_categorical: pd.DataFrame) -> None:
        original_dtype = df_categorical["color"].dtype
        encode_categoricals(df_categorical, method="label")
        assert df_categorical["color"].dtype == original_dtype

    def test_invalid_method_raises(self, df_categorical: pd.DataFrame) -> None:
        with pytest.raises(PreprocessingError, match="Unknown encoding method"):
            encode_categoricals(df_categorical, method="binary")

    def test_no_categorical_cols_noop(self, numeric_df: pd.DataFrame) -> None:
        result, summary, _ = encode_categoricals(numeric_df, method="label")
        assert result.equals(numeric_df)
        assert summary["columns_encoded"] == []

    def test_summary_label(self, df_categorical: pd.DataFrame) -> None:
        _, summary, _ = encode_categoricals(df_categorical, method="label")
        assert summary["method"] == "label"
        assert "columns_encoded" in summary

    def test_summary_onehot(self, df_categorical: pd.DataFrame) -> None:
        _, summary, _ = encode_categoricals(df_categorical, method="onehot")
        assert summary["method"] == "onehot"
        assert "new_columns" in summary


# ── scale_numerics ────────────────────────────────────────────────────────────


class TestScaleNumerics:
    def test_standard_scaler_mean_zero(self, numeric_df: pd.DataFrame) -> None:
        result, summary, scaler = scale_numerics(numeric_df, method="standard")
        assert result["a"].mean() == pytest.approx(0.0, abs=1e-10)
        assert result["a"].std(ddof=0) == pytest.approx(1.0, abs=1e-10)

    def test_minmax_range(self, numeric_df: pd.DataFrame) -> None:
        result, summary, scaler = scale_numerics(numeric_df, method="minmax")
        assert result["a"].min() == pytest.approx(0.0)
        assert result["a"].max() == pytest.approx(1.0)

    def test_robust_scaler_runs(self, numeric_df: pd.DataFrame) -> None:
        result, summary, scaler = scale_numerics(numeric_df, method="robust")
        assert result.shape == numeric_df.shape

    def test_returns_fitted_scaler(self, numeric_df: pd.DataFrame) -> None:
        from sklearn.preprocessing import StandardScaler as SK_SS

        _, _, scaler = scale_numerics(numeric_df, method="standard")
        assert isinstance(scaler, SK_SS)

    def test_column_subset(self, numeric_df: pd.DataFrame) -> None:
        result, summary, _ = scale_numerics(numeric_df, method="standard", columns=["a"])
        # 'b' and 'c' should be unchanged
        pd.testing.assert_series_equal(result["b"], numeric_df["b"])
        pd.testing.assert_series_equal(result["c"], numeric_df["c"])

    def test_original_not_modified(self, numeric_df: pd.DataFrame) -> None:
        original = numeric_df.copy()
        scale_numerics(numeric_df, method="standard")
        pd.testing.assert_frame_equal(numeric_df, original)

    def test_invalid_method_raises(self, numeric_df: pd.DataFrame) -> None:
        with pytest.raises(PreprocessingError, match="Unknown scaling method"):
            scale_numerics(numeric_df, method="normalizer")

    def test_summary_structure(self, numeric_df: pd.DataFrame) -> None:
        _, summary, _ = scale_numerics(numeric_df, method="standard")
        assert "method" in summary
        assert "columns_scaled" in summary
        assert summary["method"] == "standard"


# ── extract_datetime_features ─────────────────────────────────────────────────


class TestExtractDatetimeFeatures:
    def test_extracts_year_month_day(self, df_datetime: pd.DataFrame) -> None:
        result, summary = extract_datetime_features(df_datetime, columns=["created_date"])
        assert "created_date_year" in result.columns
        assert "created_date_month" in result.columns
        assert "created_date_day" in result.columns

    def test_extracts_dayofweek(self, df_datetime: pd.DataFrame) -> None:
        result, _ = extract_datetime_features(df_datetime, columns=["created_date"])
        assert "created_date_dayofweek" in result.columns
        # 2024-01-01 is a Monday → dayofweek == 0
        assert result["created_date_dayofweek"].iloc[0] == 0

    def test_drop_original(self, df_datetime: pd.DataFrame) -> None:
        result, _ = extract_datetime_features(
            df_datetime, columns=["created_date"], drop_original=True
        )
        assert "created_date" not in result.columns

    def test_keep_original_by_default(self, df_datetime: pd.DataFrame) -> None:
        result, _ = extract_datetime_features(df_datetime, columns=["created_date"])
        assert "created_date" in result.columns

    def test_auto_detect_datetime_col(self, df_datetime: pd.DataFrame) -> None:
        # Auto-detection picks up datetime64 columns
        result, summary = extract_datetime_features(df_datetime)
        assert "created_date" in summary["columns_processed"]

    def test_string_date_column(self) -> None:
        df = pd.DataFrame({"order_date": ["2023-01-15", "2023-06-20"], "qty": [5, 10]})
        result, summary = extract_datetime_features(df, columns=["order_date"])
        assert "order_date_year" in result.columns
        assert result["order_date_year"].iloc[0] == 2023

    def test_original_not_modified(self, df_datetime: pd.DataFrame) -> None:
        original_cols = list(df_datetime.columns)
        extract_datetime_features(df_datetime, columns=["created_date"])
        assert list(df_datetime.columns) == original_cols

    def test_summary_keys(self, df_datetime: pd.DataFrame) -> None:
        _, summary = extract_datetime_features(df_datetime, columns=["created_date"])
        assert "columns_processed" in summary
        assert "features_extracted" in summary
        assert "created_date" in summary["features_extracted"]

    def test_non_parseable_col_skipped(self) -> None:
        df = pd.DataFrame({"text_col": ["abc", "def"], "val": [1, 2]})
        # Should silently skip columns that cannot be parsed as dates
        result, summary = extract_datetime_features(df, columns=["text_col"])
        # No new columns added; no exception raised
        assert "text_col_year" not in result.columns


# ── run_preprocessing (integration) ──────────────────────────────────────────


class TestRunPreprocessing:
    def test_returns_dataframe_and_summary(self) -> None:
        df = pd.DataFrame(
            {
                "age": [25.0, np.nan, 35.0, 40.0, 28.0],
                "income": [50000.0, 60000.0, np.nan, 80000.0, 55000.0],
                "city": ["NY", "LA", "NY", "SF", None],
            }
        )
        result, summary = run_preprocessing(df)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(summary, dict)
        assert not result.isnull().any().any()

    def test_original_not_modified(self) -> None:
        df = pd.DataFrame(
            {"x": [1.0, 2.0, np.nan], "cat": ["a", "b", "a"]}
        )
        original_nulls = df.isnull().sum().sum()
        run_preprocessing(df)
        assert df.isnull().sum().sum() == original_nulls

    def test_summary_records_steps(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
        _, summary = run_preprocessing(df)
        assert "steps" in summary
        assert "missing_values" in summary["steps"]
        assert "encoding" in summary["steps"]
        assert "scaling" in summary["steps"]

    def test_duplicate_removal_step(self) -> None:
        df = pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": ["x", "x", "y"]})
        _, summary = run_preprocessing(df, remove_dupes=True)
        assert "duplicates" in summary

    def test_skip_duplicate_removal(self) -> None:
        df = pd.DataFrame({"a": [1.0, 1.0, 2.0], "b": ["x", "x", "y"]})
        _, summary = run_preprocessing(df, remove_dupes=False)
        assert "duplicates" not in summary

    def test_skip_outlier_handling(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
        _, summary = run_preprocessing(df, outlier_method=None)
        assert "outliers" not in summary

    def test_all_numeric_dataset(self, numeric_df: pd.DataFrame) -> None:
        result, summary = run_preprocessing(numeric_df)
        assert result.shape[0] == numeric_df.shape[0]
        assert result.isnull().sum().sum() == 0

    def test_summary_has_shapes(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        _, summary = run_preprocessing(df)
        assert "original_shape" in summary
        assert "final_shape" in summary
        assert summary["original_shape"] == [2, 2]

    def test_onehot_encoding_option(self) -> None:
        df = pd.DataFrame({"cat": ["a", "b", "a"], "val": [1.0, 2.0, 3.0]})
        result, summary = run_preprocessing(df, encoding_method="onehot")
        assert "cat" not in result.columns
        assert any(c.startswith("cat_") for c in result.columns)

    def test_minmax_scaling_option(self, numeric_df: pd.DataFrame) -> None:
        result, _ = run_preprocessing(numeric_df, scaling_method="minmax")
        # After minmax scaling every column should be in [0, 1]
        assert result.min().min() == pytest.approx(0.0)
        assert result.max().max() == pytest.approx(1.0)

    def test_datetime_extraction_in_pipeline(self) -> None:
        df = pd.DataFrame(
            {
                "signup_date": pd.date_range("2023-01-01", periods=3, freq="ME"),
                "amount": [100.0, 200.0, 300.0],
            }
        )
        result, summary = run_preprocessing(df, datetime_columns=["signup_date"])
        assert "signup_date_year" in result.columns
        assert "datetime_extraction" in summary
