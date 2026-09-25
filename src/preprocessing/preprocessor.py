"""Preprocessing module — cleans and transforms raw DataFrames into ML-ready form.

All operations are non-destructive: the original DataFrame is never modified.
Each public function returns a transformed copy and a structured
``preprocessing_summary`` dict that records every transformation performed,
suitable for injection into GenAI prompts or audit logs.

Leakage prevention
------------------
Scalers, encoders, and imputers are *fitted* on the data passed in.  When the
caller needs to apply the same fitted transformers to a held-out test set (i.e.
to avoid leakage), they should fit on training data and call ``transform`` on
the returned transformer objects.  The high-level :func:`run_preprocessing`
convenience function is intended for exploratory / single-dataset use only.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

__all__ = [
    "handle_missing_values",
    "remove_duplicates",
    "detect_outliers_iqr",
    "detect_outliers_zscore",
    "handle_outliers",
    "encode_categoricals",
    "scale_numerics",
    "extract_datetime_features",
    "run_preprocessing",
    "PreprocessingError",
]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PreprocessingError(ValueError):
    """Raised when a preprocessing step cannot be completed."""


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Summary = dict[str, Any]

# ---------------------------------------------------------------------------
# 1. Missing-value handling
# ---------------------------------------------------------------------------

_VALID_IMPUTE_STRATEGIES = frozenset({"mean", "median", "mode", "drop", "constant"})


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    fill_value: Any = 0,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, Summary]:
    """Impute or remove missing values.

    Parameters
    ----------
    df:
        Input DataFrame.  Not modified.
    strategy:
        One of ``"mean"``, ``"median"``, ``"mode"``, ``"drop"``,
        ``"constant"``.
    fill_value:
        Value used when *strategy* is ``"constant"``.
    columns:
        Subset of columns to operate on.  Defaults to all columns.

    Returns
    -------
    tuple[pd.DataFrame, Summary]
        Transformed copy and a summary dict.
    """
    if strategy not in _VALID_IMPUTE_STRATEGIES:
        raise PreprocessingError(
            f"Unknown imputation strategy '{strategy}'. "
            f"Choose from {sorted(_VALID_IMPUTE_STRATEGIES)}."
        )

    df = df.copy()
    cols = columns if columns is not None else df.columns.tolist()
    missing_before = df[cols].isnull().sum().to_dict()
    total_before = sum(missing_before.values())

    if strategy == "drop":
        rows_before = len(df)
        df = df.dropna(subset=cols)
        rows_dropped = rows_before - len(df)
        summary: Summary = {
            "strategy": "drop",
            "columns": cols,
            "missing_before": missing_before,
            "rows_dropped": rows_dropped,
        }
        return df, summary

    if strategy == "constant":
        df[cols] = df[cols].fillna(fill_value)
        summary = {
            "strategy": "constant",
            "fill_value": fill_value,
            "columns": cols,
            "missing_before": missing_before,
            "cells_filled": total_before,
        }
        return df, summary

    # mean / median / mode — use sklearn SimpleImputer for consistency
    sklearn_strategy = "most_frequent" if strategy == "mode" else strategy
    # Exclude datetime columns — SimpleImputer cannot handle them
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [
        c for c in cols
        if not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]

    if numeric_cols:
        imputer = SimpleImputer(strategy=sklearn_strategy)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    if cat_cols:
        # Convert Python None → np.nan so SimpleImputer recognises them,
        # then apply most_frequent imputation for non-numeric columns.
        df[cat_cols] = df[cat_cols].where(
            df[cat_cols].notna(), other=np.nan
        )
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols].astype(object))

    missing_after = df[cols].isnull().sum().to_dict()
    summary = {
        "strategy": strategy,
        "columns": cols,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "cells_imputed": total_before,
    }
    return df, summary


# ---------------------------------------------------------------------------
# 2. Duplicate-row removal
# ---------------------------------------------------------------------------


def remove_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, Summary]:
    """Remove duplicate rows.

    Parameters
    ----------
    df:
        Input DataFrame.  Not modified.
    subset:
        Column names to consider.  ``None`` means all columns.
    keep:
        ``"first"`` (default), ``"last"``, or ``False`` to drop all dupes.

    Returns
    -------
    tuple[pd.DataFrame, Summary]
    """
    rows_before = len(df)
    df = df.copy()
    df = df.drop_duplicates(subset=subset, keep=keep)  # type: ignore[arg-type]
    rows_after = len(df)
    summary: Summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": rows_before - rows_after,
        "subset": subset,
        "keep": keep,
    }
    return df, summary


# ---------------------------------------------------------------------------
# 3. Outlier detection
# ---------------------------------------------------------------------------


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> dict[str, pd.Series]:
    """Return boolean masks (True = outlier) per column using the IQR method.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numeric columns to analyse.  Defaults to all numeric columns.
    factor:
        Multiplier applied to IQR (default ``1.5``).

    Returns
    -------
    dict[str, pd.Series]
        Mapping ``column_name → boolean Series``.
    """
    cols = columns if columns is not None else df.select_dtypes(include="number").columns.tolist()
    masks: dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            raise PreprocessingError(f"Column '{col}' not found in DataFrame.")
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        masks[col] = (df[col] < lower) | (df[col] > upper)
    return masks


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 3.0,
) -> dict[str, pd.Series]:
    """Return boolean masks (True = outlier) per column using the Z-score method.

    Parameters
    ----------
    df:
        Input DataFrame.
    columns:
        Numeric columns to analyse.  Defaults to all numeric columns.
    threshold:
        Absolute Z-score threshold (default ``3.0``).

    Returns
    -------
    dict[str, pd.Series]
        Mapping ``column_name → boolean Series``.
    """
    cols = columns if columns is not None else df.select_dtypes(include="number").columns.tolist()
    masks: dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            raise PreprocessingError(f"Column '{col}' not found in DataFrame.")
        series = df[col]
        # Use median + MAD to avoid masking effect from extreme outliers
        median_ = float(np.nanmedian(series))
        mad = float(np.nanmedian(np.abs(series - median_)))
        # Scale MAD to be consistent with std (factor 1.4826 for normal dist.)
        mad_scaled = mad * 1.4826 if mad > 0.0 else float(np.nanstd(series, ddof=0))
        if mad_scaled == 0.0:
            z = np.zeros(len(series))
        else:
            z = np.abs((series.fillna(median_) - median_) / mad_scaled)
        masks[col] = pd.Series(z.to_numpy() >= threshold, index=df.index)
    return masks


def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    action: str = "cap",
    columns: list[str] | None = None,
    factor: float = 1.5,
    threshold: float = 3.0,
) -> tuple[pd.DataFrame, Summary]:
    """Detect and handle outliers.

    Parameters
    ----------
    df:
        Input DataFrame.  Not modified.
    method:
        ``"iqr"`` or ``"zscore"``.
    action:
        ``"flag"`` (add a boolean column), ``"cap"`` (Winsorize to bounds),
        or ``"drop"`` (remove rows containing outliers).
    columns:
        Numeric columns to process.  Defaults to all numeric columns.
    factor:
        IQR multiplier (only used when *method* is ``"iqr"``).
    threshold:
        Z-score threshold (only used when *method* is ``"zscore"``).

    Returns
    -------
    tuple[pd.DataFrame, Summary]
    """
    if method not in ("iqr", "zscore"):
        raise PreprocessingError(f"Unknown outlier method '{method}'. Use 'iqr' or 'zscore'.")
    if action not in ("flag", "cap", "drop"):
        raise PreprocessingError(f"Unknown outlier action '{action}'. Use 'flag', 'cap', or 'drop'.")

    df = df.copy()
    cols = columns if columns is not None else df.select_dtypes(include="number").columns.tolist()

    if method == "iqr":
        masks = detect_outliers_iqr(df, cols, factor=factor)
    else:
        masks = detect_outliers_zscore(df, cols, threshold=threshold)

    outlier_counts: dict[str, int] = {c: int(m.sum()) for c, m in masks.items()}
    combined_mask = pd.Series(False, index=df.index)
    for m in masks.values():
        combined_mask = combined_mask | m

    if action == "flag":
        df["is_outlier"] = combined_mask.astype(int)

    elif action == "cap":
        for col in cols:
            series = df[col].dropna()
            if method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - factor * iqr, q3 + factor * iqr
            else:
                # Use median + MAD consistent with detect_outliers_zscore
                median_ = float(np.nanmedian(series))
                mad = float(np.nanmedian(np.abs(series - median_)))
                mad_scaled = mad * 1.4826 if mad > 0.0 else float(np.nanstd(series, ddof=0))
                if mad_scaled == 0.0:
                    continue  # no variation — nothing to cap
                lower = median_ - threshold * mad_scaled
                upper = median_ + threshold * mad_scaled
            df[col] = df[col].clip(lower=lower, upper=upper)

    elif action == "drop":
        rows_before = len(df)
        df = df[~combined_mask]
        outlier_counts["rows_dropped"] = rows_before - len(df)

    summary: Summary = {
        "method": method,
        "action": action,
        "columns": cols,
        "outlier_counts_per_column": outlier_counts,
        "total_outlier_rows": int(combined_mask.sum()),
    }
    return df, summary


# ---------------------------------------------------------------------------
# 4. Categorical encoding
# ---------------------------------------------------------------------------


def encode_categoricals(
    df: pd.DataFrame,
    method: str = "label",
    columns: list[str] | None = None,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, Summary, dict[str, Any]]:
    """Encode categorical / object columns.

    Parameters
    ----------
    df:
        Input DataFrame.  Not modified.
    method:
        ``"label"`` (LabelEncoder) or ``"onehot"`` (OneHotEncoder).
    columns:
        Categorical columns to encode.  Defaults to all object/category columns.
    drop_first:
        Drop the first dummy column to avoid multicollinearity (only applies to
        ``"onehot"``).

    Returns
    -------
    tuple[pd.DataFrame, Summary, dict[str, Any]]
        Transformed copy, summary dict, and fitted encoder(s) keyed by column
        name so callers can apply the same encoding to new data.
    """
    if method not in ("label", "onehot"):
        raise PreprocessingError(f"Unknown encoding method '{method}'. Use 'label' or 'onehot'.")

    df = df.copy()
    cols = (
        columns
        if columns is not None
        else df.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    # Only encode columns that are actually in the DataFrame
    cols = [c for c in cols if c in df.columns]

    encoders: dict[str, Any] = {}
    new_columns: list[str] = []
    encoded_columns: list[str] = []

    if method == "label":
        for col in cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            encoded_columns.append(col)

        summary: Summary = {
            "method": "label",
            "columns_encoded": encoded_columns,
        }

    else:  # onehot
        ohe = OneHotEncoder(
            sparse_output=False,
            drop="first" if drop_first else None,
            handle_unknown="ignore",
        )
        if cols:
            encoded_array = ohe.fit_transform(df[cols].astype(str))
            feature_names = ohe.get_feature_names_out(cols).tolist()
            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
            df = df.drop(columns=cols)
            df = pd.concat([df, encoded_df], axis=1)
            encoders["onehot_encoder"] = ohe
            new_columns = feature_names

        summary = {
            "method": "onehot",
            "original_columns": cols,
            "new_columns": new_columns,
            "drop_first": drop_first,
        }

    return df, summary, encoders


# ---------------------------------------------------------------------------
# 5. Numerical scaling
# ---------------------------------------------------------------------------

_SCALER_MAP: dict[str, type] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def scale_numerics(
    df: pd.DataFrame,
    method: str = "standard",
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, Summary, Any]:
    """Scale numeric columns.

    Parameters
    ----------
    df:
        Input DataFrame.  Not modified.
    method:
        ``"standard"`` (StandardScaler), ``"minmax"`` (MinMaxScaler), or
        ``"robust"`` (RobustScaler).
    columns:
        Numeric columns to scale.  Defaults to all numeric columns.

    Returns
    -------
    tuple[pd.DataFrame, Summary, fitted_scaler]
        Transformed copy, summary dict, and the fitted scaler so it can be
        reused on test data without leakage.
    """
    if method not in _SCALER_MAP:
        raise PreprocessingError(
            f"Unknown scaling method '{method}'. "
            f"Choose from {sorted(_SCALER_MAP.keys())}."
        )

    df = df.copy()
    cols = (
        columns
        if columns is not None
        else df.select_dtypes(include="number").columns.tolist()
    )
    cols = [c for c in cols if c in df.columns]

    scaler = _SCALER_MAP[method]()
    if cols:
        df[cols] = scaler.fit_transform(df[cols])

    summary: Summary = {
        "method": method,
        "columns_scaled": cols,
    }
    return df, summary, scaler


# ---------------------------------------------------------------------------
# 6. Datetime feature extraction
# ---------------------------------------------------------------------------


def extract_datetime_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    drop_original: bool = False,
) -> tuple[pd.DataFrame, Summary]:
    """Extract year, month, day, day-of-week, hour, etc. from datetime columns.

    Parameters
    ----------
    df:
        Input DataFrame.  Not modified.
    columns:
        Columns to parse as datetimes.  Defaults to all columns whose dtype
        is already ``datetime64`` or whose name contains common date keywords.
    drop_original:
        Whether to drop the source datetime column after extraction.

    Returns
    -------
    tuple[pd.DataFrame, Summary]
    """
    df = df.copy()

    if columns is None:
        # Auto-detect: existing datetime columns + heuristic name match
        dt_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
        keywords = ("date", "time", "datetime", "timestamp", "created", "updated")
        name_cols = [
            c for c in df.columns
            if any(kw in c.lower() for kw in keywords) and c not in dt_cols
        ]
        columns = dt_cols + name_cols

    extracted: dict[str, list[str]] = {}
    for col in columns:
        if col not in df.columns:
            continue
        try:
            dt_series = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            continue

        if dt_series.isnull().all():
            continue

        prefix = col
        new_cols: list[str] = []
        for attr, feat_name in [
            ("year", f"{prefix}_year"),
            ("month", f"{prefix}_month"),
            ("day", f"{prefix}_day"),
            ("dayofweek", f"{prefix}_dayofweek"),
            ("hour", f"{prefix}_hour"),
            ("quarter", f"{prefix}_quarter"),
        ]:
            df[feat_name] = getattr(dt_series.dt, attr)
            new_cols.append(feat_name)

        if drop_original:
            df = df.drop(columns=[col])

        extracted[col] = new_cols

    summary: Summary = {
        "columns_processed": list(extracted.keys()),
        "features_extracted": extracted,
        "drop_original": drop_original,
    }
    return df, summary


# ---------------------------------------------------------------------------
# 7. High-level pipeline
# ---------------------------------------------------------------------------


def run_preprocessing(
    df: pd.DataFrame,
    *,
    missing_strategy: str = "mean",
    missing_fill_value: Any = 0,
    remove_dupes: bool = True,
    outlier_method: str | None = "iqr",
    outlier_action: str = "cap",
    encoding_method: str = "label",
    scaling_method: str = "standard",
    datetime_columns: list[str] | None = None,
    drop_datetime_originals: bool = False,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, Summary]:
    """Run the full preprocessing pipeline in a sensible default order.

    Each step is optional and can be disabled by passing the appropriate flag.
    The function is intended for exploratory / single-dataset use.  For
    train/test scenarios, use the individual functions and reuse the fitted
    transformers.

    Parameters
    ----------
    df:
        Raw input DataFrame.  Not modified.
    missing_strategy:
        Imputation strategy passed to :func:`handle_missing_values`.
    missing_fill_value:
        Fill value for constant imputation.
    remove_dupes:
        Whether to call :func:`remove_duplicates`.
    outlier_method:
        ``"iqr"``, ``"zscore"``, or ``None`` to skip outlier handling.
    outlier_action:
        ``"flag"``, ``"cap"``, or ``"drop"``.
    encoding_method:
        ``"label"`` or ``"onehot"``.
    scaling_method:
        ``"standard"``, ``"minmax"``, or ``"robust"``.
    datetime_columns:
        Columns to parse as datetimes.  ``None`` = auto-detect.
    drop_datetime_originals:
        Whether to drop source datetime columns after extraction.
    numeric_columns:
        Explicit numeric column list for scaling.  ``None`` = auto-detect.
    categorical_columns:
        Explicit categorical column list for encoding.  ``None`` = auto-detect.

    Returns
    -------
    tuple[pd.DataFrame, Summary]
        ML-ready DataFrame and a combined ``preprocessing_summary`` dict.
    """
    pipeline_summary: Summary = {
        "steps": [],
        "original_shape": list(df.shape),
    }

    # Step 1: datetime features (before missing-value handling, to avoid type confusion)
    df, dt_summary = extract_datetime_features(
        df, columns=datetime_columns, drop_original=drop_datetime_originals
    )
    if dt_summary["columns_processed"]:
        pipeline_summary["steps"].append("datetime_extraction")
        pipeline_summary["datetime_extraction"] = dt_summary

    # Step 2: missing values
    df, mv_summary = handle_missing_values(
        df, strategy=missing_strategy, fill_value=missing_fill_value
    )
    pipeline_summary["steps"].append("missing_values")
    pipeline_summary["missing_values"] = mv_summary

    # Step 3: duplicates
    if remove_dupes:
        df, dup_summary = remove_duplicates(df)
        pipeline_summary["steps"].append("duplicates")
        pipeline_summary["duplicates"] = dup_summary

    # Step 4: outliers
    if outlier_method is not None:
        df, out_summary = handle_outliers(
            df, method=outlier_method, action=outlier_action, columns=numeric_columns
        )
        pipeline_summary["steps"].append("outliers")
        pipeline_summary["outliers"] = out_summary

    # Step 5: encoding
    df, enc_summary, _ = encode_categoricals(
        df, method=encoding_method, columns=categorical_columns
    )
    pipeline_summary["steps"].append("encoding")
    pipeline_summary["encoding"] = enc_summary

    # Step 6: scaling
    df, scale_summary, _ = scale_numerics(
        df, method=scaling_method, columns=numeric_columns
    )
    pipeline_summary["steps"].append("scaling")
    pipeline_summary["scaling"] = scale_summary

    pipeline_summary["final_shape"] = list(df.shape)
    return df, pipeline_summary
