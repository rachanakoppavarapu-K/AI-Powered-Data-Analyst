"""
Rachana_AI_Data_Analyst.py
==========================
Single-file, submission-ready implementation of the AI-Powered Data Analyst.

Covers:
  - Dataset loading (CSV / Excel)
  - Data preprocessing (imputation, encoding, scaling, outliers)
  - Data profiling (statistics, null analysis, correlation)
  - EDA (univariate, bivariate, group-by, target analysis)
  - Visualization (histogram, bar, pie, heatmap, scatter, box, violin, ROC,
                   confusion matrix, feature importance, cluster scatter)
  - ML – classification, regression, clustering
  - ML evaluation metrics
  - Feature importance (model-native + permutation)
  - Natural Language Query processing
  - IBM watsonx.ai GenAI integration with safe credential handling
  - Report generation (HTML)
  - Streamlit dashboard

Entry point:
    streamlit run Rachana_AI_Data_Analyst.py

Environment variables (optional – all GenAI features degrade gracefully):
    WATSONX_API_KEY
    WATSONX_PROJECT_ID
    WATSONX_URL
    WATSONX_MODEL_ID
    MAX_UPLOAD_MB
"""

from __future__ import annotations

# ============================================================
# stdlib
# ============================================================
import base64
import csv
import io
import logging
import math
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence, Union

# ============================================================
# Load .env (best-effort)
# ============================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

# ============================================================
# Third-party imports
# ============================================================
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

import streamlit as st

logger = logging.getLogger(__name__)

# ============================================================
# ============================================================
#  SECTION 1 – DATA LOADING
# ============================================================
# ============================================================

_DEFAULT_MAX_MB: int = 50
_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".xlsx", ".xls"})


class DataLoadError(ValueError):
    """Raised when a file cannot be loaded or fails validation."""


class DataLoader:
    """Load and validate a CSV or Excel file into a :class:`pd.DataFrame`."""

    def __init__(self, max_mb: int | None = None) -> None:
        if max_mb is None:
            max_mb = int(os.getenv("MAX_UPLOAD_MB", _DEFAULT_MAX_MB))
        self.max_mb: int = max_mb

    def load(self, file: Union[str, Path, io.IOBase, object]) -> pd.DataFrame:
        name, raw = self._extract_name_and_bytes(file)
        ext = Path(name).suffix.lower()
        self._validate_extension(ext, name)
        self._validate_size(raw, name)
        df = self._parse(raw, ext, name)
        df = self._normalise_columns(df)
        self._validate_dataframe(df, name)
        return df

    def _extract_name_and_bytes(self, file: Any) -> tuple[str, bytes]:
        if isinstance(file, (str, Path)):
            path = Path(file)
            return path.name, path.read_bytes()
        if hasattr(file, "read"):
            raw: bytes = file.read()
            if hasattr(file, "seek"):
                file.seek(0)
            name: str = getattr(file, "name", "upload")
            return name, raw
        raise DataLoadError(f"Unsupported file input type: {type(file).__name__}.")

    def _validate_extension(self, ext: str, name: str) -> None:
        if ext not in _SUPPORTED_EXTENSIONS:
            raise DataLoadError(
                f"Unsupported file type '{ext}' for '{name}'. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

    def _validate_size(self, raw: bytes, name: str) -> None:
        size_mb = len(raw) / (1024 * 1024)
        if size_mb > self.max_mb:
            raise DataLoadError(
                f"File '{name}' is {size_mb:.1f} MB, exceeds {self.max_mb} MB limit."
            )

    def _parse(self, raw: bytes, ext: str, name: str) -> pd.DataFrame:
        buf = io.BytesIO(raw)
        try:
            if ext == ".csv":
                sample = raw[:4096].decode("utf-8", errors="replace")
                sep = self._detect_delimiter(sample)
                return pd.read_csv(buf, sep=sep, engine="python")
            else:
                return pd.read_excel(buf, engine="openpyxl")
        except Exception as exc:
            raise DataLoadError(f"Failed to parse '{name}': {exc}") from exc

    @staticmethod
    def _detect_delimiter(sample: str) -> str:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            return ","

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = (
            pd.Index(df.columns)
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9]+", "_", regex=True)
            .str.strip("_")
        )
        return df

    def _validate_dataframe(self, df: pd.DataFrame, name: str) -> None:
        if df.empty:
            raise DataLoadError(f"'{name}' produced an empty DataFrame.")
        if len(df.columns) == 0:
            raise DataLoadError(f"'{name}' has no columns after parsing.")


# ============================================================
# ============================================================
#  SECTION 2 – PREPROCESSING
# ============================================================
# ============================================================

class PreprocessingError(ValueError):
    """Raised when a preprocessing step cannot be completed."""


Summary = dict[str, Any]
_VALID_IMPUTE_STRATEGIES = frozenset({"mean", "median", "mode", "drop", "constant"})


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    fill_value: Any = 0,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, Summary]:
    if strategy not in _VALID_IMPUTE_STRATEGIES:
        raise PreprocessingError(f"Unknown imputation strategy '{strategy}'.")
    df = df.copy()
    cols = columns if columns is not None else df.columns.tolist()
    missing_before = df[cols].isnull().sum().to_dict()
    total_before = sum(missing_before.values())

    if strategy == "drop":
        rows_before = len(df)
        df = df.dropna(subset=cols)
        return df, {"strategy": "drop", "columns": cols,
                    "missing_before": missing_before,
                    "rows_dropped": rows_before - len(df)}

    if strategy == "constant":
        df[cols] = df[cols].fillna(fill_value)
        return df, {"strategy": "constant", "fill_value": fill_value,
                    "columns": cols, "missing_before": missing_before,
                    "cells_filled": total_before}

    sklearn_strategy = "most_frequent" if strategy == "mode" else strategy
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
        df[cat_cols] = df[cat_cols].where(df[cat_cols].notna(), other=np.nan)
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols].astype(object))

    missing_after = df[cols].isnull().sum().to_dict()
    return df, {"strategy": strategy, "columns": cols,
                "missing_before": missing_before, "missing_after": missing_after,
                "cells_imputed": total_before}


def remove_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, Summary]:
    rows_before = len(df)
    df = df.copy().drop_duplicates(subset=subset, keep=keep)  # type: ignore[arg-type]
    return df, {"rows_before": rows_before, "rows_after": len(df),
                "duplicates_removed": rows_before - len(df),
                "subset": subset, "keep": keep}


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> dict[str, pd.Series]:
    cols = columns if columns is not None else df.select_dtypes(include="number").columns.tolist()
    masks: dict[str, pd.Series] = {}
    for col in cols:
        if col not in df.columns:
            raise PreprocessingError(f"Column '{col}' not found.")
        series = df[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        masks[col] = (df[col] < q1 - factor * iqr) | (df[col] > q3 + factor * iqr)
    return masks


def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    action: str = "cap",
    columns: list[str] | None = None,
    factor: float = 1.5,
    threshold: float = 3.0,
) -> tuple[pd.DataFrame, Summary]:
    if method not in ("iqr", "zscore"):
        raise PreprocessingError(f"Unknown outlier method '{method}'.")
    if action not in ("flag", "cap", "drop"):
        raise PreprocessingError(f"Unknown outlier action '{action}'.")
    df = df.copy()
    cols = columns if columns is not None else df.select_dtypes(include="number").columns.tolist()

    if method == "iqr":
        masks = detect_outliers_iqr(df, cols, factor=factor)
    else:
        # zscore via median + MAD
        masks = {}
        for col in cols:
            series = df[col]
            median_ = float(np.nanmedian(series))
            mad = float(np.nanmedian(np.abs(series - median_)))
            mad_scaled = mad * 1.4826 if mad > 0.0 else float(np.nanstd(series, ddof=0))
            if mad_scaled == 0.0:
                z = np.zeros(len(series))
            else:
                z = np.abs((series.fillna(median_) - median_) / mad_scaled)
            masks[col] = pd.Series(z.to_numpy() >= threshold, index=df.index)

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
                median_ = float(np.nanmedian(series))
                mad = float(np.nanmedian(np.abs(series - median_)))
                mad_scaled = mad * 1.4826 if mad > 0.0 else float(np.nanstd(series, ddof=0))
                if mad_scaled == 0.0:
                    continue
                lower = median_ - threshold * mad_scaled
                upper = median_ + threshold * mad_scaled
            df[col] = df[col].clip(lower=lower, upper=upper)
    elif action == "drop":
        rows_before = len(df)
        df = df[~combined_mask]
        outlier_counts["rows_dropped"] = rows_before - len(df)

    return df, {"method": method, "action": action, "columns": cols,
                "outlier_counts_per_column": outlier_counts,
                "total_outlier_rows": int(combined_mask.sum())}


def encode_categoricals(
    df: pd.DataFrame,
    method: str = "label",
    columns: list[str] | None = None,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, Summary, dict[str, Any]]:
    if method not in ("label", "onehot"):
        raise PreprocessingError(f"Unknown encoding method '{method}'.")
    df = df.copy()
    cols = (
        columns if columns is not None
        else df.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    cols = [c for c in cols if c in df.columns]
    encoders: dict[str, Any] = {}

    if method == "label":
        for col in cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        return df, {"method": "label", "columns_encoded": cols}, encoders
    else:
        ohe = OneHotEncoder(sparse_output=False,
                            drop="first" if drop_first else None,
                            handle_unknown="ignore")
        new_columns: list[str] = []
        if cols:
            encoded_array = ohe.fit_transform(df[cols].astype(str))
            feature_names = ohe.get_feature_names_out(cols).tolist()
            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
            df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)
            encoders["onehot_encoder"] = ohe
            new_columns = feature_names
        return df, {"method": "onehot", "original_columns": cols,
                    "new_columns": new_columns, "drop_first": drop_first}, encoders


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
    if method not in _SCALER_MAP:
        raise PreprocessingError(f"Unknown scaling method '{method}'.")
    df = df.copy()
    cols = (
        columns if columns is not None
        else df.select_dtypes(include="number").columns.tolist()
    )
    cols = [c for c in cols if c in df.columns]
    scaler = _SCALER_MAP[method]()
    if cols:
        df[cols] = scaler.fit_transform(df[cols])
    return df, {"method": method, "columns_scaled": cols}, scaler


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
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, Summary]:
    pipeline_summary: Summary = {"steps": [], "original_shape": list(df.shape)}

    df, mv_summary = handle_missing_values(df, strategy=missing_strategy, fill_value=missing_fill_value)
    pipeline_summary["steps"].append("missing_values")
    pipeline_summary["missing_values"] = mv_summary

    if remove_dupes:
        df, dup_summary = remove_duplicates(df)
        pipeline_summary["steps"].append("duplicates")
        pipeline_summary["duplicates"] = dup_summary

    if outlier_method is not None:
        df, out_summary = handle_outliers(df, method=outlier_method, action=outlier_action, columns=numeric_columns)
        pipeline_summary["steps"].append("outliers")
        pipeline_summary["outliers"] = out_summary

    df, enc_summary, _ = encode_categoricals(df, method=encoding_method, columns=categorical_columns)
    pipeline_summary["steps"].append("encoding")
    pipeline_summary["encoding"] = enc_summary

    df, scale_summary, _ = scale_numerics(df, method=scaling_method, columns=numeric_columns)
    pipeline_summary["steps"].append("scaling")
    pipeline_summary["scaling"] = scale_summary

    pipeline_summary["final_shape"] = list(df.shape)
    return df, pipeline_summary


# ============================================================
# ============================================================
#  SECTION 3 – DATA PROFILING
# ============================================================
# ============================================================

def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _corr_df_to_dict(corr_df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for col in corr_df.columns:
        result[str(col)] = {str(other): _safe_float(corr_df.loc[col, other]) for other in corr_df.columns}
    return result


class DataProfiler:
    """Compute a full statistical profile of a DataFrame."""

    def __init__(self, top_n: int = 10) -> None:
        self.top_n = top_n

    def profile(self, df: pd.DataFrame) -> dict[str, Any]:
        df = df.copy()
        report: dict[str, Any] = {}
        report["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
        report["memory_usage_bytes"] = int(df.memory_usage(deep=True).sum())
        report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        report["null_counts"] = {col: int(df[col].isnull().sum()) for col in df.columns}
        report["null_percentages"] = {col: _safe_float(df[col].isnull().mean() * 100) for col in df.columns}
        report["cardinality"] = {col: int(df[col].dropna().nunique()) for col in df.columns}
        report["numeric_stats"] = self._numeric_stats(df)
        report["categorical_stats"] = self._categorical_stats(df)
        report["correlation"] = self._correlation(df)
        return report

    def _numeric_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for col in df.select_dtypes(include="number").columns:
            series = df[col].dropna()
            if series.empty:
                result[col] = {"count": 0, "mean": None, "median": None, "std": None,
                               "min": None, "max": None, "q1": None, "q3": None,
                               "iqr": None, "skewness": None, "kurtosis": None}
                continue
            values = series.to_numpy(dtype=float)
            q1 = float(np.percentile(values, 25))
            q3 = float(np.percentile(values, 75))
            try:
                skewness: float | None = float(scipy_stats.skew(values, bias=True))
                kurtosis: float | None = float(scipy_stats.kurtosis(values, bias=True))
            except Exception:
                skewness = kurtosis = None
            result[col] = {
                "count": int(series.count()),
                "mean": _safe_float(float(np.mean(values))),
                "median": _safe_float(float(np.median(values))),
                "std": _safe_float(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
                "min": _safe_float(float(np.min(values))),
                "max": _safe_float(float(np.max(values))),
                "q1": _safe_float(q1),
                "q3": _safe_float(q3),
                "iqr": _safe_float(q3 - q1),
                "skewness": _safe_float(skewness),
                "kurtosis": _safe_float(kurtosis),
            }
        return result

    def _categorical_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for col in df.select_dtypes(exclude="number").columns:
            series = df[col].dropna()
            if series.empty:
                result[col] = {"count": 0, "cardinality": 0, "mode": None, "top_values": {}}
                continue
            value_counts = series.value_counts()
            result[col] = {
                "count": int(series.count()),
                "cardinality": int(series.nunique()),
                "mode": str(value_counts.index[0]),
                "top_values": {str(k): int(v) for k, v in value_counts.head(self.top_n).items()},
            }
        return result

    def _correlation(self, df: pd.DataFrame) -> dict[str, Any]:
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return {"pearson": {}, "spearman": {}}
        return {
            "pearson": _corr_df_to_dict(numeric_df.corr(method="pearson")),
            "spearman": _corr_df_to_dict(numeric_df.corr(method="spearman")),
        }


def profile_dataframe(df: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    return DataProfiler(top_n=top_n).profile(df)


# ============================================================
# ============================================================
#  SECTION 4 – EDA ENGINE
# ============================================================
# ============================================================

EDAResult = dict[str, Any]


def _sf(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _infer_task(target: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    n_unique = target.dropna().nunique()
    return "classification" if n_unique < 20 else "regression"


def _try_skew(values: np.ndarray) -> float | None:
    try:
        return float(scipy_stats.skew(values, bias=True))
    except Exception:
        return None


def _try_kurtosis(values: np.ndarray) -> float | None:
    try:
        return float(scipy_stats.kurtosis(values, bias=True))
    except Exception:
        return None


class EDAEngine:
    def __init__(self, top_n_categories: int = 15, scatter_sample_size: int = 500, n_bins: int = 10) -> None:
        self.top_n_categories = top_n_categories
        self.scatter_sample_size = scatter_sample_size
        self.n_bins = n_bins

    def analyze(self, df: pd.DataFrame, target_column: str | None = None, task_type: str | None = None) -> EDAResult:
        df = df.copy()
        result: EDAResult = {}
        result["univariate"] = self._univariate(df)
        result["bivariate"] = self._bivariate(df)
        result["groupby"] = self._groupby(df)
        if target_column is not None and target_column in df.columns:
            resolved_task = task_type or _infer_task(df[target_column])
            result["target_analysis"] = self._target_analysis(df, target_column, resolved_task)
        return result

    def _univariate(self, df: pd.DataFrame) -> dict[str, Any]:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        return {
            "numeric": {col: self._numeric_univariate(df[col]) for col in num_cols},
            "categorical": {col: self._categorical_univariate(df[col]) for col in cat_cols},
        }

    def _numeric_univariate(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna()
        if clean.empty:
            return {"count": 0, "null_count": 0, "mean": None, "median": None, "std": None,
                    "min": None, "max": None, "q1": None, "q3": None, "iqr": None,
                    "skewness": None, "kurtosis": None, "histogram": {"counts": [], "bin_edges": []}}
        values = clean.to_numpy(dtype=float)
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        hist_counts, bin_edges = np.histogram(values, bins=self.n_bins)
        return {
            "count": int(clean.count()),
            "null_count": int(series.isnull().sum()),
            "mean": _sf(float(np.mean(values))),
            "median": _sf(float(np.median(values))),
            "std": _sf(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
            "min": _sf(float(np.min(values))),
            "max": _sf(float(np.max(values))),
            "q1": _sf(q1),
            "q3": _sf(q3),
            "iqr": _sf(q3 - q1),
            "skewness": _sf(_try_skew(values)),
            "kurtosis": _sf(_try_kurtosis(values)),
            "histogram": {"counts": hist_counts.tolist(), "bin_edges": [_sf(e) for e in bin_edges.tolist()]},
        }

    def _categorical_univariate(self, series: pd.Series) -> dict[str, Any]:
        clean = series.dropna()
        if clean.empty:
            return {"count": 0, "null_count": 0, "cardinality": 0, "mode": None, "top_values": {}, "top_values_pct": {}}
        value_counts = clean.value_counts()
        top_values = {str(k): int(v) for k, v in value_counts.head(self.top_n_categories).items()}
        return {
            "count": int(clean.count()),
            "null_count": int(series.isnull().sum()),
            "cardinality": int(clean.nunique()),
            "mode": str(value_counts.index[0]),
            "top_values": top_values,
            "top_values_pct": {k: _sf(v / len(clean) * 100) for k, v in top_values.items()},
        }

    def _bivariate(self, df: pd.DataFrame) -> dict[str, Any]:
        numeric_df = df.select_dtypes(include="number")
        corr_result: dict[str, Any] = {}
        if numeric_df.shape[1] >= 2:
            corr_result = {
                "pearson": _corr_df_to_dict(numeric_df.corr(method="pearson")),
                "spearman": _corr_df_to_dict(numeric_df.corr(method="spearman")),
            }
        else:
            corr_result = {"pearson": {}, "spearman": {}}
        scatter: dict[str, Any] = {}
        cols = numeric_df.columns.tolist()
        for i, cx in enumerate(cols):
            for cy in cols[i + 1:]:
                pair_df = numeric_df[[cx, cy]].dropna().head(self.scatter_sample_size)
                if not pair_df.empty:
                    scatter[f"{cx}_vs_{cy}"] = {"x": pair_df[cx].tolist(), "y": pair_df[cy].tolist(), "x_col": cx, "y_col": cy}
        return {"correlation": corr_result, "scatter_data": scatter}

    def _groupby(self, df: pd.DataFrame) -> dict[str, Any]:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        if not num_cols or not cat_cols:
            return {}
        result: dict[str, Any] = {}
        for cat_col in cat_cols:
            n_unique = df[cat_col].dropna().nunique()
            if n_unique == 0 or n_unique > 50:
                continue
            col_results: dict[str, Any] = {}
            for num_col in num_cols:
                pair = df[[cat_col, num_col]].dropna(subset=[num_col])
                if pair.empty:
                    continue
                grouped = pair.groupby(cat_col, sort=True)[num_col]
                col_results[num_col] = {
                    "mean": {str(k): _sf(v) for k, v in grouped.mean().items()},
                    "sum": {str(k): _sf(v) for k, v in grouped.sum().items()},
                    "count": {str(k): _sf(v) for k, v in grouped.count().items()},
                    "median": {str(k): _sf(v) for k, v in grouped.median().items()},
                }
            if col_results:
                result[cat_col] = col_results
        return result

    def _target_analysis(self, df: pd.DataFrame, target_column: str, task_type: str) -> dict[str, Any]:
        target = df[target_column]
        result: dict[str, Any] = {"column": target_column, "task_type": task_type,
                                   "null_count": int(target.isnull().sum())}
        if task_type == "classification":
            target_clean = target.dropna()
            value_counts = target_clean.value_counts(sort=True)
            n_total = len(target_clean)
            class_dist = {str(k): int(v) for k, v in value_counts.items()}
            class_pct = {str(k): _sf(v / n_total * 100) for k, v in value_counts.items()}
            result.update({
                "n_classes": int(target_clean.nunique()),
                "class_distribution": class_dist,
                "class_distribution_pct": class_pct,
                "most_common_class": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "imbalance_ratio": _sf(float(value_counts.iloc[0]) / float(value_counts.iloc[-1])) if len(value_counts) >= 2 else None,
            })
        else:
            target_clean = target.dropna()
            if not target_clean.empty:
                values = target_clean.to_numpy(dtype=float)
                q1 = float(np.percentile(values, 25))
                q3 = float(np.percentile(values, 75))
                hist_counts, bin_edges = np.histogram(values, bins=self.n_bins)
                result["distribution"] = {
                    "count": int(target_clean.count()),
                    "mean": _sf(float(np.mean(values))),
                    "median": _sf(float(np.median(values))),
                    "std": _sf(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0),
                    "min": _sf(float(np.min(values))),
                    "max": _sf(float(np.max(values))),
                    "q1": _sf(q1), "q3": _sf(q3), "iqr": _sf(q3 - q1),
                    "skewness": _sf(_try_skew(values)), "kurtosis": _sf(_try_kurtosis(values)),
                    "histogram": {"counts": hist_counts.tolist(), "bin_edges": [_sf(e) for e in bin_edges.tolist()]},
                }
        return result


def run_eda(
    df: pd.DataFrame,
    target_column: str | None = None,
    task_type: str | None = None,
    top_n_categories: int = 15,
    scatter_sample_size: int = 500,
    n_bins: int = 10,
) -> EDAResult:
    return EDAEngine(top_n_categories, scatter_sample_size, n_bins).analyze(
        df, target_column=target_column, task_type=task_type
    )


# ============================================================
# ============================================================
#  SECTION 5 – VISUALIZATION
# ============================================================
# ============================================================

def _empty_figure(message: str = "No data to display") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font={"size": 14})
    fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    return fig


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return all(c in df.columns for c in cols)


def histogram(df: pd.DataFrame, column: str, *, kde: bool = True, nbins: int = 30, title: Optional[str] = None) -> go.Figure:
    if not _require_columns(df, [column]):
        return _empty_figure(f"Column '{column}' not found.")
    series = df[column].dropna()
    if series.empty:
        return _empty_figure(f"Column '{column}' has no non-null values.")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=series, nbinsx=nbins, name="Frequency", opacity=0.7,
                               histnorm="density" if kde else ""))
    if kde and len(series) >= 2:
        try:
            kde_func = gaussian_kde(series)
            x_range = np.linspace(series.min(), series.max(), 200)
            fig.add_trace(go.Scatter(x=x_range, y=kde_func(x_range), mode="lines",
                                     name="KDE", line={"color": "firebrick", "width": 2}))
        except Exception:
            pass
    fig.update_layout(title=title or f"Distribution of {column}", xaxis_title=column,
                      yaxis_title="Density" if kde else "Count", bargap=0.05)
    return fig


def bar_chart(df: pd.DataFrame, x_col: str, y_col: Optional[str] = None, *,
              color_col: Optional[str] = None, title: Optional[str] = None, top_n: int = 20) -> go.Figure:
    if not _require_columns(df, [x_col]):
        return _empty_figure(f"Column '{x_col}' not found.")
    data = df.copy()
    if y_col is None:
        counts = data[x_col].value_counts().nlargest(top_n)
        fig = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values, name=x_col))
        fig.update_layout(title=title or f"Frequency of {x_col}", xaxis_title=x_col, yaxis_title="Count")
    else:
        if not _require_columns(df, [y_col]):
            return _empty_figure(f"Column '{y_col}' not found.")
        kwargs: dict = {"x": x_col, "y": y_col, "title": title or f"{y_col} by {x_col}"}
        if color_col and color_col in data.columns:
            kwargs["color"] = color_col
        fig = px.bar(data.head(top_n) if len(data) > top_n else data, **kwargs)
    return fig


def pie_chart(df: pd.DataFrame, column: str, *, title: Optional[str] = None, top_n: int = 10) -> go.Figure:
    if not _require_columns(df, [column]):
        return _empty_figure(f"Column '{column}' not found.")
    counts = df[column].value_counts().nlargest(top_n)
    if counts.empty:
        return _empty_figure(f"Column '{column}' has no data.")
    fig = go.Figure(go.Pie(labels=counts.index.astype(str), values=counts.values, hole=0.3))
    fig.update_layout(title=title or f"Distribution of {column}")
    return fig


def correlation_heatmap(df: pd.DataFrame, *, method: str = "pearson", title: Optional[str] = None) -> go.Figure:
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        return _empty_figure("Need at least 2 numeric columns for correlation.")
    corr = num_df.corr(method=method)
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                               colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                               text=np.round(corr.values, 2), texttemplate="%{text}", showscale=True))
    fig.update_layout(title=title or f"Correlation Heatmap ({method})")
    return fig


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, *,
                 color_col: Optional[str] = None, size_col: Optional[str] = None,
                 trendline: bool = False, title: Optional[str] = None) -> go.Figure:
    if not _require_columns(df, [x_col, y_col]):
        return _empty_figure(f"Columns '{x_col}' or '{y_col}' not found.")
    data = df.copy()
    kwargs: dict = {"x": x_col, "y": y_col, "title": title or f"{y_col} vs {x_col}"}
    if color_col and color_col in data.columns:
        kwargs["color"] = color_col
    if size_col and size_col in data.columns:
        kwargs["size"] = size_col
    if trendline:
        kwargs["trendline"] = "ols"
    return px.scatter(data, **kwargs)


def pair_plot(df: pd.DataFrame, columns: Optional[Sequence[str]] = None, *,
              color_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if columns is not None:
        num_cols = [c for c in columns if c in df.columns]
    if len(num_cols) < 2:
        return _empty_figure("Need at least 2 numeric columns for pair plot.")
    data = df[num_cols].copy()
    kwargs: dict = {"dimensions": num_cols, "title": title or "Pair Plot"}
    if color_col and color_col in df.columns:
        data[color_col] = df[color_col].values
        kwargs["color"] = color_col
    fig = px.scatter_matrix(data, **kwargs)
    fig.update_traces(diagonal_visible=True, showupperhalf=False)
    return fig


def box_plot(df: pd.DataFrame, y_col: str, *, x_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    if not _require_columns(df, [y_col]):
        return _empty_figure(f"Column '{y_col}' not found.")
    kwargs: dict = {"y": y_col, "title": title or f"Box Plot of {y_col}"}
    if x_col and _require_columns(df, [x_col]):
        kwargs["x"] = x_col
    return px.box(df, **kwargs)


def violin_plot(df: pd.DataFrame, y_col: str, *, x_col: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    if not _require_columns(df, [y_col]):
        return _empty_figure(f"Column '{y_col}' not found.")
    kwargs: dict = {"y": y_col, "box": True, "title": title or f"Violin Plot of {y_col}"}
    if x_col and _require_columns(df, [x_col]):
        kwargs["x"] = x_col
    return px.violin(df, **kwargs)


def roc_curve_chart(fpr: Sequence[float], tpr: Sequence[float], *,
                    auc: Optional[float] = None, title: Optional[str] = None) -> go.Figure:
    fpr_arr, tpr_arr = list(fpr), list(tpr)
    if not fpr_arr or not tpr_arr:
        return _empty_figure("FPR / TPR data is empty.")
    label = f"ROC (AUC = {auc:.3f})" if auc is not None else "ROC"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_arr, y=tpr_arr, mode="lines", name=label, line={"width": 2}))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line={"dash": "dash", "color": "grey"}))
    fig.update_layout(title=title or "ROC Curve", xaxis_title="False Positive Rate",
                      yaxis_title="True Positive Rate",
                      xaxis={"range": [0, 1]}, yaxis={"range": [0, 1]})
    return fig


def confusion_matrix_chart(cm: "np.ndarray", *, labels: Optional[Sequence[str]] = None,
                           title: Optional[str] = None) -> go.Figure:
    cm_arr = np.array(cm)
    if cm_arr.ndim != 2 or cm_arr.shape[0] == 0:
        return _empty_figure("Confusion matrix must be a non-empty 2-D array.")
    n = cm_arr.shape[0]
    tick_labels = list(labels) if labels else [str(i) for i in range(n)]
    fig = go.Figure(go.Heatmap(z=cm_arr, x=tick_labels, y=tick_labels, colorscale="Blues",
                               text=cm_arr, texttemplate="%{text}", showscale=True))
    fig.update_layout(title=title or "Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def feature_importance_chart(feature_names: Sequence[str], importances: Sequence[float], *,
                              top_n: int = 20, title: Optional[str] = None) -> go.Figure:
    if not feature_names or not importances:
        return _empty_figure("No feature importance data provided.")
    pairs = sorted(zip(importances, feature_names), reverse=True)[:top_n]
    imp_vals, feat_names = zip(*pairs)
    fig = go.Figure(go.Bar(x=list(imp_vals), y=list(feat_names), orientation="h"))
    fig.update_layout(title=title or "Feature Importances", xaxis_title="Importance",
                      yaxis={"autorange": "reversed"}, height=max(300, 25 * len(feat_names)))
    return fig


def cluster_scatter(df: pd.DataFrame, cluster_col: str, *,
                    x_col: Optional[str] = None, y_col: Optional[str] = None,
                    title: Optional[str] = None) -> go.Figure:
    if not _require_columns(df, [cluster_col]):
        return _empty_figure(f"Cluster column '{cluster_col}' not found.")
    num_df = df.select_dtypes(include="number").drop(columns=[cluster_col], errors="ignore")
    if num_df.shape[1] < 1:
        return _empty_figure("No numeric feature columns available.")
    use_pca = False
    if x_col and y_col and _require_columns(df, [x_col, y_col]):
        plot_df = df[[x_col, y_col, cluster_col]].dropna().copy()
        px_x, px_y = x_col, y_col
    else:
        use_pca = True
        feat_df = num_df.dropna()
        if feat_df.shape[0] < 2:
            return _empty_figure("Not enough rows to compute PCA.")
        n_components = min(2, feat_df.shape[1])
        pca = PCA(n_components=n_components)
        comps = pca.fit_transform(feat_df.values)
        col_names = ["PC1", "PC2"] if n_components == 2 else ["PC1", "PC1"]
        plot_df = pd.DataFrame(comps, columns=col_names)
        plot_df[cluster_col] = df[cluster_col].iloc[feat_df.index].values
        px_x, px_y = "PC1", "PC2" if n_components == 2 else "PC1"
    fig = px.scatter(plot_df, x=px_x, y=px_y, color=cluster_col,
                     title=title or ("Cluster Scatter (PCA)" if use_pca else "Cluster Scatter"),
                     color_continuous_scale="Viridis")
    return fig


# ============================================================
# ============================================================
#  SECTION 6 – ML: TASK DETECTION, TRAINING, EVALUATION,
#              FEATURE IMPORTANCE
# ============================================================
# ============================================================

# --- Task Detection ---

_CLASSIFICATION_UNIQUE_FRAC = 0.05
_CLASSIFICATION_UNIQUE_MAX = 20
_MIN_SUPERVISED_ROWS = 10


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class TaskDetectionResult:
    def __init__(self, task_type: TaskType, reason: str,
                 n_classes: Optional[int], is_override: bool) -> None:
        self.task_type = task_type
        self.reason = reason
        self.n_classes = n_classes
        self.is_override = is_override


def detect_task(df: pd.DataFrame, target_column: Optional[str] = None,
                user_override: Optional[TaskType] = None) -> TaskDetectionResult:
    if user_override is not None:
        n_classes: Optional[int] = None
        if target_column and target_column in df.columns:
            n_classes = int(df[target_column].nunique(dropna=True))
        return TaskDetectionResult(user_override, f"Task overridden to '{user_override.value}'.", n_classes, True)

    if target_column is None or target_column not in df.columns:
        return TaskDetectionResult(TaskType.CLUSTERING, "No target column; defaulting to clustering.", None, False)

    if len(df) < _MIN_SUPERVISED_ROWS:
        return TaskDetectionResult(TaskType.CLUSTERING,
                                   f"Dataset too small ({len(df)} rows) for supervised learning.", None, False)

    target = df[target_column].dropna()
    n_unique = int(target.nunique())
    n_rows = len(target)

    if pd.api.types.is_bool_dtype(target) or isinstance(target.dtype, pd.CategoricalDtype):
        return TaskDetectionResult(TaskType.CLASSIFICATION,
                                   f"Target '{target_column}' is boolean/categorical → classification.", n_unique, False)
    if pd.api.types.is_object_dtype(target):
        return TaskDetectionResult(TaskType.CLASSIFICATION,
                                   f"Target '{target_column}' is string/object → classification.", n_unique, False)

    unique_frac = n_unique / n_rows if n_rows > 0 else 0.0
    if n_unique <= _CLASSIFICATION_UNIQUE_MAX or unique_frac <= _CLASSIFICATION_UNIQUE_FRAC:
        return TaskDetectionResult(TaskType.CLASSIFICATION,
                                   f"Numeric target with {n_unique} unique values ({unique_frac:.1%}) → classification.",
                                   n_unique, False)

    return TaskDetectionResult(TaskType.REGRESSION,
                               f"Numeric target with {n_unique} unique values ({unique_frac:.1%}) → regression.",
                               None, False)


# --- Trainer ---

class TrainerError(ValueError):
    """Raised when training cannot proceed."""


@dataclass
class TrainResult:
    task_type: TaskType
    model_name: str
    pipeline: Any
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


def _build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str]]:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary"))])
    transformers: list[tuple[str, Any, list[str]]] = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, num_cols + cat_cols


def _encode_target(y: pd.Series) -> tuple[pd.Series, Optional[LabelEncoder]]:
    if pd.api.types.is_object_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(y):
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=y.name)
        return y_encoded, le
    return y, None


def _cv_score_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                        task_type: TaskType, n_splits: int = 5) -> np.ndarray:
    n = len(X)
    n_splits = max(min(n_splits, n // 2) if n >= 4 else 2, 2)
    if task_type == TaskType.CLASSIFICATION:
        scoring = "f1_weighted"
        if y.nunique() < 2:
            return np.array([])
        actual_splits = max(min(n_splits, int(y.value_counts().min())), 2)
        cv = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    else:
        scoring = "r2"
        cv = n_splits  # type: ignore[assignment]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        except Exception:
            scores = np.array([])
    return scores


def _train_classification(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           feature_names: list[str],
                           label_encoder: Optional[LabelEncoder]) -> list[TrainResult]:
    estimators: list[tuple[str, Any]] = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]
    if _XGBOOST_AVAILABLE:
        estimators.append(("XGBoost", XGBClassifier(n_estimators=100, random_state=42,
                                                      eval_metric="logloss", verbosity=0)))

    results: list[TrainResult] = []
    for model_name, estimator in estimators:
        pipeline = Pipeline(steps=[("preprocessor", _build_preprocessor(X_train)[0]),
                                   ("classifier", estimator)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)

        cv_pipeline = Pipeline(steps=[("preprocessor", _build_preprocessor(X_train)[0]),
                                      ("classifier", type(estimator)(**estimator.get_params()))])
        cv_scores = _cv_score_pipeline(cv_pipeline, X_train, y_train, TaskType.CLASSIFICATION)
        results.append(TrainResult(
            task_type=TaskType.CLASSIFICATION, model_name=model_name, pipeline=pipeline,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            feature_names=feature_names, label_encoder=label_encoder,
            cv_scores=cv_scores,
            cv_mean=float(cv_scores.mean()) if len(cv_scores) else float("nan"),
            cv_std=float(cv_scores.std()) if len(cv_scores) else float("nan"),
            n_clusters=None, cluster_labels=None,
        ))
    return results


def _train_regression(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series,
                       feature_names: list[str]) -> list[TrainResult]:
    estimators: list[tuple[str, Any]] = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
    if _XGBOOST_AVAILABLE:
        estimators.append(("XGBoost Regressor", XGBRegressor(n_estimators=100, random_state=42, verbosity=0)))

    results: list[TrainResult] = []
    for model_name, estimator in estimators:
        preprocessor, _ = _build_preprocessor(X_train)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", estimator)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)

        cv_pipeline = Pipeline(steps=[("preprocessor", _build_preprocessor(X_train)[0]),
                                      ("regressor", type(estimator)(**estimator.get_params()))])
        cv_scores = _cv_score_pipeline(cv_pipeline, X_train, y_train, TaskType.REGRESSION)
        results.append(TrainResult(
            task_type=TaskType.REGRESSION, model_name=model_name, pipeline=pipeline,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
            feature_names=feature_names, label_encoder=None,
            cv_scores=cv_scores,
            cv_mean=float(cv_scores.mean()) if len(cv_scores) else float("nan"),
            cv_std=float(cv_scores.std()) if len(cv_scores) else float("nan"),
            n_clusters=None, cluster_labels=None,
        ))
    return results


def _elbow_k(X_scaled: np.ndarray, k_range: range) -> tuple[int, dict[int, float]]:
    inertias: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X_scaled)
        inertias[k] = float(km.inertia_)
    if len(inertias) < 3:
        return min(k_range), inertias
    ks = list(inertias.keys())
    vals = [inertias[k] for k in ks]
    best_k = ks[0]
    best_dd = -float("inf")
    for i in range(1, len(vals) - 1):
        dd = vals[i - 1] - 2 * vals[i] + vals[i + 1]
        if dd > best_dd:
            best_dd = dd
            best_k = ks[i]
    return best_k, inertias


def _train_clustering(X: pd.DataFrame, feature_names: list[str],
                       n_clusters: Optional[int] = None) -> TrainResult:
    preprocessor, _ = _build_preprocessor(X)
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
            n_clusters, elbow_scores = _elbow_k(X_scaled, range(2, max_k + 1))

    n_clusters = max(2, min(n_clusters, n - 1))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_labels = km.fit_predict(X_scaled)

    pipeline = Pipeline(steps=[("preprocessor", _build_preprocessor(X)[0]),
                                ("kmeans", KMeans(n_clusters=n_clusters, random_state=42, n_init="auto"))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X)

    return TrainResult(
        task_type=TaskType.CLUSTERING, model_name="K-Means", pipeline=pipeline,
        X_train=X, X_test=pd.DataFrame(columns=X.columns),
        y_train=None, y_test=None, feature_names=feature_names, label_encoder=None,
        cv_scores=np.array([]), cv_mean=float("nan"), cv_std=float("nan"),
        n_clusters=n_clusters, cluster_labels=cluster_labels,
        extra={"elbow_scores": elbow_scores},
    )


def train_models(df: pd.DataFrame, target_column: Optional[str], task_type: TaskType,
                 *, test_size: float = 0.2, n_clusters: Optional[int] = None) -> list[TrainResult]:
    df = df.copy()
    if task_type == TaskType.CLUSTERING:
        feature_cols = [c for c in df.columns if c != target_column] if target_column else df.columns.tolist()
        if not feature_cols:
            raise TrainerError("No feature columns available for clustering.")
        return [_train_clustering(df[feature_cols], feature_cols, n_clusters=n_clusters)]

    if not target_column or target_column not in df.columns:
        raise TrainerError(f"Target column '{target_column}' not found.")

    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        raise TrainerError("No feature columns after removing the target.")

    X = df[feature_cols]
    y = df[target_column]
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    if len(X) < 4:
        raise TrainerError(f"Dataset has only {len(X)} complete rows; too few to train.")

    if task_type == TaskType.CLASSIFICATION:
        y, label_encoder = _encode_target(y)
        if y.nunique() < 2:
            raise TrainerError("Target column has fewer than 2 unique classes.")
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                              random_state=42, stratify=stratify)
        return _train_classification(X_train, X_test, y_train, y_test, feature_cols, label_encoder)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return _train_regression(X_train, X_test, y_train, y_test, feature_cols)


# --- Evaluator ---

@dataclass
class ModelMetrics:
    task_type: TaskType
    model_name: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_weighted: Optional[float] = None
    f1_macro: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    adj_r2: Optional[float] = None
    inertia: Optional[float] = None
    silhouette: Optional[float] = None
    davies_bouldin: Optional[float] = None
    n_clusters: Optional[int] = None
    cv_mean: float = float("nan")
    cv_std: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"task_type": self.task_type.value, "model_name": self.model_name,
                               "cv_mean": self.cv_mean, "cv_std": self.cv_std}
        for attr in ("accuracy", "precision", "recall", "f1_weighted", "f1_macro", "roc_auc",
                     "mae", "mse", "rmse", "r2", "adj_r2", "inertia", "silhouette", "davies_bouldin", "n_clusters"):
            val = getattr(self, attr)
            if val is not None:
                out[attr] = val.tolist() if isinstance(val, np.ndarray) else val
        if self.confusion_matrix is not None:
            out["confusion_matrix"] = self.confusion_matrix.tolist()
        out.update(self.extra)
        return out


def _adjusted_r2(r2: float, n: int, p: int) -> float:
    if n <= p + 1:
        return float("nan")
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def _safe_roc_auc(y_true: np.ndarray, y_proba: Optional[np.ndarray], n_classes: int) -> Optional[float]:
    if y_proba is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if n_classes == 2:
                p = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                return float(roc_auc_score(y_true, p))
            else:
                return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
    except Exception:
        return None


def evaluate_model(result: TrainResult) -> ModelMetrics:
    task = result.task_type
    metrics = ModelMetrics(task_type=task, model_name=result.model_name,
                           cv_mean=result.cv_mean, cv_std=result.cv_std)

    if task == TaskType.CLASSIFICATION:
        X_test, y_test = result.X_test, result.y_test
        if X_test is None or y_test is None or len(X_test) == 0:
            return metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = result.pipeline.predict(X_test)
        n_classes = int(y_test.nunique())
        avg = "binary" if n_classes == 2 else "weighted"
        metrics.accuracy = float(accuracy_score(y_test, y_pred))
        metrics.precision = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
        metrics.recall = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
        metrics.f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics.f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        metrics.confusion_matrix = confusion_matrix(y_test, y_pred)
        y_proba: Optional[np.ndarray] = None
        if hasattr(result.pipeline, "predict_proba"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_proba = result.pipeline.predict_proba(X_test)
            except Exception:
                y_proba = None
        metrics.roc_auc = _safe_roc_auc(y_test.to_numpy(), y_proba, n_classes)
        if result.label_encoder is not None:
            metrics.extra["class_labels"] = result.label_encoder.classes_.tolist()

    elif task == TaskType.REGRESSION:
        X_test, y_test = result.X_test, result.y_test
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

    elif task == TaskType.CLUSTERING:
        X = result.X_train
        cluster_labels = result.cluster_labels
        if cluster_labels is None or len(X) == 0:
            return metrics
        kmeans_step = result.pipeline.named_steps.get("kmeans")
        if kmeans_step is not None:
            metrics.inertia = float(kmeans_step.inertia_)
        metrics.n_clusters = result.n_clusters
        n_unique_labels = len(set(cluster_labels))
        if n_unique_labels >= 2 and len(X) >= 2:
            preprocessor = result.pipeline.named_steps.get("preprocessor")
            if preprocessor is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        X_scaled = preprocessor.transform(X)
                        metrics.silhouette = float(silhouette_score(X_scaled, cluster_labels))
                        metrics.davies_bouldin = float(davies_bouldin_score(X_scaled, cluster_labels))
                    except Exception:
                        pass
    return metrics


# --- Feature Importance ---

@dataclass(frozen=True)
class FeatureImportanceEntry:
    feature: str
    importance: float
    std: float = 0.0


@dataclass
class ImportanceResult:
    method: str
    model_name: str
    ranked: list[FeatureImportanceEntry]
    raw_feature_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "model_name": self.model_name,
            "ranked": [{"feature": e.feature, "importance": e.importance, "std": e.std} for e in self.ranked],
        }


def _get_transformed_feature_names(pipeline: Any, raw_feature_names: list[str]) -> list[str]:
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is None:
            return raw_feature_names
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return raw_feature_names


def _get_terminal_estimator(pipeline: Any) -> Optional[Any]:
    try:
        return list(pipeline.steps)[-1][1]
    except Exception:
        return None


def compute_feature_importance(result: TrainResult, method: str = "auto",
                                n_repeats: int = 10) -> ImportanceResult:
    if result.task_type == TaskType.CLUSTERING:
        n = len(result.feature_names)
        uniform = 1.0 / n if n > 0 else 0.0
        ranked = [FeatureImportanceEntry(feature=f, importance=uniform) for f in result.feature_names]
        return ImportanceResult("uniform", result.model_name, ranked, result.feature_names)

    pipeline = result.pipeline
    raw_feature_names = result.feature_names
    transformed_names = _get_transformed_feature_names(pipeline, raw_feature_names)
    estimator = _get_terminal_estimator(pipeline)
    use_native = hasattr(estimator, "feature_importances_")

    if method == "model_native" and not use_native:
        raise ValueError(f"Model '{result.model_name}' does not expose feature_importances_.")

    if (method == "auto" and use_native) or method == "model_native":
        importances: np.ndarray = estimator.feature_importances_
        n = min(len(importances), len(transformed_names))
        ranked = sorted(
            [FeatureImportanceEntry(feature=transformed_names[i], importance=float(importances[i])) for i in range(n)],
            key=lambda e: e.importance, reverse=True,
        )
        return ImportanceResult("model_native", result.model_name, ranked, raw_feature_names)

    # Permutation
    X = result.X_test if len(result.X_test) > 0 else result.X_train
    y = result.y_test if result.y_test is not None and len(result.y_test) > 0 else result.y_train
    if y is None:
        ranked = [FeatureImportanceEntry(feature=f, importance=0.0) for f in raw_feature_names]
        return ImportanceResult("permutation", result.model_name, ranked, raw_feature_names)

    scoring = "f1_weighted" if result.task_type == TaskType.CLASSIFICATION else "r2"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            perm = permutation_importance(pipeline, X, y, n_repeats=n_repeats,
                                          random_state=42, scoring=scoring)
            imp_mean = perm.importances_mean
            imp_std = perm.importances_std
        except Exception:
            n = len(X.columns)
            imp_mean = np.ones(n) / n
            imp_std = np.zeros(n)

    col_names = list(X.columns)
    ranked = sorted(
        [FeatureImportanceEntry(feature=col_names[i], importance=float(imp_mean[i]), std=float(imp_std[i]))
         for i in range(len(col_names))],
        key=lambda e: e.importance, reverse=True,
    )
    return ImportanceResult("permutation", result.model_name, ranked, raw_feature_names)


# ============================================================
# ============================================================
#  SECTION 7 – GENAI (watsonx.ai)
# ============================================================
# ============================================================

DEFAULT_WATSONX_MODEL_ID = "ibm/granite-13b-instruct-v2"
DEFAULT_WATSONX_PARAMS: dict[str, Any] = {
    "decoding_method": "greedy",
    "max_new_tokens": 512,
    "min_new_tokens": 1,
    "repetition_penalty": 1.05,
}


class WatsonxClientError(Exception):
    pass


class WatsonxClient:
    """Thin wrapper around the IBM watsonx.ai SDK.

    Credentials are loaded exclusively from environment variables:
      WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL, WATSONX_MODEL_ID

    When credentials are absent the client enters *fallback mode*:
    generate() returns None instead of raising.
    """

    def __init__(self, api_key: str | None = None, project_id: str | None = None,
                 url: str | None = None, model_id: str | None = None) -> None:
        self._api_key = api_key or os.getenv("WATSONX_API_KEY") or ""
        self._project_id = project_id or os.getenv("WATSONX_PROJECT_ID") or ""
        self._url = url or os.getenv("WATSONX_URL") or "https://us-south.ml.cloud.ibm.com"
        self._model_id = model_id or os.getenv("WATSONX_MODEL_ID") or DEFAULT_WATSONX_MODEL_ID
        self._model: Any = None

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key and self._project_id)

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
        except ImportError as exc:
            raise WatsonxClientError(
                "ibm-watsonx-ai package not installed. Run: pip install ibm-watsonx-ai"
            ) from exc
        credentials = Credentials(url=self._url, api_key=self._api_key)
        self._model = ModelInference(model_id=self._model_id, credentials=credentials,
                                     project_id=self._project_id)
        return self._model

    def generate(self, prompt: str, parameters: dict[str, Any] | None = None) -> str | None:
        if not self.is_configured:
            logger.warning("WatsonxClient: credentials not configured — GenAI unavailable.")
            return None
        params = {**DEFAULT_WATSONX_PARAMS, **(parameters or {})}
        try:
            model = self._get_model()
            response = model.generate_text(prompt=prompt, params=params)
            return str(response).strip()
        except WatsonxClientError:
            raise
        except Exception as exc:
            logger.error("WatsonxClient: API call failed — %s", exc)
            return None


# --- Prompt Builder ---

def build_dataset_summary_prompt(profile_report: dict[str, Any]) -> str:
    n_rows = profile_report.get("n_rows", 0)
    n_cols = profile_report.get("n_cols", 0)
    n_numeric = profile_report.get("n_numeric", 0)
    n_categorical = profile_report.get("n_categorical", 0)
    missing_cells = profile_report.get("missing_cells", 0)
    missing_pct = profile_report.get("missing_pct", 0.0)
    duplicate_rows = profile_report.get("duplicate_rows", 0)
    memory_usage_mb = profile_report.get("memory_usage_mb", 0.0)

    numeric_lines = [
        f"  - {col}: mean={s.get('mean', 'N/A'):.4g}, std={s.get('std', 'N/A'):.4g}, "
        f"min={s.get('min', 'N/A'):.4g}, max={s.get('max', 'N/A'):.4g}"
        for col, s in (profile_report.get("numeric_summary") or {}).items()
    ]
    category_lines = [
        f"  - {col}: " + ", ".join(f"{v}({c})" for v, c in (top_vals or [])[:3])
        for col, top_vals in (profile_report.get("top_categories") or {}).items()
    ]
    return (
        "You are a data analyst writing for a non-technical business audience.\n\n"
        "The following statistics were computed by a data-profiling tool. "
        "Do not recalculate or change any number. "
        "Write a concise, plain-English summary (3–5 sentences) of what this dataset looks like.\n\n"
        "=== Dataset Statistics (pre-computed) ===\n"
        f"Rows: {n_rows}\nColumns: {n_cols}\nNumeric columns: {n_numeric}\n"
        f"Categorical columns: {n_categorical}\nMissing cells: {missing_cells} ({missing_pct:.2f}%)\n"
        f"Duplicate rows: {duplicate_rows}\nMemory usage: {memory_usage_mb:.2f} MB\n\n"
        "Numeric column statistics:\n" + ("\n".join(numeric_lines) if numeric_lines else "  (none)") + "\n\n"
        "Top categories per categorical column:\n" + ("\n".join(category_lines) if category_lines else "  (none)") + "\n\n"
        "=== Task ===\n"
        "Write a clear, non-technical paragraph summarising the dataset. "
        "Do not introduce any numbers that are not listed above."
    )


def build_key_findings_prompt(profile_report: dict[str, Any], eda_results: dict[str, Any]) -> str:
    n_rows = profile_report.get("n_rows", 0)
    n_cols = profile_report.get("n_cols", 0)
    missing_pct = profile_report.get("missing_pct", 0.0)
    corr_lines = [
        f"  - {e['feature_a']} / {e['feature_b']}: {e['correlation']:.4f}"
        for e in (eda_results.get("top_correlations") or [])[:5]
    ]
    skew_lines = [
        f"  - {e['column']}: skewness={e['skewness']:.4f}"
        for e in (eda_results.get("skewed_columns") or [])[:5]
    ]
    outlier_lines = [
        f"  - {col}: {count} outliers"
        for col, count in (eda_results.get("outlier_counts") or {}).items()
    ]
    return (
        "You are a data analyst writing for a non-technical business audience.\n\n"
        "The numbers below were computed deterministically by statistical tools. "
        "Do not change, recalculate, or introduce any number not listed here. "
        "Summarise the three to five most important findings in bullet-point form.\n\n"
        "=== Pre-Computed Statistics ===\n"
        f"Dataset: {n_rows} rows, {n_cols} columns\nMissing data: {missing_pct:.2f}%\n\n"
        "Top correlations:\n" + ("\n".join(corr_lines) if corr_lines else "  (none)") + "\n\n"
        "Highly skewed columns:\n" + ("\n".join(skew_lines) if skew_lines else "  (none)") + "\n\n"
        "Outlier counts:\n" + ("\n".join(outlier_lines) if outlier_lines else "  (none)") + "\n\n"
        "=== Task ===\n"
        "List the key findings a business analyst should know. "
        "Use only the numbers shown above. Do not invent or estimate values."
    )


def build_eda_explanation_prompt(eda_results: dict[str, Any]) -> str:
    corr_lines = [
        f"  - {e['feature_a']} ↔ {e['feature_b']}: r = {e['correlation']:.4f}"
        for e in (eda_results.get("top_correlations") or [])[:10]
    ]
    dist_lines = [f"  - {note}" for note in (eda_results.get("distribution_notes") or [])[:10]]
    group_lines = []
    for group_col, stats_list in (eda_results.get("group_stats") or {}).items():
        for entry in (stats_list or [])[:5]:
            group_lines.append(f"  - {group_col}={entry['group']}: {entry['metric']}={entry['value']:.4g}")
    return (
        "You are a data analyst writing for a non-technical business audience.\n\n"
        "All numbers below are pre-computed. Do not recalculate, alter, or add any value. "
        "Explain what the exploratory data analysis reveals in plain language (4–6 sentences).\n\n"
        "=== Exploratory Data Analysis Results (pre-computed) ===\n"
        "Correlations:\n" + ("\n".join(corr_lines) if corr_lines else "  (none)") + "\n\n"
        "Distribution observations:\n" + ("\n".join(dist_lines) if dist_lines else "  (none)") + "\n\n"
        "Group-level statistics:\n" + ("\n".join(group_lines) if group_lines else "  (none)") + "\n\n"
        "=== Task ===\n"
        "Explain what these EDA results mean for a business analyst. "
        "Do not introduce any numbers not listed above."
    )


def build_model_explanation_prompt(metrics: dict[str, Any], feature_importance: list[dict[str, Any]]) -> str:
    task_type = metrics.get("task_type", "unknown")
    model_name = metrics.get("model_name", "unknown")
    n_samples = metrics.get("n_samples", "N/A")
    n_features = metrics.get("n_features", "N/A")
    metric_pairs = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                    ("f1", "F1 Score"), ("roc_auc", "ROC-AUC"), ("rmse", "RMSE"),
                    ("mae", "MAE"), ("r2", "R²"), ("silhouette_score", "Silhouette Score")]
    metric_lines = [f"  - {label}: {metrics[key]:.6g}" for key, label in metric_pairs if key in metrics]
    importance_lines = [
        f"  - {e['feature']}: {e['importance']:.6g}"
        for e in (feature_importance or [])[:10]
    ]
    return (
        "You are a data scientist writing for a non-technical business audience.\n\n"
        "The evaluation metrics and feature importances below were computed by a "
        "machine-learning evaluation framework. Do not change, recalculate, or add "
        "any number. Explain the model performance and key drivers in 4–6 sentences.\n\n"
        "=== Model Evaluation (pre-computed) ===\n"
        f"Task type: {task_type}\nModel: {model_name}\n"
        f"Training samples: {n_samples}\nFeatures: {n_features}\n\n"
        "Performance metrics:\n" + ("\n".join(metric_lines) if metric_lines else "  (none)") + "\n\n"
        "Top feature importances:\n" + ("\n".join(importance_lines) if importance_lines else "  (none)") + "\n\n"
        "=== Task ===\n"
        "Interpret what these results mean for the business. "
        "Do not introduce any numbers not listed above."
    )


def build_recommendations_prompt(full_context: dict[str, Any]) -> str:
    dataset_summary = full_context.get("dataset_summary") or {}
    eda_highlights = full_context.get("eda_highlights") or []
    model_metrics = full_context.get("model_metrics") or {}
    anomaly_summary = full_context.get("anomaly_summary") or {}
    business_goal = full_context.get("business_goal", "improve data-driven decision making")
    n_rows = dataset_summary.get("n_rows", "N/A")
    n_cols = dataset_summary.get("n_cols", "N/A")
    missing_pct = dataset_summary.get("missing_pct", "N/A")
    eda_lines = "\n".join(f"  - {h}" for h in eda_highlights[:5]) or "  (none)"
    metric_pairs = [("accuracy", "Accuracy"), ("f1", "F1 Score"), ("rmse", "RMSE"), ("r2", "R²")]
    metric_lines = [f"  - {label}: {model_metrics[key]:.6g}" for key, label in metric_pairs if key in model_metrics]
    metric_block = "\n".join(metric_lines) if metric_lines else "  (none)"
    total_anomalies = anomaly_summary.get("total_anomalies", "N/A")
    anomaly_pct = anomaly_summary.get("anomaly_pct", "N/A")
    anomaly_str = (f"{total_anomalies} anomalous records ({anomaly_pct:.2f}%)"
                   if isinstance(anomaly_pct, float) else str(total_anomalies))
    return (
        "You are a senior data scientist writing for a non-technical business audience.\n\n"
        "All numbers below were computed by data science tools. "
        "Do not change, recalculate, or add any number. "
        "Provide 3–5 concrete, actionable recommendations based solely on these findings.\n\n"
        "=== Analysis Summary (pre-computed) ===\n"
        f"Business goal: {business_goal}\n"
        f"Dataset: {n_rows} rows, {n_cols} columns, {missing_pct}% missing\n\n"
        "Key EDA observations:\n" + eda_lines + "\n\n"
        "Model performance:\n" + metric_block + "\n\n"
        f"Anomalies detected: {anomaly_str}\n\n"
        "=== Task ===\n"
        "Give 3–5 bullet-point recommendations. Each must reference at least one "
        "number from the list above. Do not introduce any numbers not listed above."
    )


# ============================================================
# ============================================================
#  SECTION 8 – NATURAL LANGUAGE QUERY ENGINE
# ============================================================
# ============================================================

@dataclass
class NLQueryResult:
    question: str
    query_type: str
    computed_result: Any = None
    explanation: str | None = None
    error: str | None = None


QT_AVERAGE = "AVERAGE"
QT_SUM = "SUM"
QT_MAX_CATEGORY = "MAX_CATEGORY"
QT_COUNT = "COUNT"
QT_DISTRIBUTION = "DISTRIBUTION"
QT_CORRELATION = "CORRELATION"
QT_SUMMARY = "SUMMARY"
QT_UNSUPPORTED = "UNSUPPORTED"

_NL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (QT_COUNT, re.compile(r"\b(how many|count|number of|total number|total count|n rows|num rows|row count|record count|size of|how large)\b", re.IGNORECASE)),
    (QT_AVERAGE, re.compile(r"\b(average|avg|mean|typical|median)\b", re.IGNORECASE)),
    (QT_SUM, re.compile(r"\b(total|sum|aggregate|cumulative|combined)\b", re.IGNORECASE)),
    (QT_CORRELATION, re.compile(r"\b(correlations?|correlate[sd]?|relationship between|related to|strongest relationship|most related|most correlated|strongest correlations?)\b", re.IGNORECASE)),
    (QT_MAX_CATEGORY, re.compile(r"\b(which|what).{0,40}(highest|most|largest|maximum|top|biggest|best|peak|greatest)\b|\b(highest|most|largest|maximum|top|biggest|best|peak|greatest).{0,40}(category|group|value|item|entry|class|label)\b", re.IGNORECASE)),
    (QT_DISTRIBUTION, re.compile(r"\b(distribution|spread|breakdown|frequency|histogram|value counts|how is .{1,30} distributed)\b", re.IGNORECASE)),
    (QT_SUMMARY, re.compile(r"\b(summarise|summarize|summary|describe|overview|tell me about|what does|about this dataset|what is in)\b", re.IGNORECASE)),
]


def _extract_column(question: str, columns: list[str]) -> str | None:
    q_lower = question.lower()
    for col in sorted(columns, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(col.lower()) + r"\b")
        if pattern.search(q_lower):
            return col
    return None


def _build_nl_query_prompt(question: str, query_type: str, computed_result: Any) -> str:
    def _fmt_result(result: Any) -> str:
        if isinstance(result, dict):
            return "\n".join(f"  {k}: {v}" for k, v in result.items())
        if isinstance(result, (list, tuple)):
            return "\n".join(f"  {item}" for item in result)
        if isinstance(result, pd.Series):
            return "\n".join(f"  {idx}: {val}" for idx, val in result.items()) or "  (empty)"
        return f"  {result}"

    result_str = _fmt_result(computed_result)
    return (
        "You are a data analyst explaining results to a non-technical business audience.\n\n"
        "The following result was computed deterministically by a data analysis tool. "
        "Do not recalculate, alter, or introduce any number not shown below.\n\n"
        "=== Query Details (pre-computed) ===\n"
        f"Question: {question}\nQuery type: {query_type}\nComputed result:\n{result_str}\n\n"
        "=== Task ===\n"
        "In 2–4 sentences, explain what this result means in plain business language. "
        "Reference only the numbers shown above. Do not introduce any numbers not listed."
    )


class NLQueryEngine:
    """Execute natural-language questions against a Pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, watsonx_client: Any | None = None) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self._df = df
        self._client = watsonx_client
        self._columns = list(df.columns)
        self._numeric_cols = list(df.select_dtypes(include="number").columns)
        self._categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

    def ask(self, question: str) -> NLQueryResult:
        if not question or not question.strip():
            return NLQueryResult(question=question, query_type=QT_UNSUPPORTED, error="Empty question.")
        query_type = self._classify(question)
        try:
            computed_result = self._execute(query_type, question)
        except Exception as exc:
            return NLQueryResult(question=question, query_type=query_type,
                                 error=f"Query execution failed: {exc}")
        explanation = self._explain(question, query_type, computed_result)
        return NLQueryResult(question=question, query_type=query_type,
                             computed_result=computed_result, explanation=explanation)

    def _classify(self, question: str) -> str:
        for query_type, pattern in _NL_PATTERNS:
            if pattern.search(question):
                return query_type
        return QT_UNSUPPORTED

    def _execute(self, query_type: str, question: str) -> Any:
        dispatch = {
            QT_COUNT: self._exec_count,
            QT_AVERAGE: self._exec_average,
            QT_SUM: self._exec_sum,
            QT_MAX_CATEGORY: self._exec_max_category,
            QT_DISTRIBUTION: self._exec_distribution,
            QT_CORRELATION: self._exec_correlation,
            QT_SUMMARY: self._exec_summary,
            QT_UNSUPPORTED: lambda q: None,
        }
        return dispatch.get(query_type, lambda q: None)(question)

    def _exec_count(self, _: str) -> dict[str, int]:
        return {"row_count": len(self._df)}

    def _exec_average(self, question: str) -> dict[str, float]:
        col = _extract_column(question, self._numeric_cols)
        if col:
            return {col: round(float(self._df[col].mean()), 6)}
        return {c: round(float(self._df[c].mean()), 6) for c in self._numeric_cols}

    def _exec_sum(self, question: str) -> dict[str, float]:
        col = _extract_column(question, self._numeric_cols)
        if col:
            return {col: round(float(self._df[col].sum()), 6)}
        return {c: round(float(self._df[c].sum()), 6) for c in self._numeric_cols}

    def _exec_max_category(self, question: str) -> dict[str, Any]:
        cat_col = _extract_column(question, self._categorical_cols)
        num_col = _extract_column(question, self._numeric_cols)
        if cat_col and num_col:
            grouped = self._df.groupby(cat_col)[num_col].sum()
            return {"category_column": cat_col, "value_column": num_col,
                    "top_category": str(grouped.idxmax()), "top_value": round(float(grouped.max()), 6)}
        if cat_col:
            counts = self._df[cat_col].value_counts()
            return {"category_column": cat_col, "top_category": str(counts.index[0]), "count": int(counts.iloc[0])}
        if num_col:
            return {"value_column": num_col, "max_value": round(float(self._df[num_col].max()), 6)}
        if self._categorical_cols:
            col = self._categorical_cols[0]
            counts = self._df[col].value_counts()
            return {"category_column": col, "top_category": str(counts.index[0]), "count": int(counts.iloc[0])}
        if self._numeric_cols:
            means = {c: float(self._df[c].mean()) for c in self._numeric_cols}
            top_col = max(means, key=means.__getitem__)
            return {"column_with_highest_mean": top_col, "mean_value": round(means[top_col], 6)}
        return {}

    def _exec_distribution(self, question: str) -> dict[str, Any]:
        col = _extract_column(question, self._columns) or (self._columns[0] if self._columns else None)
        if col is None:
            return {}
        if col in self._numeric_cols:
            s = self._df[col].dropna()
            return {"column": col, "mean": round(float(s.mean()), 6), "std": round(float(s.std()), 6),
                    "min": round(float(s.min()), 6), "25%": round(float(s.quantile(0.25)), 6),
                    "median": round(float(s.median()), 6), "75%": round(float(s.quantile(0.75)), 6),
                    "max": round(float(s.max()), 6)}
        counts = self._df[col].value_counts().head(10)
        return {"column": col, "value_counts": {str(k): int(v) for k, v in counts.items()}}

    def _exec_correlation(self, _: str) -> list[dict[str, Any]]:
        if len(self._numeric_cols) < 2:
            return []
        corr_matrix = self._df[self._numeric_cols].corr()
        pairs: list[dict[str, Any]] = []
        seen: set[frozenset[str]] = set()
        for col_a in self._numeric_cols:
            for col_b in self._numeric_cols:
                if col_a == col_b:
                    continue
                key = frozenset({col_a, col_b})
                if key in seen:
                    continue
                seen.add(key)
                val = corr_matrix.loc[col_a, col_b]
                if pd.notna(val):
                    pairs.append({"feature_a": col_a, "feature_b": col_b, "correlation": round(float(val), 4)})
        pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return pairs[:10]

    def _exec_summary(self, _: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "n_rows": int(len(self._df)),
            "n_cols": int(len(self._df.columns)),
            "n_numeric": int(len(self._numeric_cols)),
            "n_categorical": int(len(self._categorical_cols)),
            "missing_cells": int(self._df.isna().sum().sum()),
            "missing_pct": round(float(self._df.isna().sum().sum()) / max(self._df.size, 1) * 100, 4),
            "duplicate_rows": int(self._df.duplicated().sum()),
        }
        if self._numeric_cols:
            result["numeric_means"] = {c: round(float(self._df[c].mean()), 6) for c in self._numeric_cols}
        return result

    def _explain(self, question: str, query_type: str, computed_result: Any) -> str | None:
        if query_type == QT_UNSUPPORTED or computed_result is None:
            return None
        if self._client is None:
            return None
        prompt = _build_nl_query_prompt(question, query_type, computed_result)
        return self._client.generate(prompt)


# ============================================================
# ============================================================
#  SECTION 9 – REPORT GENERATION (inline HTML, no Jinja2 dep)
# ============================================================
# ============================================================


def _figure_to_b64(fig: Any) -> str | None:
    try:
        if not isinstance(fig, go.Figure):
            return None
        png_bytes: bytes = fig.to_image(format="png", width=800, height=400)
        return base64.b64encode(png_bytes).decode("ascii")
    except Exception:
        return None


def _render_report_html(context: dict[str, Any]) -> str:
    """Render a self-contained HTML report from the analysis context dict.

    All sections are optional — missing keys are handled gracefully.
    """
    ts = context.get("generated_at") or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = context.get("title", "Data Analysis Report")
    dataset_name = context.get("dataset_name", "Dataset")

    def _num(v: Any) -> str:
        if v is None:
            return "—"
        try:
            f = float(v)
            return "—" if (math.isnan(f) or math.isinf(f)) else f"{f:.4g}"
        except (TypeError, ValueError):
            return str(v)

    sections_html: list[str] = []

    # Data profile section
    profile = context.get("profile")
    if profile:
        shape = profile.get("shape", {})
        null_pct_dict = profile.get("null_percentages", {})
        num_stats = profile.get("numeric_stats", {})
        rows = [
            f"<tr><td>{col}</td><td>{_num(s.get('mean'))}</td><td>{_num(s.get('std'))}</td>"
            f"<td>{_num(s.get('min'))}</td><td>{_num(s.get('max'))}</td>"
            f"<td>{_num(s.get('skewness'))}</td></tr>"
            for col, s in num_stats.items()
        ]
        null_rows = [
            f"<tr><td>{col}</td><td>{_num(pct)}%</td></tr>"
            for col, pct in null_pct_dict.items() if pct and float(pct) > 0
        ]
        sections_html.append(f"""
<section>
  <h2>Data Profile</h2>
  <p>Rows: <strong>{shape.get('rows', 0):,}</strong> &nbsp;|&nbsp;
     Columns: <strong>{shape.get('columns', 0)}</strong> &nbsp;|&nbsp;
     Memory: <strong>{profile.get('memory_usage_bytes', 0) / (1024*1024):.2f} MB</strong></p>
  {"<h3>Numeric Statistics</h3><table><thead><tr><th>Column</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Skewness</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>" if rows else ""}
  {"<h3>Missing Values (columns with nulls)</h3><table><thead><tr><th>Column</th><th>Null %</th></tr></thead><tbody>" + "".join(null_rows) + "</tbody></table>" if null_rows else ""}
</section>""")

    # EDA section
    eda = context.get("eda")
    if eda:
        top_corrs = eda.get("top_correlations", [])
        skewed = eda.get("skewed_columns", [])
        corr_rows = [
            f"<tr><td>{e.get('feature_a')}</td><td>{e.get('feature_b')}</td><td>{_num(e.get('correlation'))}</td></tr>"
            for e in top_corrs[:10]
        ]
        skew_rows = [
            f"<tr><td>{e.get('column')}</td><td>{_num(e.get('skewness'))}</td></tr>"
            for e in skewed[:10]
        ]
        sections_html.append(f"""
<section>
  <h2>Exploratory Data Analysis</h2>
  {"<h3>Top Correlations</h3><table><thead><tr><th>Feature A</th><th>Feature B</th><th>Pearson r</th></tr></thead><tbody>" + "".join(corr_rows) + "</tbody></table>" if corr_rows else ""}
  {"<h3>Highly Skewed Columns</h3><table><thead><tr><th>Column</th><th>Skewness</th></tr></thead><tbody>" + "".join(skew_rows) + "</tbody></table>" if skew_rows else ""}
</section>""")

    # ML section
    ml = context.get("ml_metrics")
    if ml:
        task = ml.get("task_type", "unknown")
        model = ml.get("model_name", "unknown")
        metric_html = ""
        if task == "classification":
            metric_html = (
                f"<ul><li>Accuracy: {_num(ml.get('accuracy'))}</li>"
                f"<li>F1 (weighted): {_num(ml.get('f1_weighted'))}</li>"
                f"<li>ROC-AUC: {_num(ml.get('roc_auc'))}</li>"
                f"<li>CV Mean: {_num(ml.get('cv_mean'))}</li></ul>"
            )
        elif task == "regression":
            metric_html = (
                f"<ul><li>RMSE: {_num(ml.get('rmse'))}</li>"
                f"<li>MAE: {_num(ml.get('mae'))}</li>"
                f"<li>R²: {_num(ml.get('r2'))}</li>"
                f"<li>CV Mean: {_num(ml.get('cv_mean'))}</li></ul>"
            )
        else:
            metric_html = (
                f"<ul><li>Clusters: {ml.get('n_clusters')}</li>"
                f"<li>Silhouette: {_num(ml.get('silhouette'))}</li>"
                f"<li>Inertia: {_num(ml.get('inertia'))}</li></ul>"
            )
        sections_html.append(f"""
<section>
  <h2>Machine Learning Results</h2>
  <p>Task: <strong>{task}</strong> &nbsp;|&nbsp; Best Model: <strong>{model}</strong></p>
  {metric_html}
</section>""")

    # Feature importance
    fi = context.get("feature_importance")
    if fi and fi.get("entries"):
        fi_rows = [
            f"<tr><td>{e.get('feature')}</td><td>{_num(e.get('importance'))}</td></tr>"
            for e in fi["entries"][:20]
        ]
        sections_html.append(f"""
<section>
  <h2>Feature Importance</h2>
  <table><thead><tr><th>Feature</th><th>Importance</th></tr></thead>
  <tbody>{"".join(fi_rows)}</tbody></table>
</section>""")

    # GenAI insights
    genai = context.get("genai_insights")
    if genai:
        insight_items = "".join(
            f"<div class='insight'><h3>{label}</h3><p>{text}</p></div>"
            for label, text in genai.items()
        )
        sections_html.append(f"<section><h2>AI-Generated Insights</h2>{insight_items}</section>")

    # NL query results
    nl_results = context.get("nl_query_results")
    if nl_results:
        nl_items: list[str] = []
        for r in (nl_results if isinstance(nl_results[0], dict) else
                  [{"question": getattr(x, "question", ""), "computed_result": getattr(x, "computed_result", None),
                    "explanation": getattr(x, "explanation", None), "error": getattr(x, "error", None)}
                   for x in nl_results]):
            q = r.get("question", "")
            result_str = str(r.get("computed_result", ""))[:500]
            expl = r.get("explanation") or ""
            err = r.get("error") or ""
            nl_items.append(
                f"<div class='nl-item'><p><strong>Q:</strong> {q}</p>"
                f"{'<p class=\"error\">' + err + '</p>' if err else '<p>' + result_str + '</p>'}"
                f"{'<p class=\"explanation\">' + expl + '</p>' if expl else ''}</div>"
            )
        sections_html.append(f"<section><h2>Natural Language Queries</h2>{''.join(nl_items)}</section>")

    body = "\n".join(sections_html) if sections_html else "<p>No analysis data available.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 960px;
           margin: 0 auto; padding: 2rem; background: #fff; color: #1f2328; line-height: 1.6; }}
    h1 {{ border-bottom: 2px solid #3b82d4; padding-bottom: .5rem; }}
    h2 {{ color: #3b82d4; margin-top: 2rem; }}
    h3 {{ color: #57606a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th {{ background: #f7f8fa; border: 1px solid #e5e7eb; padding: .5rem .75rem; text-align: left; }}
    td {{ border: 1px solid #e5e7eb; padding: .4rem .75rem; }}
    tr:nth-child(even) td {{ background: #f7f8fa; }}
    section {{ margin-bottom: 2rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 1.5rem; }}
    .insight {{ background: #f0f7ff; border-left: 4px solid #3b82d4; padding: 1rem; margin: .5rem 0; }}
    .nl-item {{ background: #f7f8fa; border: 1px solid #e5e7eb; padding: 1rem; margin: .5rem 0; border-radius: 4px; }}
    .explanation {{ font-style: italic; color: #57606a; }}
    .error {{ color: #c0392b; }}
    footer {{ text-align: center; font-size: 12px; color: #57606a; border-top: 1px solid #e5e7eb;
              margin-top: 2rem; padding-top: 1rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p><strong>Dataset:</strong> {dataset_name} &nbsp;|&nbsp; <strong>Generated:</strong> {ts}</p>
  {body}
  <footer>Made with IBM Bob &bull; AI-Powered Data Analyst</footer>
</body>
</html>"""


class ReportGenerator:
    """Assemble pipeline outputs into a self-contained HTML report."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self._output_dir = Path(output_dir) if output_dir else Path(
            os.environ.get("TMPDIR", os.environ.get("TEMP", tempfile.gettempdir()))
        )

    def generate(self, context: dict[str, Any], *,
                 output_path: str | Path | None = None, export_pdf: bool = False) -> str:
        html_str = self.render_html(context)
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = self._output_dir / f"report_{ts}.html"
        else:
            html_path = Path(output_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_str, encoding="utf-8")
        if export_pdf:
            try:
                from weasyprint import HTML
                HTML(string=html_str).write_pdf(str(html_path.with_suffix(".pdf")))
            except Exception as exc:
                warnings.warn(f"PDF export failed: {exc}", RuntimeWarning, stacklevel=2)
        return str(html_path)

    def render_html(self, context: dict[str, Any]) -> str:
        return _render_report_html(context)


# ============================================================
# ============================================================
#  SECTION 10 – STREAMLIT DASHBOARD
# ============================================================
# ============================================================

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session-state init
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "df": None, "filename": None, "target_column": None, "task_type": None,
        "profile": None, "eda_result": None,
        "ml_trained": False, "ml_train_results": [], "ml_metrics": [], "ml_task_type": None,
        "nl_query_history": [],
        "report_html_path": None, "report_html_str": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Cached computation helpers
# ---------------------------------------------------------------------------
def _compute_profile(df: pd.DataFrame) -> dict[str, Any]:
    cached = st.session_state.get("profile")
    sig = (df.shape, tuple(df.columns))
    if cached is None or st.session_state.get("_profile_sig") != sig:
        with st.spinner("Profiling dataset…"):
            cached = profile_dataframe(df)
        st.session_state["profile"] = cached
        st.session_state["_profile_sig"] = sig
    return cached


def _compute_eda(df: pd.DataFrame, target_column: str | None, task_type: Any) -> dict[str, Any]:
    cached = st.session_state.get("eda_result")
    task_val = task_type.value if task_type else None
    sig = (df.shape, tuple(df.columns), target_column, task_val)
    if cached is None or st.session_state.get("_eda_sig") != sig:
        with st.spinner("Running EDA…"):
            cached = run_eda(df, target_column=target_column, task_type=task_val)
        st.session_state["eda_result"] = cached
        st.session_state["_eda_sig"] = sig
    return cached


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar() -> dict[str, Any]:
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.header("1. Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file", type=["csv", "xlsx", "xls"],
        help="Maximum 50 MB. Supported formats: CSV, Excel (.xlsx, .xls).",
    )
    df: pd.DataFrame | None = None
    filename: str | None = None
    if uploaded_file is not None:
        filename = uploaded_file.name
        try:
            with st.sidebar.spinner("Loading dataset…"):
                df = DataLoader().load(uploaded_file)
            st.sidebar.success(f"✅ Loaded **{filename}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
        except DataLoadError as exc:
            st.sidebar.error(f"❌ {exc}")
        except Exception as exc:
            st.sidebar.error(f"❌ Unexpected error: {exc}")

    target_column: str | None = None
    if df is not None:
        st.sidebar.header("2. Target Column")
        col_options = ["(None — unsupervised)"] + list(df.columns)
        target_sel = st.sidebar.selectbox("Select target column", options=col_options, index=0)
        if target_sel != "(None — unsupervised)":
            target_column = target_sel

    task_type: TaskType | None = None
    if df is not None:
        st.sidebar.header("3. Task Type")
        task_options = {"Auto-detect": None, "Classification": TaskType.CLASSIFICATION,
                        "Regression": TaskType.REGRESSION, "Clustering": TaskType.CLUSTERING}
        task_sel = st.sidebar.selectbox("ML task type", options=list(task_options.keys()), index=0)
        task_type = task_options[task_sel]

    model_names: list[str] = []
    if df is not None and task_type != TaskType.CLUSTERING and target_column is not None:
        st.sidebar.header("4. Models")
        avail = {"classification": ["Logistic Regression", "Random Forest", "XGBoost"],
                 "regression": ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"]}
        task_key = (task_type.value if task_type else "classification")
        if task_key == "clustering":
            task_key = "classification"
        choices = avail.get(task_key, avail["classification"])
        model_names = st.sidebar.multiselect("Select models to train", options=choices, default=choices)

    st.sidebar.markdown("---")
    st.sidebar.caption("AI Data Analyst • Powered by IBM watsonx.ai")
    return {"df": df, "filename": filename, "target_column": target_column,
            "task_type": task_type, "model_names": model_names}


# ---------------------------------------------------------------------------
# Data tab
# ---------------------------------------------------------------------------
def _render_data_tab(df: pd.DataFrame, filename: str | None) -> None:
    st.header("📋 Dataset Preview")
    if filename:
        st.caption(f"File: **{filename}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Memory", f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    st.markdown("#### First 100 rows")
    st.dataframe(df.head(100), use_container_width=True)
    with st.expander("Column info"):
        col_info = [{"Column": col, "Dtype": str(df[col].dtype),
                     "Non-null": int(df[col].notna().sum()), "Null": int(df[col].isna().sum()),
                     "Unique": int(df[col].nunique()),
                     "Sample": str(df[col].dropna().iloc[0]) if df[col].notna().any() else "—"}
                    for col in df.columns]
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)


# ---------------------------------------------------------------------------
# EDA tab
# ---------------------------------------------------------------------------
def _fmt_stat(value: Any) -> str:
    if value is None:
        return "—"
    try:
        f = float(value)
        return "—" if f != f else f"{f:.4g}"
    except (TypeError, ValueError):
        return str(value)


def _render_eda_tab(df: pd.DataFrame, target_column: str | None, task_type: str | None) -> None:
    st.header("📊 Exploratory Data Analysis")
    with st.spinner("Computing data profile…"):
        profile = profile_dataframe(df)
    # Overview metrics
    shape = profile.get("shape", {})
    n_rows = shape.get("rows", len(df))
    n_cols = shape.get("columns", len(df.columns))
    null_counts = profile.get("null_counts", {})
    total_nulls = sum(null_counts.values())
    null_pct = total_nulls / max(n_rows * n_cols, 1) * 100
    num_cols_count = len(df.select_dtypes(include="number").columns)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{n_rows:,}")
    c2.metric("Columns", n_cols)
    c3.metric("Numeric", num_cols_count)
    c4.metric("Categorical", n_cols - num_cols_count)
    c5.metric("Missing %", f"{null_pct:.1f}%")
    c6, c7 = st.columns(2)
    c6.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
    c7.metric("Memory", f"{profile.get('memory_usage_bytes', 0) / (1024*1024):.2f} MB")

    st.markdown("---")
    st.subheader("Descriptive Statistics")
    tab_num, tab_cat = st.tabs(["Numeric Columns", "Categorical Columns"])
    with tab_num:
        num_stats = profile.get("numeric_stats", {})
        if num_stats:
            rows = [{"Column": col, "Count": s.get("count"), "Mean": _fmt_stat(s.get("mean")),
                     "Std": _fmt_stat(s.get("std")), "Min": _fmt_stat(s.get("min")),
                     "Q1": _fmt_stat(s.get("q1")), "Median": _fmt_stat(s.get("median")),
                     "Q3": _fmt_stat(s.get("q3")), "Max": _fmt_stat(s.get("max")),
                     "Skewness": _fmt_stat(s.get("skewness"))} for col, s in num_stats.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No numeric columns found.")
    with tab_cat:
        cat_stats = profile.get("categorical_stats", {})
        if cat_stats:
            rows = [{"Column": col, "Count": s.get("count"), "Unique": s.get("cardinality"),
                     "Mode": s.get("mode"),
                     "Top Values": ", ".join(f"{k}({v})" for k, v in list(s.get("top_values", {}).items())[:3])}
                    for col, s in cat_stats.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No categorical columns found.")

    st.markdown("---")
    st.subheader("Visualisations")
    with st.spinner("Running EDA analysis…"):
        eda_result = run_eda(df, target_column=target_column, task_type=task_type)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if len(num_cols) >= 2:
        st.markdown("#### Correlation Heatmap")
        st.plotly_chart(correlation_heatmap(df, method="pearson"), use_container_width=True)

    st.markdown("#### Univariate Distributions")
    if num_cols:
        sel_num = st.selectbox("Select a numeric column", options=num_cols, key="eda_hist_col")
        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(histogram(df, sel_num), use_container_width=True)
        with cb:
            st.plotly_chart(box_plot(df, sel_num), use_container_width=True)
    if cat_cols:
        sel_cat = st.selectbox("Select a categorical column", options=cat_cols, key="eda_bar_col")
        st.plotly_chart(bar_chart(df, sel_cat), use_container_width=True)

    if len(num_cols) >= 2:
        st.markdown("#### Bivariate Scatter")
        sc1, sc2 = st.columns(2)
        with sc1:
            x_sel = st.selectbox("X axis", options=num_cols, index=0, key="eda_sc_x")
        with sc2:
            y_sel = st.selectbox("Y axis", options=num_cols, index=min(1, len(num_cols)-1), key="eda_sc_y")
        c_opts = ["(none)"] + cat_cols
        c_sel = st.selectbox("Colour by", options=c_opts, index=0, key="eda_sc_c")
        st.plotly_chart(scatter_plot(df, x_sel, y_sel, color_col=None if c_sel == "(none)" else c_sel, trendline=True),
                        use_container_width=True)

    if target_column and target_column in df.columns:
        ta = eda_result.get("target_analysis")
        if ta:
            st.markdown(f"#### Target Column: `{target_column}`")
            ttype = ta.get("task_type", "unknown")
            st.caption(f"Detected task: **{ttype}**")
            if ttype == "classification":
                dist = ta.get("class_distribution", {})
                if dist:
                    dist_df = pd.DataFrame([{"Class": k, "Count": v} for k, v in dist.items()])
                    st.plotly_chart(bar_chart(dist_df, "Class", "Count",
                                             title=f"Class distribution of {target_column}"),
                                   use_container_width=True)
            else:
                if target_column in num_cols:
                    st.plotly_chart(histogram(df, target_column), use_container_width=True)

    if len(num_cols) >= 2:
        with st.expander("Pair Plot (select columns)"):
            max_c = min(5, len(num_cols))
            pair_sel = st.multiselect("Columns for pair plot", options=num_cols, default=num_cols[:max_c], key="eda_pair")
            if len(pair_sel) >= 2:
                st.plotly_chart(pair_plot(df, columns=pair_sel), use_container_width=True)


# ---------------------------------------------------------------------------
# ML tab
# ---------------------------------------------------------------------------
def _pick_best_model(results: list[TrainResult], metrics: list[ModelMetrics],
                     task: TaskType) -> tuple[TrainResult, ModelMetrics]:
    if len(results) == 1:
        return results[0], metrics[0]
    best_idx, best_score = 0, -float("inf")
    for i, m in enumerate(metrics):
        if task == TaskType.CLASSIFICATION:
            score = m.f1_weighted if m.f1_weighted is not None else -float("inf")
        elif task == TaskType.REGRESSION:
            r2 = m.r2 if m.r2 is not None else -float("inf")
            score = r2 if not math.isnan(r2) else -float("inf")
        else:
            score = m.silhouette if m.silhouette is not None else -float("inf")
        if score > best_score:
            best_score, best_idx = score, i
    return results[best_idx], metrics[best_idx]


def _pct(v: float | None) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def _flt(v: float | None) -> str:
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4g}"


def _render_ml_tab(df: pd.DataFrame, target_column: str | None,
                   task_type_override: TaskType | None, selected_model_names: list[str]) -> dict[str, Any]:
    st.header("🤖 Machine Learning")
    detection = detect_task(df, target_column=target_column, user_override=task_type_override)
    resolved_task = detection.task_type
    info_parts = [f"**Task:** {resolved_task.value}"]
    if detection.is_override:
        info_parts.append("*(user override)*")
    if detection.n_classes:
        info_parts.append(f"**Classes:** {detection.n_classes}")
    st.info("  |  ".join(info_parts) + f"\n\n_{detection.reason}_")

    if st.button("🚀 Train Models", key="btn_train"):
        st.session_state.update({"ml_trained": False, "ml_train_results": [], "ml_metrics": [],
                                  "ml_task_type": resolved_task})
        with st.spinner("Training models…"):
            try:
                all_results = train_models(df, target_column=target_column, task_type=resolved_task)
            except TrainerError as exc:
                st.error(f"❌ Training failed: {exc}")
                return {"train_results": [], "metrics": [], "best_result": None, "best_metrics": None, "task_type": resolved_task}
            except Exception as exc:
                st.error(f"❌ Unexpected error: {exc}")
                return {"train_results": [], "metrics": [], "best_result": None, "best_metrics": None, "task_type": resolved_task}

        if selected_model_names:
            filtered = [r for r in all_results if r.model_name in selected_model_names]
            all_results = filtered or all_results

        all_metrics = [evaluate_model(r) for r in all_results]
        st.session_state.update({"ml_trained": True, "ml_train_results": all_results,
                                  "ml_metrics": all_metrics, "ml_task_type": resolved_task})
        st.success(f"✅ Trained {len(all_results)} model(s).")

    if not st.session_state.get("ml_trained"):
        st.info("Click **Train Models** to start training.")
        return {"train_results": [], "metrics": [], "best_result": None, "best_metrics": None, "task_type": resolved_task}

    all_results = st.session_state.get("ml_train_results", [])
    all_metrics = st.session_state.get("ml_metrics", [])
    task = st.session_state.get("ml_task_type", resolved_task)
    if not all_results:
        return {"train_results": [], "metrics": [], "best_result": None, "best_metrics": None, "task_type": task}

    # Comparison table
    st.subheader("Model Comparison")
    rows = []
    for m in all_metrics:
        row: dict[str, Any] = {"Model": m.model_name}
        if task == TaskType.CLASSIFICATION:
            row.update({"Accuracy": _pct(m.accuracy), "F1 (weighted)": _pct(m.f1_weighted),
                        "ROC-AUC": _pct(m.roc_auc), "CV Mean": _pct(m.cv_mean)})
        elif task == TaskType.REGRESSION:
            row.update({"RMSE": _flt(m.rmse), "MAE": _flt(m.mae),
                        "R²": _flt(m.r2), "CV Mean": _flt(m.cv_mean)})
        else:
            row.update({"Inertia": _flt(m.inertia), "Silhouette": _flt(m.silhouette),
                        "Davies-Bouldin": _flt(m.davies_bouldin), "Clusters": m.n_clusters})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    best_result, best_metrics = _pick_best_model(all_results, all_metrics, task)
    st.subheader(f"Best Model: {best_metrics.model_name}")

    if task == TaskType.CLASSIFICATION:
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Accuracy", _pct(best_metrics.accuracy))
        cc2.metric("F1 (weighted)", _pct(best_metrics.f1_weighted))
        cc3.metric("ROC-AUC", _pct(best_metrics.roc_auc))
        cc4.metric("CV", f"{_flt(best_metrics.cv_mean)} ± {_flt(best_metrics.cv_std)}")
        if best_metrics.confusion_matrix is not None:
            st.markdown("#### Confusion Matrix")
            st.plotly_chart(confusion_matrix_chart(best_metrics.confusion_matrix,
                                                    labels=best_metrics.extra.get("class_labels")),
                            use_container_width=True)
    elif task == TaskType.REGRESSION:
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("RMSE", _flt(best_metrics.rmse))
        cc2.metric("MAE", _flt(best_metrics.mae))
        cc3.metric("R²", _flt(best_metrics.r2))
        cc4.metric("CV Mean", _flt(best_metrics.cv_mean))
    else:
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Clusters", str(best_metrics.n_clusters))
        cc2.metric("Silhouette", _flt(best_metrics.silhouette))
        cc3.metric("Inertia", _flt(best_metrics.inertia))
        if best_result.cluster_labels is not None:
            st.markdown("#### Cluster Scatter")
            plot_df = best_result.X_train.copy()
            plot_df["cluster"] = best_result.cluster_labels.astype(str)
            st.plotly_chart(cluster_scatter(plot_df, "cluster"), use_container_width=True)

    st.markdown("---")
    if task != TaskType.CLUSTERING:
        st.subheader("Feature Importance")
        try:
            with st.spinner("Computing feature importances…"):
                imp_result = compute_feature_importance(best_result, method="auto", n_repeats=5)
            names = [e.feature for e in imp_result.ranked]
            scores = [e.importance for e in imp_result.ranked]
            st.plotly_chart(feature_importance_chart(names, scores, top_n=20,
                                                      title=f"Feature Importance — {imp_result.model_name}"),
                            use_container_width=True)
            st.caption(f"Method: **{imp_result.method}**")
        except Exception as exc:
            st.warning(f"Feature importance unavailable: {exc}")

    return {"train_results": all_results, "metrics": all_metrics,
            "best_result": best_result, "best_metrics": best_metrics, "task_type": task}


# ---------------------------------------------------------------------------
# Insights tab
# ---------------------------------------------------------------------------
def _get_watsonx_client() -> WatsonxClient | None:
    client = WatsonxClient()
    return client if client.is_configured else None


def _flatten_profile_for_prompt(profile: dict[str, Any]) -> dict[str, Any]:
    shape = profile.get("shape", {})
    n_rows = shape.get("rows", 0)
    n_cols = shape.get("columns", 0)
    null_counts = profile.get("null_counts", {})
    total_nulls = sum(null_counts.values())
    num_stats = profile.get("numeric_stats", {})
    cat_stats = profile.get("categorical_stats", {})
    return {
        "n_rows": n_rows, "n_cols": n_cols,
        "n_numeric": len(num_stats), "n_categorical": len(cat_stats),
        "missing_cells": total_nulls,
        "missing_pct": total_nulls / max(n_rows * n_cols, 1) * 100,
        "duplicate_rows": 0,
        "memory_usage_mb": profile.get("memory_usage_bytes", 0) / (1024 * 1024),
        "numeric_summary": {col: {"mean": s.get("mean"), "std": s.get("std"),
                                   "min": s.get("min"), "max": s.get("max")}
                            for col, s in num_stats.items() if s.get("mean") is not None},
        "top_categories": {col: list(s.get("top_values", {}).items())[:3] for col, s in cat_stats.items()},
    }


def _flatten_eda_for_prompt(eda_result: dict[str, Any]) -> dict[str, Any]:
    bivariate = eda_result.get("bivariate", {})
    pearson = bivariate.get("correlation", {}).get("pearson", {})
    top_corrs: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()
    for col_a, others in pearson.items():
        for col_b, val in others.items():
            if col_a == col_b:
                continue
            key = frozenset({col_a, col_b})
            if key in seen:
                continue
            seen.add(key)
            if val is not None:
                top_corrs.append({"feature_a": col_a, "feature_b": col_b, "correlation": val})
    top_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    univariate = eda_result.get("univariate", {})
    numeric_uni = univariate.get("numeric", {})
    skewed = [{"column": col, "skewness": s.get("skewness", 0)}
              for col, s in numeric_uni.items()
              if s.get("skewness") is not None and abs(s.get("skewness", 0)) > 1]
    skewed.sort(key=lambda x: abs(x["skewness"]), reverse=True)
    return {
        "top_correlations": top_corrs[:10], "skewed_columns": skewed[:10], "outlier_counts": {},
        "distribution_notes": [f"{col}: skewness={s.get('skewness', 0):.2f}"
                                for col, s in numeric_uni.items()
                                if s.get("skewness") is not None][:10],
        "group_stats": {},
    }


def _render_nl_result(result: NLQueryResult) -> None:
    with st.container():
        st.markdown(f"**Q: {result.question}**")
        if result.error:
            st.error(f"Error: {result.error}")
        else:
            st.markdown(f"*Query type:* `{result.query_type}`")
            if result.computed_result is not None:
                if isinstance(result.computed_result, dict):
                    st.dataframe(pd.DataFrame([{"Key": k, "Value": v}
                                               for k, v in result.computed_result.items()]),
                                 use_container_width=True)
                elif isinstance(result.computed_result, list):
                    if result.computed_result and isinstance(result.computed_result[0], dict):
                        st.dataframe(pd.DataFrame(result.computed_result), use_container_width=True)
                    else:
                        st.write(result.computed_result)
                else:
                    st.write(result.computed_result)
            if result.explanation:
                st.success(result.explanation)
        st.divider()


def _render_insights_tab(df: pd.DataFrame, profile: dict[str, Any] | None,
                          eda_result: dict[str, Any] | None,
                          best_metrics: Any | None, best_result: Any | None) -> list[NLQueryResult]:
    st.header("💡 AI Insights")
    client = _get_watsonx_client()
    genai_available = client is not None

    if not genai_available:
        st.warning(
            "⚠️ **GenAI unavailable** — IBM watsonx.ai credentials are not configured.\n\n"
            "Set `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` to enable AI narrative insights. "
            "Deterministic computed results are still shown."
        )

    st.subheader("🔍 Ask a Question About Your Data")
    nl_history: list[NLQueryResult] = st.session_state.get("nl_query_history", [])
    with st.form("nl_query_form", clear_on_submit=True):
        question = st.text_input("Your question",
                                  placeholder="e.g. What is the average age? How many rows?")
        submitted = st.form_submit_button("Ask")
    if submitted and question.strip():
        with st.spinner("Querying data…"):
            result = NLQueryEngine(df, watsonx_client=client).ask(question)
        nl_history.append(result)
        st.session_state["nl_query_history"] = nl_history

    for res in reversed(nl_history[-5:]):
        _render_nl_result(res)

    st.markdown("---")
    st.subheader("📝 AI-Generated Insights")
    if not genai_available:
        st.info("Configure watsonx.ai credentials to generate narrative insights.")
    else:
        assert client is not None
        sections = ["Dataset Summary", "Key Findings", "EDA Explanation", "Model Explanation", "Recommendations"]
        selected_section = st.selectbox("Generate insight for:", options=sections, key="genai_section")
        if st.button("✨ Generate Insight", key="btn_gen_insight"):
            with st.spinner(f"Generating {selected_section}…"):
                try:
                    flat_profile = _flatten_profile_for_prompt(profile) if profile else None
                    flat_eda = _flatten_eda_for_prompt(eda_result) if eda_result else None
                    prompt: str | None = None
                    if selected_section == "Dataset Summary" and flat_profile:
                        prompt = build_dataset_summary_prompt(flat_profile)
                    elif selected_section == "Key Findings" and flat_profile and flat_eda:
                        prompt = build_key_findings_prompt(flat_profile, flat_eda)
                    elif selected_section == "EDA Explanation" and flat_eda:
                        prompt = build_eda_explanation_prompt(flat_eda)
                    elif selected_section == "Model Explanation" and best_metrics:
                        md = best_metrics.to_dict()
                        imp_list: list[dict] = []
                        if best_result:
                            try:
                                imp = compute_feature_importance(best_result, method="auto", n_repeats=5)
                                imp_list = [{"feature": e.feature, "importance": e.importance} for e in imp.ranked[:10]]
                            except Exception:
                                pass
                        md["n_samples"] = len(best_result.X_train) if best_result else "N/A"
                        md["n_features"] = len(best_result.feature_names) if best_result else "N/A"
                        prompt = build_model_explanation_prompt(md, imp_list)
                    elif selected_section == "Recommendations":
                        ctx: dict[str, Any] = {}
                        if flat_profile:
                            ctx["dataset_summary"] = flat_profile
                        if flat_eda:
                            ctx["eda_highlights"] = [
                                f"{e['feature_a']} ↔ {e['feature_b']} = {e['correlation']:.4f}"
                                for e in flat_eda.get("top_correlations", [])[:3]
                            ]
                        if best_metrics:
                            ctx["model_metrics"] = best_metrics.to_dict()
                        prompt = build_recommendations_prompt(ctx)

                    if prompt is None:
                        st.warning(f"Cannot generate '{selected_section}': required data not available yet.")
                    else:
                        insight_text = client.generate(prompt)
                        if insight_text:
                            st.session_state[f"genai_{selected_section}"] = insight_text
                        else:
                            st.warning("GenAI returned no response.")
                except Exception as exc:
                    st.error(f"GenAI call failed: {exc}")

        cached = st.session_state.get(f"genai_{selected_section}")
        if cached:
            st.markdown(f"**{selected_section}**")
            st.info(cached)

    return nl_history


# ---------------------------------------------------------------------------
# Report tab
# ---------------------------------------------------------------------------
def _extract_top_corrs(pearson: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[frozenset[str]] = set()
    pairs: list[dict[str, Any]] = []
    for col_a, others in pearson.items():
        for col_b, val in others.items():
            if col_a == col_b:
                continue
            k = frozenset({col_a, col_b})
            if k in seen:
                continue
            seen.add(k)
            if val is not None:
                pairs.append({"feature_a": col_a, "feature_b": col_b, "correlation": val})
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs[:10]


def _render_report_tab(df: pd.DataFrame | None, filename: str | None,
                        profile: dict[str, Any] | None, eda_result: dict[str, Any] | None,
                        best_metrics: Any | None, best_result: Any | None,
                        nl_query_results: list[Any]) -> None:
    st.header("📄 Report")
    if df is None:
        st.info("Upload a dataset to generate a report.")
        return

    with st.expander("Report options", expanded=True):
        report_title = st.text_input("Report title",
                                      value=f"Data Analysis Report — {filename or 'Dataset'}")
        include_sections = st.multiselect("Sections to include",
                                           options=["Data Profile", "EDA", "ML Results", "Insights", "NL Queries"],
                                           default=["Data Profile", "EDA", "ML Results", "Insights", "NL Queries"])

    if st.button("📋 Generate Report", key="btn_gen_report"):
        with st.spinner("Assembling report…"):
            context: dict[str, Any] = {"title": report_title, "dataset_name": filename or "Uploaded Dataset"}

            if "Data Profile" in include_sections and profile:
                context["profile"] = profile
                shape = profile.get("shape", {})
                null_counts = profile.get("null_counts", {})
                context["data_quality"] = {
                    "null_percentages": profile.get("null_percentages", {}),
                    "outlier_info": {},
                    "notes": [f"Dataset has {shape.get('rows', 0):,} rows and {shape.get('columns', 0)} columns.",
                               f"Total missing cells: {sum(null_counts.values()):,}"],
                }

            if "EDA" in include_sections and eda_result:
                bivariate = eda_result.get("bivariate", {})
                pearson = bivariate.get("correlation", {}).get("pearson", {})
                univariate = eda_result.get("univariate", {})
                numeric_uni = univariate.get("numeric", {})
                skewed = [{"column": col, "skewness": s.get("skewness", 0)}
                          for col, s in numeric_uni.items()
                          if s.get("skewness") is not None and abs(s.get("skewness", 0)) > 1]
                context["eda"] = {"top_correlations": _extract_top_corrs(pearson),
                                   "skewed_columns": skewed, "charts": []}

            if "ML Results" in include_sections and best_metrics:
                context["ml_metrics"] = best_metrics.to_dict()
                if best_result is not None:
                    try:
                        imp = compute_feature_importance(best_result, method="auto", n_repeats=5)
                        context["feature_importance"] = {
                            "entries": [{"feature": e.feature, "importance": e.importance} for e in imp.ranked[:20]],
                        }
                    except Exception:
                        pass

            if "Insights" in include_sections:
                genai_insights: dict[str, str] = {}
                for sec in ["Dataset Summary", "Key Findings", "EDA Explanation", "Model Explanation", "Recommendations"]:
                    cached = st.session_state.get(f"genai_{sec}")
                    if cached:
                        genai_insights[sec] = cached
                if genai_insights:
                    context["genai_insights"] = genai_insights

            if "NL Queries" in include_sections and nl_query_results:
                context["nl_query_results"] = nl_query_results

            try:
                generator = ReportGenerator(output_dir=tempfile.gettempdir())
                html_path = generator.generate(context)
                st.session_state["report_html_path"] = html_path
                st.session_state["report_html_str"] = Path(html_path).read_text(encoding="utf-8")
                st.success(f"✅ Report generated: `{html_path}`")
            except Exception as exc:
                st.error(f"❌ Report generation failed: {exc}")
                return

    html_str: str | None = st.session_state.get("report_html_str")
    if html_str:
        st.markdown("---")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("⬇️ Download HTML Report", data=html_str.encode("utf-8"),
                           file_name=f"report_{ts}.html", mime="text/html", key="btn_dl_report")
        with st.expander("📖 Preview Report", expanded=False):
            st.components.v1.html(html_str, height=800, scrolling=True)


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
def _render_landing() -> None:
    st.markdown("---")
    st.markdown("""
## Welcome! 👋

This application provides end-to-end data analytics in five steps:

| Tab | What it does |
|---|---|
| **📋 Data** | Preview the raw dataset and column info |
| **📊 EDA** | Automated profiling, statistics, and interactive charts |
| **🤖 ML** | Train & compare ML models; evaluate metrics & feature importance |
| **💡 Insights** | Ask natural-language questions; generate AI narrative insights |
| **📄 Report** | Build and download a self-contained HTML report |

### Getting Started
1. **Upload a CSV or Excel file** using the sidebar on the left.
2. *(Optional)* Select a **target column** if you want supervised ML.
3. *(Optional)* Override the **task type** (classification / regression / clustering).
4. Navigate through the tabs.

### Sample Datasets
Three sample datasets are available in the `datasets/` folder:
- `titanic_like.csv` — classification (survival prediction)
- `house_prices.csv` — regression (price prediction)
- `customers.csv` — clustering / mixed analysis
""")
    st.info(
        "🔑 **GenAI Insights** require IBM watsonx.ai credentials. "
        "Set `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` in your environment or `.env` file. "
        "All deterministic analytics work without credentials."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _init_session_state()
    st.title("🔬 AI-Powered Data Analyst")
    st.caption(
        "Upload a CSV or Excel dataset, explore it with automated EDA, "
        "train ML models, and generate AI-driven insights — powered by IBM watsonx.ai."
    )

    sidebar_state = _render_sidebar()
    df = sidebar_state["df"]
    filename = sidebar_state["filename"]
    target_column = sidebar_state["target_column"]
    task_type_override = sidebar_state["task_type"]
    model_names = sidebar_state["model_names"]

    # Reset state on dataset change
    prev_sig = st.session_state.get("_dataset_sig")
    current_sig = (filename, df.shape if df is not None else None)
    if prev_sig != current_sig:
        st.session_state.update({
            "ml_trained": False, "ml_train_results": [], "ml_metrics": [],
            "profile": None, "eda_result": None, "nl_query_history": [],
            "report_html_path": None, "report_html_str": None, "_dataset_sig": current_sig,
        })

    if df is None:
        _render_landing()
        return

    tab_data, tab_eda, tab_ml, tab_insights, tab_report = st.tabs(
        ["📋 Data", "📊 EDA", "🤖 ML", "💡 Insights", "📄 Report"]
    )

    with tab_data:
        _render_data_tab(df, filename)

    with tab_eda:
        _compute_profile(df)
        _compute_eda(df, target_column, task_type_override)
        task_str = task_type_override.value if task_type_override else None
        _render_eda_tab(df, target_column=target_column, task_type=task_str)

    with tab_ml:
        ml_state = _render_ml_tab(df, target_column=target_column,
                                   task_type_override=task_type_override,
                                   selected_model_names=model_names)

    with tab_insights:
        profile = st.session_state.get("profile")
        eda_result = st.session_state.get("eda_result")
        best_result_ins = best_metrics_ins = None
        ml_results = st.session_state.get("ml_train_results", [])
        ml_metrics_list = st.session_state.get("ml_metrics", [])
        if ml_results and ml_metrics_list:
            ml_task = st.session_state.get("ml_task_type")
            if ml_task:
                best_result_ins, best_metrics_ins = _pick_best_model(ml_results, ml_metrics_list, ml_task)
        _render_insights_tab(df=df, profile=profile, eda_result=eda_result,
                              best_metrics=best_metrics_ins, best_result=best_result_ins)

    with tab_report:
        report_ml_results = st.session_state.get("ml_train_results", [])
        report_ml_metrics = st.session_state.get("ml_metrics", [])
        report_best_result = report_best_metrics = None
        if report_ml_results and report_ml_metrics:
            report_task = st.session_state.get("ml_task_type")
            if report_task:
                report_best_result, report_best_metrics = _pick_best_model(
                    report_ml_results, report_ml_metrics, report_task
                )
        _render_report_tab(
            df=df, filename=filename,
            profile=st.session_state.get("profile"),
            eda_result=st.session_state.get("eda_result"),
            best_metrics=report_best_metrics,
            best_result=report_best_result,
            nl_query_results=st.session_state.get("nl_query_history", []),
        )


if __name__ == "__main__":
    main()
