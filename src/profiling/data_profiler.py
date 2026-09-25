"""Profiling module — computes a comprehensive statistical profile of a DataFrame.

All calculations are deterministic (no randomness, no GenAI).
The original DataFrame is never modified.
Outputs are structured as a plain ``ProfileReport`` dict suitable for consumption
by the GenAI prompt-builder or any downstream reporting step.

Design notes
------------
* Numeric statistics use ``pandas`` / ``scipy`` — reproducible across runs.
* Correlation matrices are computed for both Pearson and Spearman methods.
* Empty or fully-null columns are handled gracefully (stats default to ``None``).
* Unsupported dtypes (e.g. ``object`` columns containing mixed types) are
  categorised as ``"other"`` and receive only null / cardinality stats.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

__all__ = ["DataProfiler", "profile_dataframe"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DataProfiler:
    """Compute a full statistical profile of a :class:`pd.DataFrame`.

    Parameters
    ----------
    top_n:
        Number of most-frequent values to record in categorical frequency
        distributions. Defaults to 10.
    """

    def __init__(self, top_n: int = 10) -> None:
        self.top_n = top_n

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def profile(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return a ``ProfileReport`` dict for *df*.

        The returned dict has the following top-level keys:

        * ``shape``               — ``{"rows": int, "columns": int}``
        * ``memory_usage_bytes``  — total DataFrame memory in bytes
        * ``dtypes``              — per-column dtype string
        * ``null_counts``         — per-column null count
        * ``null_percentages``    — per-column null percentage (0–100)
        * ``cardinality``         — per-column unique non-null value count
        * ``numeric_stats``       — per-column numeric summary (or ``{}`` if none)
        * ``categorical_stats``   — per-column categorical summary (or ``{}`` if none)
        * ``correlation``         — ``{"pearson": …, "spearman": …}``

        Parameters
        ----------
        df:
            Input DataFrame. Must not be empty.

        Returns
        -------
        dict[str, Any]
            Structured profile report. All numeric values are Python built-ins
            (``int``, ``float``, or ``None``) so the dict is JSON-serialisable.
        """
        # Work on a copy so the original is never mutated.
        df = df.copy()

        report: dict[str, Any] = {}

        report["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
        report["memory_usage_bytes"] = int(df.memory_usage(deep=True).sum())
        report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        report["null_counts"] = {
            col: int(df[col].isnull().sum()) for col in df.columns
        }
        report["null_percentages"] = {
            col: _safe_float(df[col].isnull().mean() * 100) for col in df.columns
        }
        report["cardinality"] = {
            col: int(df[col].dropna().nunique()) for col in df.columns
        }

        report["numeric_stats"] = self._numeric_stats(df)
        report["categorical_stats"] = self._categorical_stats(df)
        report["correlation"] = self._correlation(df)

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _numeric_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return per-column statistics for all numeric columns."""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        result: dict[str, Any] = {}

        for col in numeric_cols:
            series = df[col].dropna()

            if series.empty:
                result[col] = _empty_numeric_stats()
                continue

            values = series.to_numpy(dtype=float)

            q1 = float(np.percentile(values, 25))
            q3 = float(np.percentile(values, 75))

            skewness: float | None
            kurtosis: float | None
            try:
                skewness = float(scipy_stats.skew(values, bias=True))
                kurtosis = float(scipy_stats.kurtosis(values, bias=True))
            except Exception:
                skewness = None
                kurtosis = None

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
                "skewness": _safe_float(skewness) if skewness is not None else None,
                "kurtosis": _safe_float(kurtosis) if kurtosis is not None else None,
            }

        return result

    def _categorical_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return per-column statistics for all non-numeric columns."""
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()
        result: dict[str, Any] = {}

        for col in cat_cols:
            series = df[col].dropna()

            if series.empty:
                result[col] = _empty_categorical_stats()
                continue

            value_counts = series.value_counts()
            top_n_values = {
                str(k): int(v)
                for k, v in value_counts.head(self.top_n).items()
            }
            mode_val = value_counts.index[0] if not value_counts.empty else None

            result[col] = {
                "count": int(series.count()),
                "cardinality": int(series.nunique()),
                "mode": str(mode_val) if mode_val is not None else None,
                "top_values": top_n_values,
            }

        return result

    def _correlation(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return Pearson and Spearman correlation matrices for numeric columns.

        Matrices are represented as nested dicts ``{col_a: {col_b: value}}``.
        If fewer than two numeric columns exist, both matrices are empty dicts.
        """
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.shape[1] < 2:
            return {"pearson": {}, "spearman": {}}

        pearson_df = numeric_df.corr(method="pearson")
        spearman_df = numeric_df.corr(method="spearman")

        return {
            "pearson": _corr_df_to_dict(pearson_df),
            "spearman": _corr_df_to_dict(spearman_df),
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def profile_dataframe(df: pd.DataFrame, top_n: int = 10) -> dict[str, Any]:
    """Compute and return a full statistical profile for *df*.

    Convenience wrapper around :class:`DataProfiler`.

    Parameters
    ----------
    df:
        Input DataFrame.
    top_n:
        Number of most-frequent values to include in categorical stats.

    Returns
    -------
    dict[str, Any]
        Structured ``ProfileReport`` dictionary.
    """
    return DataProfiler(top_n=top_n).profile(df)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    """Convert *value* to a Python ``float``, returning ``None`` for NaN/Inf."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _empty_numeric_stats() -> dict[str, Any]:
    """Return a zeroed-out numeric stats dict for an all-null column."""
    return {
        "count": 0,
        "mean": None,
        "median": None,
        "std": None,
        "min": None,
        "max": None,
        "q1": None,
        "q3": None,
        "iqr": None,
        "skewness": None,
        "kurtosis": None,
    }


def _empty_categorical_stats() -> dict[str, Any]:
    """Return an empty categorical stats dict for an all-null column."""
    return {
        "count": 0,
        "cardinality": 0,
        "mode": None,
        "top_values": {},
    }


def _corr_df_to_dict(corr_df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Convert a correlation DataFrame to a nested plain-dict representation."""
    result: dict[str, dict[str, float | None]] = {}
    for col in corr_df.columns:
        result[str(col)] = {
            str(other): _safe_float(corr_df.loc[col, other])
            for other in corr_df.columns
        }
    return result
