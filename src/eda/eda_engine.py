"""EDA module — structured exploratory data analysis of a DataFrame.

All calculations are deterministic (no randomness, no GenAI).
The original DataFrame is never modified.
Outputs are plain Python dicts suitable for consumption by the
visualization layer, GenAI prompt-builder, or reporting module.

Design notes
------------
* Univariate analysis covers both numeric and categorical columns.
* Bivariate analysis provides correlation matrices and pairwise scatter data.
* Group-by analysis aggregates numeric columns by categorical groupers.
* Target-variable analysis supports classification (class distribution) and
  regression (distribution statistics + per-numeric-column means by target bin).
* Missing values and unsuitable column types are handled safely — no hard
  failures for degenerate inputs.
* All numeric results are Python built-ins (int, float, None) so the returned
  dicts are JSON-serialisable.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["EDAEngine", "run_eda"]

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

EDAResult = dict[str, Any]

# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class EDAEngine:
    """Compute a structured EDA report for a :class:`pd.DataFrame`.

    Parameters
    ----------
    top_n_categories:
        Maximum number of top categories to retain in frequency tables.
        Defaults to 15.
    scatter_sample_size:
        Maximum number of rows to include in scatter-plot data (random sample
        drawn *without* randomness — uses the first N rows after sorting).
        Defaults to 500.
    n_bins:
        Number of equal-width bins used when creating numeric histograms and
        when binning a continuous target for regression relationship analysis.
        Defaults to 10.
    """

    def __init__(
        self,
        top_n_categories: int = 15,
        scatter_sample_size: int = 500,
        n_bins: int = 10,
    ) -> None:
        self.top_n_categories = top_n_categories
        self.scatter_sample_size = scatter_sample_size
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        task_type: str | None = None,
    ) -> EDAResult:
        """Return a full EDA result dict for *df*.

        Parameters
        ----------
        df:
            Input DataFrame. Not modified.
        target_column:
            Name of the target / label column for supervised-learning context.
            When ``None``, target-variable analysis is skipped.
        task_type:
            ``"classification"``, ``"regression"``, or ``None`` to auto-detect
            based on the target column's dtype and cardinality.

        Returns
        -------
        EDAResult
            Dict with keys:
            ``"univariate"``, ``"bivariate"``, ``"groupby"``,
            ``"target_analysis"`` (present only when *target_column* is given).
        """
        # Never mutate the caller's DataFrame.
        df = df.copy()

        result: EDAResult = {}
        result["univariate"] = self._univariate(df)
        result["bivariate"] = self._bivariate(df)
        result["groupby"] = self._groupby(df)

        if target_column is not None and target_column in df.columns:
            resolved_task = task_type or _infer_task(df[target_column])
            result["target_analysis"] = self._target_analysis(
                df, target_column, resolved_task
            )

        return result

    # ------------------------------------------------------------------
    # 1. Univariate analysis
    # ------------------------------------------------------------------

    def _univariate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute per-column univariate statistics."""
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

        return {
            "numeric": {col: self._numeric_univariate(df[col]) for col in numeric_cols},
            "categorical": {
                col: self._categorical_univariate(df[col]) for col in categorical_cols
            },
        }

    def _numeric_univariate(self, series: pd.Series) -> dict[str, Any]:
        """Summary statistics + histogram bins for a numeric series."""
        clean = series.dropna()

        if clean.empty:
            return _empty_numeric_univariate()

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
            "histogram": {
                "counts": hist_counts.tolist(),
                "bin_edges": [_sf(e) for e in bin_edges.tolist()],
            },
        }

    def _categorical_univariate(self, series: pd.Series) -> dict[str, Any]:
        """Frequency table + summary for a non-numeric series."""
        clean = series.dropna()

        if clean.empty:
            return _empty_categorical_univariate()

        value_counts = clean.value_counts()
        top_values = {
            str(k): int(v) for k, v in value_counts.head(self.top_n_categories).items()
        }
        mode_val = value_counts.index[0] if not value_counts.empty else None

        return {
            "count": int(clean.count()),
            "null_count": int(series.isnull().sum()),
            "cardinality": int(clean.nunique()),
            "mode": str(mode_val) if mode_val is not None else None,
            "top_values": top_values,
            "top_values_pct": {
                k: _sf(v / len(clean) * 100) for k, v in top_values.items()
            },
        }

    # ------------------------------------------------------------------
    # 2. Bivariate analysis
    # ------------------------------------------------------------------

    def _bivariate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Correlation matrices and scatter-plot data for numeric columns."""
        numeric_df = df.select_dtypes(include="number")

        return {
            "correlation": self._correlation(numeric_df),
            "scatter_data": self._scatter_data(numeric_df),
        }

    def _correlation(self, numeric_df: pd.DataFrame) -> dict[str, Any]:
        """Pearson and Spearman correlation matrices as nested dicts."""
        if numeric_df.shape[1] < 2:
            return {"pearson": {}, "spearman": {}}

        pearson = numeric_df.corr(method="pearson")
        spearman = numeric_df.corr(method="spearman")

        return {
            "pearson": _corr_to_dict(pearson),
            "spearman": _corr_to_dict(spearman),
        }

    def _scatter_data(self, numeric_df: pd.DataFrame) -> dict[str, Any]:
        """Return x/y arrays for every unique numeric column pair.

        Data is capped at *scatter_sample_size* rows (first N after dropping
        nulls on both columns) to keep payloads small.
        """
        cols = numeric_df.columns.tolist()
        scatter: dict[str, Any] = {}

        for i, col_x in enumerate(cols):
            for col_y in cols[i + 1 :]:
                pair_df = numeric_df[[col_x, col_y]].dropna()
                if pair_df.empty:
                    continue
                # Take first N rows (deterministic, no random sampling)
                pair_df = pair_df.head(self.scatter_sample_size)
                key = f"{col_x}_vs_{col_y}"
                scatter[key] = {
                    "x": pair_df[col_x].tolist(),
                    "y": pair_df[col_y].tolist(),
                    "x_col": col_x,
                    "y_col": col_y,
                }

        return scatter

    # ------------------------------------------------------------------
    # 3. Group-by analysis
    # ------------------------------------------------------------------

    def _groupby(self, df: pd.DataFrame) -> dict[str, Any]:
        """Aggregate numeric columns grouped by each categorical column.

        Aggregations computed: mean, sum, count, median.
        """
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

        if not numeric_cols or not categorical_cols:
            return {}

        groupby_results: dict[str, Any] = {}

        for cat_col in categorical_cols:
            # Skip high-cardinality columns (more than 50 unique values) to
            # avoid producing extremely large result dicts.
            n_unique = df[cat_col].dropna().nunique()
            if n_unique == 0 or n_unique > 50:
                continue

            col_results: dict[str, Any] = {}
            for num_col in numeric_cols:
                pair = df[[cat_col, num_col]].dropna(subset=[num_col])
                if pair.empty:
                    continue

                grouped = pair.groupby(cat_col, sort=True)[num_col]
                col_results[num_col] = {
                    "mean": _series_to_dict(grouped.mean()),
                    "sum": _series_to_dict(grouped.sum()),
                    "count": _series_to_dict(grouped.count()),
                    "median": _series_to_dict(grouped.median()),
                }

            if col_results:
                groupby_results[cat_col] = col_results

        return groupby_results

    # ------------------------------------------------------------------
    # 4. Target-variable analysis
    # ------------------------------------------------------------------

    def _target_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str,
    ) -> dict[str, Any]:
        """Return target-specific analysis depending on *task_type*."""
        target = df[target_column]
        result: dict[str, Any] = {
            "column": target_column,
            "task_type": task_type,
            "null_count": int(target.isnull().sum()),
        }

        if task_type == "classification":
            result.update(self._classification_target(df, target_column))
        else:  # regression (or unknown — fall back to numeric treatment)
            result.update(self._regression_target(df, target_column))

        return result

    def _classification_target(
        self, df: pd.DataFrame, target_column: str
    ) -> dict[str, Any]:
        """Class distribution and balance metrics for a classification target."""
        target = df[target_column].dropna()
        value_counts = target.value_counts(sort=True)
        n_total = len(target)
        class_dist = {str(k): int(v) for k, v in value_counts.items()}
        class_pct = {str(k): _sf(v / n_total * 100) for k, v in value_counts.items()}

        most_common = value_counts.index[0] if not value_counts.empty else None
        least_common = value_counts.index[-1] if not value_counts.empty else None
        imbalance_ratio: float | None = None
        if len(value_counts) >= 2:
            imbalance_ratio = _sf(
                float(value_counts.iloc[0]) / float(value_counts.iloc[-1])
            )

        return {
            "n_classes": int(target.nunique()),
            "class_distribution": class_dist,
            "class_distribution_pct": class_pct,
            "most_common_class": str(most_common) if most_common is not None else None,
            "least_common_class": str(least_common) if least_common is not None else None,
            "imbalance_ratio": imbalance_ratio,
        }

    def _regression_target(
        self, df: pd.DataFrame, target_column: str
    ) -> dict[str, Any]:
        """Distribution stats + per-feature mean-by-bin for a regression target."""
        target = df[target_column].dropna()

        if target.empty:
            return {"distribution": _empty_numeric_univariate(), "feature_means": {}}

        values = target.to_numpy(dtype=float)
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        hist_counts, bin_edges = np.histogram(values, bins=self.n_bins)

        distribution: dict[str, Any] = {
            "count": int(target.count()),
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
            "histogram": {
                "counts": hist_counts.tolist(),
                "bin_edges": [_sf(e) for e in bin_edges.tolist()],
            },
        }

        # For each numeric feature (excluding target), compute mean target value
        # per equal-width bin of the feature — useful for spotting relationships.
        feature_cols = [
            c
            for c in df.select_dtypes(include="number").columns
            if c != target_column
        ]
        feature_means: dict[str, Any] = {}
        for feat_col in feature_cols:
            try:
                pair = df[[feat_col, target_column]].dropna()
                if pair.empty:
                    continue
                bins = pd.cut(pair[feat_col], bins=self.n_bins, duplicates="drop")
                grouped = pair.groupby(bins, observed=True)[target_column].mean()
                feature_means[feat_col] = {
                    str(interval): _sf(val) for interval, val in grouped.items()
                }
            except Exception:
                continue

        return {"distribution": distribution, "feature_means": feature_means}


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def run_eda(
    df: pd.DataFrame,
    target_column: str | None = None,
    task_type: str | None = None,
    top_n_categories: int = 15,
    scatter_sample_size: int = 500,
    n_bins: int = 10,
) -> EDAResult:
    """Run a full EDA and return a structured result dict.

    Convenience wrapper around :class:`EDAEngine`.

    Parameters
    ----------
    df:
        Input DataFrame. Not modified.
    target_column:
        Optional target / label column name.
    task_type:
        ``"classification"`` or ``"regression"`` (auto-detected when ``None``).
    top_n_categories:
        Max categories to retain in frequency tables.
    scatter_sample_size:
        Max rows per scatter-plot pair.
    n_bins:
        Number of histogram / binning bins.

    Returns
    -------
    EDAResult
        Structured EDA dictionary.
    """
    return EDAEngine(
        top_n_categories=top_n_categories,
        scatter_sample_size=scatter_sample_size,
        n_bins=n_bins,
    ).analyze(df, target_column=target_column, task_type=task_type)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _sf(value: Any) -> float | None:
    """Convert *value* to a Python float, returning None for NaN/Inf/None."""
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
    """Heuristically decide ``"classification"`` or ``"regression"``."""
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    n_unique = target.dropna().nunique()
    # Low cardinality numeric target → treat as classification
    if n_unique < 20:
        return "classification"
    return "regression"


def _corr_to_dict(
    corr_df: pd.DataFrame,
) -> dict[str, dict[str, float | None]]:
    """Convert a correlation DataFrame to a nested plain-dict."""
    result: dict[str, dict[str, float | None]] = {}
    for col in corr_df.columns:
        result[str(col)] = {
            str(other): _sf(corr_df.loc[col, other]) for other in corr_df.columns
        }
    return result


def _series_to_dict(series: pd.Series) -> dict[str, float | None]:
    """Convert an aggregated Series (index = category) to a plain dict."""
    return {str(k): _sf(v) for k, v in series.items()}


def _try_skew(values: np.ndarray) -> float | None:
    """Compute skewness, returning None on failure."""
    try:
        from scipy import stats as scipy_stats

        return float(scipy_stats.skew(values, bias=True))
    except Exception:
        return None


def _try_kurtosis(values: np.ndarray) -> float | None:
    """Compute excess kurtosis, returning None on failure."""
    try:
        from scipy import stats as scipy_stats

        return float(scipy_stats.kurtosis(values, bias=True))
    except Exception:
        return None


def _empty_numeric_univariate() -> dict[str, Any]:
    return {
        "count": 0,
        "null_count": 0,
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
        "histogram": {"counts": [], "bin_edges": []},
    }


def _empty_categorical_univariate() -> dict[str, Any]:
    return {
        "count": 0,
        "null_count": 0,
        "cardinality": 0,
        "mode": None,
        "top_values": {},
        "top_values_pct": {},
    }
