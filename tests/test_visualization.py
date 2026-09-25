"""Tests for src/visualization/chart_builder.py — ST-06."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization.chart_builder import (
    bar_chart,
    box_plot,
    cluster_scatter,
    confusion_matrix_chart,
    correlation_heatmap,
    feature_importance_chart,
    histogram,
    pair_plot,
    pie_chart,
    roc_curve_chart,
    scatter_plot,
    time_series_chart,
    violin_plot,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    """DataFrame with purely numeric columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, 50),
            "b": rng.normal(5, 2, 50),
            "c": rng.uniform(0, 10, 50),
        }
    )


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    """DataFrame with numeric and categorical columns."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "value": rng.normal(100, 15, 40),
            "score": rng.uniform(0, 1, 40),
            "category": np.tile(["A", "B", "C", "D"], 10),
            "group": np.tile(["X", "Y"], 20),
        }
    )


@pytest.fixture()
def time_df() -> pd.DataFrame:
    """DataFrame with a datetime column and a numeric series."""
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "date": dates,
            "sales": rng.integers(100, 500, 20).astype(float),
            "costs": rng.integers(50, 200, 20).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _is_figure(obj: object) -> bool:
    return isinstance(obj, go.Figure)


# ---------------------------------------------------------------------------
# 1. Histogram / KDE
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_returns_figure(self, numeric_df: pd.DataFrame) -> None:
        fig = histogram(numeric_df, "a")
        assert _is_figure(fig)

    def test_with_kde(self, numeric_df: pd.DataFrame) -> None:
        fig = histogram(numeric_df, "b", kde=True)
        assert _is_figure(fig)
        # Histogram + KDE scatter trace
        assert len(fig.data) >= 2

    def test_without_kde(self, numeric_df: pd.DataFrame) -> None:
        fig = histogram(numeric_df, "a", kde=False)
        assert _is_figure(fig)
        assert len(fig.data) == 1

    def test_missing_column_returns_empty_figure(self, numeric_df: pd.DataFrame) -> None:
        fig = histogram(numeric_df, "nonexistent")
        assert _is_figure(fig)

    def test_all_null_returns_empty_figure(self) -> None:
        df = pd.DataFrame({"x": [np.nan] * 5})
        fig = histogram(df, "x")
        assert _is_figure(fig)

    def test_custom_title(self, numeric_df: pd.DataFrame) -> None:
        fig = histogram(numeric_df, "a", title="My Title")
        assert fig.layout.title.text == "My Title"


# ---------------------------------------------------------------------------
# 2. Bar chart
# ---------------------------------------------------------------------------


class TestBarChart:
    def test_frequency_bar(self, mixed_df: pd.DataFrame) -> None:
        fig = bar_chart(mixed_df, "category")
        assert _is_figure(fig)
        assert len(fig.data) >= 1

    def test_xy_bar(self, mixed_df: pd.DataFrame) -> None:
        fig = bar_chart(mixed_df, "category", "value")
        assert _is_figure(fig)

    def test_missing_x_col(self, mixed_df: pd.DataFrame) -> None:
        fig = bar_chart(mixed_df, "missing_col")
        assert _is_figure(fig)

    def test_missing_y_col(self, mixed_df: pd.DataFrame) -> None:
        fig = bar_chart(mixed_df, "category", "missing_y")
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# 3. Pie chart
# ---------------------------------------------------------------------------


class TestPieChart:
    def test_returns_figure(self, mixed_df: pd.DataFrame) -> None:
        fig = pie_chart(mixed_df, "category")
        assert _is_figure(fig)

    def test_labels_present(self, mixed_df: pd.DataFrame) -> None:
        fig = pie_chart(mixed_df, "category")
        pie_trace = fig.data[0]
        assert hasattr(pie_trace, "labels")

    def test_missing_column(self, mixed_df: pd.DataFrame) -> None:
        fig = pie_chart(mixed_df, "no_such_col")
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------------------


class TestCorrelationHeatmap:
    def test_returns_figure(self, numeric_df: pd.DataFrame) -> None:
        fig = correlation_heatmap(numeric_df)
        assert _is_figure(fig)

    def test_heatmap_trace_present(self, numeric_df: pd.DataFrame) -> None:
        fig = correlation_heatmap(numeric_df)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_only_one_numeric_col_returns_empty(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        fig = correlation_heatmap(df)
        assert _is_figure(fig)

    def test_spearman_method(self, numeric_df: pd.DataFrame) -> None:
        fig = correlation_heatmap(numeric_df, method="spearman")
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# 5. Scatter plot
# ---------------------------------------------------------------------------


class TestScatterPlot:
    def test_returns_figure(self, numeric_df: pd.DataFrame) -> None:
        fig = scatter_plot(numeric_df, "a", "b")
        assert _is_figure(fig)

    def test_with_color(self, mixed_df: pd.DataFrame) -> None:
        fig = scatter_plot(mixed_df, "value", "score", color_col="category")
        assert _is_figure(fig)

    def test_missing_column(self, numeric_df: pd.DataFrame) -> None:
        fig = scatter_plot(numeric_df, "a", "missing")
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# 6. Pair plot
# ---------------------------------------------------------------------------


class TestPairPlot:
    def test_returns_figure(self, numeric_df: pd.DataFrame) -> None:
        fig = pair_plot(numeric_df)
        assert _is_figure(fig)

    def test_with_explicit_columns(self, numeric_df: pd.DataFrame) -> None:
        fig = pair_plot(numeric_df, columns=["a", "b"])
        assert _is_figure(fig)

    def test_single_column_returns_empty(self, numeric_df: pd.DataFrame) -> None:
        fig = pair_plot(numeric_df, columns=["a"])
        assert _is_figure(fig)

    def test_with_color_col(self, mixed_df: pd.DataFrame) -> None:
        fig = pair_plot(mixed_df, columns=["value", "score"], color_col="category")
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# 7. Box plot
# ---------------------------------------------------------------------------


class TestBoxPlot:
    def test_returns_figure(self, mixed_df: pd.DataFrame) -> None:
        fig = box_plot(mixed_df, "value")
        assert _is_figure(fig)

    def test_grouped(self, mixed_df: pd.DataFrame) -> None:
        fig = box_plot(mixed_df, "value", x_col="category")
        assert _is_figure(fig)

    def test_missing_y_col(self, mixed_df: pd.DataFrame) -> None:
        fig = box_plot(mixed_df, "missing")
        assert _is_figure(fig)

    def test_missing_x_col(self, mixed_df: pd.DataFrame) -> None:
        fig = box_plot(mixed_df, "value", x_col="no_col")
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# 8. Violin plot
# ---------------------------------------------------------------------------


class TestViolinPlot:
    def test_returns_figure(self, mixed_df: pd.DataFrame) -> None:
        fig = violin_plot(mixed_df, "value")
        assert _is_figure(fig)

    def test_grouped(self, mixed_df: pd.DataFrame) -> None:
        fig = violin_plot(mixed_df, "value", x_col="group")
        assert _is_figure(fig)

    def test_missing_y_col(self, mixed_df: pd.DataFrame) -> None:
        fig = violin_plot(mixed_df, "not_there")
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# 9. ROC curve
# ---------------------------------------------------------------------------


class TestRocCurve:
    def test_returns_figure(self) -> None:
        fpr = [0.0, 0.1, 0.3, 0.6, 1.0]
        tpr = [0.0, 0.5, 0.7, 0.9, 1.0]
        fig = roc_curve_chart(fpr, tpr, auc=0.82)
        assert _is_figure(fig)

    def test_two_traces(self) -> None:
        """Should have ROC trace + diagonal baseline."""
        fig = roc_curve_chart([0, 1], [0, 1])
        assert len(fig.data) == 2

    def test_empty_data(self) -> None:
        fig = roc_curve_chart([], [])
        assert _is_figure(fig)

    def test_auc_in_legend_label(self) -> None:
        fig = roc_curve_chart([0, 0.5, 1], [0, 0.8, 1], auc=0.9)
        assert "0.900" in fig.data[0].name


# ---------------------------------------------------------------------------
# 10. Confusion matrix heatmap
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_returns_figure(self) -> None:
        cm = np.array([[50, 5], [3, 42]])
        fig = confusion_matrix_chart(cm)
        assert _is_figure(fig)

    def test_with_labels(self) -> None:
        cm = np.array([[10, 2], [1, 15]])
        fig = confusion_matrix_chart(cm, labels=["cat", "dog"])
        assert _is_figure(fig)
        heatmap = fig.data[0]
        assert list(heatmap.x) == ["cat", "dog"]

    def test_multiclass(self) -> None:
        cm = np.eye(3, dtype=int) * 10
        fig = confusion_matrix_chart(cm, labels=["A", "B", "C"])
        assert _is_figure(fig)

    def test_invalid_input(self) -> None:
        fig = confusion_matrix_chart(np.array([]))
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# 11. Feature importance bar chart
# ---------------------------------------------------------------------------


class TestFeatureImportance:
    def test_returns_figure(self) -> None:
        names = ["feat_a", "feat_b", "feat_c"]
        imps = [0.5, 0.3, 0.2]
        fig = feature_importance_chart(names, imps)
        assert _is_figure(fig)

    def test_top_n(self) -> None:
        names = [f"f{i}" for i in range(30)]
        imps = list(range(30, 0, -1))
        fig = feature_importance_chart(names, imps, top_n=10)
        assert _is_figure(fig)
        # Only top 10 bars
        assert len(fig.data[0].x) == 10

    def test_empty_input(self) -> None:
        fig = feature_importance_chart([], [])
        assert _is_figure(fig)

    def test_sorted_descending(self) -> None:
        names = ["low", "high", "mid"]
        imps = [0.1, 0.9, 0.5]
        fig = feature_importance_chart(names, imps)
        # First bar should be the highest importance
        assert fig.data[0].y[0] == "high"


# ---------------------------------------------------------------------------
# 12. Cluster scatter
# ---------------------------------------------------------------------------


class TestClusterScatter:
    def test_explicit_xy(self, numeric_df: pd.DataFrame) -> None:
        df = numeric_df.copy()
        df["cluster"] = np.tile([0, 1], 25)
        fig = cluster_scatter(df, "cluster", x_col="a", y_col="b")
        assert _is_figure(fig)

    def test_pca_fallback(self, numeric_df: pd.DataFrame) -> None:
        df = numeric_df.copy()
        df["cluster"] = np.tile([0, 1], 25)
        fig = cluster_scatter(df, "cluster")
        assert _is_figure(fig)

    def test_missing_cluster_col(self, numeric_df: pd.DataFrame) -> None:
        fig = cluster_scatter(numeric_df, "no_cluster")
        assert _is_figure(fig)

    def test_no_numeric_features(self) -> None:
        df = pd.DataFrame({"cluster": ["A", "B", "A"], "cat": ["x", "y", "x"]})
        fig = cluster_scatter(df, "cluster")
        assert _is_figure(fig)


# ---------------------------------------------------------------------------
# 13. Time-series line chart
# ---------------------------------------------------------------------------


class TestTimeSeriesChart:
    def test_returns_figure(self, time_df: pd.DataFrame) -> None:
        fig = time_series_chart(time_df, "date", ["sales"])
        assert _is_figure(fig)

    def test_multiple_value_cols(self, time_df: pd.DataFrame) -> None:
        fig = time_series_chart(time_df, "date", ["sales", "costs"])
        assert _is_figure(fig)
        assert len(fig.data) == 2

    def test_missing_time_col(self, time_df: pd.DataFrame) -> None:
        fig = time_series_chart(time_df, "no_date", ["sales"])
        assert _is_figure(fig)

    def test_missing_value_col(self, time_df: pd.DataFrame) -> None:
        fig = time_series_chart(time_df, "date", ["missing_col"])
        assert _is_figure(fig)

    def test_string_dates_parsed(self) -> None:
        df = pd.DataFrame(
            {"ts": ["2023-01-01", "2023-01-02", "2023-01-03"], "val": [1.0, 2.0, 3.0]}
        )
        fig = time_series_chart(df, "ts", ["val"])
        assert _is_figure(fig)
        assert len(fig.data) == 1
