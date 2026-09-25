"""EDA tab component — data profile, statistics, and EDA visualisations."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.eda.eda_engine import run_eda
from src.profiling.data_profiler import profile_dataframe
from src.visualization.chart_builder import (
    bar_chart,
    box_plot,
    correlation_heatmap,
    histogram,
    pair_plot,
    scatter_plot,
)


def render_eda_tab(df: pd.DataFrame, target_column: str | None, task_type: str | None) -> None:
    """Render the EDA tab.

    Parameters
    ----------
    df:
        The loaded (raw) DataFrame.
    target_column:
        Name of the target column, or ``None``.
    task_type:
        Task type string (``"classification"`` / ``"regression"``), or ``None``.
    """
    st.header("📊 Exploratory Data Analysis")

    # ------------------------------------------------------------------
    # 1. Data profile overview
    # ------------------------------------------------------------------
    with st.spinner("Computing data profile…"):
        profile = profile_dataframe(df)

    _render_profile_overview(df, profile)

    st.markdown("---")

    # ------------------------------------------------------------------
    # 2. Descriptive statistics tables
    # ------------------------------------------------------------------
    _render_statistics(df, profile)

    st.markdown("---")

    # ------------------------------------------------------------------
    # 3. EDA engine results
    # ------------------------------------------------------------------
    with st.spinner("Running EDA analysis…"):
        eda_result = run_eda(df, target_column=target_column, task_type=task_type)

    _render_visualisations(df, eda_result, target_column)


# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------


def _render_profile_overview(df: pd.DataFrame, profile: dict[str, Any]) -> None:
    """Render high-level dataset profile metrics."""
    st.subheader("Dataset Overview")

    shape = profile.get("shape", {})
    n_rows = shape.get("rows", len(df))
    n_cols = shape.get("columns", len(df.columns))
    mem_bytes = profile.get("memory_usage_bytes", 0)
    mem_mb = mem_bytes / (1024 * 1024)

    null_counts = profile.get("null_counts", {})
    total_nulls = sum(null_counts.values())
    null_pct = total_nulls / max(n_rows * n_cols, 1) * 100

    num_cols = len(df.select_dtypes(include="number").columns)
    cat_cols = n_cols - num_cols

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", f"{n_rows:,}")
    col2.metric("Columns", n_cols)
    col3.metric("Numeric", num_cols)
    col4.metric("Categorical", cat_cols)
    col5.metric("Missing %", f"{null_pct:.1f}%")

    col6, col7 = st.columns(2)
    dup_count = int(df.duplicated().sum())
    col6.metric("Duplicate Rows", f"{dup_count:,}")
    col7.metric("Memory", f"{mem_mb:.2f} MB")

    # Column-level null summary
    if total_nulls > 0:
        with st.expander("Null counts per column"):
            null_df = pd.DataFrame(
                [
                    {
                        "Column": col,
                        "Null Count": cnt,
                        "Null %": f"{cnt / n_rows * 100:.1f}%",
                    }
                    for col, cnt in null_counts.items()
                    if cnt > 0
                ]
            )
            st.dataframe(null_df, use_container_width=True)


def _render_statistics(df: pd.DataFrame, profile: dict[str, Any]) -> None:
    """Render descriptive statistics tables."""
    st.subheader("Descriptive Statistics")

    tab_num, tab_cat = st.tabs(["Numeric Columns", "Categorical Columns"])

    with tab_num:
        num_stats = profile.get("numeric_stats", {})
        if num_stats:
            rows = []
            for col, stats in num_stats.items():
                rows.append(
                    {
                        "Column": col,
                        "Count": stats.get("count"),
                        "Mean": _fmt(stats.get("mean")),
                        "Std": _fmt(stats.get("std")),
                        "Min": _fmt(stats.get("min")),
                        "Q1": _fmt(stats.get("q1")),
                        "Median": _fmt(stats.get("median")),
                        "Q3": _fmt(stats.get("q3")),
                        "Max": _fmt(stats.get("max")),
                        "Skewness": _fmt(stats.get("skewness")),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with tab_cat:
        cat_stats = profile.get("categorical_stats", {})
        if cat_stats:
            rows = []
            for col, stats in cat_stats.items():
                top_vals = stats.get("top_values", {})
                top_str = ", ".join(f"{k}({v})" for k, v in list(top_vals.items())[:3])
                rows.append(
                    {
                        "Column": col,
                        "Count": stats.get("count"),
                        "Unique": stats.get("cardinality"),
                        "Mode": stats.get("mode"),
                        "Top Values": top_str,
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No categorical columns found.")


def _render_visualisations(
    df: pd.DataFrame,
    eda_result: dict[str, Any],
    target_column: str | None,
) -> None:
    """Render EDA charts."""
    st.subheader("Visualisations")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    # ---- Correlation heatmap ----
    if len(num_cols) >= 2:
        st.markdown("#### Correlation Heatmap")
        fig = correlation_heatmap(df, method="pearson")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Univariate distributions ----
    st.markdown("#### Univariate Distributions")
    if num_cols:
        selected_num = st.selectbox(
            "Select a numeric column to visualise",
            options=num_cols,
            key="eda_hist_col",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(histogram(df, selected_num), use_container_width=True)
        with col_b:
            st.plotly_chart(box_plot(df, selected_num), use_container_width=True)

    if cat_cols:
        selected_cat = st.selectbox(
            "Select a categorical column to visualise",
            options=cat_cols,
            key="eda_bar_col",
        )
        st.plotly_chart(bar_chart(df, selected_cat), use_container_width=True)

    # ---- Scatter / bivariate ----
    if len(num_cols) >= 2:
        st.markdown("#### Bivariate Scatter")
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            x_sel = st.selectbox("X axis", options=num_cols, index=0, key="eda_scatter_x")
        with sc_col2:
            y_default = 1 if len(num_cols) > 1 else 0
            y_sel = st.selectbox("Y axis", options=num_cols, index=y_default, key="eda_scatter_y")
        color_options = ["(none)"] + cat_cols
        color_sel = st.selectbox("Colour by", options=color_options, index=0, key="eda_scatter_c")
        color_col = None if color_sel == "(none)" else color_sel
        st.plotly_chart(
            scatter_plot(df, x_sel, y_sel, color_col=color_col, trendline=True),
            use_container_width=True,
        )

    # ---- Target analysis ----
    if target_column and target_column in df.columns:
        target_analysis = eda_result.get("target_analysis")
        if target_analysis:
            st.markdown(f"#### Target Column: `{target_column}`")
            task = target_analysis.get("task_type", "unknown")
            st.caption(f"Detected task: **{task}**")
            if task == "classification":
                dist = target_analysis.get("class_distribution", {})
                if dist:
                    dist_df = pd.DataFrame(
                        [{"Class": k, "Count": v} for k, v in dist.items()]
                    )
                    fig = bar_chart(dist_df, "Class", "Count", title=f"Class distribution of {target_column}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                if target_column in num_cols:
                    st.plotly_chart(histogram(df, target_column, title=f"Distribution of {target_column}"), use_container_width=True)

    # ---- Pair plot (small datasets) ----
    if len(num_cols) >= 2:
        with st.expander("Pair Plot (select columns)"):
            max_cols = min(5, len(num_cols))
            pair_sel = st.multiselect(
                "Columns for pair plot",
                options=num_cols,
                default=num_cols[:max_cols],
                key="eda_pair_sel",
            )
            if len(pair_sel) >= 2:
                color_pair = None
                if cat_cols and target_column and target_column in cat_cols:
                    color_pair = target_column
                st.plotly_chart(
                    pair_plot(df, columns=pair_sel, color_col=color_pair),
                    use_container_width=True,
                )


def _fmt(value: Any) -> str:
    """Format a float to 4 significant figures."""
    if value is None:
        return "—"
    try:
        f = float(value)
        if f != f:
            return "—"
        return f"{f:.4g}"
    except (TypeError, ValueError):
        return str(value)
