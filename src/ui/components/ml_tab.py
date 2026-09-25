"""ML tab component — model training, comparison, evaluation, and feature importance."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src.ml.evaluator import ModelMetrics, evaluate_model
from src.ml.feature_importance import compute_feature_importance
from src.ml.task_detector import TaskDetectionResult, TaskType, detect_task
from src.ml.trainer import TrainResult, TrainerError, train_models
from src.visualization.chart_builder import (
    cluster_scatter,
    confusion_matrix_chart,
    feature_importance_chart,
)


def render_ml_tab(
    df: pd.DataFrame,
    target_column: str | None,
    task_type_override: TaskType | None,
    selected_model_names: list[str],
) -> dict[str, Any]:
    """Render the ML tab and return trained results for downstream use.

    Parameters
    ----------
    df:
        The raw DataFrame.
    target_column:
        Selected target column, or ``None`` for clustering.
    task_type_override:
        User-specified :class:`TaskType`, or ``None`` to auto-detect.
    selected_model_names:
        List of model names to keep (empty list = keep all).

    Returns
    -------
    dict with keys:
        ``train_results``  — list of :class:`TrainResult`
        ``metrics``        — list of :class:`ModelMetrics`
        ``best_result``    — best :class:`TrainResult` by primary metric
        ``best_metrics``   — :class:`ModelMetrics` for the best model
        ``task_type``      — resolved :class:`TaskType`
    """
    st.header("🤖 Machine Learning")

    # ------------------------------------------------------------------
    # 1. Task detection
    # ------------------------------------------------------------------
    detection: TaskDetectionResult = detect_task(
        df, target_column=target_column, user_override=task_type_override
    )
    resolved_task = detection.task_type

    info_parts = [f"**Task:** {resolved_task.value}"]
    if detection.is_override:
        info_parts.append("*(user override)*")
    if detection.n_classes:
        info_parts.append(f"**Classes:** {detection.n_classes}")
    st.info("  |  ".join(info_parts) + f"\n\n_{detection.reason}_")

    # ------------------------------------------------------------------
    # 2. Train button
    # ------------------------------------------------------------------
    if st.button("🚀 Train Models", key="btn_train_models"):
        st.session_state["ml_trained"] = False
        st.session_state["ml_train_results"] = []
        st.session_state["ml_metrics"] = []
        st.session_state["ml_task_type"] = resolved_task

        with st.spinner("Training models… this may take a moment."):
            try:
                all_results: list[TrainResult] = train_models(
                    df,
                    target_column=target_column,
                    task_type=resolved_task,
                )
            except TrainerError as exc:
                st.error(f"❌ Training failed: {exc}")
                return _empty_ml_state(resolved_task)
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Unexpected error during training: {exc}")
                return _empty_ml_state(resolved_task)

        # Filter to selected models if any specified
        if selected_model_names:
            filtered = [r for r in all_results if r.model_name in selected_model_names]
            all_results = filtered if filtered else all_results

        # Evaluate
        all_metrics: list[ModelMetrics] = [evaluate_model(r) for r in all_results]

        st.session_state["ml_trained"] = True
        st.session_state["ml_train_results"] = all_results
        st.session_state["ml_metrics"] = all_metrics
        st.session_state["ml_task_type"] = resolved_task
        st.success(f"✅ Trained {len(all_results)} model(s) successfully.")

    # ------------------------------------------------------------------
    # 3. Display results (if available in session state)
    # ------------------------------------------------------------------
    if not st.session_state.get("ml_trained"):
        st.info("Click **Train Models** to start training.")
        return _empty_ml_state(resolved_task)

    all_results = st.session_state.get("ml_train_results", [])
    all_metrics = st.session_state.get("ml_metrics", [])
    task = st.session_state.get("ml_task_type", resolved_task)

    if not all_results:
        return _empty_ml_state(resolved_task)

    _render_model_comparison(all_results, all_metrics, task)

    st.markdown("---")

    best_result, best_metrics = _pick_best_model(all_results, all_metrics, task)
    _render_model_details(best_result, best_metrics, task)

    st.markdown("---")

    _render_feature_importance(best_result, task)

    return {
        "train_results": all_results,
        "metrics": all_metrics,
        "best_result": best_result,
        "best_metrics": best_metrics,
        "task_type": task,
    }


# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------


def _render_model_comparison(
    results: list[TrainResult],
    metrics: list[ModelMetrics],
    task: TaskType,
) -> None:
    """Render a model comparison table."""
    st.subheader("Model Comparison")

    rows = []
    for m in metrics:
        row: dict[str, Any] = {"Model": m.model_name}
        if task == TaskType.CLASSIFICATION:
            row["Accuracy"] = _pct(m.accuracy)
            row["F1 (weighted)"] = _pct(m.f1_weighted)
            row["F1 (macro)"] = _pct(m.f1_macro)
            row["ROC-AUC"] = _pct(m.roc_auc)
            row["CV Mean"] = _pct(m.cv_mean)
        elif task == TaskType.REGRESSION:
            row["RMSE"] = _flt(m.rmse)
            row["MAE"] = _flt(m.mae)
            row["R²"] = _flt(m.r2)
            row["Adj R²"] = _flt(m.adj_r2)
            row["CV Mean"] = _flt(m.cv_mean)
        else:  # CLUSTERING
            row["Inertia"] = _flt(m.inertia)
            row["Silhouette"] = _flt(m.silhouette)
            row["Davies-Bouldin"] = _flt(m.davies_bouldin)
            row["Clusters"] = m.n_clusters
        rows.append(row)

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_model_details(
    result: TrainResult,
    metrics: ModelMetrics,
    task: TaskType,
) -> None:
    """Render detailed evaluation for a single model."""
    st.subheader(f"Best Model: {metrics.model_name}")

    if task == TaskType.CLASSIFICATION:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", _pct(metrics.accuracy))
        c2.metric("F1 (weighted)", _pct(metrics.f1_weighted))
        c3.metric("ROC-AUC", _pct(metrics.roc_auc))
        c4.metric("CV Mean ± Std", f"{_flt(metrics.cv_mean)} ± {_flt(metrics.cv_std)}")

        if metrics.confusion_matrix is not None:
            st.markdown("#### Confusion Matrix")
            labels = metrics.extra.get("class_labels")
            st.plotly_chart(
                confusion_matrix_chart(metrics.confusion_matrix, labels=labels),
                use_container_width=True,
            )

    elif task == TaskType.REGRESSION:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", _flt(metrics.rmse))
        c2.metric("MAE", _flt(metrics.mae))
        c3.metric("R²", _flt(metrics.r2))
        c4.metric("CV Mean", _flt(metrics.cv_mean))

    else:  # CLUSTERING
        c1, c2, c3 = st.columns(3)
        c1.metric("Clusters", str(metrics.n_clusters))
        c2.metric("Silhouette", _flt(metrics.silhouette))
        c3.metric("Inertia", _flt(metrics.inertia))

        # Cluster scatter
        if result.cluster_labels is not None:
            st.markdown("#### Cluster Scatter")
            plot_df = result.X_train.copy()
            plot_df["cluster"] = result.cluster_labels.astype(str)
            st.plotly_chart(
                cluster_scatter(plot_df, "cluster"),
                use_container_width=True,
            )


def _render_feature_importance(result: TrainResult, task: TaskType) -> None:
    """Render feature importance for the best model."""
    if task == TaskType.CLUSTERING:
        return  # No supervised importance for clustering

    st.subheader("Feature Importance")
    try:
        with st.spinner("Computing feature importances…"):
            imp_result = compute_feature_importance(result, method="auto", n_repeats=5)
        names = [e.feature for e in imp_result.ranked]
        scores = [e.importance for e in imp_result.ranked]
        st.plotly_chart(
            feature_importance_chart(names, scores, top_n=20, title=f"Feature Importance — {imp_result.model_name}"),
            use_container_width=True,
        )
        st.caption(f"Method: **{imp_result.method}**")
        with st.expander("Full importance table"):
            imp_df = pd.DataFrame(
                [{"Feature": e.feature, "Importance": e.importance, "Std": e.std} for e in imp_result.ranked]
            )
            st.dataframe(imp_df, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Feature importance unavailable: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_best_model(
    results: list[TrainResult],
    metrics: list[ModelMetrics],
    task: TaskType,
) -> tuple[TrainResult, ModelMetrics]:
    """Return the best (result, metrics) pair by primary metric."""
    if len(results) == 1:
        return results[0], metrics[0]

    best_idx = 0
    best_score = -float("inf")

    for i, m in enumerate(metrics):
        if task == TaskType.CLASSIFICATION:
            score = m.f1_weighted if m.f1_weighted is not None else -float("inf")
        elif task == TaskType.REGRESSION:
            r2 = m.r2 if m.r2 is not None else -float("inf")
            score = r2 if not math.isnan(r2) else -float("inf")
        else:  # CLUSTERING — prefer highest silhouette
            score = m.silhouette if m.silhouette is not None else -float("inf")

        if score > best_score:
            best_score = score
            best_idx = i

    return results[best_idx], metrics[best_idx]


def _empty_ml_state(task: TaskType) -> dict[str, Any]:
    return {
        "train_results": [],
        "metrics": [],
        "best_result": None,
        "best_metrics": None,
        "task_type": task,
    }


def _pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.4f}"


def _flt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.4g}"
