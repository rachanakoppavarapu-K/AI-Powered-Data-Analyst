"""Tests for the Streamlit UI components.

Because Streamlit is a runtime dependency that may not be installed in the
CI test environment (or is not invokable without a running server), ALL tests
that exercise UI modules must mock the ``streamlit`` module via
``patch.dict("sys.modules", ...)``.

What is tested
--------------
- sidebar.render_sidebar — returns the correct keys; DataLoader errors
  are surfaced without raising.
- eda_tab._fmt — numeric formatting helper.
- eda_tab.render_eda_tab — returns without error given a valid DataFrame.
- ml_tab._pick_best_model — picks the highest-scoring model.
- ml_tab._empty_ml_state — returns the expected empty structure.
- ml_tab._pct / _flt — numeric formatting helpers.
- ml_tab.render_ml_tab — returns empty state when ml_trained is False.
- insights_tab._flatten_profile — correctly converts DataProfiler output.
- insights_tab._flatten_eda — correctly extracts top correlations.
- report_tab._extract_top_correlations — returns correct top-N pairs.
- report_tab.render_report_tab — renders without error when df is None.
- dashboard._init_session_state — initialises all required keys.
"""

from __future__ import annotations

import math
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """A small mixed DataFrame suitable for testing all tabs."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60],
            "salary": [30000, 45000, 50000, 55000, 60000, 65000, 70000, 80000],
            "score": [0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
            "survived": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


@pytest.fixture()
def sample_profile(sample_df: pd.DataFrame) -> dict[str, Any]:
    """A minimal profile dict mirroring DataProfiler output."""
    from src.profiling.data_profiler import profile_dataframe

    return profile_dataframe(sample_df)


@pytest.fixture()
def sample_eda(sample_df: pd.DataFrame) -> dict[str, Any]:
    """A minimal EDA dict from the EDAEngine."""
    from src.eda.eda_engine import run_eda

    return run_eda(sample_df, target_column="survived", task_type="classification")


# ---------------------------------------------------------------------------
# Streamlit mock factory
# ---------------------------------------------------------------------------


def _make_st_mock() -> MagicMock:
    """Create a MagicMock that satisfies common st.* call patterns."""
    st = MagicMock()

    # Widgets return sensible defaults so component code can consume them
    st.selectbox.return_value = "(None — unsupervised)"
    st.multiselect.return_value = []
    st.text_input.return_value = ""
    st.button.return_value = False

    # columns() must return the correct number of mock context managers
    def _columns(n, *args, **kwargs):
        return [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]

    st.columns.side_effect = _columns

    # sidebar mirrors the same behaviour
    st.sidebar.columns.side_effect = _columns
    st.sidebar.file_uploader.return_value = None
    st.sidebar.selectbox.return_value = "(None — unsupervised)"
    st.sidebar.multiselect.return_value = []
    st.sidebar.text_input.return_value = ""

    # Context-manager helpers
    def _ctx_mgr():
        m = MagicMock()
        m.__enter__ = MagicMock(return_value=m)
        m.__exit__ = MagicMock(return_value=False)
        return m

    st.spinner.side_effect = lambda *a, **kw: _ctx_mgr()
    st.expander.side_effect = lambda *a, **kw: _ctx_mgr()
    st.form.side_effect = lambda *a, **kw: _ctx_mgr()
    st.tabs.return_value = [_ctx_mgr() for _ in range(5)]

    # Session state as a plain dict
    st.session_state = {}

    return st


def _patched_st_modules(st_mock: MagicMock) -> dict[str, MagicMock]:
    """Return the sys.modules patch dict for streamlit and common sub-modules."""
    patches: dict[str, MagicMock] = {"streamlit": st_mock}
    for sub in ("streamlit.components", "streamlit.components.v1"):
        m = MagicMock()
        m.html = MagicMock()
        patches[sub] = m
    return patches


# ---------------------------------------------------------------------------
# 1. sidebar tests
# ---------------------------------------------------------------------------


class TestSidebar:
    def test_render_sidebar_no_upload(self) -> None:
        """render_sidebar returns correct keys when no file is uploaded."""
        st_mock = _make_st_mock()

        with patch.dict("sys.modules", _patched_st_modules(st_mock)):
            # Force reimport so the module picks up the mocked streamlit
            for key in list(sys.modules.keys()):
                if "src.ui.components.sidebar" in key:
                    del sys.modules[key]
            from src.ui.components.sidebar import render_sidebar

            result = render_sidebar()

        assert set(result.keys()) >= {"df", "filename", "target_column", "task_type", "model_names"}
        assert result["df"] is None
        assert result["filename"] is None

    def test_render_sidebar_bad_file(self) -> None:
        """render_sidebar does not raise when DataLoader raises DataLoadError."""
        st_mock = _make_st_mock()
        fake_file = MagicMock()
        fake_file.name = "bad_file.txt"
        fake_file.read.return_value = b"garbage"
        st_mock.sidebar.file_uploader.return_value = fake_file

        with patch.dict("sys.modules", _patched_st_modules(st_mock)):
            for key in list(sys.modules.keys()):
                if "src.ui.components.sidebar" in key:
                    del sys.modules[key]
            from src.ui.components.sidebar import render_sidebar

            result = render_sidebar()

        assert result["df"] is None


# ---------------------------------------------------------------------------
# 2. eda_tab tests — pure helper functions (no st.* calls)
# ---------------------------------------------------------------------------


class TestEdaTabHelpers:
    """Test pure helper functions in eda_tab that do not call st.*."""

    def _import_fmt(self) -> Any:
        """Import _fmt with streamlit mocked."""
        st_mock = _make_st_mock()
        with patch.dict("sys.modules", _patched_st_modules(st_mock)):
            for key in list(sys.modules.keys()):
                if "src.ui.components.eda_tab" in key:
                    del sys.modules[key]
            from src.ui.components.eda_tab import _fmt  # noqa: PLC0415

            return _fmt

    def test_fmt_normal(self) -> None:
        """_fmt returns 4-sig-fig string for normal floats."""
        _fmt = self._import_fmt()
        result = _fmt(3.14159)
        # 4 sig figs of 3.14159 is "3.142"
        assert result == "3.142"

    def test_fmt_none(self) -> None:
        """_fmt returns '—' for None."""
        _fmt = self._import_fmt()
        assert _fmt(None) == "—"

    def test_fmt_nan(self) -> None:
        """_fmt returns '—' for NaN."""
        _fmt = self._import_fmt()
        assert _fmt(float("nan")) == "—"

    def test_fmt_zero(self) -> None:
        """_fmt handles zero correctly."""
        _fmt = self._import_fmt()
        assert _fmt(0.0) == "0"


# ---------------------------------------------------------------------------
# 3. ml_tab helper tests
# ---------------------------------------------------------------------------


class TestMlTabHelpers:
    """Test pure helper functions in ml_tab that do not call st.*."""

    def _import_helpers(self) -> tuple[Any, Any, Any, Any]:
        """Import _pick_best_model, _empty_ml_state, _pct, _flt with st mocked."""
        st_mock = _make_st_mock()
        with patch.dict("sys.modules", _patched_st_modules(st_mock)):
            for key in list(sys.modules.keys()):
                if "src.ui.components.ml_tab" in key:
                    del sys.modules[key]
            from src.ui.components.ml_tab import (  # noqa: PLC0415
                _empty_ml_state,
                _flt,
                _pct,
                _pick_best_model,
            )
            return _pick_best_model, _empty_ml_state, _pct, _flt

    def test_pick_best_model_classification(self) -> None:
        """_pick_best_model selects the model with the highest F1."""
        from src.ml.evaluator import ModelMetrics
        from src.ml.task_detector import TaskType

        _pick_best_model, _, _, _ = self._import_helpers()

        m1 = ModelMetrics(task_type=TaskType.CLASSIFICATION, model_name="A")
        m1.f1_weighted = 0.7
        m2 = ModelMetrics(task_type=TaskType.CLASSIFICATION, model_name="B")
        m2.f1_weighted = 0.9

        r1, r2 = MagicMock(), MagicMock()
        _, best_m = _pick_best_model([r1, r2], [m1, m2], TaskType.CLASSIFICATION)
        assert best_m.model_name == "B"

    def test_pick_best_model_regression(self) -> None:
        """_pick_best_model selects the model with the highest R²."""
        from src.ml.evaluator import ModelMetrics
        from src.ml.task_detector import TaskType

        _pick_best_model, _, _, _ = self._import_helpers()

        m1 = ModelMetrics(task_type=TaskType.REGRESSION, model_name="LR")
        m1.r2 = 0.5
        m2 = ModelMetrics(task_type=TaskType.REGRESSION, model_name="RF")
        m2.r2 = 0.85

        r1, r2 = MagicMock(), MagicMock()
        _, best_m = _pick_best_model([r1, r2], [m1, m2], TaskType.REGRESSION)
        assert best_m.model_name == "RF"

    def test_pick_best_model_single(self) -> None:
        """_pick_best_model returns the single model when only one is present."""
        from src.ml.evaluator import ModelMetrics
        from src.ml.task_detector import TaskType

        _pick_best_model, _, _, _ = self._import_helpers()

        m = ModelMetrics(task_type=TaskType.CLASSIFICATION, model_name="Solo")
        r = MagicMock()
        _, best_m = _pick_best_model([r], [m], TaskType.CLASSIFICATION)
        assert best_m.model_name == "Solo"

    def test_empty_ml_state(self) -> None:
        """_empty_ml_state returns expected empty structure."""
        from src.ml.task_detector import TaskType

        _, _empty_ml_state, _, _ = self._import_helpers()

        state = _empty_ml_state(TaskType.CLASSIFICATION)
        assert state["train_results"] == []
        assert state["metrics"] == []
        assert state["best_result"] is None
        assert state["best_metrics"] is None
        assert state["task_type"] == TaskType.CLASSIFICATION

    def test_pct_formatting(self) -> None:
        """_pct formats floats to 4 decimal places."""
        _, _, _pct, _ = self._import_helpers()
        assert _pct(0.9123) == "0.9123"
        assert _pct(None) == "—"
        assert _pct(float("nan")) == "—"

    def test_flt_formatting(self) -> None:
        """_flt formats floats to 4 significant figures."""
        _, _, _, _flt = self._import_helpers()
        # Python's :.4g uses banker's rounding; accept either rounding outcome
        assert _flt(1234.5) in ("1234", "1235")
        assert _flt(0.001234) == "0.001234"
        assert _flt(None) == "—"
        assert _flt(float("nan")) == "—"

    def test_render_ml_tab_no_training(self, sample_df: pd.DataFrame) -> None:
        """render_ml_tab returns empty state when ml_trained is False."""
        st_mock = _make_st_mock()
        st_mock.button.return_value = False
        st_mock.session_state = {"ml_trained": False}

        with patch.dict("sys.modules", _patched_st_modules(st_mock)):
            for key in list(sys.modules.keys()):
                if "src.ui.components.ml_tab" in key:
                    del sys.modules[key]
            from src.ui.components.ml_tab import render_ml_tab  # noqa: PLC0415

            result = render_ml_tab(
                sample_df,
                target_column="survived",
                task_type_override=None,
                selected_model_names=[],
            )

        assert result["train_results"] == []


# ---------------------------------------------------------------------------
# 4. insights_tab tests
# ---------------------------------------------------------------------------


class TestInsightsTab:
    def _import_helpers(self) -> tuple[Any, Any]:
        st_mock = _make_st_mock()
        with patch.dict("sys.modules", _patched_st_modules(st_mock)):
            for key in list(sys.modules.keys()):
                if "src.ui.components.insights_tab" in key:
                    del sys.modules[key]
            from src.ui.components.insights_tab import _flatten_eda, _flatten_profile  # noqa: PLC0415

            return _flatten_profile, _flatten_eda

    def test_flatten_profile(self, sample_profile: dict[str, Any]) -> None:
        """_flatten_profile converts DataProfiler output correctly."""
        _flatten_profile, _ = self._import_helpers()
        flat = _flatten_profile(sample_profile)
        assert "n_rows" in flat
        assert "n_cols" in flat
        assert "missing_cells" in flat
        assert "missing_pct" in flat
        assert isinstance(flat["n_rows"], int)
        assert isinstance(flat["missing_pct"], float)

    def test_flatten_eda_top_correlations(self, sample_eda: dict[str, Any]) -> None:
        """_flatten_eda extracts top correlations without duplicate pairs."""
        _, _flatten_eda = self._import_helpers()
        flat = _flatten_eda(sample_eda)
        corrs = flat.get("top_correlations", [])
        seen: set[frozenset[str]] = set()
        for entry in corrs:
            key = frozenset({entry["feature_a"], entry["feature_b"]})
            assert key not in seen, "Duplicate correlation pair found"
            seen.add(key)

    def test_flatten_eda_sorted_by_abs_correlation(self, sample_eda: dict[str, Any]) -> None:
        """_flatten_eda returns correlations sorted by absolute value descending."""
        _, _flatten_eda = self._import_helpers()
        flat = _flatten_eda(sample_eda)
        corrs = flat.get("top_correlations", [])
        abs_vals = [abs(e["correlation"]) for e in corrs]
        assert abs_vals == sorted(abs_vals, reverse=True)


# ---------------------------------------------------------------------------
# 5. report_tab tests
# ---------------------------------------------------------------------------


class TestReportTab:
    def _import_helpers(self) -> Any:
        st_mock = _make_st_mock()
        with patch.dict("sys.modules", _patched_st_modules(st_mock)):
            for key in list(sys.modules.keys()):
                if "src.ui.components.report_tab" in key:
                    del sys.modules[key]
            from src.ui.components.report_tab import _extract_top_correlations  # noqa: PLC0415

            return _extract_top_correlations

    def test_extract_top_correlations_empty(self) -> None:
        """_extract_top_correlations returns empty list for empty input."""
        _extract = self._import_helpers()
        assert _extract({}) == []

    def test_extract_top_correlations_dedup(self) -> None:
        """_extract_top_correlations deduplicates symmetric pairs."""
        _extract = self._import_helpers()
        pearson = {
            "a": {"a": 1.0, "b": 0.9, "c": 0.3},
            "b": {"a": 0.9, "b": 1.0, "c": 0.2},
            "c": {"a": 0.3, "b": 0.2, "c": 1.0},
        }
        result = _extract(pearson)
        assert len(result) == 3
        assert result[0]["correlation"] >= result[1]["correlation"]

    def test_extract_top_correlations_ordering(self) -> None:
        """_extract_top_correlations returns strongest correlations first."""
        _extract = self._import_helpers()
        pearson = {
            "x": {"x": 1.0, "y": -0.8, "z": 0.1},
            "y": {"x": -0.8, "y": 1.0, "z": 0.5},
            "z": {"x": 0.1, "y": 0.5, "z": 1.0},
        }
        result = _extract(pearson)
        assert abs(result[0]["correlation"]) == pytest.approx(0.8, abs=1e-6)

    def test_render_report_tab_no_df(self) -> None:
        """render_report_tab shows info message and does not raise when df is None."""
        st_mock = _make_st_mock()
        st_mock.session_state = {}

        with patch.dict("sys.modules", _patched_st_modules(st_mock)):
            for key in list(sys.modules.keys()):
                if "src.ui.components.report_tab" in key:
                    del sys.modules[key]
            from src.ui.components.report_tab import render_report_tab  # noqa: PLC0415

            render_report_tab(
                df=None,
                filename=None,
                profile=None,
                eda_result=None,
                best_metrics=None,
                best_result=None,
                nl_query_results=[],
            )
        assert st_mock.info.called


# ---------------------------------------------------------------------------
# 6. dashboard tests
# ---------------------------------------------------------------------------


class TestDashboard:
    def test_init_session_state(self) -> None:
        """_init_session_state adds all required keys with correct defaults."""
        st_mock = _make_st_mock()
        st_mock.session_state = {}

        with patch.dict("sys.modules", _patched_st_modules(st_mock)):
            for key in list(sys.modules.keys()):
                if "src.ui.dashboard" in key:
                    del sys.modules[key]
            from src.ui.dashboard import _init_session_state  # noqa: PLC0415

            _init_session_state()

        required_keys = [
            "df", "filename", "target_column", "task_type",
            "profile", "eda_result",
            "ml_trained", "ml_train_results", "ml_metrics", "ml_task_type",
            "nl_query_history",
            "report_html_path", "report_html_str",
        ]
        for key in required_keys:
            assert key in st_mock.session_state, f"Missing session state key: {key}"

    def test_init_session_state_does_not_overwrite(self) -> None:
        """_init_session_state preserves existing session state values."""
        st_mock = _make_st_mock()
        st_mock.session_state = {"df": "already_set", "ml_trained": True}

        with patch.dict("sys.modules", _patched_st_modules(st_mock)):
            for key in list(sys.modules.keys()):
                if "src.ui.dashboard" in key:
                    del sys.modules[key]
            from src.ui.dashboard import _init_session_state  # noqa: PLC0415

            _init_session_state()

        assert st_mock.session_state["df"] == "already_set"
        assert st_mock.session_state["ml_trained"] is True
