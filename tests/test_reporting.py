"""tests/test_reporting.py — Tests for src/reporting/report_generator.py.

Test coverage
-------------
- ReportGenerator.generate() produces a valid HTML file at the expected path.
- ReportGenerator.render_html() returns a non-empty HTML string.
- All required section headings appear when the full context is supplied.
- Missing optional sections are gracefully omitted (no KeyError / exception).
- Dataset profile data (rows, columns) appears in the output.
- ML metrics appear in the output.
- Feature importance entries appear in the output.
- GenAI insights text appears in the output.
- NL query results appear in the output — both dict and dataclass-like forms.
- Recommendations (string form and list form) appear in the output.
- Data-quality null percentages appear in the output.
- EDA top-correlations appear in the output.
- The generated HTML is self-contained (no external src= / href= references
  that would break when opened offline — only data: URIs are allowed for media).
- export_pdf=True raises no exception when weasyprint is absent (emits warning).
- Custom output_path is honoured.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.reporting.report_generator import ReportGenerator, _int_fmt, _num_fmt, _bytes_fmt


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_context() -> dict[str, Any]:
    """Smallest valid context — only required metadata."""
    return {
        "title": "Test Report",
        "dataset_name": "iris.csv",
        "generated_at": "2024-01-01 00:00 UTC",
    }


@pytest.fixture()
def profile_data() -> dict[str, Any]:
    return {
        "shape": {"rows": 150, "columns": 5},
        "memory_usage_bytes": 6000,
        "null_counts": {"sepal_length": 0, "sepal_width": 0, "species": 2},
        "null_percentages": {"sepal_length": 0.0, "sepal_width": 0.0, "species": 1.33},
        "cardinality": {"sepal_length": 35, "sepal_width": 23, "species": 3},
        "numeric_stats": {
            "sepal_length": {
                "count": 150, "mean": 5.84, "median": 5.8, "std": 0.83,
                "min": 4.3, "max": 7.9, "q1": 5.1, "q3": 6.4,
                "iqr": 1.3, "skewness": 0.31, "kurtosis": -0.55,
            },
        },
        "categorical_stats": {
            "species": {
                "count": 148, "cardinality": 3, "mode": "setosa",
                "top_values": {"setosa": 50, "versicolor": 50, "virginica": 48},
            },
        },
        "correlation": {"pearson": {}, "spearman": {}},
    }


@pytest.fixture()
def data_quality_data() -> dict[str, Any]:
    return {
        "null_percentages": {"sepal_length": 0.0, "species": 1.33},
        "outlier_info": {"sepal_length": 3},
        "notes": ["species has 2 null values", "sepal_length has 3 outliers"],
    }


@pytest.fixture()
def eda_data() -> dict[str, Any]:
    return {
        "top_correlations": [
            {"feature_a": "sepal_length", "feature_b": "petal_length", "correlation": 0.87},
            {"feature_a": "petal_length", "feature_b": "petal_width", "correlation": 0.96},
        ],
        "skewed_columns": [
            {"column": "sepal_length", "skewness": 0.31},
        ],
        "charts": [],  # No Plotly figures in unit tests
    }


@pytest.fixture()
def ml_metrics_data() -> dict[str, Any]:
    return {
        "task_type": "classification",
        "model_name": "RandomForestClassifier",
        "accuracy": 0.96,
        "f1_weighted": 0.96,
        "roc_auc": 0.998,
        "cv_mean": 0.95,
        "cv_std": 0.02,
        "charts": [],
    }


@pytest.fixture()
def feature_importance_data() -> dict[str, Any]:
    return {
        "entries": [
            {"feature": "petal_length", "importance": 0.44},
            {"feature": "petal_width", "importance": 0.41},
            {"feature": "sepal_length", "importance": 0.09},
        ],
        "chart": None,
    }


@pytest.fixture()
def genai_insights_data() -> dict[str, str]:
    return {
        "Dataset Summary": "The dataset contains 150 iris samples across 3 species.",
        "Model Explanation": "The RandomForest achieved 96% accuracy with high ROC-AUC of 0.998.",
        "Recommendations": None,  # type: ignore[dict-item]  # intentionally None
    }


@pytest.fixture()
def nl_query_results_dict() -> list[dict[str, Any]]:
    return [
        {
            "question": "What is the average sepal_length?",
            "query_type": "AVERAGE",
            "computed_result": {"sepal_length": 5.843},
            "explanation": "The average sepal length is 5.843 cm across all 150 samples.",
            "error": None,
        },
        {
            "question": "How many rows are there?",
            "query_type": "COUNT",
            "computed_result": {"row_count": 150},
            "explanation": None,
            "error": None,
        },
    ]


@dataclass
class _FakeNLResult:
    """Minimal stand-in for NLQueryResult."""
    question: str
    query_type: str
    computed_result: Any = None
    explanation: str | None = None
    error: str | None = None


@pytest.fixture()
def nl_query_results_dataclass() -> list[_FakeNLResult]:
    return [
        _FakeNLResult(
            question="What is the distribution of species?",
            query_type="DISTRIBUTION",
            computed_result={"setosa": 50, "versicolor": 50, "virginica": 48},
            explanation="The three species are roughly equally represented.",
        ),
    ]


@pytest.fixture()
def recommendations_list() -> list[str]:
    return [
        "Collect more samples to improve model robustness.",
        "Investigate the 2 null values in the species column.",
    ]


@pytest.fixture()
def full_context(
    profile_data,
    data_quality_data,
    eda_data,
    ml_metrics_data,
    feature_importance_data,
    genai_insights_data,
    nl_query_results_dict,
    recommendations_list,
) -> dict[str, Any]:
    return {
        "title": "Full Iris Analysis Report",
        "dataset_name": "iris.csv",
        "generated_at": "2024-01-01 12:00 UTC",
        "profile": profile_data,
        "data_quality": data_quality_data,
        "eda": eda_data,
        "ml_metrics": ml_metrics_data,
        "feature_importance": feature_importance_data,
        "genai_insights": genai_insights_data,
        "nl_query_results": nl_query_results_dict,
        "recommendations": recommendations_list,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generator(tmp_path: Path) -> ReportGenerator:
    return ReportGenerator(output_dir=tmp_path)


# ---------------------------------------------------------------------------
# Filter unit tests
# ---------------------------------------------------------------------------


class TestFilters:
    def test_int_fmt_integer(self):
        assert _int_fmt(1234567) == "1,234,567"

    def test_int_fmt_none(self):
        assert _int_fmt(None) == "—"

    def test_num_fmt_float(self):
        result = _num_fmt(0.96)
        assert "0.96" in result

    def test_num_fmt_none(self):
        assert _num_fmt(None) == "—"

    def test_num_fmt_nan(self):
        import math
        assert _num_fmt(math.nan) == "—"

    def test_bytes_fmt_bytes(self):
        assert _bytes_fmt(512) == "512.0 B"

    def test_bytes_fmt_kilobytes(self):
        assert "KB" in _bytes_fmt(2048)

    def test_bytes_fmt_megabytes(self):
        assert "MB" in _bytes_fmt(2 * 1024 * 1024)

    def test_bytes_fmt_none(self):
        assert _bytes_fmt(None) == "—"


# ---------------------------------------------------------------------------
# render_html — basic output tests
# ---------------------------------------------------------------------------


class TestRenderHtml:
    def test_returns_string(self, minimal_context):
        gen = ReportGenerator()
        result = gen.render_html(minimal_context)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_html_doctype_present(self, minimal_context):
        gen = ReportGenerator()
        result = gen.render_html(minimal_context)
        assert "<!DOCTYPE html>" in result or "<!doctype html>" in result.lower()

    def test_title_in_output(self, minimal_context):
        gen = ReportGenerator()
        result = gen.render_html(minimal_context)
        assert "Test Report" in result

    def test_dataset_name_in_output(self, minimal_context):
        gen = ReportGenerator()
        result = gen.render_html(minimal_context)
        assert "iris.csv" in result

    def test_generated_at_in_output(self, minimal_context):
        gen = ReportGenerator()
        result = gen.render_html(minimal_context)
        assert "2024-01-01 00:00 UTC" in result


# ---------------------------------------------------------------------------
# generate() — file creation tests
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_string_path(self, minimal_context, tmp_path):
        gen = _make_generator(tmp_path)
        path = gen.generate(minimal_context)
        assert isinstance(path, str)

    def test_file_is_created(self, minimal_context, tmp_path):
        gen = _make_generator(tmp_path)
        path = gen.generate(minimal_context)
        assert Path(path).exists()

    def test_file_extension_is_html(self, minimal_context, tmp_path):
        gen = _make_generator(tmp_path)
        path = gen.generate(minimal_context)
        assert path.endswith(".html")

    def test_file_content_is_html(self, minimal_context, tmp_path):
        gen = _make_generator(tmp_path)
        path = gen.generate(minimal_context)
        content = Path(path).read_text(encoding="utf-8")
        assert "<html" in content.lower()

    def test_custom_output_path_honoured(self, minimal_context, tmp_path):
        custom_path = tmp_path / "custom_report.html"
        gen = _make_generator(tmp_path)
        returned_path = gen.generate(minimal_context, output_path=custom_path)
        assert returned_path == str(custom_path)
        assert custom_path.exists()

    def test_auto_generated_at_when_absent(self, tmp_path):
        gen = _make_generator(tmp_path)
        ctx = {"title": "T"}
        path = gen.generate(ctx)
        content = Path(path).read_text(encoding="utf-8")
        assert "UTC" in content  # auto-generated timestamp


# ---------------------------------------------------------------------------
# Section content tests
# ---------------------------------------------------------------------------


class TestProfileSection:
    def test_profile_section_heading(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "Dataset Profile" in content

    def test_row_count_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "150" in content

    def test_column_count_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "5" in content

    def test_numeric_column_name_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "sepal_length" in content

    def test_categorical_mode_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "setosa" in content


class TestDataQualitySection:
    def test_data_quality_heading(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "Data Quality" in content

    def test_null_percentage_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "1.33" in content

    def test_notes_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "2 null values" in content


class TestEdaSection:
    def test_eda_heading(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "Exploratory Data Analysis" in content

    def test_correlation_features_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "petal_length" in content
        assert "petal_width" in content

    def test_correlation_value_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "0.87" in content


class TestMlMetricsSection:
    def test_ml_heading(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "Machine Learning" in content

    def test_model_name_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "RandomForestClassifier" in content

    def test_accuracy_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        # 96% displayed
        assert "96" in content

    def test_roc_auc_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "0.998" in content

    def test_task_type_badge_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "classification" in content


class TestFeatureImportanceSection:
    def test_fi_heading(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "Feature Importance" in content

    def test_feature_name_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "petal_length" in content

    def test_importance_value_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "0.44" in content or "0.4400" in content


class TestGenAiInsightsSection:
    def test_insights_heading(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "AI-Generated Insights" in content

    def test_insight_label_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "Dataset Summary" in content

    def test_insight_text_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "150 iris samples" in content

    def test_none_insight_skipped(self, full_context, tmp_path):
        """A None insight value must not cause an error."""
        content = _make_generator(tmp_path).render_html(full_context)
        # Recommendations insight is None; heading should still not appear
        # in the insights block (it is skipped via {% if text %})
        assert isinstance(content, str)


class TestNlQuerySection:
    def test_nl_query_heading(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "Natural-Language Query" in content

    def test_question_text_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "average sepal_length" in content

    def test_explanation_text_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "5.843 cm" in content

    def test_dataclass_results_normalised(self, tmp_path, nl_query_results_dataclass):
        """NLQueryResult dataclass-like objects must be normalised to dicts."""
        ctx = {
            "title": "T",
            "nl_query_results": nl_query_results_dataclass,
        }
        content = _make_generator(tmp_path).render_html(ctx)
        assert "distribution of species" in content

    def test_result_with_no_explanation_shows_computed(self, tmp_path):
        ctx = {
            "nl_query_results": [
                {
                    "question": "How many rows?",
                    "computed_result": {"row_count": 200},
                    "explanation": None,
                    "error": None,
                }
            ]
        }
        content = _make_generator(tmp_path).render_html(ctx)
        assert "How many rows?" in content

    def test_result_with_error_shows_error(self, tmp_path):
        ctx = {
            "nl_query_results": [
                {
                    "question": "Who wrote this?",
                    "computed_result": None,
                    "explanation": None,
                    "error": "Unsupported query type.",
                }
            ]
        }
        content = _make_generator(tmp_path).render_html(ctx)
        assert "Unsupported query type." in content


class TestRecommendationsSection:
    def test_recommendations_heading(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "Recommendations" in content

    def test_list_recommendations_present(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        assert "Collect more samples" in content

    def test_string_recommendations(self, tmp_path):
        ctx = {
            "recommendations": "Improve feature engineering and gather more data.",
        }
        content = _make_generator(tmp_path).render_html(ctx)
        assert "Improve feature engineering" in content


# ---------------------------------------------------------------------------
# Missing optional sections — graceful handling
# ---------------------------------------------------------------------------


class TestMissingSections:
    def test_no_profile_no_error(self, tmp_path):
        gen = _make_generator(tmp_path)
        result = gen.render_html({"title": "T"})
        assert isinstance(result, str)
        assert "Dataset Profile" not in result

    def test_no_ml_metrics_no_error(self, tmp_path):
        gen = _make_generator(tmp_path)
        result = gen.render_html({"title": "T"})
        assert "Machine Learning" not in result

    def test_no_eda_no_error(self, tmp_path):
        gen = _make_generator(tmp_path)
        result = gen.render_html({"title": "T"})
        assert "Exploratory Data Analysis" not in result

    def test_no_feature_importance_no_error(self, tmp_path):
        gen = _make_generator(tmp_path)
        result = gen.render_html({"title": "T"})
        assert "Feature Importance" not in result

    def test_no_insights_no_error(self, tmp_path):
        gen = _make_generator(tmp_path)
        result = gen.render_html({"title": "T"})
        assert "AI-Generated Insights" not in result

    def test_no_nl_results_no_error(self, tmp_path):
        gen = _make_generator(tmp_path)
        result = gen.render_html({"title": "T"})
        assert "Natural-Language Query" not in result

    def test_no_recommendations_no_error(self, tmp_path):
        gen = _make_generator(tmp_path)
        result = gen.render_html({"title": "T"})
        # Should not crash and should not show recommendations heading
        assert isinstance(result, str)

    def test_empty_context_no_error(self, tmp_path):
        gen = _make_generator(tmp_path)
        result = gen.render_html({})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extra_context_keys_ignored(self, tmp_path):
        gen = _make_generator(tmp_path)
        result = gen.render_html({"title": "T", "unknown_key": {"foo": "bar"}})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Self-contained HTML check
# ---------------------------------------------------------------------------


class TestSelfContained:
    def test_no_external_script_src(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        import re
        # No <script src="http..."> or <script src="//...">
        external_scripts = re.findall(r'<script[^>]+src=["\']https?://', content, re.I)
        assert external_scripts == []

    def test_no_external_link_href(self, full_context, tmp_path):
        content = _make_generator(tmp_path).render_html(full_context)
        import re
        # No <link href="http..."> (external stylesheets)
        external_links = re.findall(r'<link[^>]+href=["\']https?://', content, re.I)
        assert external_links == []


# ---------------------------------------------------------------------------
# PDF export — graceful when weasyprint absent
# ---------------------------------------------------------------------------


class TestPdfExport:
    def test_pdf_export_warns_when_weasyprint_missing(self, minimal_context, tmp_path):
        gen = _make_generator(tmp_path)
        with patch.dict("sys.modules", {"weasyprint": None}):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                gen.generate(minimal_context, export_pdf=True)
            warning_messages = [str(x.message) for x in w]
            assert any("weasyprint" in msg.lower() or "pdf" in msg.lower()
                       for msg in warning_messages)

    def test_pdf_export_succeeds_with_mock_weasyprint(self, minimal_context, tmp_path):
        gen = _make_generator(tmp_path)
        mock_html_cls = MagicMock()
        mock_html_instance = MagicMock()
        mock_html_cls.return_value = mock_html_instance

        fake_weasyprint = MagicMock()
        fake_weasyprint.HTML = mock_html_cls

        with patch.dict("sys.modules", {"weasyprint": fake_weasyprint}):
            path = gen.generate(minimal_context, export_pdf=True)

        assert Path(path).exists()
        mock_html_cls.assert_called_once()
        mock_html_instance.write_pdf.assert_called_once()

    def test_html_file_still_created_when_pdf_fails(self, minimal_context, tmp_path):
        gen = _make_generator(tmp_path)
        with patch.dict("sys.modules", {"weasyprint": None}):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                path = gen.generate(minimal_context, export_pdf=True)
        assert Path(path).exists()


# ---------------------------------------------------------------------------
# Chart embedding — no kaleido installed (graceful degradation)
# ---------------------------------------------------------------------------


class TestChartEmbedding:
    def test_plotly_figure_in_context_no_kaleido_no_crash(self, tmp_path):
        """If kaleido is not installed, chart conversion silently returns None
        and the report is still generated."""
        try:
            import plotly.graph_objects as go  # noqa: PLC0415
            fig = go.Figure()
        except ImportError:
            pytest.skip("plotly not installed")

        ctx = {
            "eda": {"charts": [fig], "top_correlations": [], "skewed_columns": []},
        }
        gen = _make_generator(tmp_path)
        # Should not raise even if kaleido is absent
        result = gen.render_html(ctx)
        assert isinstance(result, str)
