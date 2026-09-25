"""
Tests for src/nl_query/nl_query_engine.py.

Coverage areas
--------------
- Question classification (regex patterns) for all supported query types
- Pandas execution correctness for each query type
- Column extraction from free-text questions
- build_nl_query_prompt embeds computed values verbatim (deterministic firewall)
- NLQueryResult structure and field correctness
- LLM integration: prompt is passed to client; client result is surfaced
- Unsupported questions handled safely without raising
- Empty/None question edge cases
- DataFrames without numeric or categorical columns
- watsonx_client=None returns explanation=None but still computes result
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.nl_query.nl_query_engine import (
    NLQueryEngine,
    NLQueryResult,
    QT_AVERAGE,
    QT_CORRELATION,
    QT_COUNT,
    QT_DISTRIBUTION,
    QT_MAX_CATEGORY,
    QT_SUMMARY,
    QT_SUM,
    QT_UNSUPPORTED,
    _extract_column,
    build_nl_query_prompt,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small mixed dataset with numeric and categorical columns."""
    return pd.DataFrame(
        {
            "age": [25, 32, 41, 28, 55, 22, 37, 44, 30, 60],
            "salary": [40000, 55000, 70000, 45000, 90000, 35000, 62000, 75000, 50000, 95000],
            "sales": [200, 300, 450, 250, 600, 180, 350, 500, 280, 620],
            "department": ["HR", "IT", "IT", "HR", "Finance", "HR", "IT", "Finance", "HR", "Finance"],
            "region": ["North", "South", "North", "East", "South", "East", "North", "South", "East", "North"],
        }
    )


@pytest.fixture()
def engine(sample_df: pd.DataFrame) -> NLQueryEngine:
    """NLQueryEngine with no watsonx client (deterministic-only mode)."""
    return NLQueryEngine(sample_df)


@pytest.fixture()
def engine_with_client(sample_df: pd.DataFrame) -> tuple[NLQueryEngine, MagicMock]:
    """NLQueryEngine with a mocked watsonx client."""
    mock_client = MagicMock()
    mock_client.generate.return_value = "This is a mock LLM explanation."
    eng = NLQueryEngine(sample_df, watsonx_client=mock_client)
    return eng, mock_client


# ---------------------------------------------------------------------------
# 1. NLQueryResult dataclass
# ---------------------------------------------------------------------------


class TestNLQueryResult:
    def test_default_fields(self):
        result = NLQueryResult(question="q", query_type="AVERAGE")
        assert result.question == "q"
        assert result.query_type == "AVERAGE"
        assert result.computed_result is None
        assert result.explanation is None
        assert result.error is None

    def test_all_fields_set(self):
        result = NLQueryResult(
            question="q",
            query_type="SUM",
            computed_result={"salary": 99.0},
            explanation="Some text",
            error=None,
        )
        assert result.computed_result == {"salary": 99.0}
        assert result.explanation == "Some text"


# ---------------------------------------------------------------------------
# 2. Question classification
# ---------------------------------------------------------------------------


class TestClassification:
    """Verify _classify maps questions to the correct QT_* constants."""

    @pytest.mark.parametrize("question", [
        "How many records are in this dataset?",
        "What is the total count of rows?",
        "Count the number of entries",
        "How many rows are present?",
        "What is the row count?",
    ])
    def test_count_questions(self, engine: NLQueryEngine, question: str):
        assert engine._classify(question) == QT_COUNT

    @pytest.mark.parametrize("question", [
        "What is the average salary?",
        "What is the mean age of employees?",
        "Show me the avg sales",
        "What is the typical salary?",
    ])
    def test_average_questions(self, engine: NLQueryEngine, question: str):
        assert engine._classify(question) == QT_AVERAGE

    @pytest.mark.parametrize("question", [
        "What is the total sales?",
        "What is the sum of salary?",
        "Show me the aggregate revenue",
        "What is the combined salary?",
    ])
    def test_sum_questions(self, engine: NLQueryEngine, question: str):
        assert engine._classify(question) == QT_SUM

    @pytest.mark.parametrize("question", [
        "Which department has the highest salary?",
        "What category has the most sales?",
        "Which region has the largest revenue?",
        "Which group has the maximum value?",
    ])
    def test_max_category_questions(self, engine: NLQueryEngine, question: str):
        assert engine._classify(question) == QT_MAX_CATEGORY

    @pytest.mark.parametrize("question", [
        "What is the distribution of age?",
        "Show me the spread of salary",
        "What is the breakdown of department?",
        "How is salary distributed?",
        "Give me the frequency of region values",
    ])
    def test_distribution_questions(self, engine: NLQueryEngine, question: str):
        assert engine._classify(question) == QT_DISTRIBUTION

    @pytest.mark.parametrize("question", [
        "What are the strongest correlations?",
        "Which features are most correlated?",
        "Show me the correlation between columns",
        "What is the relationship between age and salary?",
    ])
    def test_correlation_questions(self, engine: NLQueryEngine, question: str):
        assert engine._classify(question) == QT_CORRELATION

    @pytest.mark.parametrize("question", [
        "Summarise the dataset",
        "Give me a summary",
        "Describe the data",
        "Provide an overview of this dataset",
        "Tell me about this dataset",
    ])
    def test_summary_questions(self, engine: NLQueryEngine, question: str):
        assert engine._classify(question) == QT_SUMMARY

    @pytest.mark.parametrize("question", [
        "What is the meaning of life?",
        "Predict next year's revenue",
        "Build a model for me",
        "asdfghjkl",
    ])
    def test_unsupported_questions(self, engine: NLQueryEngine, question: str):
        assert engine._classify(question) == QT_UNSUPPORTED


# ---------------------------------------------------------------------------
# 3. Column extraction
# ---------------------------------------------------------------------------


class TestExtractColumn:
    def test_exact_column_name(self):
        assert _extract_column("What is the average salary?", ["salary", "age"]) == "salary"

    def test_case_insensitive(self):
        assert _extract_column("What is the SALARY average?", ["salary", "age"]) == "salary"

    def test_longest_match_wins(self):
        # "total_sales" should match before "sales"
        result = _extract_column("What is the total_sales?", ["sales", "total_sales"])
        assert result == "total_sales"

    def test_no_match_returns_none(self):
        assert _extract_column("What is the revenue?", ["salary", "age"]) is None

    def test_word_boundary_respected(self):
        # "age" should NOT match inside "average" — word-boundary guard
        result = _extract_column("What is the average?", ["age"])
        # "average" does not contain the word "age" at a word boundary, so None
        assert result is None

    def test_empty_column_list(self):
        assert _extract_column("anything", []) is None


# ---------------------------------------------------------------------------
# 4. COUNT execution
# ---------------------------------------------------------------------------


class TestCountExecution:
    def test_returns_row_count(self, engine: NLQueryEngine, sample_df: pd.DataFrame):
        result = engine.ask("How many records are in the dataset?")
        assert result.query_type == QT_COUNT
        assert result.computed_result == {"row_count": len(sample_df)}
        assert result.error is None

    def test_row_count_value(self, engine: NLQueryEngine):
        result = engine.ask("How many rows?")
        assert result.computed_result["row_count"] == 10


# ---------------------------------------------------------------------------
# 5. AVERAGE execution
# ---------------------------------------------------------------------------


class TestAverageExecution:
    def test_average_named_column(self, engine: NLQueryEngine, sample_df: pd.DataFrame):
        result = engine.ask("What is the average salary?")
        assert result.query_type == QT_AVERAGE
        assert result.error is None
        expected = round(float(sample_df["salary"].mean()), 6)
        assert result.computed_result == {"salary": expected}

    def test_average_all_numeric_when_no_column_named(self, engine: NLQueryEngine, sample_df: pd.DataFrame):
        result = engine.ask("What is the average value?")
        assert result.query_type == QT_AVERAGE
        assert result.error is None
        # Should return a dict with all numeric columns
        for col in ["age", "salary", "sales"]:
            assert col in result.computed_result

    def test_average_result_is_dict(self, engine: NLQueryEngine):
        result = engine.ask("What is the mean age?")
        assert isinstance(result.computed_result, dict)

    def test_average_correct_value(self, engine: NLQueryEngine, sample_df: pd.DataFrame):
        result = engine.ask("What is the average age?")
        expected = round(float(sample_df["age"].mean()), 6)
        assert result.computed_result["age"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 6. SUM execution
# ---------------------------------------------------------------------------


class TestSumExecution:
    def test_sum_named_column(self, engine: NLQueryEngine, sample_df: pd.DataFrame):
        result = engine.ask("What is the total salary?")
        assert result.query_type == QT_SUM
        assert result.error is None
        expected = round(float(sample_df["salary"].sum()), 6)
        assert result.computed_result == {"salary": expected}

    def test_sum_all_numeric_when_no_column_named(self, engine: NLQueryEngine):
        result = engine.ask("What is the sum of all values?")
        assert isinstance(result.computed_result, dict)
        assert len(result.computed_result) > 0

    def test_sum_correct_value(self, engine: NLQueryEngine, sample_df: pd.DataFrame):
        result = engine.ask("What is the sum of sales?")
        expected = round(float(sample_df["sales"].sum()), 6)
        assert result.computed_result["sales"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 7. MAX_CATEGORY execution
# ---------------------------------------------------------------------------


class TestMaxCategoryExecution:
    def test_categorical_col_mentioned(self, engine: NLQueryEngine, sample_df: pd.DataFrame):
        result = engine.ask("Which department has the highest salary?")
        assert result.query_type == QT_MAX_CATEGORY
        assert result.error is None
        # Should identify a department
        assert "top_category" in result.computed_result
        assert result.computed_result["top_category"] in sample_df["department"].unique()

    def test_result_contains_expected_keys(self, engine: NLQueryEngine):
        result = engine.ask("Which department has the highest salary?")
        cr = result.computed_result
        assert "category_column" in cr
        assert "top_category" in cr

    def test_fallback_when_no_column_named(self, engine: NLQueryEngine):
        result = engine.ask("Which category has the highest value?")
        assert result.query_type == QT_MAX_CATEGORY
        assert result.computed_result is not None
        assert result.error is None


# ---------------------------------------------------------------------------
# 8. DISTRIBUTION execution
# ---------------------------------------------------------------------------


class TestDistributionExecution:
    def test_numeric_column_distribution(self, engine: NLQueryEngine):
        result = engine.ask("What is the distribution of salary?")
        assert result.query_type == QT_DISTRIBUTION
        assert result.error is None
        cr = result.computed_result
        assert cr["column"] == "salary"
        for key in ("mean", "std", "min", "median", "max"):
            assert key in cr

    def test_categorical_column_distribution(self, engine: NLQueryEngine, sample_df: pd.DataFrame):
        result = engine.ask("What is the distribution of department?")
        assert result.query_type == QT_DISTRIBUTION
        assert result.error is None
        cr = result.computed_result
        assert cr["column"] == "department"
        assert "value_counts" in cr
        total = sum(cr["value_counts"].values())
        assert total == len(sample_df)

    def test_numeric_stats_correct(self, engine: NLQueryEngine, sample_df: pd.DataFrame):
        result = engine.ask("Show me the distribution of age")
        cr = result.computed_result
        assert cr["mean"] == pytest.approx(round(float(sample_df["age"].mean()), 6))
        assert cr["min"] == pytest.approx(round(float(sample_df["age"].min()), 6))
        assert cr["max"] == pytest.approx(round(float(sample_df["age"].max()), 6))


# ---------------------------------------------------------------------------
# 9. CORRELATION execution
# ---------------------------------------------------------------------------


class TestCorrelationExecution:
    def test_returns_list(self, engine: NLQueryEngine):
        result = engine.ask("What are the strongest correlations?")
        assert result.query_type == QT_CORRELATION
        assert result.error is None
        assert isinstance(result.computed_result, list)

    def test_pairs_have_required_keys(self, engine: NLQueryEngine):
        result = engine.ask("Which features are most correlated?")
        for pair in result.computed_result:
            assert "feature_a" in pair
            assert "feature_b" in pair
            assert "correlation" in pair

    def test_sorted_by_abs_correlation(self, engine: NLQueryEngine):
        result = engine.ask("Show me the correlations")
        pairs = result.computed_result
        if len(pairs) >= 2:
            assert abs(pairs[0]["correlation"]) >= abs(pairs[1]["correlation"])

    def test_no_self_pairs(self, engine: NLQueryEngine):
        result = engine.ask("Show correlations")
        for pair in result.computed_result:
            assert pair["feature_a"] != pair["feature_b"]

    def test_no_duplicate_pairs(self, engine: NLQueryEngine):
        result = engine.ask("Show correlations")
        seen: set[frozenset[str]] = set()
        for pair in result.computed_result:
            key = frozenset({pair["feature_a"], pair["feature_b"]})
            assert key not in seen, f"Duplicate pair: {pair}"
            seen.add(key)

    def test_correlation_values_in_range(self, engine: NLQueryEngine):
        result = engine.ask("Show correlations")
        for pair in result.computed_result:
            assert -1.0 <= pair["correlation"] <= 1.0

    def test_single_numeric_col_returns_empty(self):
        df = pd.DataFrame({"x": [1, 2, 3], "cat": ["a", "b", "c"]})
        eng = NLQueryEngine(df)
        result = eng.ask("What are the strongest correlations?")
        assert result.computed_result == []


# ---------------------------------------------------------------------------
# 10. SUMMARY execution
# ---------------------------------------------------------------------------


class TestSummaryExecution:
    def test_returns_dict(self, engine: NLQueryEngine):
        result = engine.ask("Summarise the dataset")
        assert result.query_type == QT_SUMMARY
        assert isinstance(result.computed_result, dict)
        assert result.error is None

    def test_required_keys_present(self, engine: NLQueryEngine):
        result = engine.ask("Give me a summary of the data")
        cr = result.computed_result
        for key in ("n_rows", "n_cols", "n_numeric", "n_categorical", "missing_cells", "missing_pct"):
            assert key in cr, f"Missing key: {key}"

    def test_n_rows_correct(self, engine: NLQueryEngine, sample_df: pd.DataFrame):
        result = engine.ask("Describe the dataset")
        assert result.computed_result["n_rows"] == len(sample_df)

    def test_n_cols_correct(self, engine: NLQueryEngine, sample_df: pd.DataFrame):
        result = engine.ask("Describe the data")
        assert result.computed_result["n_cols"] == len(sample_df.columns)

    def test_numeric_means_present(self, engine: NLQueryEngine):
        result = engine.ask("Overview of the dataset")
        assert "numeric_means" in result.computed_result

    def test_missing_cells_zero_for_clean_data(self, engine: NLQueryEngine):
        result = engine.ask("Summarise")
        assert result.computed_result["missing_cells"] == 0


# ---------------------------------------------------------------------------
# 11. UNSUPPORTED questions
# ---------------------------------------------------------------------------


class TestUnsupportedQuestions:
    def test_unsupported_type(self, engine: NLQueryEngine):
        result = engine.ask("What is the meaning of life?")
        assert result.query_type == QT_UNSUPPORTED

    def test_unsupported_computed_result_is_none(self, engine: NLQueryEngine):
        result = engine.ask("Predict future sales")
        assert result.computed_result is None

    def test_unsupported_explanation_is_none(self, engine: NLQueryEngine):
        result = engine.ask("Build me a machine learning model")
        assert result.explanation is None

    def test_unsupported_no_error(self, engine: NLQueryEngine):
        result = engine.ask("Random gibberish question xyz")
        assert result.error is None

    def test_empty_question_returns_unsupported(self, engine: NLQueryEngine):
        result = engine.ask("")
        assert result.query_type == QT_UNSUPPORTED
        assert result.error is not None

    def test_whitespace_only_question(self, engine: NLQueryEngine):
        result = engine.ask("   ")
        assert result.query_type == QT_UNSUPPORTED
        assert result.error is not None


# ---------------------------------------------------------------------------
# 12. LLM integration (mocked client)
# ---------------------------------------------------------------------------


class TestLLMIntegration:
    def test_explanation_returned_when_client_configured(
        self, engine_with_client: tuple[NLQueryEngine, MagicMock]
    ):
        eng, mock_client = engine_with_client
        result = eng.ask("How many records are in the dataset?")
        assert result.explanation == "This is a mock LLM explanation."

    def test_client_generate_called_once(
        self, engine_with_client: tuple[NLQueryEngine, MagicMock]
    ):
        eng, mock_client = engine_with_client
        eng.ask("What is the average salary?")
        mock_client.generate.assert_called_once()

    def test_prompt_contains_computed_result(
        self, engine_with_client: tuple[NLQueryEngine, MagicMock]
    ):
        eng, mock_client = engine_with_client
        eng.ask("How many rows?")
        called_prompt = mock_client.generate.call_args[0][0]
        # The row count (10) must appear in the prompt
        assert "10" in called_prompt

    def test_prompt_contains_query_type(
        self, engine_with_client: tuple[NLQueryEngine, MagicMock]
    ):
        eng, mock_client = engine_with_client
        eng.ask("What is the average age?")
        called_prompt = mock_client.generate.call_args[0][0]
        assert QT_AVERAGE in called_prompt

    def test_prompt_contains_original_question(
        self, engine_with_client: tuple[NLQueryEngine, MagicMock]
    ):
        eng, mock_client = engine_with_client
        question = "What is the average salary?"
        eng.ask(question)
        called_prompt = mock_client.generate.call_args[0][0]
        assert question in called_prompt

    def test_client_not_called_for_unsupported(
        self, engine_with_client: tuple[NLQueryEngine, MagicMock]
    ):
        eng, mock_client = engine_with_client
        eng.ask("What is the meaning of life?")
        mock_client.generate.assert_not_called()

    def test_explanation_none_when_no_client(self, engine: NLQueryEngine):
        result = engine.ask("How many rows?")
        assert result.explanation is None

    def test_client_returns_none_explanation_surfaced(
        self, sample_df: pd.DataFrame
    ):
        mock_client = MagicMock()
        mock_client.generate.return_value = None
        eng = NLQueryEngine(sample_df, watsonx_client=mock_client)
        result = eng.ask("How many records?")
        assert result.explanation is None


# ---------------------------------------------------------------------------
# 13. build_nl_query_prompt — deterministic firewall
# ---------------------------------------------------------------------------


class TestBuildNLQueryPrompt:
    def test_returns_string(self):
        prompt = build_nl_query_prompt("How many rows?", QT_COUNT, {"row_count": 10})
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_question_injected(self):
        question = "What is the average salary?"
        prompt = build_nl_query_prompt(question, QT_AVERAGE, {"salary": 55000.0})
        assert question in prompt

    def test_query_type_injected(self):
        prompt = build_nl_query_prompt("Q", QT_SUM, {"sales": 3000.0})
        assert QT_SUM in prompt

    def test_computed_result_injected_dict(self):
        prompt = build_nl_query_prompt("Q", QT_COUNT, {"row_count": 42})
        assert "42" in prompt

    def test_computed_result_injected_list(self):
        prompt = build_nl_query_prompt(
            "Q", QT_CORRELATION,
            [{"feature_a": "age", "feature_b": "salary", "correlation": 0.75}],
        )
        assert "0.75" in prompt
        assert "age" in prompt
        assert "salary" in prompt

    def test_llm_not_asked_to_calculate(self):
        prompt = build_nl_query_prompt("Q", QT_AVERAGE, {"age": 35.0})
        assert not re.search(r"\bcalculate\b", prompt, re.IGNORECASE)
        assert not re.search(r"\bcompute\b", prompt, re.IGNORECASE)

    def test_do_not_introduce_numbers_instruction(self):
        prompt = build_nl_query_prompt("Q", QT_AVERAGE, {"age": 35.0})
        assert "Do not introduce any numbers" in prompt

    def test_numeric_value_present_in_prompt(self):
        prompt = build_nl_query_prompt("Q", QT_SUMMARY, {"n_rows": 1234})
        assert "1234" in prompt


# ---------------------------------------------------------------------------
# 14. Edge cases — DataFrames with missing data
# ---------------------------------------------------------------------------


class TestEdgeCasesDataFrame:
    def test_all_numeric_df(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        eng = NLQueryEngine(df)
        result = eng.ask("Summarise the dataset")
        assert result.error is None
        assert result.computed_result["n_categorical"] == 0

    def test_all_categorical_df(self):
        df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
        eng = NLQueryEngine(df)
        result = eng.ask("Summarise the dataset")
        assert result.error is None
        assert result.computed_result["n_numeric"] == 0

    def test_df_with_missing_values(self):
        df = pd.DataFrame({"val": [1.0, None, 3.0, None, 5.0]})
        eng = NLQueryEngine(df)
        result = eng.ask("Summarise the dataset")
        assert result.computed_result["missing_cells"] == 2

    def test_single_row_df(self):
        df = pd.DataFrame({"x": [99]})
        eng = NLQueryEngine(df)
        result = eng.ask("How many rows?")
        assert result.computed_result["row_count"] == 1

    def test_invalid_df_type_raises(self):
        with pytest.raises(TypeError):
            NLQueryEngine("not a dataframe")  # type: ignore[arg-type]

    def test_correlation_no_numeric_cols(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        eng = NLQueryEngine(df)
        result = eng.ask("What are the strongest correlations?")
        assert result.computed_result == []
        assert result.error is None


# ---------------------------------------------------------------------------
# 15. ask() result structure — always returns NLQueryResult
# ---------------------------------------------------------------------------


class TestAskResultStructure:
    def test_result_is_nl_query_result(self, engine: NLQueryEngine):
        result = engine.ask("How many rows?")
        assert isinstance(result, NLQueryResult)

    def test_question_preserved(self, engine: NLQueryEngine):
        q = "What is the total sales?"
        result = engine.ask(q)
        assert result.question == q

    def test_no_exception_for_any_question(self, engine: NLQueryEngine):
        questions = [
            "?",
            "1 + 1",
            "SELECT * FROM table",
            "import os; os.system('rm -rf /')",
            "What is the average salary of employees aged over 30?",
        ]
        for q in questions:
            result = engine.ask(q)
            assert isinstance(result, NLQueryResult)

    def test_deterministic_results_across_calls(self, engine: NLQueryEngine):
        """Identical questions must produce identical computed results."""
        r1 = engine.ask("How many rows?")
        r2 = engine.ask("How many rows?")
        assert r1.computed_result == r2.computed_result
