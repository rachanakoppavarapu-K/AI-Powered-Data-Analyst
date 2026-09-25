"""Tests for src/ingestion/data_loader.py — ST-02."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.data_loader import DataLoadError, DataLoader


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def loader() -> DataLoader:
    return DataLoader(max_mb=50)


def _csv_bytes(content: str) -> io.BytesIO:
    buf = io.BytesIO(content.encode())
    buf.name = "test.csv"
    return buf


def _excel_bytes(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    buf.name = "test.xlsx"
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf


# ── Happy-path: CSV ───────────────────────────────────────────────────────────


class TestCSVLoading:
    def test_simple_csv(self, loader: DataLoader) -> None:
        csv = _csv_bytes("a,b,c\n1,2,3\n4,5,6\n")
        df = loader.load(csv)
        assert list(df.columns) == ["a", "b", "c"]
        assert len(df) == 2

    def test_semicolon_delimiter(self, loader: DataLoader) -> None:
        csv = _csv_bytes("x;y;z\n10;20;30\n")
        df = loader.load(csv)
        assert list(df.columns) == ["x", "y", "z"]

    def test_tab_delimiter(self, loader: DataLoader) -> None:
        csv = _csv_bytes("col1\tcol2\n1\t2\n3\t4\n")
        df = loader.load(csv)
        assert "col1" in df.columns

    def test_pipe_delimiter(self, loader: DataLoader) -> None:
        csv = _csv_bytes("a|b|c\n1|2|3\n")
        df = loader.load(csv)
        assert list(df.columns) == ["a", "b", "c"]

    def test_returns_dataframe(self, loader: DataLoader) -> None:
        csv = _csv_bytes("name,age\nAlice,30\nBob,25\n")
        df = loader.load(csv)
        assert isinstance(df, pd.DataFrame)

    def test_load_from_path(self, loader: DataLoader, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text("x,y\n1,2\n3,4\n")
        df = loader.load(p)
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 2

    def test_load_from_string_path(self, loader: DataLoader, tmp_path: Path) -> None:
        p = tmp_path / "data.csv"
        p.write_text("col_a,col_b\n5,6\n")
        df = loader.load(str(p))
        assert len(df) == 1


# ── Happy-path: Excel ─────────────────────────────────────────────────────────


class TestExcelLoading:
    def test_xlsx_loading(self, loader: DataLoader) -> None:
        source = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        buf = _excel_bytes(source)
        df = loader.load(buf)
        assert list(df.columns) == ["foo", "bar"]
        assert len(df) == 2

    def test_xlsx_from_path(self, loader: DataLoader, tmp_path: Path) -> None:
        p = tmp_path / "data.xlsx"
        source = pd.DataFrame({"col1": [10, 20], "col2": ["a", "b"]})
        source.to_excel(str(p), index=False)
        df = loader.load(p)
        assert len(df) == 2


# ── Column normalisation ──────────────────────────────────────────────────────


class TestColumnNormalisation:
    def test_lowercase(self, loader: DataLoader) -> None:
        csv = _csv_bytes("Name,AGE,City\nAlice,30,NY\n")
        df = loader.load(csv)
        assert list(df.columns) == ["name", "age", "city"]

    def test_spaces_to_underscores(self, loader: DataLoader) -> None:
        csv = _csv_bytes("First Name,Last Name\nJohn,Doe\n")
        df = loader.load(csv)
        assert "first_name" in df.columns
        assert "last_name" in df.columns

    def test_special_chars_replaced(self, loader: DataLoader) -> None:
        csv = _csv_bytes("Col-A,Col.B,Col (C)\n1,2,3\n")
        df = loader.load(csv)
        for col in df.columns:
            assert col.isidentifier(), f"Column '{col}' is not a valid identifier"

    def test_leading_trailing_underscores_stripped(self, loader: DataLoader) -> None:
        csv = _csv_bytes("_col_,__x__\n1,2\n")
        df = loader.load(csv)
        # After normalise, leading/trailing underscores from the stripping step are removed
        for col in df.columns:
            assert not col.startswith("_"), col
            assert not col.endswith("_"), col


# ── Validation errors ─────────────────────────────────────────────────────────


class TestValidationErrors:
    def test_unsupported_extension_raises(self, loader: DataLoader) -> None:
        buf = io.BytesIO(b"some content")
        buf.name = "data.json"
        with pytest.raises(DataLoadError, match="Unsupported file type"):
            loader.load(buf)

    def test_file_too_large_raises(self, tmp_path: Path) -> None:
        small_loader = DataLoader(max_mb=0)  # 0 MB limit → any file is too large
        p = tmp_path / "big.csv"
        p.write_text("a,b\n1,2\n")
        with pytest.raises(DataLoadError, match="exceeds the"):
            small_loader.load(p)

    def test_empty_csv_raises(self, loader: DataLoader) -> None:
        # A CSV with headers but no rows produces an empty DataFrame
        buf = _csv_bytes("a,b,c\n")
        with pytest.raises(DataLoadError, match="empty"):
            loader.load(buf)

    def test_corrupt_excel_raises(self, loader: DataLoader) -> None:
        buf = io.BytesIO(b"not an excel file at all")
        buf.name = "broken.xlsx"
        with pytest.raises(DataLoadError, match="Failed to parse"):
            loader.load(buf)

    def test_unsupported_input_type_raises(self, loader: DataLoader) -> None:
        with pytest.raises(DataLoadError, match="Unsupported file input type"):
            loader.load(12345)  # type: ignore[arg-type]


# ── Reference datasets ────────────────────────────────────────────────────────


class TestReferenceDatasets:
    """Smoke-test that all three reference datasets load cleanly."""

    datasets_dir = Path(__file__).parent.parent / "datasets"

    @pytest.mark.parametrize(
        "filename",
        ["titanic_like.csv", "house_prices.csv", "customers.csv"],
    )
    def test_reference_dataset_loads(
        self, loader: DataLoader, filename: str
    ) -> None:
        path = self.datasets_dir / filename
        if not path.exists():
            pytest.skip(f"Reference dataset '{filename}' not yet generated.")
        df = loader.load(path)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert len(df.columns) > 0
