"""Data ingestion module — loads and validates CSV / Excel files into DataFrames."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Union

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_MB: int = 50
_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".xlsx", ".xls"})

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DataLoadError(ValueError):
    """Raised when a file cannot be loaded or fails validation."""


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------


class DataLoader:
    """Load and validate a CSV or Excel file into a clean :class:`pd.DataFrame`.

    Parameters
    ----------
    max_mb:
        Maximum allowed file size in megabytes.  Defaults to the value of the
        ``MAX_UPLOAD_MB`` environment variable, or 50 MB when that variable is
        not set.
    """

    def __init__(self, max_mb: int | None = None) -> None:
        if max_mb is None:
            max_mb = int(os.getenv("MAX_UPLOAD_MB", _DEFAULT_MAX_MB))
        self.max_mb: int = max_mb

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self, file: Union[str, Path, io.IOBase, "UploadedFile"]  # type: ignore[name-defined]
    ) -> pd.DataFrame:
        """Load *file* and return a validated, normalised :class:`pd.DataFrame`.

        Parameters
        ----------
        file:
            A file path (:class:`str` / :class:`~pathlib.Path`), an open
            binary-mode file-like object, or a Streamlit ``UploadedFile``.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with normalised column names.

        Raises
        ------
        DataLoadError
            If the file extension is unsupported, the file exceeds the size
            limit, or the resulting DataFrame is empty / has no columns.
        """
        name, raw = self._extract_name_and_bytes(file)
        ext = Path(name).suffix.lower()

        self._validate_extension(ext, name)
        self._validate_size(raw, name)

        df = self._parse(raw, ext, name)
        df = self._normalise_columns(df)
        self._validate_dataframe(df, name)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_name_and_bytes(
        self, file: Union[str, Path, io.IOBase, object]
    ) -> tuple[str, bytes]:
        """Return *(filename, raw_bytes)* from any supported input type."""
        if isinstance(file, (str, Path)):
            path = Path(file)
            return path.name, path.read_bytes()

        # Streamlit UploadedFile and generic file-like objects both expose
        # `.read()` and a `.name` attribute.
        if hasattr(file, "read"):
            raw: bytes = file.read()  # type: ignore[union-attr]
            # Rewind if possible (e.g. BytesIO)
            if hasattr(file, "seek"):
                file.seek(0)  # type: ignore[union-attr]
            name: str = getattr(file, "name", "upload")
            return name, raw

        raise DataLoadError(
            f"Unsupported file input type: {type(file).__name__}. "
            "Expected a file path, BytesIO, or Streamlit UploadedFile."
        )

    def _validate_extension(self, ext: str, name: str) -> None:
        if ext not in _SUPPORTED_EXTENSIONS:
            raise DataLoadError(
                f"Unsupported file type '{ext}' for '{name}'. "
                f"Supported types: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

    def _validate_size(self, raw: bytes, name: str) -> None:
        size_mb = len(raw) / (1024 * 1024)
        if size_mb > self.max_mb:
            raise DataLoadError(
                f"File '{name}' is {size_mb:.1f} MB, which exceeds the "
                f"{self.max_mb} MB limit."
            )

    def _parse(self, raw: bytes, ext: str, name: str) -> pd.DataFrame:
        buf = io.BytesIO(raw)
        try:
            if ext == ".csv":
                # Sniff delimiter from first 4 KB
                sample = raw[:4096].decode("utf-8", errors="replace")
                sep = self._detect_delimiter(sample)
                return pd.read_csv(buf, sep=sep, engine="python")
            else:
                return pd.read_excel(buf, engine="openpyxl")
        except Exception as exc:
            raise DataLoadError(
                f"Failed to parse '{name}': {exc}"
            ) from exc

    @staticmethod
    def _detect_delimiter(sample: str) -> str:
        """Return the most likely CSV delimiter for *sample*."""
        import csv

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            return ","  # default fallback

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase, strip, and replace spaces/special chars with underscores."""
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
