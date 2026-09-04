"""Security tests for CSV formula injection sanitization (SEC-TASK-03 / CWE-1236)."""

import io

import pandas as pd
import pytest

from src.views.export_view import (
    FORMULA_PREFIXES,
    get_csv_bytes,
    get_parquet_bytes,
    sanitize_dataframe_for_csv,
)


@pytest.fixture
def malicious_formula_df() -> pd.DataFrame:
    """Fixture with various formula injection payloads in string columns."""
    return pd.DataFrame(
        {
            "payloads": [
                "=cmd|'/C calc'!A0",
                "-2+3",
                "@SUM(A1:A10)",
                "+12345",
                "=HYPERLINK('http://malicious.com','Click')",
                "\t=cmd",
                "\r=calc",
                "normal_text",
            ],
            "safe_strings": [
                "hello world",
                "foo_bar",
                "123abc",
                "",
                "test",
                "sample",
                "value",
                "another",
            ],
        }
    )


def test_formula_prefixes_constant_defined():
    """Verifies that FORMULA_PREFIXES contains standard spreadsheet injection trigger characters."""
    assert isinstance(FORMULA_PREFIXES, tuple)
    for char in ("=", "+", "-", "@", "\t", "\r"):
        assert char in FORMULA_PREFIXES


def test_csv_export_escapes_formula_prefixes(malicious_formula_df):
    """Verifies that get_csv_bytes prefixes dangerous formula starters with a single quote."""
    csv_bytes = get_csv_bytes(malicious_formula_df)
    csv_text = csv_bytes.decode("utf-8")

    # Verify that dangerous strings are prefixed with single quote in CSV text
    assert "'=cmd|'/C calc'!A0" in csv_text
    assert "'-2+3" in csv_text
    assert "'@SUM(A1:A10)" in csv_text
    assert "'+12345" in csv_text
    assert "'=HYPERLINK('http://malicious.com','Click')" in csv_text
    assert "'\t=cmd" in csv_text
    assert "'\r=calc" in csv_text

    # Safe strings must not be prefixed
    assert "normal_text" in csv_text
    assert "'normal_text" not in csv_text
    assert "hello world" in csv_text
    assert "'hello world" not in csv_text


def test_csv_export_preserves_numeric_columns():
    """Verifies that legitimate numeric values (integers, floats, negative numbers) are not modified."""
    df = pd.DataFrame(
        {
            "integers": [-42, -1, 0, 100],
            "floats": [-42.5, -0.01, 3.14, 99.9],
            "text": ["-formula", "safe", "+formula", "regular"],
        }
    )
    csv_bytes = get_csv_bytes(df)
    csv_text = csv_bytes.decode("utf-8")

    lines = [line for line in csv_text.strip().splitlines() if line]
    # Header
    assert lines[0] == "integers,floats,text"
    # Row 1: -42,-42.5,'-formula
    assert "-42,-42.5,'-formula" in lines[1]
    # Row 2: -1,-0.01,safe
    assert "-1,-0.01,safe" in lines[2]
    # Row 4: 100,99.9,regular
    assert "100,99.9,regular" in lines[4]


def test_csv_export_does_not_mutate_original_dataframe(malicious_formula_df):
    """Verifies that the original dataframe is never mutated in-place by CSV sanitization."""
    original_val = malicious_formula_df.loc[0, "payloads"]
    assert original_val == "=cmd|'/C calc'!A0"

    _ = get_csv_bytes(malicious_formula_df)

    # In-memory dataframe must retain its exact value without prefix
    assert malicious_formula_df.loc[0, "payloads"] == "=cmd|'/C calc'!A0"


def test_sanitize_dataframe_for_csv_categorical_and_nulls():
    """Verifies handling of categorical dtypes and null/missing values."""
    df = pd.DataFrame(
        {
            "cat_col": pd.Categorical(["=evil", "safe", "+also_evil", None]),
            "str_col": ["@at", None, pd.NA, "ok"],
        }
    )
    sanitized = sanitize_dataframe_for_csv(df)

    assert sanitized["cat_col"].iloc[0] == "'=evil"
    assert sanitized["cat_col"].iloc[1] == "safe"
    assert sanitized["cat_col"].iloc[2] == "'+also_evil"
    assert pd.isna(sanitized["cat_col"].iloc[3])

    assert sanitized["str_col"].iloc[0] == "'@at"
    assert pd.isna(sanitized["str_col"].iloc[1])
    assert pd.isna(sanitized["str_col"].iloc[2])
    assert sanitized["str_col"].iloc[3] == "ok"


def test_sanitize_dataframe_for_csv_column_headers():
    """Verifies that column header names starting with formula characters are also escaped."""
    df = pd.DataFrame(
        {
            "=danger_header": [1, 2],
            "@another": [3, 4],
            "normal_header": [5, 6],
        }
    )
    sanitized = sanitize_dataframe_for_csv(df)

    assert "'=danger_header" in sanitized.columns
    assert "'@another" in sanitized.columns
    assert "normal_header" in sanitized.columns


def test_parquet_export_unaffected():
    """Verifies that Parquet binary export is not modified with unnecessary formula quotes."""
    df = pd.DataFrame(
        {
            "str_col": ["=cmd", "@sum", "text"],
            "num_col": [-10.5, 0.0, 5.2],
        }
    )
    parquet_bytes = get_parquet_bytes(df)
    assert isinstance(parquet_bytes, bytes)
    assert len(parquet_bytes) > 0

    # Read back from Parquet buffer to ensure values are unaltered
    df_read = pd.read_parquet(io.BytesIO(parquet_bytes))
    assert df_read["str_col"].tolist() == ["=cmd", "@sum", "text"]
    assert df_read["num_col"].tolist() == [-10.5, 0.0, 5.2]
