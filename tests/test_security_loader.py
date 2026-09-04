"""Security tests for file loader guardrails and cache bounds (SEC-TASK-02 / CWE-400, CWE-776)."""

import io
from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
import pytest
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec

from src.loader import (
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    _get_file_size,
    load_tabular_file,
)


def _create_uploaded_file(name: str, data: bytes, size: int | None = None) -> UploadedFile:
    """Helper to construct a Streamlit UploadedFile with optional custom size."""
    rec = UploadedFileRec(
        file_id="sec-test-file-id",
        name=name,
        type="text/csv" if name.endswith(".csv") else "application/octet-stream",
        data=data,
    )
    up = UploadedFile(rec, file_urls=MagicMock())
    if size is not None:
        object.__setattr__(up, "size", size)
    return up


def test_loader_constants_defined():
    """Verifies that upload size limit constants are properly defined."""
    assert isinstance(MAX_UPLOAD_SIZE_MB, int)
    assert MAX_UPLOAD_SIZE_MB == 100
    assert MAX_UPLOAD_SIZE_BYTES == 100 * 1024 * 1024


def test_loader_cache_has_retention_bounds():
    """Verifies that st.cache_data on load_tabular_file configures max_entries and ttl."""
    assert hasattr(load_tabular_file, "_info"), "load_tabular_file must be wrapped by st.cache_data"
    info: Any = getattr(load_tabular_file, "_info", None)
    assert info is not None
    assert getattr(info, "max_entries", None) is not None, (
        "max_entries must not be None (memory bounding required)"
    )
    assert getattr(info, "max_entries", None) == 20, (
        f"Expected max_entries=20, got {getattr(info, 'max_entries', None)}"
    )
    assert getattr(info, "ttl", None) is not None, "ttl must not be None (cache expiry required)"
    assert getattr(info, "ttl", None) == 3600, (
        f"Expected ttl=3600, got {getattr(info, 'ttl', None)}"
    )
    assert getattr(info, "show_spinner", None) is False


def test_get_file_size_helper():
    """Verifies _get_file_size helper correctly detects sizes for various inputs."""
    # BytesIO
    b = io.BytesIO(b"1234567890")
    assert _get_file_size(b) == 10

    # UploadedFile
    up = _create_uploaded_file("test.csv", b"a,b\n1,2\n", size=1024)
    assert _get_file_size(up) == 1024


def test_loader_rejects_uploaded_file_exceeding_max_size():
    """Verifies that load_tabular_file rejects uploaded files exceeding MAX_UPLOAD_SIZE_BYTES."""
    oversized_size = MAX_UPLOAD_SIZE_BYTES + 1024  # Exceeds 100 MB limit
    fake_file = _create_uploaded_file(
        "large_dataset.csv",
        b"col1,col2\n1,2\n",
        size=oversized_size,
    )

    with pytest.raises(ValueError, match=r"(?i)excede|supera|l[ií]mite|100 MB"):
        load_tabular_file(fake_file, "large_dataset.csv")


def test_loader_accepts_valid_uploaded_file():
    """Verifies that legitimate files within the size limit load successfully."""
    csv_content = b"col1,col2\n10,20\n30,40\n"
    valid_file = _create_uploaded_file("valid.csv", csv_content)

    df, msg = load_tabular_file(valid_file, "valid.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert "✓ Cargado" in msg
    assert "2 filas, 2 cols" in msg


def test_loader_accepts_bytesio_buffer():
    """Verifies that standard BytesIO objects within size limit load properly."""
    b = io.BytesIO(b"x,y\n1.5,2.5\n3.5,4.5\n")
    df, _msg = load_tabular_file(b, "buffer.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_loader_rejects_empty_file():
    """Verifies that empty files raise a ValueError."""
    b = io.BytesIO(b"")
    with pytest.raises(ValueError, match=r"(?i)vac[ií]o"):
        load_tabular_file(b, "empty.csv")


def test_loader_rejects_unsupported_extension():
    """Verifies that unsupported file extensions raise ValueError."""
    b = io.BytesIO(b"some content")
    with pytest.raises(ValueError, match=r"(?i)no soportada"):
        load_tabular_file(b, "data.txt")


def test_loader_handles_latin1_encoding_safely():
    """Verifies that CSV with latin-1 encoded text falls back safely without ReDoS."""
    # 'años' in latin-1 is b'a\xf1os'
    latin1_csv = "nombre,años\nJosé,30\nMaría,25\n".encode("latin-1")
    b = io.BytesIO(latin1_csv)
    df, _msg = load_tabular_file(b, "personas_latin1.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert "años" in df.columns or "a\xf1os" in df.columns


def test_loader_error_does_not_leak_filesystem_paths():
    """Verifies that loader exceptions do not leak filesystem absolute paths (CWE-209)."""
    import re

    corrupted_data = io.BytesIO(b"NOT_A_PARQUET_FILE_CORRUPTED_BYTES")
    simulated_path = "/home/secret_server/data/corrupted.parquet"

    with pytest.raises(Exception) as exc_info:
        load_tabular_file(corrupted_data, simulated_path)

    error_msg = str(exc_info.value)
    # Must not contain server filesystem paths
    assert not re.search(r"(/home/|/var/|/tmp/|/usr/|/etc/|/opt/|[a-zA-Z]:\\)", error_msg), (
        f"Error message leaked internal filesystem path: {error_msg}"
    )
    # Must contain the clean filename and a user-friendly generic explanation
    assert "corrupted.parquet" in error_msg
    assert simulated_path not in error_msg


def test_loader_error_does_not_leak_internal_exception_details():
    """Verifies that corrupted files produce clean, generic error messages without C++/pyarrow internals."""
    malformed_csv = io.BytesIO(b"a,b\n1,2\n1,2,3,4\n5,6\n")
    with pytest.raises(RuntimeError) as exc_info:
        load_tabular_file(malformed_csv, "/var/data/bad_structure.csv")

    error_msg = str(exc_info.value)
    assert "/var/data" not in error_msg
    assert "bad_structure.csv" in error_msg
    assert "Expected 2 fields" not in error_msg
    assert "C error" not in error_msg
    assert "No se pudo procesar el archivo 'bad_structure.csv'" in error_msg


def test_mask_sensitive_paths_function():
    """Verifies mask_sensitive_paths helper redacts sensitive system directories and paths."""
    from src.loader import mask_sensitive_paths

    assert "/home/user" not in mask_sensitive_paths("Error at /home/user/project/file.py")
    assert "/var/log" not in mask_sensitive_paths("Failed to read /var/log/syslog: access denied")
    assert "C:\\Users\\admin" not in mask_sensitive_paths("Failed at C:\\Users\\admin\\secrets.txt")
    assert mask_sensitive_paths(cast(Any, 123)) == "123"


def test_loader_validation_errors_do_not_leak_absolute_paths():
    """Verifies that validation errors (empty, oversized, unsupported extension) do not leak paths."""
    # Empty file
    b_empty = io.BytesIO(b"")
    with pytest.raises(ValueError) as exc:
        load_tabular_file(b_empty, "/home/secret/empty.csv")
    assert "/home/secret" not in str(exc.value)
    assert "empty.csv" in str(exc.value)

    # Oversized file
    b_large = _create_uploaded_file(
        "huge.csv",
        b"x,y\n",
        size=MAX_UPLOAD_SIZE_BYTES + 1024,
    )
    with pytest.raises(ValueError) as exc:
        load_tabular_file(b_large, "/tmp/user/sensitive/huge.csv")
    assert "/tmp/user/sensitive" not in str(exc.value)
    assert "huge.csv" in str(exc.value)

    # Unsupported extension
    b_unsupported = io.BytesIO(b"dummy")
    with pytest.raises(ValueError) as exc:
        load_tabular_file(b_unsupported, "C:\\Users\\Admin\\Documents\\test.pdf")
    assert "C:\\Users\\Admin" not in str(exc.value)
    assert "test.pdf" not in str(exc.value) or ".pdf" in str(exc.value)
