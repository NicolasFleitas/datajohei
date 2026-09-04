"""Regression tests: sidebar must reject unknown/missing file extensions gracefully."""

import contextlib

import pandas as pd
import pytest
import streamlit as st

from src.views import sidebar_view


class _AttrDict(dict):
    """Dict stub mimicking Streamlit session_state attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None


class _FakeUploaded:
    """Minimal stand-in for Streamlit's UploadedFile."""

    def __init__(self, name: str):
        self.name = name


@contextlib.contextmanager
def _dummy_sidebar():
    yield


def _install_sidebar_harness(monkeypatch, uploaded_name, df_loaded):
    """Stub Streamlit primitives so render_sidebar runs headless.

    Returns (session_store, error_messages).
    """
    store: dict = _AttrDict()
    errors: list = []

    def fake_button(label, *args, **kwargs):
        # Only confirm the "use uploaded file" action; every other button is off.
        return "Usar archivo cargado" in str(label)

    def fake_columns(n):
        @contextlib.contextmanager
        def _col():
            yield

        return [_col(), _col()][:n]

    monkeypatch.setattr(st, "sidebar", _dummy_sidebar())
    monkeypatch.setattr(st, "session_state", store)
    monkeypatch.setattr(st, "title", lambda *a, **k: None)
    monkeypatch.setattr(st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(st, "divider", lambda *a, **k: None)
    monkeypatch.setattr(st, "text", lambda *a, **k: None)
    monkeypatch.setattr(st, "info", lambda *a, **k: None)
    monkeypatch.setattr(st, "rerun", lambda *a, **k: None)
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: _FakeUploaded(uploaded_name))
    monkeypatch.setattr(st, "button", fake_button)
    monkeypatch.setattr(st, "columns", fake_columns)
    monkeypatch.setattr(st, "error", errors.append)
    monkeypatch.setattr(
        sidebar_view,
        "load_tabular_file",
        lambda _file, _name: (df_loaded, "loaded"),
    )
    return store, errors


@pytest.mark.parametrize("bad_name", ["report.txt", "datafile", "/tmp/evil/datafile"])
def test_sidebar_unknown_or_missing_extension_shows_error(monkeypatch, bad_name):
    """Unknown/missing extension must show a friendly error and skip PipelineManager."""
    df = pd.DataFrame({"a": [1, 2]})
    store, errors = _install_sidebar_harness(monkeypatch, bad_name, df)

    sidebar_view.render_sidebar()

    assert "pipeline" not in store
    assert len(errors) == 1
    msg = str(errors[0])
    assert ".csv" in msg and ".parquet" in msg and ".xlsx" in msg
    # No internal paths, tracebacks, or KeyError details may leak to the user.
    assert "/tmp" not in msg and "/home" not in msg
    assert "Traceback" not in msg and "KeyError" not in msg


def test_sidebar_known_extension_still_builds_pipeline(monkeypatch):
    """Valid .csv upload must keep building the PipelineManager (no error)."""
    df = pd.DataFrame({"a": [1, 2]})
    store, errors = _install_sidebar_harness(monkeypatch, "dataset.csv", df)

    sidebar_view.render_sidebar()

    assert "pipeline" in store
    assert errors == []
    assert store["file_name"] == "dataset.csv"
