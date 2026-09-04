"""Tests para el helper de limpieza de estado de widgets y la regresión del
crash por columna eliminada/transformada (JD-01)."""

import pytest
from streamlit.testing.v1 import AppTest

from src.widget_state import reset_stale_choice


@pytest.fixture
def fake_session_state(monkeypatch):
    """Reemplaza st.session_state por un dict plano para aislar reset_stale_choice."""
    import streamlit as st

    store: dict = {}
    monkeypatch.setattr(st, "session_state", store)
    return store


def test_reset_stale_choice_deletes_when_not_in_options(fake_session_state):
    """Valor persistido ausente de las opciones actuales debe eliminarse."""
    fake_session_state["null_col"] = "edad"
    reset_stale_choice("null_col", ["salario_mensual", "score_desempeno"])
    assert "null_col" not in fake_session_state


def test_reset_stale_choice_keeps_when_in_options(fake_session_state):
    """Valor persistido presente en las opciones debe conservarse."""
    fake_session_state["null_col"] = "edad"
    reset_stale_choice("null_col", ["edad", "salario_mensual"])
    assert fake_session_state["null_col"] == "edad"


def test_reset_stale_choice_keeps_sentinel_present_in_options(fake_session_state):
    """Sentinelas como 'Ninguno' que figuran en options no deben eliminarse."""
    fake_session_state["bar_hue"] = "Ninguno"
    reset_stale_choice("bar_hue", ["Ninguno", "edad", "salario_mensual"])
    assert fake_session_state["bar_hue"] == "Ninguno"


def test_reset_stale_choice_is_noop_when_absent(fake_session_state):
    """Clave inexistente no debe generar error ni crear entrada."""
    reset_stale_choice("null_col", ["edad"])
    assert "null_col" not in fake_session_state


def _cleaning_drop_script():
    """Render de limpieza cuyo push_state_fn aplica la transformación al df
    persistido en session_state, reproduciendo el rerun real de la app."""
    import pandas as pd
    import streamlit as st

    from src.views.cleaning_view import render_cleaning_tab

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame({"A": [1, 2, 3], "edad": [1.0, None, 3.0]})

    def push(new_df, _cmd):
        st.session_state.df = new_df

    render_cleaning_tab(st.session_state.df, push_state_fn=push)


def test_drop_col_no_crash_on_rerun():
    """Reproduce JD-01: aplicar drop_col sobre una columna con nulos y verificar
    que el rerun no lanza StreamlitAPIException (la columna desaparece de las
    opciones del selectbox)."""
    at = AppTest.from_function(_cleaning_drop_script).run()
    at.selectbox(key="null_col").set_value("edad")
    at.selectbox(key="null_strat").set_value("drop_col")
    at.button(key="apply_null").click().run()
    assert len(at.exception) == 0, f"Excepción tras drop_col: {at.exception}"
    assert "edad" not in at.session_state["df"].columns
    assert "null_col" not in at.session_state


def _cleaning_fill_script():
    """Igual que _cleaning_drop_script pero la imputación 'media' deja la
    columna en el df (ya sin nulos), por lo que sale de cols_with_nulls."""
    import pandas as pd
    import streamlit as st

    from src.views.cleaning_view import render_cleaning_tab

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame({"A": [1, 2, 3], "edad": [1.0, None, 3.0]})

    def push(new_df, _cmd):
        st.session_state.df = new_df

    render_cleaning_tab(st.session_state.df, push_state_fn=push)


def test_fill_col_no_crash_when_leaving_cols_with_nulls():
    """Regresión: al rellenar nulos la columna sigue existiendo pero sale de
    cols_with_nulls; el valor persistido de null_col debe limpiarse sin crashear."""
    at = AppTest.from_function(_cleaning_fill_script).run()
    at.selectbox(key="null_col").set_value("edad")
    at.selectbox(key="null_strat").set_value("media")
    at.button(key="apply_null").click().run()
    assert len(at.exception) == 0, f"Excepción tras imputación media: {at.exception}"
    assert "edad" in at.session_state["df"].columns  # la columna NO se eliminó
    assert "null_col" not in at.session_state  # pero el valor obsoleto sí
