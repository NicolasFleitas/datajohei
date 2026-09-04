"""Utilidades para mantener consistente el estado de widgets keyed de Streamlit."""

import streamlit as st


def reset_stale_choice(key: str, options) -> None:
    """Si el valor persistido de un widget keyed ya no está entre las opciones
    actuales (columna eliminada, transformada o sin nulos), lo limpia para
    evitar StreamlitAPIException al re-render. Los sentinelas tipo 'Ninguno'
    deben figurar en 'options' para no ser eliminados.

    Args:
        key: Clave del widget en st.session_state (p. ej. "null_col").
        options: Iterables de opciones válidas actuales del selectbox.
    """
    current = st.session_state.get(key)
    if current is not None and current not in set(options):
        del st.session_state[key]
