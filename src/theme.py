"""Utilidades compartidas de detección de tema para las vistas de la aplicación."""

import streamlit as st


def is_dark_theme() -> bool:
    """Detecta si el tema activo en Streamlit es dark."""
    try:
        if hasattr(st, "context") and hasattr(st.context, "theme"):
            return getattr(st.context.theme, "type", "dark") == "dark"
    except Exception:
        pass
    return False
