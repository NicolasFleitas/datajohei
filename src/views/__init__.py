"""Módulo de vistas modularizadas para Streamlit."""

from src.views.cleaning_view import render_cleaning_tab
from src.views.diagnostic_view import render_diagnostic_tab
from src.views.export_view import render_export_tab
from src.views.sidebar_view import render_sidebar
from src.views.viz_view import render_viz_tab

__all__ = [
    "render_cleaning_tab",
    "render_diagnostic_tab",
    "render_export_tab",
    "render_sidebar",
    "render_viz_tab",
]
