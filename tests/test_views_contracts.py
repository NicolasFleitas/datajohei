"""Tests de arquitectura y contratos para la capa de presentación (src/views/)."""

import inspect
from pathlib import Path

import pandas as pd


def test_views_exports_and_callables():
    """Verifica que src.views exporta todas las funciones requeridas y son callables."""
    from src import views

    expected_functions = [
        "render_diagnostic_tab",
        "render_cleaning_tab",
        "render_viz_tab",
        "render_export_tab",
        "render_sidebar",
    ]

    for func_name in expected_functions:
        assert hasattr(views, func_name), f"src.views debe exportar '{func_name}'"
        func = getattr(views, func_name)
        assert callable(func), f"'{func_name}' debe ser una función ejecutable"


def test_views_submodules_exist():
    """Verifica la existencia y exports de los submódulos específicos de vistas."""
    from src.views.cleaning_view import render_cleaning_tab
    from src.views.diagnostic_view import render_diagnostic_tab
    from src.views.export_view import render_export_tab
    from src.views.sidebar_view import render_sidebar
    from src.views.viz_view import render_viz_tab

    assert callable(render_diagnostic_tab)
    assert callable(render_cleaning_tab)
    assert callable(render_viz_tab)
    assert callable(render_export_tab)
    assert callable(render_sidebar)


def test_views_signatures():
    """Verifica que las firmas de las funciones de vista reciban los parámetros mínimos esperados."""
    from src.views import (
        render_cleaning_tab,
        render_diagnostic_tab,
        render_export_tab,
        render_sidebar,
        render_viz_tab,
    )

    # render_diagnostic_tab(df)
    sig_diag = inspect.signature(render_diagnostic_tab)
    assert "df" in sig_diag.parameters

    # render_cleaning_tab(df, ...)
    sig_clean = inspect.signature(render_cleaning_tab)
    assert "df" in sig_clean.parameters

    # render_viz_tab(df)
    sig_viz = inspect.signature(render_viz_tab)
    assert "df" in sig_viz.parameters

    # render_export_tab(df, ...)
    sig_exp = inspect.signature(render_export_tab)
    assert "df" in sig_exp.parameters

    # render_sidebar(...)
    inspect.signature(render_sidebar)
    assert callable(render_sidebar)


def test_export_view_helpers():
    """Verifica las funciones utilitarias de exportación (CSV y Parquet)."""
    from src.views.export_view import get_csv_bytes, get_parquet_bytes

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    csv_bytes = get_csv_bytes(df)
    assert isinstance(csv_bytes, bytes)
    assert b"a,b" in csv_bytes

    parquet_bytes = get_parquet_bytes(df)
    assert isinstance(parquet_bytes, bytes)
    assert len(parquet_bytes) > 0


def test_app_loc_under_threshold():
    """Verifica que app.py no exceda las 100 líneas de código (arquitectura liviana)."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    assert app_path.exists(), "app.py debe existir en la raíz del proyecto"

    content = app_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    total_loc = len(lines)

    assert total_loc <= 100, f"app.py excede el límite de 100 LOC: tiene {total_loc} líneas"
