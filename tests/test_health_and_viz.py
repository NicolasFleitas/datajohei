import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src import health
from src.health import compute_health_metrics, missing_summary
from src.synthetic import generate_messy_dataset
from src.views.export_view import get_csv_bytes, get_parquet_bytes
from src.viz import (
    plot_bivariate_bar,
    plot_correlation_heatmap,
    plot_scatter,
    plot_univariate_categorical,
    plot_univariate_numeric,
    plot_univariate_temporal,
    resolve_topn_state,
    topn_slider_config,
)


def test_compute_health_metrics():
    df = generate_messy_dataset()
    metrics = compute_health_metrics(df)

    assert metrics["shape"] == df.shape
    assert metrics["total_missing"] == int(df.isna().sum().sum())
    assert metrics["dup_count"] == int(df.duplicated().sum())
    assert len(metrics["dtypes_df"]) == df.shape[1]
    assert "ejemplo" in metrics["dtypes_df"].columns


def test_compute_health_metrics_all_nan_col():
    df = pd.DataFrame(
        {
            "a": [np.nan, np.nan, np.nan],
            "b": [1, 2, 3],
        }
    )
    metrics = compute_health_metrics(df)
    assert metrics["shape"] == (3, 2)
    assert metrics["total_missing"] == 3
    row_a = metrics["dtypes_df"][metrics["dtypes_df"]["columna"] == "a"].iloc[0]
    assert row_a["nulos"] == 3


@pytest.mark.parametrize("n_cats", [0, 1, 2, 3, 4, 5, 7, 10, 29, 30, 120])
def test_topn_slider_config_bounds_are_always_valid(n_cats):
    """La configuración del slider Top N debe cumplir min < max y min <= value <= max."""
    topn_min, topn_max, default = topn_slider_config(n_cats)
    assert topn_min < topn_max
    assert topn_min <= default <= topn_max
    assert topn_min >= 3


def test_topn_slider_config_low_cardinality_regression():
    """JD-01: columnas constantes o todo-NaN (0/1/2 categorías) producen un slider válido."""
    assert topn_slider_config(0) == (3, 4, 4)
    assert topn_slider_config(1) == (3, 4, 4)
    assert topn_slider_config(2) == (3, 4, 4)


def test_topn_slider_config_preserves_healthy_behavior():
    """Columnas con cardinalidad sana mantienen el comportamiento previo al fix."""
    assert topn_slider_config(30) == (3, 30, 10)
    assert topn_slider_config(100) == (3, 30, 10)
    assert topn_slider_config(12) == (3, 12, 10)
    assert topn_slider_config(7) == (3, 7, 7)
    assert topn_slider_config(5) == (3, 5, 5)


def test_resolve_topn_state_purges_stale_out_of_bounds_state():
    """R-JD01-a: estado persistido (25) fuera de los límites de una columna de 4
    categorías (max=7) se detecta como stale y ordena la purga, con valor default."""
    value, purge = resolve_topn_state(25, 3, 7)
    assert value is None
    assert purge is True


def test_resolve_topn_state_clean_state_passthrough():
    """R-JD01-a: estado persistido dentro de límites pasa intacto y sin purga."""
    assert resolve_topn_state(5, 3, 7) == (5, False)


def test_resolve_topn_state_absent_key_is_noop():
    """R-JD01-a: sin estado previo no hay purga ni valor efectivo."""
    assert resolve_topn_state(None, 3, 7) == (None, False)


@pytest.mark.parametrize("boundary", [3, 7])
def test_resolve_topn_state_bounds_are_inclusive(boundary):
    """Los límites del slider son inclusivos: un valor en el borde es estado sano."""
    assert resolve_topn_state(boundary, 3, 7) == (boundary, False)


def test_plot_univariate_numeric_large_data():
    df = pd.DataFrame({"nums": np.random.randn(60_000)})
    fig, _n_out = plot_univariate_numeric(df, "nums")
    assert fig is not None
    plt.close(fig)


def test_plot_scatter_large_data_sampling():
    df = pd.DataFrame(
        {
            "x": np.random.randn(15_000),
            "y": np.random.randn(15_000),
        }
    )
    fig = plot_scatter(df, "x", "y")
    assert fig is not None
    plt.close(fig)


def test_plot_correlation_heatmap():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "c": [2.0, 4.0, 6.0, 8.0],
        }
    )
    fig = plot_correlation_heatmap(df)
    assert fig is not None
    plt.close(fig)


def test_plot_correlation_heatmap_single_col_none():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    fig = plot_correlation_heatmap(df)
    assert fig is None


def test_export_helpers_csv_parquet():
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    csv_b = get_csv_bytes(df)
    assert isinstance(csv_b, bytes)
    assert b"col1,col2" in csv_b

    parquet_b = get_parquet_bytes(df)
    assert isinstance(parquet_b, bytes)
    assert len(parquet_b) > 0


def test_health_module_has_no_streamlit_dependency():
    """Valida mediante introspección que streamlit no es importado por src.health."""
    source = inspect.getsource(health)
    assert "import streamlit" not in source
    assert "from streamlit" not in source
    assert "streamlit" not in health.__dict__
    assert "st" not in health.__dict__


def test_missing_summary_orders_and_filters_columns_with_nulls():
    """Devuelve [columna, nulos, % nulos] descendente y excluye columnas completas."""
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, np.nan, 4.0],
            "b": [np.nan, 2.0, 3.0, 4.0],
            "c": [1, 2, 3, 4],
        }
    )
    summary = missing_summary(df)
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == ["columna", "nulos", "% nulos"]
    assert list(summary["columna"]) == ["a", "b"]
    assert list(summary["nulos"]) == [2, 1]
    assert list(summary["% nulos"]) == [50.0, 25.0]


def test_missing_summary_without_nulls_returns_empty():
    """Cero nulos en todo el dataset -> resultado vacío (no se requiere muestreo)."""
    df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
    summary = missing_summary(df)
    assert summary.empty
    assert list(summary.columns) == ["columna", "nulos", "% nulos"]


def test_missing_summary_all_null_column_is_100_pct():
    """Columna totalmente nula -> 100% de faltantes y primer lugar del ranking."""
    df = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, np.nan]})
    summary = missing_summary(df)
    assert list(summary["columna"]) == ["a", "b"]
    row_a = summary.iloc[0]
    assert row_a["nulos"] == 3
    assert row_a["% nulos"] == 100.0


def test_dark_mode_plots_render_successfully():
    """Valida que todas las funciones de visualización renderizan correctamente con dark_mode=True."""
    df = pd.DataFrame(
        {
            "nums": np.random.randn(100),
            "nums2": np.random.randn(100),
            "cats": ["A", "B", "C", "D"] * 25,
            "dates": pd.date_range("2024-01-01", periods=100),
        }
    )

    fig1, n_out = plot_univariate_numeric(df, "nums", dark_mode=True)
    assert isinstance(fig1, Figure)
    assert isinstance(n_out, int)
    plt.close(fig1)

    fig2 = plot_univariate_temporal(df, "dates", dark_mode=True)
    assert isinstance(fig2, Figure)
    plt.close(fig2)

    fig3 = plot_univariate_categorical(df, "cats", dark_mode=True)
    assert isinstance(fig3, Figure)
    plt.close(fig3)

    fig4 = plot_correlation_heatmap(df, dark_mode=True)
    assert isinstance(fig4, Figure)
    plt.close(fig4)

    fig5 = plot_scatter(df, "nums", "nums2", hue="cats", dark_mode=True)
    assert isinstance(fig5, Figure)
    plt.close(fig5)

    fig6 = plot_bivariate_bar(df, "cats", "nums", hue_col="Ninguno", dark_mode=True)
    assert isinstance(fig6, Figure)
    plt.close(fig6)
