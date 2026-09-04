"""Módulo de presentación para la pestaña de visualización y exploración dinámica."""

import logging
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from src.theme import is_dark_theme
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
from src.widget_state import reset_stale_choice

logger = logging.getLogger(__name__)


def _close_fig(fig: Figure) -> None:
    """Renderiza la figura en Streamlit y libera la memoria de matplotlib."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def render_viz_tab(df: pd.DataFrame) -> None:
    """Renderiza la pestaña de visualización univariada, bivariada y multivariada."""
    st.header("Exploración y Visualización Dinámica")
    dark_mode = is_dark_theme()

    viz_tab1, viz_tab2 = st.tabs(
        ["Univariado", "Bivariado / Multivariado"],
        on_change="rerun",
        key="viz_tabs",
    )

    with viz_tab1:
        reset_stale_choice("uni_col", df.columns)
        col_sel = str(st.selectbox("Selecciona columna", df.columns, key="uni_col"))
        ser = cast(pd.Series, df[col_sel])

        if pd.api.types.is_datetime64_any_dtype(ser):
            fig = plot_univariate_temporal(df, col_sel, dark_mode=dark_mode)
            _close_fig(fig)
        else:
            is_numeric = pd.api.types.is_numeric_dtype(ser) and ser.nunique() > 10
            if is_numeric:
                fig, n_out = plot_univariate_numeric(df, col_sel, dark_mode=dark_mode)
                _close_fig(fig)
                if n_out > 0:
                    st.caption(f"{n_out} valores fuera del rango P99.9")
                st.write(ser.describe().to_frame().T)
            else:
                fig = plot_univariate_categorical(df, col_sel, dark_mode=dark_mode)
                _close_fig(fig)
                st.write(ser.value_counts().head(20))

    with viz_tab2:
        st.subheader("📊 Gráfico de Barras — Relación entre Variables")
        with st.container(border=True):
            categorical_cols = df.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            if not categorical_cols:
                st.info(
                    "No hay columnas categóricas para el eje X. Casteá alguna columna a 'category' en Limpieza."
                )
            elif not numeric_cols:
                st.info(
                    "No hay columnas numéricas para el eje Y. Casteá alguna columna a 'numeric' en Limpieza."
                )
            else:
                bar_col1, bar_col2, bar_col3 = st.columns(3)
                with bar_col1:
                    reset_stale_choice("bar_cat", categorical_cols)
                    bar_cat = str(
                        st.selectbox(
                            "Eje X (categórica)",
                            categorical_cols,
                            key="bar_cat",
                            help="Columna categórica para agrupar",
                        )
                    )
                with bar_col2:
                    reset_stale_choice("bar_num", numeric_cols)
                    bar_num = str(
                        st.selectbox(
                            "Eje Y (numérica)",
                            numeric_cols,
                            key="bar_num",
                            help="Columna numérica a agregar",
                        )
                    )
                with bar_col3:
                    bar_agg = str(
                        st.selectbox(
                            "Agregación",
                            ["mean", "sum", "median", "count", "min", "max", "std"],
                            key="bar_agg",
                            help="Función de agregación aplicada a la columna numérica",
                        )
                    )

                bar_col4, bar_col5 = st.columns(2)
                with bar_col4:
                    all_cols_plus_none = ["Ninguno", *df.columns.tolist()]
                    reset_stale_choice("bar_hue", all_cols_plus_none)
                    bar_hue = str(
                        st.selectbox(
                            "Agrupar por color (hue)",
                            all_cols_plus_none,
                            key="bar_hue",
                            help="Columna categórica para segmentar las barras",
                        )
                    )
                with bar_col5:
                    cat_ser = cast(pd.Series, df[bar_cat])
                    topn_min, topn_max, topn_default = topn_slider_config(
                        int(cat_ser.nunique(dropna=True))
                    )
                    # Decide clamp/purga del estado persistido que quedó fuera
                    # de los nuevos límites al cambiar de columna (evita
                    # StreamlitAPIException); delega en el helper puro de src.viz.
                    effective_topn, purge_topn = resolve_topn_state(
                        st.session_state.get("bar_topn"), topn_min, topn_max
                    )
                    if purge_topn:
                        del st.session_state["bar_topn"]
                    bar_topn = int(
                        st.slider(
                            "Mostrar Top N categorías",
                            min_value=topn_min,
                            max_value=topn_max,
                            value=effective_topn if effective_topn is not None else topn_default,
                            key="bar_topn",
                            help="Limita la cantidad de categorías visibles",
                        )
                    )

                try:
                    fig_bar = plot_bivariate_bar(
                        df,
                        cat_col=bar_cat,
                        num_col=bar_num,
                        aggfunc=bar_agg,
                        hue_col=bar_hue,
                        top_n=bar_topn,
                        dark_mode=dark_mode,
                    )
                    _close_fig(fig_bar)
                except ValueError as e:
                    logger.exception("Error al renderizar gráfico de barras: %s", e)
                    st.warning(str(e))

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Mapa de Calor Correlación")
            corr_method = str(
                st.selectbox("Método", ["pearson", "spearman", "kendall"], key="corr_method")
            )
            fig_corr = plot_correlation_heatmap(df, corr_method, dark_mode=dark_mode)
            if fig_corr:
                _close_fig(fig_corr)
            else:
                st.warning("Se necesitan al menos 2 columnas numéricas")

        with col2:
            st.subheader("Scatterplot Dinámico")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            all_cols = df.columns.tolist()
            if len(numeric_cols) >= 2:
                reset_stale_choice("scatter_x", numeric_cols)
                x_axis = str(st.selectbox("Eje X", numeric_cols, key="scatter_x"))
                reset_stale_choice("scatter_y", numeric_cols)
                y_axis = str(
                    st.selectbox(
                        "Eje Y",
                        numeric_cols,
                        index=1 if len(numeric_cols) > 1 else 0,
                        key="scatter_y",
                    )
                )
                reset_stale_choice("scatter_hue", ["Ninguno", *all_cols])
                hue_col = str(
                    st.selectbox("Color (hue)", ["Ninguno", *all_cols], key="scatter_hue")
                )
                try:
                    fig_sc = plot_scatter(df, x_axis, y_axis, hue_col, dark_mode=dark_mode)
                    _close_fig(fig_sc)
                except ValueError as e:
                    logger.exception("Error al renderizar scatterplot: %s", e)
                    st.warning(str(e))
            else:
                st.info("Se necesitan al menos 2 columnas numéricas para scatterplot")
