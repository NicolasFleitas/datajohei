"""Módulo de presentación para la pestaña de diagnóstico."""

import pandas as pd
import streamlit as st

from src.health import compute_health_metrics, missing_summary


def render_diagnostic_tab(df: pd.DataFrame) -> None:
    """Renderiza el panel de diagnóstico de salud del dataset."""
    st.header("Diagnóstico de Salud del Dataset")
    metrics = compute_health_metrics(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas / Columnas", f"{metrics['shape'][0]} / {metrics['shape'][1]}")
    c2.metric("% Nulos Global", f"{metrics['pct_missing_global']:.2f}%")
    c3.metric("Celdas Nulas", f"{metrics['total_missing']:,}")
    c4.metric("Filas Duplicadas", f"{metrics['dup_count']:,}", delta_color="inverse")

    st.divider()

    st.subheader("Tipos y Calidad por Columna")
    st.dataframe(
        metrics["dtypes_df"].style.background_gradient(subset=["% nulos"], cmap="Reds"),
        width="stretch",
        height=350,
    )

    st.divider()

    with st.expander("Vista previa - 5 primeras filas", expanded=True):
        st.dataframe(df.head(), width="stretch")

    st.subheader("Valores Faltantes")
    summary = missing_summary(df)
    total_rows = df.shape[0]
    pct_complete_rows = (float(df.notna().all(axis=1).mean()) * 100) if total_rows else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Valores Faltantes", f"{metrics['total_missing']:,}")
    m2.metric("% Filas Completas", f"{pct_complete_rows:.1f}%")
    m3.metric("Columnas Afectadas", len(summary))

    if summary.empty:
        st.success("Sin valores faltantes: todas las columnas están completas.")
    else:
        st.caption("Columnas con al menos un valor faltante, ordenadas de mayor a menor.")
        st.dataframe(
            summary.style.background_gradient(subset=["% nulos"], cmap="Reds"),
            width="stretch",
            height=min(350, 38 * (len(summary) + 1)),
        )
