"""Módulo de presentación para la pestaña de exportación y generación de código."""

import io
import json
import logging
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _sanitize_cell_value(val: Any) -> Any:
    """Escapes formula injection trigger characters in string cells."""
    if isinstance(val, str) and len(val) > 0 and val.startswith(FORMULA_PREFIXES):
        return f"'{val}"
    return val


def sanitize_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a sanitized copy of DataFrame with escaped formula prefixes in text fields and headers."""
    if df.empty:
        return df.copy()

    df_clean = df.copy()
    for col in df_clean.columns:
        series = df_clean[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            df_clean[col] = series.astype(object).apply(_sanitize_cell_value)
        elif pd.api.types.is_string_dtype(series) or series.dtype == object:
            df_clean[col] = series.apply(_sanitize_cell_value)

    df_clean.columns = [
        _sanitize_cell_value(c) if isinstance(c, str) else c for c in df_clean.columns
    ]
    return df_clean


@st.cache_data(max_entries=20, ttl=3600)
def get_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serializa el DataFrame a CSV en bytes con caché, sanitizando fórmulas."""
    sanitized_df = sanitize_dataframe_for_csv(df)
    return sanitized_df.to_csv(index=False).encode("utf-8")


@st.cache_data(max_entries=20, ttl=3600)
def get_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serializa el DataFrame a Parquet en bytes con caché."""
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    return buffer.getvalue()


def render_export_tab(df: pd.DataFrame, pipeline) -> None:
    """Renderiza la pestaña de exportación del dataset y código reproducible."""
    script = pipeline.to_script()
    ipynb_json = pipeline.to_notebook()
    transform_count = len(pipeline.commands)

    st.header("Exportación y Código Generado")

    col_exp1, col_exp2 = st.columns([1, 2])

    with col_exp1:
        st.subheader("Descarga Dataset Limpio")
        csv_data = get_csv_bytes(df)
        st.download_button(
            "📥 Descargar CSV",
            data=csv_data,
            file_name="dataset_limpio.csv",
            mime="text/csv",
            width="stretch",
        )
        try:
            parquet_data = get_parquet_bytes(df)
            st.download_button(
                "📥 Descargar Parquet",
                data=parquet_data,
                file_name="dataset_limpio.parquet",
                mime="application/octet-stream",
                width="stretch",
            )
        except Exception as e:
            logger.exception("Error al exportar a Parquet: %s", e)
            st.caption(f"Parquet requiere pyarrow: {e}")

        st.divider()
        st.subheader("Resumen Final")
        st.metric("Filas finales", df.shape[0])
        st.metric("Columnas finales", df.shape[1])
        st.metric("Transformaciones", transform_count)

    with col_exp2:
        st.subheader("🐍 Código Python Auto-generado")
        st.code(script, language="python", line_numbers=True)

        st.download_button(
            "📥 Descargar .py",
            data=script,
            file_name="pipeline_limpieza.py",
            mime="text/x-python",
            width="stretch",
        )

        ipynb_str = json.dumps(ipynb_json)
        st.download_button(
            "📥 Descargar Notebook .ipynb",
            data=ipynb_str,
            file_name="pipeline_limpieza.ipynb",
            mime="application/json",
            width="stretch",
        )

    st.divider()
    with st.expander("Ver DataFrame final"):
        if df.shape[0] > 20_000:
            st.caption(
                f"Dataset grande ({df.shape[0]:,} filas): se muestra una muestra de 2,000 filas"
            )
            st.dataframe(df.head(2000), width="stretch")
        else:
            st.dataframe(df, width="stretch")
