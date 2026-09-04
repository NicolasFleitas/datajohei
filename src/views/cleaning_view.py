"""Módulo de presentación para la pestaña de limpieza interactiva."""

import contextlib
import logging
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import streamlit as st

from src.cleaning import (
    CastCommand,
    DropDuplicatesCommand,
    ImputeCommand,
    OutlierFilterCommand,
    detect_outliers_iqr,
    detect_outliers_zscore,
    strategies_for_dtype,
)
from src.loader import mask_sensitive_paths
from src.widget_state import reset_stale_choice

logger = logging.getLogger(__name__)


def _format_outlier_pct(n_out: int, n_rows: int) -> str:
    """Porcentaje de outliers sobre el total de filas; 'N/A' si no hay filas."""
    return "N/A" if n_rows == 0 else f"{n_out / n_rows * 100:.1f}%"


def render_cleaning_tab(df: pd.DataFrame, push_state_fn: Callable | None = None) -> None:
    """Renderiza el pipeline interactivo de limpieza con sub-pestañas."""
    st.header("Pipeline Interactivo de Limpieza")

    tab_imputa, tab_tipos, tab_dup, tab_out = st.tabs(
        ["🧮 Imputación", "🔄 Tipos", "📋 Duplicados", "📊 Outliers"],
        on_change="rerun",
        key="clean_tabs",
    )

    with tab_imputa, st.container(border=True):
        st.subheader("Valores Nulos")
        cols_with_nulls = df.columns[df.isna().any()].tolist()
        if not cols_with_nulls:
            st.success("¡Sin nulos!")
        else:
            reset_stale_choice("null_col", cols_with_nulls)
            target_col = str(st.selectbox("Columna con nulos", cols_with_nulls, key="null_col"))
            strategy = str(
                st.selectbox(
                    "Estrategia",
                    strategies_for_dtype(df[target_col].dtype),
                    key="null_strat",
                )
            )
            const_val = None
            if strategy == "constante":
                const_val = st.text_input("Valor constante", value="0")
                with contextlib.suppress(ValueError, TypeError):
                    const_val = float(const_val) if "." in const_val else int(const_val)

            if st.button("Aplicar imputación", key="apply_null"):
                try:
                    cmd = ImputeCommand(target_col, strategy, constant=const_val)
                    new_df, _ = cmd.apply(df)
                    if push_state_fn:
                        push_state_fn(new_df, cmd)
                    st.success(f"Aplicado {strategy} en {target_col}")
                    st.rerun()
                except Exception as e:
                    logger.exception("Error al aplicar imputación: %s", e)
                    st.error(f"Error: {mask_sensitive_paths(str(e))}")

    with tab_tipos, st.container(border=True):
        st.subheader("Tipos de Datos")
        reset_stale_choice("cast_col", df.columns)
        col_to_cast = str(st.selectbox("Columna a convertir", df.columns, key="cast_col"))
        st.caption(f"Tipo actual: {df[col_to_cast].dtype}")
        target_type = str(
            st.selectbox(
                "Convertir a",
                ["numeric", "datetime", "string", "category", "int", "float", "bool"],
                key="cast_type",
            )
        )
        if st.button("Castear", key="apply_cast"):
            try:
                cmd = CastCommand(col_to_cast, target_type)
                new_df, cast_report = cmd.apply(df)
                if push_state_fn:
                    push_state_fn(new_df, cmd)
                if cast_report["fractional"]:
                    st.warning(
                        f"{cast_report['fractional']} valores fraccionarios redondeados en {col_to_cast}"
                    )
                if cast_report["uninterpretable"]:
                    st.warning(
                        f"{cast_report['uninterpretable']} valores no interpretables -> pd.NA en {col_to_cast}"
                    )
                st.success("Conversión aplicada")
                st.rerun()
            except Exception as e:
                logger.exception("Error al convertir tipo de datos: %s", e)
                st.error(mask_sensitive_paths(str(e)))

    with tab_dup, st.container(border=True):
        st.subheader("Duplicados")

        n_dup = int(df.duplicated().sum())
        st.write(f"Filas duplicadas encontradas: **{n_dup}**")

        if n_dup:
            with st.expander("Ver registros duplicados"):
                dups_head = df[df.duplicated(keep=False)].head(50)
                st.dataframe(dups_head, width="stretch")

        if st.button("Eliminar duplicados", key="dedup", disabled=(n_dup == 0)):
            cmd = DropDuplicatesCommand()
            new_df, report = cmd.apply(df)
            removed = report["removed"]
            if push_state_fn:
                push_state_fn(new_df, cmd)
            st.success(f"Eliminados {removed} duplicados ({df.shape[0]} → {new_df.shape[0]} filas)")
            st.rerun()

    with tab_out, st.container(border=True):
        st.subheader("Outliers")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if df.shape[0] == 0:
            st.warning("El dataset está vacío: no hay filas para analizar.")
        elif not numeric_cols:
            st.warning("No hay columnas numéricas")
        else:
            reset_stale_choice("out_col", numeric_cols)
            out_col = str(st.selectbox("Columna numérica", numeric_cols, key="out_col"))
            method = str(st.radio("Método", ["IQR", "Z-Score"], horizontal=True))
            if method == "IQR":
                thresh_val = float(st.slider("Factor IQR", 0.5, 3.0, 1.5, 0.1))
                full_mask = detect_outliers_iqr(cast(pd.Series, df[out_col]), factor=thresh_val)
            else:
                thresh_val = float(st.slider("Umbral Z-Score", 1.0, 5.0, 3.0, 0.1))
                full_mask = detect_outliers_zscore(
                    cast(pd.Series, df[out_col]), threshold=thresh_val
                )
            n_out = int(full_mask.sum())

            st.metric("Outliers detectados", n_out, _format_outlier_pct(n_out, df.shape[0]))
            if n_out == 0:
                st.warning(
                    "No se detectaron outliers: columna constante, sin valores válidos o sin extremos"
                )

            if st.button(f"Filtrar {n_out} outliers", key="filter_out", disabled=(n_out == 0)):
                cmd = OutlierFilterCommand(
                    out_col,
                    method=method,
                    factor_or_thresh=thresh_val,
                )
                new_df, report = cmd.apply(df)
                if push_state_fn:
                    push_state_fn(new_df, cmd)
                st.success(f"Filtrados {n_out} outliers")
                st.rerun()
