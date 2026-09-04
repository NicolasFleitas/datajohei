"""Visualizador interactivo de distribuciones y pipeline de limpieza — Orquestador."""

import logging

import pandas as pd
import streamlit as st

from src.cleaning import TransformationCommand
from src.pipeline import PipelineManager
from src.synthetic import generate_messy_dataset
from src.views import (
    render_cleaning_tab,
    render_diagnostic_tab,
    render_export_tab,
    render_sidebar,
    render_viz_tab,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)


def init_state():
    if "pipeline" not in st.session_state:
        df = generate_messy_dataset()
        st.session_state.pipeline = PipelineManager(df)
        st.session_state.df_original = df.copy()
        st.session_state.df_current = df.copy()
        st.session_state.file_name = "dataset_sintetico.csv"


def push_state(new_df: pd.DataFrame, command: TransformationCommand):
    st.session_state.pipeline.push(command, new_df)
    st.session_state.df_current = st.session_state.pipeline.df_current


def undo_last():
    if st.session_state.pipeline.can_undo:
        st.session_state.pipeline.undo()
        st.session_state.df_current = st.session_state.pipeline.df_current
        st.toast("✓ Último cambio revertido", icon="↩️")
    else:
        st.warning("No hay cambios para revertir")


def reset_state():
    st.session_state.pipeline.reset()
    st.session_state.df_current = st.session_state.pipeline.df_current
    st.rerun()


def main():
    st.set_page_config(
        page_title="DataJohei - Limpieza",
        page_icon="🧹",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    render_sidebar(on_undo=undo_last, on_reset=reset_state)

    t1, t2, t3, t4 = st.tabs(
        ["📊 Diagnóstico", "🧼 Limpieza", "📈 Visualización", "💾 Exportar & Código"],
        on_change="rerun",
        key="main_tabs",
    )
    df = st.session_state.df_current
    if t1.open:
        with t1:
            render_diagnostic_tab(df)
    if t2.open:
        with t2:
            render_cleaning_tab(df, push_state_fn=push_state)
    if t3.open:
        with t3:
            render_viz_tab(df)
    if t4.open:
        with t4:
            render_export_tab(df, pipeline=st.session_state.pipeline)


if __name__ == "__main__":
    main()
