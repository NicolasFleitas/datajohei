import logging
from collections.abc import Callable

import streamlit as st

from src.loader import get_clean_filename, load_tabular_file, mask_sensitive_paths
from src.pipeline import PipelineManager
from src.synthetic import generate_messy_dataset

logger = logging.getLogger(__name__)

_EXT_TO_SOURCE_KIND = {"csv": "csv", "parquet": "parquet", "xlsx": "xlsx"}


def render_sidebar(
    on_undo: Callable | None = None,
    on_reset: Callable | None = None,
) -> None:
    """Renderiza los controles de la barra lateral."""
    with st.sidebar:
        st.title("🧹 DataJohei")
        st.caption("Pandas · Seaborn · Streamlit")

        uploaded = st.file_uploader(
            "Cargar dataset",
            type=["csv", "parquet", "xlsx"],
            help="Soporta .csv, .parquet y .xlsx (máx. 100 MB)",
        )

        if uploaded:
            try:
                df_loaded, msg = load_tabular_file(uploaded, uploaded.name)
                if st.button("Usar archivo cargado", type="primary"):
                    # Sanitize to basename so no internal path ever leaks downstream.
                    clean_name = get_clean_filename(uploaded.name)
                    ext = clean_name.lower().rsplit(".", 1)[-1] if "." in clean_name else ""
                    source_kind = _EXT_TO_SOURCE_KIND.get(ext)
                    if source_kind is None:
                        # Unknown or missing extension: friendly error, no PipelineManager.
                        st.error("Tipo de archivo no soportado. Usa .csv, .parquet o .xlsx.")
                    else:
                        st.session_state.pipeline = PipelineManager(
                            df_loaded,
                            source_kind=source_kind,
                            source_name=clean_name,
                        )
                        st.session_state.df_original = df_loaded.copy()
                        st.session_state.df_current = df_loaded.copy()
                        st.session_state.file_name = clean_name
                        st.rerun()
                st.info(msg)
            except Exception as e:
                logger.exception("Error al cargar archivo en sidebar: %s", e)
                st.error(mask_sensitive_paths(str(e)))

        if st.button("🔄 Usar dataset sintético sucio"):
            df_syn = generate_messy_dataset()
            st.session_state.pipeline = PipelineManager(df_syn)
            st.session_state.df_original = df_syn.copy()
            st.session_state.df_current = df_syn.copy()
            st.session_state.file_name = "dataset_sintetico.csv"
            st.rerun()

        st.divider()
        st.subheader("📜 Historial de Transformaciones")
        transform_log = (
            st.session_state.pipeline.transform_log if "pipeline" in st.session_state else []
        )

        if transform_log:
            for i, log in enumerate(reversed(transform_log[-10:]), 1):
                st.text(f"{len(transform_log) - i + 1}. {log}")
        else:
            st.caption("Sin transformaciones aún")

        col_undo, col_reset = st.columns(2)
        with col_undo:
            st.button("↩️ Undo", on_click=on_undo, width="stretch")
        with col_reset:
            if st.button("Reset Total", width="stretch"):
                if on_reset:
                    on_reset()
                else:
                    if "pipeline" in st.session_state:
                        st.session_state.pipeline.reset()
                        st.session_state.df_current = st.session_state.pipeline.df_current
                    st.rerun()

        st.divider()
        df_current = (
            st.session_state.pipeline.df_current
            if "pipeline" in st.session_state
            else st.session_state.get("df_current", None)
        )
        if df_current is not None:
            st.caption(f"Dataset actual: {df_current.shape[0]} filas × {df_current.shape[1]} cols")
