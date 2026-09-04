"""Carga de archivos tabulares (CSV, Parquet, Excel) con detección de encoding."""

import logging
import re
from pathlib import Path

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_MB = 100
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024


def mask_sensitive_paths(text: str) -> str:
    """Enmascara rutas absolutas del sistema de archivos en mensajes de error o logs."""
    if not isinstance(text, str):
        text = str(text)
    # Redact Windows paths (e.g. C:\Users\... or C:/Users/...)
    text = re.sub(
        r"[a-zA-Z]:[\\/](?:[^\\/\s:;\"',><]+[\\/])*([^\\/\s:;\"',><]+)",
        r"[PATH]/\1",
        text,
    )
    text = re.sub(r"[a-zA-Z]:[\\/]", "[PATH]/", text)
    # Redact Unix absolute paths (e.g. /home/..., /var/..., /tmp/..., /usr/..., /etc/..., /opt/...)
    text = re.sub(
        r"/(?:home|var|tmp|usr|etc|opt|root|srv|proc|sys|dev)(?:/[^\s:;\"',><]+)*/([^\s:;\"',><]+)",
        r"[PATH]/\1",
        text,
    )
    text = re.sub(
        r"/(?:home|var|tmp|usr|etc|opt|root|srv|proc|sys|dev)(?:/[^\s:;\"',><]+)*",
        r"[PATH]",
        text,
    )
    return text


def get_clean_filename(filename: str) -> str:
    """Extrae el nombre base de archivo sanitizado sin rutas del sistema."""
    if not filename:
        return "archivo"
    return Path(filename).name


def _get_file_size(file) -> int:
    """Obtiene el tamaño en bytes de un archivo o buffer sin consumirlo."""
    if hasattr(file, "size") and isinstance(file.size, int):
        return file.size
    if hasattr(file, "getbuffer"):
        try:
            return file.getbuffer().nbytes
        except Exception:
            pass
    if hasattr(file, "seek") and hasattr(file, "tell"):
        current_pos = file.tell()
        file.seek(0, 2)
        size = file.tell()
        file.seek(current_pos)
        return size
    if isinstance(file, (str, Path)):
        return Path(file).stat().st_size
    return 0


@st.cache_data(max_entries=20, ttl=3600, show_spinner=False)
def load_tabular_file(file, filename: str) -> tuple[pd.DataFrame, str]:
    """Carga un archivo tabular (CSV, Parquet, XLSX) y devuelve (DataFrame, mensaje)."""
    clean_filename = get_clean_filename(filename)
    file_size = _get_file_size(file)

    if file_size == 0 and hasattr(file, "seek"):
        raise ValueError(f"El archivo '{clean_filename}' está vacío")
    if file_size > MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"El archivo '{clean_filename}' ({file_size / (1024 * 1024):.1f} MB) "
            f"supera el límite máximo permitido de {MAX_UPLOAD_SIZE_MB} MB."
        )

    try:
        ext = clean_filename.lower().split(".")[-1]

        if ext == "csv":
            try:
                df = pd.read_csv(file)
            except pd.errors.EmptyDataError:
                raise ValueError(f"El archivo '{clean_filename}' está vacío") from None
            except UnicodeDecodeError:
                if hasattr(file, "seek"):
                    file.seek(0)
                try:
                    df = pd.read_csv(file, encoding="latin-1")
                except Exception as e:
                    logger.exception(
                        "Error al decodificar CSV %s con latin-1: %s",
                        clean_filename,
                        e,
                    )
                    raise RuntimeError(
                        f"No se pudo procesar el archivo '{clean_filename}'. "
                        "Verifique que el formato y la codificación sean válidos."
                    ) from None
            except Exception as e:
                logger.exception("Error al leer CSV %s: %s", clean_filename, e)
                raise RuntimeError(
                    f"No se pudo procesar el archivo '{clean_filename}'. "
                    "Verifique que el formato y la codificación sean válidos."
                ) from None
        elif ext in ["parquet", "pq"]:
            try:
                df = pd.read_parquet(file)
            except Exception as e:
                logger.exception("Error al leer Parquet %s: %s", clean_filename, e)
                raise RuntimeError(
                    f"No se pudo procesar el archivo '{clean_filename}'. "
                    "Verifique que el formato y la codificación sean válidos."
                ) from None
        elif ext in ["xlsx", "xls"]:
            try:
                df = pd.read_excel(file, engine="openpyxl")
            except Exception as e:
                logger.exception("Error al leer Excel %s: %s", clean_filename, e)
                raise RuntimeError(
                    f"No se pudo procesar el archivo '{clean_filename}'. "
                    "Verifique que el formato y la codificación sean válidos."
                ) from None
        else:
            raise ValueError(f"Extensión .{ext} no soportada. Usa .csv, .parquet, .xlsx")

        if df.empty:
            raise ValueError(f"El archivo '{clean_filename}' está vacío")

        return df, f"✓ Cargado {clean_filename} -> {df.shape[0]} filas, {df.shape[1]} cols"

    except ValueError as ve:
        raise ValueError(mask_sensitive_paths(str(ve))) from None
    except RuntimeError:
        raise
    except Exception as e:
        logger.exception("Error inesperado procesando %s: %s", clean_filename, e)
        raise RuntimeError(
            f"No se pudo procesar el archivo '{clean_filename}'. "
            "Verifique que el formato y la codificación sean válidos."
        ) from None
