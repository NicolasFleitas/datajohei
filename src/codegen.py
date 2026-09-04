"""Generador dinámico de scripts Pandas a partir de las transformaciones aplicadas."""

import datetime
import inspect
import sys
from pathlib import Path
from typing import Any

from src.synthetic import generate_messy_dataset

HEADER = """# Código auto-generado por DataJohei
# {timestamp}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

"""

_FILE_READERS: dict[str, tuple[str, str]] = {
    "csv": ("pd.read_csv", ".csv"),
    "parquet": ("pd.read_parquet", ".parquet"),
    "xlsx": ("pd.read_excel", ".xlsx"),
}


# Caracteres de control C0 (U+0000-U+001F), DEL (U+007F) y C1 (U+0080-U+009F):
# no tienen representación visible y pueden corromper el artefacto exportado.
_CONTROL_CHARS: dict[int, None] = dict.fromkeys([*range(0x20), *range(0x7F, 0xA0)])


def _sanitize_comment(text: str) -> str:
    """Elimina caracteres de control (C0/DEL/C1) y colapsa todo whitespace
    (incluyendo saltos de línea y tabs) para que un comentario jamás pueda
    romper su línea `#` e inyectar código ejecutable ni corromper el artefacto."""
    cleaned = str(text).translate(_CONTROL_CHARS)
    return " ".join(cleaned.split())


def _safe_filename_literal(name: str | None, default_ext: str) -> str:
    """Normaliza el nombre del archivo subido a un literal Python seguro entre comillas dobles.

    Reduce el nombre a su componente base, colapsa whitespace y elimina comillas dobles
    y barras invertidas, de modo que un nombre de archivo hostil jamás pueda romper la
    cadena emitida ni inyectar código en el script exportado (mismo criterio que
    `_sanitize_comment`, pero aplicado al interior de una cadena entrecomillada).
    """
    raw = Path(str(name or "")).name
    cleaned = " ".join(raw.split()).replace('"', "").replace("\\", "")
    return cleaned if cleaned else f"dataset{default_ext}"


def build_load_code(kind: str = "synthetic", name: str | None = None) -> str:
    """Construye el bloque de carga de datos autocontenido según la fuente registrada.

    Para fuentes de archivo emite la lectura real (`pd.read_csv`/`read_parquet`/
    `read_excel`) con el nombre original del archivo; para la fuente sintética embebe
    el código fuente completo de `generate_messy_dataset` vía `inspect.getsource`,
    de modo que el artefacto exportado no contenga referencias a `src.*` y sea
    ejecutable fuera del repositorio.
    """
    if kind == "synthetic":
        # `inspect.getsource` lee el código instalado en el momento de generación
        # (siempre dentro de la app); el artefacto emitido nunca referencia `src.*`.
        source = inspect.getsource(generate_messy_dataset)
        return (
            "# Carga inicial: dataset sintético embebido (el script es autónomo).\n"
            f"{source}\n"
            "df = generate_messy_dataset()\n\n"
        )
    reader = _FILE_READERS.get(kind)
    if reader is None:
        raise ValueError(f"Fuente de datos no soportada: {kind!r}")
    read_fn, ext = reader
    filename = _safe_filename_literal(name, ext)
    return (
        "# Carga inicial\n"
        f'# Coloca el archivo "{filename}" junto a este script antes de ejecutarlo.\n'
        f'df = {read_fn}("{filename}")\n\n'
    )


class CodeGenerator:
    """Acumula pasos de transformación y genera un script Python reproducible."""

    def __init__(self):
        self.steps: list[dict] = []
        self._source_kind: str = "synthetic"
        self._source_name: str | None = None

    def set_data_source(self, kind: str = "synthetic", name: str | None = None) -> None:
        """Registra la fuente original de los datos para emitir una carga reproducible.

        Args:
            kind: Tipo de fuente ("synthetic" | "csv" | "parquet" | "xlsx").
            name: Nombre del archivo subido (None para datos sintéticos).
        """
        self._source_kind = kind
        self._source_name = name

    def add(self, step: Any, comment: str = ""):
        """Registra un paso de transformación con su código y comentario.
        Acepta tanto cadenas de código como instancias de TransformationCommand.

        El comentario se sanitiza en este límite de emisión: los nombres de
        columnas provenientes del usuario pueden contener saltos de línea, y un
        comentario multilínea escaparía del prefijo `#` al exportar .py/.ipynb.
        """
        if hasattr(step, "to_code"):
            to_code_fn = step.to_code
            code = str(to_code_fn())
            if not comment and hasattr(step, "description"):
                comment = str(step.description)
        else:
            code = str(step)
        self.steps.append({"code": code, "comment": _sanitize_comment(comment)})

    def generate_script(self) -> str:
        """Genera el script completo con todos los pasos registrados."""
        ts = datetime.datetime.now().isoformat()
        load_block = build_load_code(self._source_kind, self._source_name)
        script = HEADER.format(timestamp=ts) + load_block
        for i, s in enumerate(self.steps, 1):
            if s["comment"]:
                script += f"\n# Paso {i}: {s['comment']}\n"
            script += s["code"] + "\n"

        script += """
# --- Visualización final (ejemplo) ---
# fig, axes = plt.subplots(1,2, figsize=(12,4))
# sns.histplot(df.select_dtypes(include=np.number).iloc[:,0], kde=True, ax=axes[0])
# sns.boxplot(y=df.select_dtypes(include=np.number).iloc[:,0], ax=axes[1])
# plt.show()
"""
        return script


def export_ipynb_json(code_steps: list[dict], preamble_cell: dict | None = None) -> dict:
    """Exporta los pasos a una estructura de Jupyter Notebook (nbformat v4).

    Args:
        code_steps: Pasos de transformación ya sanitizados.
        preamble_cell: Celda opcional de código (dict nbformat) insertada antes de los
            pasos, típicamente el bloque de carga de datos autónomo.
    """
    cells = []
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Pipeline de Limpieza Auto-generado\n", "Generado por DataJohei"],
        }
    )
    if preamble_cell is not None:
        cells.append(preamble_cell)
    for step in code_steps:
        src = (f"# {step['comment']}\n" if step["comment"] else "") + step["code"]
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [line + "\n" for line in src.split("\n")],
            }
        )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
