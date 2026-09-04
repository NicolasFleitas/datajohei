"""Módulo de gestión de pipeline de transformaciones y estado reproducible (Event Sourcing)."""

from collections.abc import Iterator
from typing import Any

import pandas as pd

from src.cleaning import TransformationCommand
from src.codegen import CodeGenerator, build_load_code, export_ipynb_json

_SOURCE_KINDS = frozenset({"synthetic", "csv", "parquet", "xlsx"})


class PipelineManager:
    """Gestiona la secuencia de comandos de transformación aplicados a un DataFrame.

    Implementa el patrón Event Sourcing / Replay-based Undo para evitar la
    retención de múltiples copias completas de DataFrames en memoria (O(K * size)).
    El estado almacena únicamente `df_original` inmutable y la lista ordenada de
    comandos `TransformationCommand`.
    """

    def __init__(
        self,
        df_original: pd.DataFrame,
        source_kind: str = "synthetic",
        source_name: str | None = None,
    ):
        """Args:
        df_original: DataFrame original inmutable.
        source_kind: Origen de los datos ("synthetic" | "csv" | "parquet" | "xlsx").
        source_name: Nombre del archivo subido (None para datos sintéticos).
        """
        if source_kind not in _SOURCE_KINDS:
            raise ValueError(f"source_kind no soportado: {source_kind!r}")
        self._df_original: pd.DataFrame = df_original.copy()
        self._df_current: pd.DataFrame = df_original.copy()
        self._commands: list[TransformationCommand] = []
        self._source_kind: str = source_kind
        self._source_name: str | None = source_name

    @property
    def source_kind(self) -> str:
        """Devuelve el tipo de fuente de datos original del pipeline."""
        return self._source_kind

    @property
    def source_name(self) -> str | None:
        """Devuelve el nombre del archivo de origen (None para datos sintéticos)."""
        return self._source_name

    @property
    def df_original(self) -> pd.DataFrame:
        """Devuelve el DataFrame original inmutable."""
        return self._df_original

    @property
    def df_current(self) -> pd.DataFrame:
        """Devuelve el DataFrame con todas las transformaciones aplicadas."""
        return self._df_current

    @property
    def commands(self) -> list[TransformationCommand]:
        """Devuelve la lista ordenada de comandos aplicados."""
        return self._commands

    @property
    def transform_log(self) -> list[str]:
        """Devuelve el historial legible de transformaciones."""
        return [cmd.description for cmd in self._commands]

    @property
    def can_undo(self) -> bool:
        """Indica si existen transformaciones que puedan revertirse."""
        return len(self._commands) > 0

    def __len__(self) -> int:
        return len(self._commands)

    def __iter__(self) -> Iterator[TransformationCommand]:
        return iter(self._commands)

    def __getitem__(self, index: int) -> TransformationCommand:
        return self._commands[index]

    def apply(self, command: TransformationCommand) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Aplica un comando de transformación sobre el DataFrame actual y lo registra.

        Args:
            command: Instancia de TransformationCommand a aplicar.

        Returns:
            Tupla con el nuevo DataFrame y los metadatos de la operación.
        """
        new_df, metadata = command.apply(self._df_current)
        self._commands.append(command)
        self._df_current = new_df
        return new_df, metadata

    def push(
        self,
        command: TransformationCommand,
        df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Registra un comando ya ejecutado o lo aplica si no se proveyó el DataFrame resultante."""
        if df is not None:
            self._commands.append(command)
            self._df_current = df
            return df
        return self.apply(command)[0]

    def undo(self) -> TransformationCommand | None:
        """Revierte el último comando reejecutando la secuencia restante sobre df_original.

        Returns:
            El comando revertido, o None si no había comandos aplicados.
        """
        if not self._commands:
            return None

        last_command = self._commands.pop()
        df = self._df_original.copy()
        for cmd in self._commands:
            df, _ = cmd.apply(df)
        self._df_current = df
        return last_command

    def reset(self) -> pd.DataFrame:
        """Vacía todos los comandos y restaura el estado al DataFrame original.

        Returns:
            El DataFrame original restablecido.
        """
        self._commands.clear()
        self._df_current = self._df_original.copy()
        return self._df_current

    def to_script(self) -> str:
        """Genera el script Python reproducible correspondiente a los comandos actuales."""
        cg = CodeGenerator()
        cg.set_data_source(self._source_kind, self._source_name)
        for cmd in self._commands:
            cg.add(cmd, comment=cmd.description)
        return cg.generate_script()

    def to_notebook(self) -> dict:
        """Genera la estructura de Jupyter Notebook (.ipynb) de los comandos actuales.

        La primera celda de código contiene el bloque de carga autónomo, de modo que
        el notebook sea reproducible fuera del repositorio.
        """
        cg = CodeGenerator()
        cg.set_data_source(self._source_kind, self._source_name)
        for cmd in self._commands:
            cg.add(cmd, comment=cmd.description)
        load_code = build_load_code(self._source_kind, self._source_name)
        preamble_cell = {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in load_code.split("\n")],
        }
        return export_ipynb_json(cg.steps, preamble_cell=preamble_cell)
