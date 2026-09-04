"""Tests unitarios y de integración para PipelineManager (Gestión de Estado y Replay-based Undo)."""

import pandas as pd
import pytest

from src.cleaning import (
    CastCommand,
    DropDuplicatesCommand,
    ImputeCommand,
    OutlierFilterCommand,
)
from src.pipeline import PipelineManager
from src.synthetic import generate_messy_dataset


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Fixture con dataset sintético reproducible."""
    return generate_messy_dataset(seed=42)


def test_pipeline_initialization(sample_df):
    """Verifica el estado inicial de PipelineManager."""
    pipeline = PipelineManager(sample_df)

    assert pipeline.df_original is not None
    pd.testing.assert_frame_equal(pipeline.df_current, sample_df)
    pd.testing.assert_frame_equal(pipeline.df_original, sample_df)
    assert len(pipeline.commands) == 0
    assert pipeline.transform_log == []
    assert not pipeline.can_undo


def test_pipeline_apply_sequence(sample_df):
    """Aplica una secuencia de comandos y verifica el estado resultante."""
    pipeline = PipelineManager(sample_df)

    cmd1 = ImputeCommand(col="score_desempeno", strategy="media")
    cmd2 = CastCommand(col="edad", target_type="int")
    cmd3 = OutlierFilterCommand(col="salario_mensual", method="IQR", factor_or_thresh=1.5)

    df1, _ = pipeline.apply(cmd1)
    assert not bool(df1["score_desempeno"].isna().any())
    pd.testing.assert_frame_equal(pipeline.df_current, df1)
    assert len(pipeline.commands) == 1

    df2, _ = pipeline.apply(cmd2)
    assert str(df2["edad"].dtype) == "Int64"
    pd.testing.assert_frame_equal(pipeline.df_current, df2)
    assert len(pipeline.commands) == 2

    df3, _ = pipeline.apply(cmd3)
    pd.testing.assert_frame_equal(pipeline.df_current, df3)
    assert len(pipeline.commands) == 3

    assert pipeline.can_undo
    assert len(pipeline.transform_log) == 3
    assert pipeline.transform_log[0] == cmd1.description
    assert pipeline.transform_log[1] == cmd2.description
    assert pipeline.transform_log[2] == cmd3.description

    # Comparamos contra aplicación manual secuencial
    expected_df = sample_df.copy()
    expected_df, _ = cmd1.apply(expected_df)
    expected_df, _ = cmd2.apply(expected_df)
    expected_df, _ = cmd3.apply(expected_df)
    pd.testing.assert_frame_equal(pipeline.df_current, expected_df)


def test_pipeline_undo_restores_previous_state(sample_df):
    """Verifica que undo() reconstruye exactamente el estado previo reejecutando comandos."""
    pipeline = PipelineManager(sample_df)

    cmd1 = ImputeCommand(col="score_desempeno", strategy="media")
    cmd2 = CastCommand(col="edad", target_type="int")
    cmd3 = DropDuplicatesCommand()

    pipeline.apply(cmd1)
    pipeline.apply(cmd2)
    expected_after_2 = pipeline.df_current.copy()

    pipeline.apply(cmd3)
    assert len(pipeline.commands) == 3

    # Undo del 3er comando
    popped = pipeline.undo()
    assert popped is cmd3
    assert len(pipeline.commands) == 2
    pd.testing.assert_frame_equal(pipeline.df_current, expected_after_2)

    # Undo del 2do comando
    popped2 = pipeline.undo()
    assert popped2 is cmd2
    assert len(pipeline.commands) == 1

    # Undo del 1er comando -> vuelve al original
    popped1 = pipeline.undo()
    assert popped1 is cmd1
    assert len(pipeline.commands) == 0
    pd.testing.assert_frame_equal(pipeline.df_current, sample_df)
    assert not pipeline.can_undo

    # Undo en pipeline vacío debe ser safe (retorna None)
    assert pipeline.undo() is None
    pd.testing.assert_frame_equal(pipeline.df_current, sample_df)


def test_pipeline_reset_clears_commands(sample_df):
    """Verifica que reset() restaura el DataFrame original y vacía los comandos."""
    pipeline = PipelineManager(sample_df)

    pipeline.apply(ImputeCommand(col="score_desempeno", strategy="mediana"))
    pipeline.apply(CastCommand(col="salario_mensual", target_type="numeric"))

    assert len(pipeline.commands) == 2
    assert pipeline.can_undo

    pipeline.reset()

    assert len(pipeline.commands) == 0
    assert pipeline.transform_log == []
    assert not pipeline.can_undo
    pd.testing.assert_frame_equal(pipeline.df_current, sample_df)


def test_pipeline_generates_matching_script(sample_df):
    """Verifica que to_script() y to_notebook() generan artefactos coherentes con CodeGenerator."""
    pipeline = PipelineManager(sample_df)

    cmd1 = ImputeCommand(col="score_desempeno", strategy="media")
    cmd2 = CastCommand(col="edad", target_type="int")
    pipeline.apply(cmd1)
    pipeline.apply(cmd2)

    script = pipeline.to_script()
    assert isinstance(script, str)
    assert cmd1.to_code() in script
    assert cmd2.to_code() in script

    nb_json = pipeline.to_notebook()
    assert isinstance(nb_json, dict)
    assert nb_json["nbformat"] == 4
    code_cells = [c for c in nb_json["cells"] if c["cell_type"] == "code"]
    # Celda de carga autónoma + una celda por comando aplicado (JD-05)
    assert len(code_cells) == 3
    assert "df = generate_messy_dataset()" in "".join(code_cells[0]["source"])


def test_pipeline_memory_efficiency_no_snapshot_stack(sample_df):
    """Verifica que PipelineManager no almacena una lista histórica de DataFrames clonados."""
    pipeline = PipelineManager(sample_df)

    for _ in range(5):
        pipeline.apply(ImputeCommand(col="score_desempeno", strategy="media"))

    # No debe existir un atributo 'history' con DataFrames
    assert not hasattr(pipeline, "history") or not isinstance(
        getattr(pipeline, "history", None), list
    )
