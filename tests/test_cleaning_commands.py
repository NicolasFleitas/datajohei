"""Tests unitarios para la arquitectura de comandos de transformación (TransformationCommand)."""

import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    CastCommand,
    DropDuplicatesCommand,
    ImputeCommand,
    OutlierFilterCommand,
    TransformationCommand,
)
from src.codegen import CodeGenerator, export_ipynb_json
from src.synthetic import generate_messy_dataset


def test_impute_commands_apply_and_code():
    """Para cada estrategia de imputación, validar apply() y sintaxis de to_code()."""
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, np.nan, 4.0, 5.0],
            "cat": ["A", "B", np.nan, "A", "A"],
            "mixed": [10.0, np.nan, np.nan, 40.0, 50.0],
        }
    )

    strategies = [
        ("media", "num", None),
        ("mediana", "num", None),
        ("moda", "cat", None),
        ("ffill", "mixed", None),
        ("bfill", "mixed", None),
        ("constante", "num", 99.0),
        ("drop_rows", "num", None),
        ("drop_col", "cat", None),
    ]

    for strat, col, const_val in strategies:
        cmd = ImputeCommand(col=col, strategy=strat, constant=const_val)
        assert isinstance(cmd, TransformationCommand)
        assert isinstance(cmd.description, str) and len(cmd.description) > 0

        code = cmd.to_code()
        assert isinstance(code, str) and len(code) > 0
        compile(code, "<test>", "exec")

        df_res, report = cmd.apply(df)
        assert isinstance(df_res, pd.DataFrame)
        assert isinstance(report, dict)

        if strat == "drop_col":
            assert col not in df_res.columns
        elif strat == "drop_rows":
            assert not bool(df_res[col].isna().any())
            assert len(df_res) == 4
        else:
            assert not bool(df_res[col].isna().any())


# JD-04: media sobre Int64 anulable no explota; ensancha a float64 e imputa la media exacta
def test_impute_media_int64_nullable_widens_to_float():
    df = pd.DataFrame({"edad": pd.array([25, 30, None, 41], dtype="Int64")})

    df_out, _report = ImputeCommand(col="edad", strategy="media").apply(df)

    out = df_out["edad"]
    assert str(out.dtype) == "float64"
    # media exacta de los válidos: (25 + 30 + 41) / 3 = 32.0
    assert out.tolist() == [25.0, 30.0, 32.0, 41.0]


def test_impute_mediana_int64_nullable_widens_to_float():
    # cantidad par de válidos: la mediana es fraccional (1.5), caso que rompía el dtype
    df = pd.DataFrame({"edad": pd.array([1, 2, None], dtype="Int64")})

    df_out, _report = ImputeCommand(col="edad", strategy="mediana").apply(df)

    out = df_out["edad"]
    assert str(out.dtype) == "float64"
    assert out.tolist() == [1.0, 2.0, 1.5]


# JD-04 (documentado como intencional): int64 sin nulos también se ensancha a float64
def test_impute_media_plain_int_without_na_widens_to_float():
    df = pd.DataFrame({"edad": [10, 20, 30]})

    df_out, _report = ImputeCommand(col="edad", strategy="media").apply(df)

    out = df_out["edad"]
    assert str(out.dtype) == "float64"
    assert out.tolist() == [10.0, 20.0, 30.0]


def test_impute_media_float64_unchanged_behavior():
    df = pd.DataFrame({"saldo": [1.0, np.nan, 3.0]})

    df_out, _report = ImputeCommand(col="saldo", strategy="media").apply(df)

    out = df_out["saldo"]
    assert str(out.dtype) == "float64"  # dtype y semántica intactos para floats
    assert out.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("strategy", ["media", "mediana"])
def test_impute_media_mediana_to_code_exec_parity_int64(strategy):
    """JD-04: el snippet exportado reproduce apply() en un Int64 con NA
    (mismos valores Y mismo dtype) ejecutándose standalone."""
    df = pd.DataFrame({"edad": pd.array([25, 30, None, 41], dtype="Int64")})
    cmd = ImputeCommand(col="edad", strategy=strategy)
    df_applied, _report = cmd.apply(df)

    code = cmd.to_code()
    compile(code, "<test>", "exec")

    local_ns = {"df": df.copy(), "pd": pd}
    exec(code, local_ns)

    pd.testing.assert_series_equal(df_applied["edad"], local_ns["df"]["edad"])


@pytest.mark.parametrize(
    "target_type",
    ["numeric", "datetime", "string", "category", "int", "float", "bool"],
)
def test_cast_commands_parity(target_type):
    """Verifica paridad total: apply() vs exec(to_code()) produce DataFrames idénticos."""
    df = pd.DataFrame(
        {
            "val": ["1", "2.5", "invalid", None, "true", "0", "2023-01-01", "Sí"],
        }
    )

    cmd = CastCommand(col="val", target_type=target_type)
    df_applied, _report = cmd.apply(df)

    code = cmd.to_code()
    compile(code, "<test>", "exec")

    # Ejecución de to_code() en un namespace aislado
    local_ns = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
    }
    exec(code, local_ns)
    df_exec = local_ns["df"]

    pd.testing.assert_frame_equal(df_applied, df_exec)


def test_outlier_and_dedup_commands():
    """Valida comandos de outliers (IQR y Z-Score) y deduplicación."""
    df = generate_messy_dataset()

    # IQR
    cmd_iqr = OutlierFilterCommand(col="salario_mensual", method="IQR", factor_or_thresh=1.5)
    df_iqr, report_iqr = cmd_iqr.apply(df)
    assert report_iqr["outliers_detected"] == 7  # seed 42, default_rng determinista
    assert len(df_iqr) == len(df) - 7
    code_iqr = cmd_iqr.to_code()
    compile(code_iqr, "<test>", "exec")

    # Z-Score
    cmd_z = OutlierFilterCommand(col="salario_mensual", method="Z-Score", factor_or_thresh=3.0)
    df_z, report_z = cmd_z.apply(df)
    assert report_z["outliers_detected"] == 4  # seed 42, default_rng determinista
    assert len(df_z) == len(df) - 4
    code_z = cmd_z.to_code()
    compile(code_z, "<test>", "exec")

    # Deduplicación
    expected_dups = int(df.duplicated().sum())
    cmd_dedup = DropDuplicatesCommand()
    df_dedup, report_dedup = cmd_dedup.apply(df)
    assert report_dedup["removed"] == expected_dups
    assert len(df_dedup) == len(df) - expected_dups
    code_dedup = cmd_dedup.to_code()
    assert "drop_duplicates" in code_dedup
    compile(code_dedup, "<test>", "exec")


def test_codegen_integration_with_commands():
    """Verifica que CodeGenerator acepta TransformationCommand y exporta script y notebook."""
    cg = CodeGenerator()
    cmd1 = ImputeCommand(col="salario_mensual", strategy="media")
    cmd2 = CastCommand(col="salario_mensual", target_type="int")
    cmd3 = OutlierFilterCommand(col="salario_mensual", method="IQR", factor_or_thresh=1.5)
    cmd4 = DropDuplicatesCommand()

    cg.add(cmd1)
    cg.add(cmd2)
    cg.add(cmd3)
    cg.add(cmd4)

    assert len(cg.steps) == 4
    assert cg.steps[0]["code"] == cmd1.to_code()
    assert cg.steps[0]["comment"] == cmd1.description

    script = cg.generate_script()
    assert "fillna" in script
    assert "round().astype('Int64')" in script
    assert "detect_outliers_iqr" in script
    assert "drop_duplicates" in script

    nb = export_ipynb_json(cg.steps)
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 4
