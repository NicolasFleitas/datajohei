"""Fidelidad del código exportado y detección de dead code."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from src.cleaning import ImputeCommand, impute_column
from src.codegen import CodeGenerator, export_ipynb_json
from src.pipeline import PipelineManager
from src.synthetic import generate_messy_dataset
from src.views.export_view import get_csv_bytes, get_parquet_bytes

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PY = REPO_ROOT / "app.py"

# Strings de imputación exportados por ImputeCommand. Deben
# ser assignment-based, idénticos en semántica a src.cleaning.impute_column.
# media/mediana llevan guard JD-04: un dtype entero se ensancha a float64
# porque el estadístico puede ser fraccional.
IMPUT_EXPORT_TMPL = {
    "media": (
        "if pd.api.types.is_integer_dtype(df['{col}'].dtype):\n"
        "    df['{col}'] = df['{col}'].astype('float64')\n"
        "df['{col}'] = df['{col}'].fillna(df['{col}'].mean())"
    ),
    "mediana": (
        "if pd.api.types.is_integer_dtype(df['{col}'].dtype):\n"
        "    df['{col}'] = df['{col}'].astype('float64')\n"
        "df['{col}'] = df['{col}'].fillna(df['{col}'].median())"
    ),
    "moda": "df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])",
    "ffill": "df['{col}'] = df['{col}'].ffill()",
    "bfill": "df['{col}'] = df['{col}'].bfill()",
}

# — columnas: float64 con NaN del dataset sintético (seed 42, determinista)
NUMERIC_WITH_NAN = "score_desempeno"
OTHER_NUMERIC_WITH_NAN = "anios_experiencia"
STRING_WITH_NULL = "departamento"


def _run_exported_script(steps) -> pd.DataFrame:
    """Genera el script .py con CodeGenerator (mecanismo real de la app) y lo
    EJECUTA en un subproceso aislado. Devuelve el DataFrame final (pickle
    preserva dtypes/nullables exactos). Nada queda en el repo."""
    cg = CodeGenerator()
    for item in steps:
        if isinstance(item, tuple):
            cg.add(item[0], comment=item[1])
        else:
            cg.add(item)
    script = cg.generate_script()
    script += "import sys\ndf.to_pickle(sys.argv[1])\n"
    with tempfile.TemporaryDirectory() as td:
        runner = Path(td) / "exported_pipeline.py"
        out_pkl = Path(td) / "result.pkl"
        runner.write_text(script, encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.run(
            [sys.executable, str(runner), str(out_pkl)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"export falló:\n{proc.stdout}\n{proc.stderr}"
        return cast(pd.DataFrame, pd.read_pickle(out_pkl))


def test_generate_notebook_cells_removed():
    """generate_notebook_cells() ya no existe (dead code eliminado)."""
    assert "generate_notebook_cells" not in dir(CodeGenerator)


def test_script_contains_int_round_astype():
    """El script .py exportado replica el cast int con .round().astype('Int64')."""
    cg = CodeGenerator()
    code = "df['salario_mensual'] = pd.to_numeric(df['salario_mensual'], errors='coerce').round().astype('Int64')"
    cg.add(code, comment="Cast salario_mensual -> int")
    assert ".round().astype('Int64')" in cg.generate_script()


def test_script_contains_bool_mapper():
    """El script exportado lleva el mapper bool explícito (alias + pd.NA)."""
    cg = CodeGenerator()
    code = (
        "# Cast bool: mapper explícito\n"
        "_BOOL_ALIASES = {'si': True, 'sí': True, 'yes': True, 'true': True, '1': True, 'no': False, 'false': False, '0': False}\n"
        "def _cast_bool(v):\n"
        "    if pd.isna(v):\n"
        "        return pd.NA\n"
        "    return _BOOL_ALIASES.get(str(v).strip().lower(), pd.NA)\n"
        "df['flag'] = df['flag'].map(_cast_bool).astype('boolean')"
    )
    cg.add(code, comment="Cast flag -> bool")
    script = cg.generate_script()
    assert "def _cast_bool" in script
    assert "_BOOL_ALIASES" in script
    assert "pd.NA" in script


def test_export_ipynb_json_valid_nbformat():
    """export_ipynb_json() produce un notebook nbformat 4 válido."""
    steps = [{"code": "df = df.copy()", "comment": "probando"}]
    nb = export_ipynb_json(steps)
    assert nb["nbformat"] == 4
    assert nb["cells"][0]["cell_type"] == "markdown"
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 1
    # el comentario se antepone como línea del cell
    assert "".join(code_cells[0]["source"]) == "# probando\ndf = df.copy()\n"
    assert "execution_count" in code_cells[0]


# ejecución del export vs runtime (regresión pandas 3 CoW)
def test_app_imputation_strings_assignment_based():
    """ImputeCommand exporta imputación assignment-based (nunca chained
    inplace=True, que bajo pandas 3 CoW es NO-OP)."""
    src = APP_PY.read_text(encoding="utf-8")
    # chained inplace son NO-OP bajo CoW
    buggy_patterns = [
        "df['{target_col}'].fillna(df['{target_col}'].mean(), inplace=True)",
        "df['{target_col}'].fillna(df['{target_col}'].median(), inplace=True)",
        "df['{target_col}'].fillna(df['{target_col}'].mode()[0], inplace=True)",
        "df['{target_col}'].ffill(inplace=True)",
        "df['{target_col}'].bfill(inplace=True)",
    ]
    assert not any(p in src for p in buggy_patterns)
    for strat, tmpl in IMPUT_EXPORT_TMPL.items():
        cmd = ImputeCommand(col="{col}", strategy=strat)
        assert cmd.to_code() == tmpl


@pytest.mark.parametrize(
    ("col", "strategy"),
    [
        (NUMERIC_WITH_NAN, "media"),
        (NUMERIC_WITH_NAN, "mediana"),
        (STRING_WITH_NULL, "moda"),
        (NUMERIC_WITH_NAN, "ffill"),
        (NUMERIC_WITH_NAN, "bfill"),
    ],
)
def test_export_imputation_strategy_equivalence(col, strategy):
    """El comando de imputación exportado y ejecutado produce el mismo
    DataFrame que impute_column (runtime), para cada estrategia por separado."""
    runtime = impute_column(generate_messy_dataset(), col, strategy)
    cmd = ImputeCommand(col=col, strategy=strategy)
    exported = _run_exported_script([cmd])
    pd.testing.assert_frame_equal(runtime, exported)


def test_export_imputation_multistep_equivalence():
    """Pipeline de 3 pasos — media numérica, moda string y ffill — el script
    exportado y ejecutado produce el mismo DataFrame que el runtime."""
    cmds = [
        ImputeCommand(col=NUMERIC_WITH_NAN, strategy="media"),
        ImputeCommand(col=STRING_WITH_NULL, strategy="moda"),
        ImputeCommand(col=OTHER_NUMERIC_WITH_NAN, strategy="ffill"),
    ]
    df = generate_messy_dataset()
    runtime = impute_column(
        impute_column(impute_column(df, NUMERIC_WITH_NAN, "media"), STRING_WITH_NULL, "moda"),
        OTHER_NUMERIC_WITH_NAN,
        "ffill",
    )
    exported = _run_exported_script(cmds)
    assert exported.shape == runtime.shape
    assert not bool(exported[NUMERIC_WITH_NAN].isna().any())  # media imputó
    assert not bool(exported[STRING_WITH_NULL].isna().any())  # moda imputó
    assert not bool(exported[OTHER_NUMERIC_WITH_NAN].isna().any())  # ffill imputó
    pd.testing.assert_frame_equal(runtime, exported)


# — JD-05: exports reproducibles con la fuente de datos real —


def test_pipeline_manager_source_properties():
    """PipelineManager expone source_kind/source_name con defaults sintéticos."""
    pm_default = PipelineManager(pd.DataFrame({"a": [1]}))
    assert pm_default.source_kind == "synthetic"
    assert pm_default.source_name is None

    pm_csv = PipelineManager(pd.DataFrame({"a": [1]}), source_kind="csv", source_name="ventas.csv")
    assert pm_csv.source_kind == "csv"
    assert pm_csv.source_name == "ventas.csv"

    with pytest.raises(ValueError):
        PipelineManager(pd.DataFrame({"a": [1]}), source_kind="hdf5")


def test_synthetic_export_is_standalone():
    """El .py de una fuente sintética define generate_messy_dataset inline y no
    referencia el paquete src (reproducibilidad fuera del repo)."""
    pm = PipelineManager(generate_messy_dataset())
    script = pm.to_script()
    assert "from src." not in script
    assert "import src" not in script
    assert "def generate_messy_dataset(" in script
    assert "df = generate_messy_dataset()" in script


def test_exported_synthetic_script_runs_outside_repo():
    """Prueba funcional de autonomía (JD-05): el script sintético exportado se escribe
    en un directorio temporal FUERA del repo y se ejecuta ahí con entorno limpio (sin
    PYTHONPATH ni cwd del proyecto). Debe producir un DataFrame con filas."""
    pm = PipelineManager(generate_messy_dataset())
    script = pm.to_script() + (
        "import pickle\nwith open('result.pkl', 'wb') as fh:\n    pickle.dump(df, fh)\n"
    )
    with tempfile.TemporaryDirectory() as td:
        runner = Path(td) / "exported_pipeline.py"
        runner.write_text(script, encoding="utf-8")
        clean_env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", td)}
        proc = subprocess.run(
            [sys.executable, str(runner)],
            cwd=td,
            env=clean_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"export autónomo falló:\n{proc.stdout}\n{proc.stderr}"
        df_out = pd.read_pickle(Path(td) / "result.pkl")
        assert len(df_out) > 0


@pytest.mark.parametrize(
    ("kind", "filename", "expected_reader"),
    [
        ("csv", "reporte_2024.csv", 'df = pd.read_csv("reporte_2024.csv")'),
        ("parquet", "clientes.parquet", 'df = pd.read_parquet("clientes.parquet")'),
        ("xlsx", "inventario.xlsx", 'df = pd.read_excel("inventario.xlsx")'),
    ],
)
def test_file_source_emits_real_reader(kind, filename, expected_reader):
    """Los managers con fuente de archivo emiten la lectura del nombre real."""
    pm = PipelineManager(pd.DataFrame({"a": [1]}), source_kind=kind, source_name=filename)
    script = pm.to_script()
    assert expected_reader in script
    assert "junto a este script antes de ejecutarlo" in script
    assert "from src." not in script


def test_notebook_includes_load_cell_before_steps():
    """to_notebook() antepone la celda de carga autónoma a las celdas de transformación."""
    pm = PipelineManager(generate_messy_dataset(), source_kind="xlsx", source_name="empleados.xlsx")
    pm.apply(ImputeCommand(col=NUMERIC_WITH_NAN, strategy="media"))
    nb = pm.to_notebook()
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 2
    load_src = "".join(code_cells[0]["source"])
    assert 'df = pd.read_excel("empleados.xlsx")' in load_src
    assert NUMERIC_WITH_NAN in "".join(code_cells[1]["source"])  # paso tras la carga
    assert "src." not in json.dumps(nb)


def test_export_cache_decorators_are_bounded():
    """JD-06: get_csv_bytes/get_parquet_bytes declaran caché acotada como src.loader."""
    assert get_csv_bytes._info.max_entries == 20
    assert get_csv_bytes._info.ttl == 3600
    assert get_parquet_bytes._info.max_entries == 20
    assert get_parquet_bytes._info.ttl == 3600
