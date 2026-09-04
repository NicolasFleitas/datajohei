"""Security tests for CodeGen identifier sanitization (SEC-TASK-01 / CWE-94, CWE-116)."""

import ast

import numpy as np
import pandas as pd
import pytest

from src.cleaning import (
    CastCommand,
    DropDuplicatesCommand,
    ImputeCommand,
    OutlierFilterCommand,
)
from src.codegen import _sanitize_comment
from src.pipeline import PipelineManager

MALICIOUS_COL = "col'] = 1\nimport os; os.system('echo pwned')\n#"
SPECIAL_CHARS_COLS = [
    "col'with'single_quote",
    'col"with"double_quote',
    "col\nwith\nnewline",
    "col\\with\\backslash",
    "col\twith\ttab",
]


@pytest.mark.parametrize(
    "cmd_factory",
    [
        lambda col: ImputeCommand(col=col, strategy="media"),
        lambda col: ImputeCommand(col=col, strategy="mediana"),
        lambda col: ImputeCommand(col=col, strategy="moda"),
        lambda col: ImputeCommand(col=col, strategy="ffill"),
        lambda col: ImputeCommand(col=col, strategy="bfill"),
        lambda col: ImputeCommand(col=col, strategy="constante", constant=0),
        lambda col: ImputeCommand(col=col, strategy="drop_rows"),
        lambda col: ImputeCommand(col=col, strategy="drop_col"),
        lambda col: CastCommand(col=col, target_type="numeric"),
        lambda col: CastCommand(col=col, target_type="datetime"),
        lambda col: CastCommand(col=col, target_type="string"),
        lambda col: CastCommand(col=col, target_type="category"),
        lambda col: CastCommand(col=col, target_type="int"),
        lambda col: CastCommand(col=col, target_type="float"),
        lambda col: CastCommand(col=col, target_type="bool"),
        lambda col: OutlierFilterCommand(col=col, method="IQR"),
        lambda col: OutlierFilterCommand(col=col, method="Z-Score"),
        lambda col: DropDuplicatesCommand(subset=[col]),
    ],
)
def test_codegen_prevents_code_injection_in_column_names(cmd_factory):
    """Verifies that malicious payloads in column names cannot inject statements."""
    cmd = cmd_factory(MALICIOUS_COL)
    code = cmd.to_code()

    # Parse AST
    parsed = ast.parse(code)

    # Check for injected AST nodes (e.g. import os or calls to os.system)
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "os", f"Injected 'import os' node found in code: {code}"
        if isinstance(node, ast.ImportFrom):
            assert node.module == "src.cleaning", f"Injected ImportFrom found in code: {code}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "system", f"Injected os.system call found in code: {code}"

    # Ensure the malicious string exists solely as a string literal constant in AST
    constants = [
        node.value
        for node in ast.walk(parsed)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    assert any(MALICIOUS_COL in c or c == MALICIOUS_COL for c in constants), (
        f"Column name not preserved as literal constant: {code}"
    )


@pytest.mark.parametrize("special_col", SPECIAL_CHARS_COLS)
@pytest.mark.parametrize(
    "cmd_factory",
    [
        lambda col: ImputeCommand(col=col, strategy="media"),
        lambda col: ImputeCommand(col=col, strategy="drop_rows"),
        lambda col: ImputeCommand(col=col, strategy="drop_col"),
        lambda col: CastCommand(col=col, target_type="numeric"),
        lambda col: CastCommand(col=col, target_type="bool"),
        lambda col: OutlierFilterCommand(col=col, method="IQR"),
        lambda col: OutlierFilterCommand(col=col, method="Z-Score"),
        lambda col: DropDuplicatesCommand(subset=[col]),
    ],
)
def test_codegen_handles_special_characters_safely(cmd_factory, special_col):
    """Verifies that column names with quotes, newlines, and backslashes parse cleanly."""
    cmd = cmd_factory(special_col)
    code = cmd.to_code()
    # ast.parse must succeed without SyntaxError
    parsed = ast.parse(code)
    assert parsed is not None


HOSTILE_COMMENT_COL = "edad'; import os\nos.system('id')"
SMUGGLED_STATEMENTS = {"import os", "os.system('id')"}


def _assert_no_os_execution(tree: ast.AST, artifact: str) -> None:
    """Recorre el AST rechazando `import os` y llamadas a os.system ejecutables."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "os" for alias in node.names), (
                f"{artifact}: 'import os' ejecutable encontrado"
            )
        if isinstance(node, ast.ImportFrom):
            assert node.module != "os", f"{artifact}: 'from os import' ejecutable encontrado"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "system", f"{artifact}: llamada os.system ejecutiva encontrada"


def _assert_no_loose_smuggled_lines(lines: list[str], artifact: str) -> None:
    """Ninguna línea del artefacto puede ser la sentencia contrabandeada suelta."""
    for line in lines:
        assert line.strip() not in SMUGGLED_STATEMENTS, (
            f"{artifact}: sentencia ejecutable suelta fuera de comentario: {line!r}"
        )


def test_pipeline_export_neutralizes_comment_injection():
    """JD-02: un nombre de columna hostil que fluye hacia la descripción de un comando
    real debe quedar confinado al comentario en los artefactos .py/.ipynb exportados."""
    df = pd.DataFrame({HOSTILE_COMMENT_COL: [1.0, np.nan, 3.0]})
    pipeline = PipelineManager(df_original=df)
    pipeline.apply(ImputeCommand(col=HOSTILE_COMMENT_COL, strategy="media"))

    script = pipeline.to_script()
    notebook = pipeline.to_notebook()

    # El .py emitido parsea sin SyntaxError y no contiene código contrabandeado
    _assert_no_os_execution(ast.parse(script), ".py")
    _assert_no_loose_smuggled_lines(script.splitlines(), ".py")
    # Y el payload quedó confinado dentro de una línea de comentario
    assert any(
        line.lstrip().startswith("#") and "os.system('id')" in line for line in script.splitlines()
    ), "El payload hostil no aparece confinado en un comentario del .py"

    # Cada celda de código del notebook parsea limpia y sin líneas ejecutables sueltas
    for idx, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        cell_src = "".join(cell["source"])
        _assert_no_os_execution(ast.parse(cell_src), f".ipynb cell {idx}")
        _assert_no_loose_smuggled_lines(cell["source"], f".ipynb cell {idx}")
        # Sin saltos de línea crudos dentro de líneas de comentario
        for line in cell["source"]:
            if line.lstrip().startswith("#"):
                assert "\n" not in line.rstrip("\n"), f".ipynb cell {idx}: comentario multilínea"


def _assert_no_control_chars(text: str, artifact: str) -> None:
    """Rechaza cualquier carácter C0, DEL o C1 en el texto analizado."""
    offenders = [ch for ch in text if ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F]
    assert not offenders, f"{artifact}: caracteres de control supervivientes: {offenders!r}"


def test_sanitize_comment_strips_control_characters():
    """R-JD02-a: C0/DEL/C1 (p. ej. \\x00) se eliminan junto con el colapso de whitespace."""
    raw = "\x00paso\x01\tcon\b control\x1b[31m\x7f y\x85 C1"
    cleaned = _sanitize_comment(raw)
    _assert_no_control_chars(cleaned, "_sanitize_comment")
    # El texto visible sobrevive salvo los caracteres eliminados.
    assert "[31m" in cleaned and "paso" in cleaned and "y C1" in cleaned


def test_pipeline_export_strips_control_chars_and_compiles():
    """R-JD02-a: un \\x00 en el nombre de columna no corrompe el artefacto exportado.

    Ninguna línea de comentario del .py/.ipynb conserva caracteres de control y
    ambos artefactos compilan como Python válido.
    """
    control_col = "col\x00_inject\x1b"
    df = pd.DataFrame({control_col: [1.0, np.nan, 3.0]})
    pipeline = PipelineManager(df_original=df)
    pipeline.apply(ImputeCommand(col=control_col, strategy="media"))

    script = pipeline.to_script()
    notebook = pipeline.to_notebook()

    # Sin bytes NUL crudos en todo el artefacto .py
    assert b"\x00" not in script.encode()
    # Las líneas de comentario están libres de C0/DEL/C1
    for line in script.splitlines():
        if line.lstrip().startswith("#"):
            _assert_no_control_chars(line, ".py comment")
    # Y el script compila como Python válido
    compile(script, "<export>", "exec")

    for idx, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        cell_src = "".join(cell["source"])
        for line in cell_src.splitlines():
            if line.lstrip().startswith("#"):
                _assert_no_control_chars(line, f".ipynb cell {idx} comment")
        compile(cell_src, f"<export cell {idx}>", "exec")
