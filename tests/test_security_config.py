"""Security tests for configuration and repository hygiene (SEC-TASK-05 / CWE-538)."""

from pathlib import Path


def test_gitignore_contains_security_exclusions():
    """Verifies that .gitignore contains rules to exclude secrets, keys, and local dataset dumps."""
    project_root = Path(__file__).resolve().parent.parent
    gitignore_path = project_root / ".gitignore"

    assert gitignore_path.is_file(), ".gitignore file must exist at project root"

    content = gitignore_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    required_patterns = [
        ".env",
        "*.env",
        ".streamlit/secrets.toml",
        "*.pem",
        "*.key",
        "*.csv",
        "*.parquet",
        "*.xlsx",
        "*.pkl",
    ]

    missing_patterns = [pattern for pattern in required_patterns if pattern not in lines]

    assert not missing_patterns, (
        f".gitignore is missing required security exclusion patterns: {missing_patterns}"
    )


def test_pyproject_contains_ruff_configuration():
    """Verifies that pyproject.toml defines [tool.ruff] configuration targeting Python 3.12."""
    import tomllib

    project_root = Path(__file__).resolve().parent.parent
    pyproject_path = project_root / "pyproject.toml"

    assert pyproject_path.is_file(), "pyproject.toml must exist at project root"

    config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert "tool" in config, "pyproject.toml must define a [tool] table"
    assert "ruff" in config["tool"], "pyproject.toml must define [tool.ruff]"
    ruff_conf = config["tool"]["ruff"]
    assert ruff_conf.get("target-version") == "py312"
    assert "lint" in ruff_conf
    assert "select" in ruff_conf["lint"]
