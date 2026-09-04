"""Punto de entrada de la aplicación Streamlit."""

import sys
from pathlib import Path


def main() -> None:
    """Ejecuta la app de Streamlit asegurando que `src` sea importable."""
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(root / "app.py")]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
