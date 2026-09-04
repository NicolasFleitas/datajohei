# DataJohei

Visualizador interactivo de limpieza y exploración de datos. Cargás un dataset, diagnosticás su calidad, aplicás transformaciones y exportás el resultado junto con el código Python reproducible.

## Requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes y entornos)

## Instalación

```bash
git clone https://github.com/NicolasFleitas/datajohei.git && cd datajohei
uv sync
```

## Ejecución

Local:

```bash
uv run streamlit run app.py
```

Abre `http://localhost:8501`.

Con Docker:

```bash
docker compose up --build
```

Abre `http://localhost:8501`.

## Flujo de trabajo

| Paso | Qué hacés | Qué obtenés |
| ---- | --------- | ----------- |
| 1. Diagnóstico | Cargás un CSV, Parquet o XLSX (o probás con el dataset sintético) | Métricas generales, tipos por columna, vista previa y matriz de nulos |
| 2. Limpieza | Imputás nulos, convertís tipos, eliminás duplicados y filtrás outliers | Historial con deshacer y reset al estado original |
| 3. Visualización | Explorás una o dos columnas | Histogramas, boxplots, correlaciones, scatterplots y barras agregadas |
| 4. Exportación | Descargás el resultado | CSV o Parquet, script Python reproducible y notebook Jupyter |

## Arquitectura

- `src/` — lógica de dominio (limpieza, diagnóstico, visualización, exportación).
- `src/views/` — interfaz organizada por pestaña.
- `app.py` / `main.py` — puntos de entrada.
- `tests/` — suite automatizada (`uv run pytest tests/ -q`).

Cada transformación queda registrada y es reproducible: el historial permite deshacer y genera el código exportado.

## Stack

| Componente | Tecnología |
| ------------ | ----------- |
| Framework | Streamlit 1.59+ |
| Datos | Pandas 3.0+, NumPy 2.5+ |
| Gráficos | Seaborn 0.13+, Matplotlib 3.11+ |
| Archivos | PyArrow, OpenPyXL |
| Linter / Formatter | Ruff (target: py312) |
| Tests | pytest 9.1+ |
| Entorno y Paquetes | uv |

## Configuración y límites

- Límite de carga: 100 MB por archivo.
- Puerto por defecto: 8501.
- Sin variables de entorno requeridas para uso local.
