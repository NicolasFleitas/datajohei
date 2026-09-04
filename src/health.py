"""Cálculo de métricas de salud del dataset y resumen numérico de valores faltantes."""

from typing import Any, cast

import numpy as np
import pandas as pd


def _first_valid_val(s: pd.Series) -> Any:
    idx = s.first_valid_index()
    return s.loc[idx] if idx is not None else np.nan


def compute_health_metrics(df: pd.DataFrame) -> dict:
    """Calcula métricas generales del dataset: dimensiones, nulos, duplicados
    y resumen de tipos por columna."""
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())
    pct_missing = (total_missing / total_cells * 100) if total_cells else 0
    dup_count = int(df.duplicated().sum())

    dtypes_df = pd.DataFrame(
        {
            "columna": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "nulos": df.isna().sum().values,
            "% nulos": (df.isna().mean() * 100).round(2).values,
            "unicos": df.nunique().values,
            "ejemplo": [_first_valid_val(cast(pd.Series, df[c])) for c in df.columns],
        }
    )
    dtypes_df["ejemplo"] = dtypes_df["ejemplo"].astype(str)
    return {
        "shape": df.shape,
        "pct_missing_global": pct_missing,
        "total_missing": total_missing,
        "dup_count": dup_count,
        "dtypes_df": dtypes_df,
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen numérico de valores faltantes por columna.

    Devuelve un DataFrame con columnas [columna, nulos, % nulos], ordenado
    descendente por cantidad de nulos e incluyendo únicamente columnas con
    al menos un valor faltante. Vacío si el dataset no tiene nulos.
    """
    null_counts = df.isna().sum()
    summary = pd.DataFrame(
        {
            "columna": null_counts.index,
            "nulos": null_counts.to_numpy(dtype=int),
            "% nulos": (df.isna().mean() * 100).round(2).to_numpy(),
        }
    )
    summary = summary[summary["nulos"] > 0]
    return summary.sort_values("nulos", ascending=False, ignore_index=True)
