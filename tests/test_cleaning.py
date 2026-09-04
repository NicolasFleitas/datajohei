"""Tests unitarios del núcleo de limpieza."""

from typing import cast

import numpy as np
import pandas as pd

from src.cleaning import (
    CAST_BOOL_ALIASES,
    cast_column,
    detect_outliers_iqr,
    detect_outliers_zscore,
    strategies_for_dtype,
)
from src.synthetic import generate_messy_dataset


# cast int redondea fraccionarios
def test_cast_int_rounds_fractions():
    df = pd.DataFrame({"salario_mensual": ["46233.62", "100.0", None, "notnum"]})
    df_out, report = cast_column(df, "salario_mensual", "int")

    assert df_out["salario_mensual"].dtype == pd.Int64Dtype()
    assert df_out["salario_mensual"].tolist() == [46234, 100, pd.NA, pd.NA]
    assert report["fractional"] == 1  # 46233.62 → 46234; 100.0 no es fraccional
    assert report["uninterpretable"] == 0


def test_cast_int_non_numeric_to_na():
    df = pd.DataFrame({"col": ["0", "abc", "xyz"]})
    df_out, report = cast_column(df, "col", "int")

    assert df_out["col"].dtype == pd.Int64Dtype()
    assert df_out["col"].isna().sum() == 2  # abc, xyz → pd.NA
    assert df_out["col"].iloc[0] == 0
    assert report["fractional"] == 0


# cast bool con mapper explícito
def test_cast_bool_aliases():
    df = pd.DataFrame({"flag": ["Sí", "  no ", "TRUE", "1", "0", True, False, 1.0, 0.0]})
    df_out, report = cast_column(df, "flag", "bool")

    assert str(df_out["flag"].dtype) == "boolean"
    assert df_out["flag"].tolist() == [True, False, True, True, False, True, False, True, False]
    assert report["uninterpretable"] == 0
    assert report["fractional"] == 0


def test_cast_bool_uninterpretable_to_na():
    df = pd.DataFrame({"mod": ["quizá", "si", "no"]})
    df_out, report = cast_column(df, "mod", "bool")

    assert str(df_out["mod"].dtype) == "boolean"
    assert df_out["mod"].tolist() == [pd.NA, True, False]
    assert report["uninterpretable"] == 1  # "quizá" → pd.NA, nunca error


def test_CAST_BOOL_ALIASES_key_set():
    # case/space-insensitive: el mapper usa str(v).strip().lower()
    assert CAST_BOOL_ALIASES == {
        "si": True,
        "sí": True,
        "yes": True,
        "true": True,
        "1": True,
        "no": False,
        "false": False,
        "0": False,
    }
    assert all(k == k.strip().lower() for k in CAST_BOOL_ALIASES)


# estrategias por dtype
def test_strategies_for_dtype_only_numeric_offers_mean_median():
    numeric = strategies_for_dtype(pd.Series([1.0, 2.0, 3.0]).dtype)
    assert "media" in numeric and "mediana" in numeric

    for dtype in [
        pd.Series(["a", None]).dtype,  # str / StringDtype
        pd.CategoricalDtype(),
        pd.BooleanDtype(),
        np.dtype("datetime64[s]"),
    ]:
        strat = strategies_for_dtype(dtype)
        assert "media" not in strat, f"media no debe ofrecerse para {dtype}"
        assert "mediana" not in strat, f"mediana no debe ofrecerse para {dtype}"

    # drop_rows/drop_col son universales
    for dtype in [
        pd.Series([1.0, 2.0]).dtype,
        pd.Series(["a", "b"]).dtype,
        pd.BooleanDtype(),
    ]:
        strat = strategies_for_dtype(dtype)
        assert "drop_rows" in strat and "drop_col" in strat


# JD-04: Int64 anulable sigue ofreciendo media/mediana (se ensancha el dtype, no se oculta)
def test_strategies_for_dtype_int64_nullable_keeps_media_mediana():
    strat = strategies_for_dtype(pd.Int64Dtype())
    assert "media" in strat
    assert "mediana" in strat


# JD-03: porcentaje de outliers tolera datasets sin filas
def test_format_outlier_pct_handles_zero_rows():
    from src.views.cleaning_view import _format_outlier_pct

    assert _format_outlier_pct(0, 0) == "N/A"
    assert _format_outlier_pct(3, 0) == "N/A"
    assert _format_outlier_pct(5, 200) == "2.5%"


# IQR robusto
def test_detect_outliers_iqr_nan_aware_aligned():
    df = generate_messy_dataset()
    mask = detect_outliers_iqr(cast(pd.Series, df["salario_mensual"]))

    assert mask.index.equals(df.index)  # alineada
    assert mask.dtype == bool
    assert not mask[df["salario_mensual"].isna()].any()  # NaN nunca outlier
    # seed 42: la cola lognatural marca 16 outliers IQR en total
    assert mask.sum() == 7  # IQR 1.5 en sintético (seed 42, default_rng determinista)


def test_detect_outliers_iqr_constant_zero():
    mask = detect_outliers_iqr(pd.Series([5.0] * 10))
    assert mask.sum() == 0
    assert mask.dtype == bool


def test_detect_outliers_iqr_all_nan_zero():
    s = pd.Series([np.nan] * 7, index=[3, 5, 7, 9, 11, 13, 15])
    mask = detect_outliers_iqr(s)  # sin excepción
    assert mask.sum() == 0
    assert mask.index.tolist() == [3, 5, 7, 9, 11, 13, 15]


# Z-score index-safe
def test_detect_outliers_zscore_preserves_nan_and_index():
    df = generate_messy_dataset()
    mask = detect_outliers_zscore(cast(pd.Series, df["salario_mensual"]))

    assert mask.index.equals(df.index)
    assert mask.dtype == bool
    assert not mask[df["salario_mensual"].isna()].any()  # NaN nunca marcados
    assert mask.sum() == 4  # z > 3.0 en sintético (seed 42, default_rng determinista)


def test_detect_outliers_zscore_all_nan_zero():
    s = pd.Series([np.nan] * 4, index=[1, 2, 3, 4])
    mask = detect_outliers_zscore(s)  # no excepción
    assert mask.sum() == 0
    assert mask.dtype == bool


def test_detect_outliers_zscore_constant_zero():
    mask = detect_outliers_zscore(pd.Series([7.0, 7.0, 7.0]))
    assert mask.sum() == 0
    assert mask.dtype == bool
