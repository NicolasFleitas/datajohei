"""Tests del generador sintético: determinismo, honoring de n y pureza del RNG global."""

import numpy as np
import pandas as pd

from src.synthetic import generate_messy_dataset

EXPECTED_COLUMNS = [
    "id_empleado",
    "edad",
    "salario_mensual",
    "departamento",
    "fecha_ingreso",
    "score_desempeno",
    "ciudad",
    "anios_experiencia",
    "modalidad",
]


def test_same_seed_is_deterministic():
    """(a) Misma semilla produce DataFrames idénticos (contrato de reproducibilidad)."""
    df_a = generate_messy_dataset(seed=42)
    df_b = generate_messy_dataset(seed=42)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_different_seed_produces_different_data():
    """Semillas distintas producen valores distintos con la misma estructura."""
    df_a = generate_messy_dataset(seed=42)
    df_b = generate_messy_dataset(seed=43)
    assert list(df_a.columns) == list(df_b.columns)
    assert not df_a.equals(df_b)


def test_default_row_count_is_235():
    """Default n=200 sin cap + 35 duplicados = 235 filas exactas."""
    assert len(generate_messy_dataset()) == 235


def test_n_is_honored_without_silent_cap():
    """(b) n=500 ya no se recorta silenciosamente a 200: 500 base + 35 duplicados."""
    assert len(generate_messy_dataset(n=500)) == 535


def test_small_n_yields_at_least_n_rows():
    """(b) Con n=50 el resultado conserva al menos las filas solicitadas."""
    small = generate_messy_dataset(n=50)
    assert len(small) >= 50
    assert list(small.columns) == EXPECTED_COLUMNS


def test_structure_preserved_for_defaults():
    """Estructura canónica: columnas, nulos por columna y patrones de fechas sucias."""
    df = generate_messy_dataset()
    assert list(df.columns) == EXPECTED_COLUMNS
    # Conteos mínimos de nulos: los canónicos (35/55/30/40/25/10) solo pueden
    # crecer por las 35 filas duplicadas, nunca reducirse.
    nulls = df.isna().sum()
    for col, min_nulls in [
        ("edad", 35),
        ("salario_mensual", 55),
        ("departamento", 30),
        ("score_desempeno", 40),
        ("anios_experiencia", 25),
        ("fecha_ingreso", 10),
    ]:
        assert int(nulls[col]) >= min_nulls, col
    # Patrones de fechas inválidas presentes.
    fechas = df["fecha_ingreso"]
    assert int((fechas == "31/02/2021").sum()) >= 10
    assert int((fechas == "not_a_date").sum()) >= 5
    # Duplicados inyectados e id compuesto.
    assert int(df.duplicated().sum()) > 0
    assert df["id_empleado"].str.contains("-").all()


def test_generator_does_not_mutate_global_numpy_rng():
    """(c) La generación no avanza ni muta el RNG global de NumPy.

    Se fija la secuencia global, se registra su continuación esperada y se
    verifica que tras generar un dataset la secuencia global continúa igual.
    """
    np.random.seed(123)
    np.random.random()  # consume un valor; la continuación queda determinada
    expected_next = np.random.random()

    np.random.seed(123)
    np.random.random()
    generate_messy_dataset(n=50, seed=7)
    actual_next = np.random.random()

    assert actual_next == expected_next
