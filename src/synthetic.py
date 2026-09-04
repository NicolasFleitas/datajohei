"""Generador de dataset sintético con problemas comunes de calidad de datos."""

import numpy as np
import pandas as pd


def generate_messy_dataset(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Genera un dataset de empleados con problemas de calidad inyectados:
    nulos, outliers, duplicados, fechas inválidas y tipos mixtos.

    La generación usa un Generador de NumPy local (`np.random.default_rng`),
    sin mutar el estado global del proceso. Con la misma `seed` el resultado
    es determinista y reproducible.

    Aritmética de filas: el resultado final contiene exactamente `n` filas base
    más las 35 filas duplicadas que se agregan al final (muestreadas sin
    reposición sobre las `n` base), barajadas antes de devolverse; es decir,
    `n + 35` filas en total (por defecto 200 + 35 = 235).

    Los conteos canónicos de inyección (35/55/30/40/25 nulos, 12 outliers,
    25 fechas inválidas) se aplican completos cuando `n` alcanza para
    indexarlos sin reposición; si no, se recortan a `n` (requiere `n >= 35`
    para el bloque de duplicados).

    Args:
        n: Cantidad de filas base solicitadas (sin cap silencioso).
        seed: Semilla del generador local (por defecto 42).

    Returns:
        DataFrame con las columnas canónicas y problemas de calidad inyectados.
    """
    rng = np.random.default_rng(seed)
    base_n = max(int(n), 0)

    edades = rng.normal(35, 10, base_n)
    salarios = rng.lognormal(mean=10.2, sigma=0.6, size=base_n)
    departamentos = rng.choice(
        ["Ventas", "TI", "RRHH", "Marketing", "Finanzas"],
        size=base_n,
        p=[0.25, 0.25, 0.20, 0.15, 0.15],
    )
    fechas = pd.date_range("2019-01-01", "2024-12-31", periods=base_n)
    scores = rng.beta(2, 5, base_n) * 100
    ciudades = rng.choice(
        ["Asunción", "Lambaré", "Luque", "Fernando de la Mora", "San Lorenzo"], base_n
    )
    antiguedad = rng.exponential(scale=3, size=base_n)

    df = pd.DataFrame(
        {
            "id_empleado": np.arange(1000, 1000 + base_n),
            "edad": edades.round(),
            "salario_mensual": salarios.round(2),
            "departamento": departamentos,
            "fecha_ingreso": rng.choice(fechas, size=base_n),
            "score_desempeno": scores.round(1),
            "ciudad": ciudades,
            "anios_experiencia": antiguedad.round(1),
            "modalidad": rng.choice(["Remoto", "Presencial", "Híbrido"], base_n),
        }
    )

    for col, n_nulls in [
        ("edad", 35),
        ("salario_mensual", 55),
        ("departamento", 30),
        ("score_desempeno", 40),
        ("anios_experiencia", 25),
    ]:
        idx = rng.choice(df.index.to_numpy(), min(n_nulls, base_n), replace=False)
        df.loc[idx, col] = np.nan

    outlier_idx = rng.choice(df.index.to_numpy(), min(12, base_n), replace=False)
    df.loc[outlier_idx[:6], "salario_mensual"] *= rng.uniform(4, 8, len(outlier_idx[:6]))
    df.loc[outlier_idx[6:9], "edad"] = rng.choice([16, 82, 90], len(outlier_idx[6:9]))
    df.loc[outlier_idx[9:], "score_desempeno"] = rng.choice([5, 99.9], len(outlier_idx[9:]))

    df["fecha_ingreso"] = df["fecha_ingreso"].astype(str)
    messy_dates_idx = rng.choice(df.index.to_numpy(), min(25, base_n), replace=False)
    df.loc[messy_dates_idx[:10], "fecha_ingreso"] = "31/02/2021"
    df.loc[messy_dates_idx[10:15], "fecha_ingreso"] = "not_a_date"
    df.loc[messy_dates_idx[15:], "fecha_ingreso"] = np.nan

    dup = df.sample(min(35, base_n), random_state=seed).copy()
    df = pd.concat([df, dup], ignore_index=True)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df["id_empleado"] = df["id_empleado"].astype(str) + "-" + df["ciudad"].str[:2]

    return df
