"""Lógica de limpieza e imputación de datos y arquitectura de comandos."""

import contextlib
from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from scipy import stats

CAST_BOOL_ALIASES = {
    "si": True,
    "sí": True,
    "yes": True,
    "true": True,
    "1": True,
    "no": False,
    "false": False,
    "0": False,
}

_IMPUT_ALL = ["media", "mediana", "moda", "ffill", "bfill", "constante", "drop_rows", "drop_col"]
_IMPUT_NON_NUMERIC = ["moda", "ffill", "bfill", "constante", "drop_rows", "drop_col"]


class TransformationCommand(ABC):
    """Clase base abstracta para todos los comandos de transformación."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Ejecuta la transformación sobre el DataFrame dado.

        Devuelve una tupla (nuevo_df, reporte_o_metadata).
        """

    @abstractmethod
    def to_code(self) -> str:
        """Emite el snippet de código Python reproducible exacto."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Mensaje descriptivo de la operación para historial y logs."""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.description}>"


class ImputeCommand(TransformationCommand):
    """Comando de imputación de valores nulos."""

    def __init__(self, col: str, strategy: str, constant: Any = None):
        self.col = col
        self.strategy = strategy
        self.constant = constant

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = df.copy()
        if self.strategy in ("media", "mediana"):
            ser = df[self.col]
            stat = ser.mean() if self.strategy == "media" else ser.median()
            if pd.api.types.is_integer_dtype(ser.dtype):
                # El estadístico puede ser fraccional: se ensancha el dtype entero a float.
                ser = ser.astype("float64")
            df[self.col] = ser.fillna(stat)
        elif self.strategy == "moda":
            mode = df[self.col].mode()
            if not mode.empty:
                df[self.col] = df[self.col].fillna(mode.iloc[0])
        elif self.strategy == "ffill":
            df[self.col] = df[self.col].ffill()
        elif self.strategy == "bfill":
            df[self.col] = df[self.col].bfill()
        elif self.strategy == "constante":
            df[self.col] = df[self.col].fillna(self.constant)
        elif self.strategy == "drop_rows":
            df = df.dropna(subset=[self.col])
        elif self.strategy == "drop_col":
            df = df.drop(columns=[self.col])
        else:
            raise ValueError(f"Estrategia de imputación desconocida: {self.strategy}")
        return df, {}

    def to_code(self) -> str:
        if self.strategy == "media":
            return (
                f"if pd.api.types.is_integer_dtype(df[{self.col!r}].dtype):\n"
                f"    df[{self.col!r}] = df[{self.col!r}].astype('float64')\n"
                f"df[{self.col!r}] = df[{self.col!r}].fillna(df[{self.col!r}].mean())"
            )
        elif self.strategy == "mediana":
            return (
                f"if pd.api.types.is_integer_dtype(df[{self.col!r}].dtype):\n"
                f"    df[{self.col!r}] = df[{self.col!r}].astype('float64')\n"
                f"df[{self.col!r}] = df[{self.col!r}].fillna(df[{self.col!r}].median())"
            )
        elif self.strategy == "moda":
            return f"df[{self.col!r}] = df[{self.col!r}].fillna(df[{self.col!r}].mode()[0])"
        elif self.strategy == "ffill":
            return f"df[{self.col!r}] = df[{self.col!r}].ffill()"
        elif self.strategy == "bfill":
            return f"df[{self.col!r}] = df[{self.col!r}].bfill()"
        elif self.strategy == "constante":
            return f"df[{self.col!r}] = df[{self.col!r}].fillna({self.constant!r})"
        elif self.strategy == "drop_rows":
            return f"df = df.dropna(subset=[{self.col!r}])"
        elif self.strategy == "drop_col":
            return f"df = df.drop(columns=[{self.col!r}])"
        else:
            raise ValueError(f"Estrategia desconocida para code: {self.strategy}")

    @property
    def description(self) -> str:
        return f"Imputación {self.strategy} en {self.col}"


class CastCommand(TransformationCommand):
    """Comando de conversión y casteo de tipos de datos."""

    def __init__(self, col: str, target_type: str):
        self.col = col
        self.target_type = target_type

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = df.copy()
        report = {"fractional": 0, "uninterpretable": 0}
        try:
            if self.target_type == "numeric":
                df[self.col] = pd.to_numeric(df[self.col], errors="coerce")
            elif self.target_type == "datetime":
                df[self.col] = pd.to_datetime(
                    df[self.col], errors="coerce", format="mixed", dayfirst=True
                )
            elif self.target_type == "string":
                df[self.col] = df[self.col].astype("string")
            elif self.target_type == "category":
                df[self.col] = df[self.col].astype("category")
            elif self.target_type == "int":
                ser = cast(pd.Series, pd.to_numeric(df[self.col], errors="coerce"))
                floor_ser = pd.Series(
                    np.floor(ser.to_numpy(dtype=float, na_value=np.nan)), index=ser.index
                )
                report["fractional"] = int((ser.notna() & (ser != floor_ser)).sum())
                df[self.col] = ser.round().astype("Int64")
            elif self.target_type == "float":
                df[self.col] = cast(pd.Series, pd.to_numeric(df[self.col], errors="coerce")).astype(
                    "float64"
                )
            elif self.target_type == "bool":
                mapped, n_unint = _map_bool_series(cast(pd.Series, df[self.col]))
                report["uninterpretable"] = n_unint
                df[self.col] = mapped
            else:
                raise ValueError(f"Tipo de destino no soportado: {self.target_type}")
        except Exception as e:
            raise ValueError(f"No se pudo convertir {self.col} a {self.target_type}: {e}") from e
        return df, report

    def to_code(self) -> str:
        if self.target_type == "numeric":
            return f"df[{self.col!r}] = pd.to_numeric(df[{self.col!r}], errors='coerce')"
        elif self.target_type == "datetime":
            return f"df[{self.col!r}] = pd.to_datetime(df[{self.col!r}], errors='coerce', format='mixed', dayfirst=True)"
        elif self.target_type == "string":
            return f"df[{self.col!r}] = df[{self.col!r}].astype('string')"
        elif self.target_type == "category":
            return f"df[{self.col!r}] = df[{self.col!r}].astype('category')"
        elif self.target_type == "int":
            return f"df[{self.col!r}] = pd.to_numeric(df[{self.col!r}], errors='coerce').round().astype('Int64')"
        elif self.target_type == "float":
            return f"df[{self.col!r}] = pd.to_numeric(df[{self.col!r}], errors='coerce').astype('float64')"
        elif self.target_type == "bool":
            aliases_literal = ", ".join(f"{k!r}: {v}" for k, v in CAST_BOOL_ALIASES.items())
            return (
                "# Cast bool: mapper explícito (aliases case/space-insensitive + nativo bool/0-1) -> pd.NA\n"
                f"_BOOL_ALIASES = {{{aliases_literal}}}\n"
                "def _cast_bool(v):\n"
                "    if pd.isna(v):\n"
                "        return pd.NA\n"
                "    if isinstance(v, (bool, np.bool_)):\n"
                "        return bool(v)\n"
                "    if isinstance(v, (int, float, np.integer, np.floating)):\n"
                "        fv = float(v)\n"
                "        if fv == 1.0:\n"
                "            return True\n"
                "        if fv == 0.0:\n"
                "            return False\n"
                "        return pd.NA\n"
                "    if isinstance(v, str):\n"
                "        return _BOOL_ALIASES.get(v.strip().lower(), pd.NA)\n"
                "    return pd.NA\n"
                f"df[{self.col!r}] = df[{self.col!r}].map(_cast_bool).astype('boolean')"
            )
        else:
            raise ValueError(f"Tipo desconocido para code: {self.target_type}")

    @property
    def description(self) -> str:
        return f"Cast {self.col} -> {self.target_type}"


class OutlierFilterCommand(TransformationCommand):
    """Comando de filtrado de outliers (IQR / Z-Score)."""

    def __init__(self, col: str, method: str = "IQR", factor_or_thresh: float = 1.5):
        self.col = col
        self.method = method
        self.factor_or_thresh = factor_or_thresh

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        method_norm = self.method.upper().replace("-", "").replace("_", "").replace(" ", "")
        col_ser = cast(pd.Series, df[self.col])
        if method_norm == "IQR":
            mask = detect_outliers_iqr(col_ser, factor=self.factor_or_thresh)
        elif method_norm in ("ZSCORE", "Z"):
            mask = detect_outliers_zscore(col_ser, threshold=self.factor_or_thresh)
        else:
            raise ValueError(f"Método de outlier no soportado: {self.method}")

        n_out = int(mask.sum())
        new_df = df.loc[~mask].copy()
        report = {"outliers_detected": n_out, "mask": mask}
        return new_df, report

    def to_code(self) -> str:
        method_norm = self.method.upper().replace("-", "").replace("_", "").replace(" ", "")
        if method_norm == "IQR":
            return (
                f"from src.cleaning import detect_outliers_iqr\n"
                f"mask = detect_outliers_iqr(df[{self.col!r}], {self.factor_or_thresh})\n"
                f"df = df.loc[~mask]"
            )
        elif method_norm in ("ZSCORE", "Z"):
            return (
                f"from src.cleaning import detect_outliers_zscore\n"
                f"mask = detect_outliers_zscore(df[{self.col!r}], {self.factor_or_thresh})\n"
                f"df = df.loc[~mask]"
            )
        else:
            raise ValueError(f"Método desconocido para code: {self.method}")

    @property
    def description(self) -> str:
        return f"Filtrado outliers {self.method} {self.col}"


class DropDuplicatesCommand(TransformationCommand):
    """Comando de eliminación de filas duplicadas."""

    def __init__(
        self,
        subset: list[str] | None = None,
        keep: Literal["first", "last", False] = "first",
    ):
        self.subset = subset
        self.keep: Literal["first", "last", False] = keep

    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        keep_arg = cast(Literal["first", "last", False], self.keep)
        new_df = df.drop_duplicates(subset=self.subset, keep=keep_arg)
        removed = df.shape[0] - new_df.shape[0]
        return new_df, {"removed": removed}

    def to_code(self) -> str:
        if self.subset is None and self.keep == "first":
            return "df = df.drop_duplicates()"
        elif self.subset is None:
            return f"df = df.drop_duplicates(keep={self.keep!r})"
        else:
            return f"df = df.drop_duplicates(subset={self.subset!r}, keep={self.keep!r})"

    @property
    def description(self) -> str:
        return "Eliminar duplicados"


def impute_column(
    df: pd.DataFrame, col: str, strategy: str, constant=None, copy: bool = True
) -> pd.DataFrame:
    """Imputa valores nulos en una columna según la estrategia indicada."""
    cmd = ImputeCommand(col=col, strategy=strategy, constant=constant)
    new_df, _ = cmd.apply(df)
    return new_df


def strategies_for_dtype(dtype) -> list:
    """Devuelve las estrategias de imputación aplicables a un dtype.
    Excluye media/mediana para tipos no numéricos (incluido bool)."""
    if isinstance(dtype, type) and not isinstance(dtype, np.dtype):
        with contextlib.suppress(TypeError):
            dtype = np.dtype(dtype)
    is_numeric = pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
    return list(_IMPUT_ALL) if is_numeric else list(_IMPUT_NON_NUMERIC)


def _map_bool_series(series: pd.Series):
    """Convierte una serie a booleano usando aliases de texto y valores nativos.
    Los valores no interpretables se convierten en pd.NA.
    Devuelve (serie_convertida, cantidad_no_interpretable)."""
    out = []
    n_unint = 0
    for v in series:
        if pd.isna(v):
            out.append(pd.NA)
        elif isinstance(v, (bool, np.bool_)):
            out.append(bool(v))
        elif isinstance(v, (int, float, np.integer, np.floating)):
            f = float(v)
            if f == 1.0:
                out.append(True)
            elif f == 0.0:
                out.append(False)
            else:
                out.append(pd.NA)
                n_unint += 1
        elif isinstance(v, str):
            key = v.strip().lower()
            if key in CAST_BOOL_ALIASES:
                out.append(CAST_BOOL_ALIASES[key])
            else:
                out.append(pd.NA)
                n_unint += 1
        else:
            out.append(pd.NA)
            n_unint += 1
    return pd.Series(out, index=series.index, dtype="boolean"), n_unint


def cast_column(df: pd.DataFrame, col: str, target_type: str):
    """Convierte una columna al tipo indicado.
    Devuelve (df_modificado, reporte) con el conteo de valores afectados."""
    cmd = CastCommand(col=col, target_type=target_type)
    return cmd.apply(df)


def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Genera una máscara booleana con los outliers detectados por el método IQR.
    Los NaN nunca se marcan como outliers; columnas constantes devuelven ceros."""
    if series.empty or series.isna().all():
        return pd.Series(False, index=series.index, dtype=bool)

    valid = cast(pd.Series, pd.to_numeric(series, errors="coerce"))
    if valid.notna().sum() == 0:
        return pd.Series(False, index=series.index, dtype=bool)

    valid_values = cast(pd.Series, valid.dropna())
    q1 = float(valid_values.quantile(0.25))
    q3 = float(valid_values.quantile(0.75))
    iqr = q3 - q1
    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index, dtype=bool)

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = pd.Series(False, index=series.index, dtype=bool)
    mask.loc[valid.index] = (valid < lower) | (valid > upper)
    return mask


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Genera una máscara booleana con los outliers detectados por Z-score.
    Columnas constantes o vacías devuelven una máscara de ceros."""
    mask = pd.Series(False, index=series.index, dtype=bool)
    valid = cast(pd.Series, pd.to_numeric(series, errors="coerce")).dropna()
    if len(valid) == 0 or valid.nunique() <= 1:
        return mask
    z = np.abs(np.asarray(stats.zscore(valid.to_numpy(dtype=float)), dtype=float))
    mask.loc[valid.index] = z > threshold
    return mask
