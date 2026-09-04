"""Visualizaciones univariadas y bivariadas con Matplotlib / Seaborn.

Todas las funciones devuelven objetos matplotlib.Figure y se regeneran en
cada rerun (no usan caché porque Figure no es serializable con pickle)."""

from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

HUE_MAX_CATEGORIES = 20
TOP_N_SLIDER_MIN = 3
TOP_N_SLIDER_DEFAULT = 10
TOP_N_MAX_CATEGORIES = 30


def topn_slider_config(n_cats: int) -> tuple[int, int, int]:
    """Deriva (min_value, max_value, value) coherentes para el slider Top N.

    Con cardinalidad baja (0, 1 o 2 categorías: columnas constantes o todo-NaN)
    garantiza igualmente min_value < max_value y min_value <= value <= max_value,
    de modo que el widget nunca falle con StreamlitAPIException.
    """
    effective = max(int(n_cats), 0)
    max_value = max(TOP_N_SLIDER_MIN + 1, min(effective, TOP_N_MAX_CATEGORIES))
    default_value = min(TOP_N_SLIDER_DEFAULT, max_value)
    return TOP_N_SLIDER_MIN, max_value, default_value


def resolve_topn_state(
    persisted_value: int | None,
    min_value: int,
    max_value: int,
) -> tuple[int | None, bool]:
    """Resuelve el valor persistido del slider Top N frente a los límites vigentes.

    Función pura que encapsula la decisión de clamp/purga: si el estado keyed
    quedó fuera de los nuevos límites (p. ej. al cambiar de columna), debe
    eliminarse antes de recrear el widget; si está sano o ausente, se respeta.

    Args:
        persisted_value: Valor actual en session_state bajo la key del slider
            (None si la key aún no existe).
        min_value: Límite inferior vigente del slider.
        max_value: Límite superior vigente del slider.

    Returns:
        Tupla `(value, purge)` donde `value` es el valor efectivo a usar
        (el persistido si es coherente; None si debe usarse el default) y
        `purge` indica si el estado persistido quedó fuera de límites y debe
        eliminarse de session_state.
    """
    if persisted_value is None:
        return None, False
    if not min_value <= persisted_value <= max_value:
        return None, True
    return int(persisted_value), False


THEME_PALETTES = {
    "light": {
        "fig_bg": "#FFFFFF",
        "ax_bg": "#F8FAFC",
        "text": "#0F172A",
        "muted_text": "#475569",
        "grid": "#E2E8F0",
        "spine": "#CBD5E1",
        "primary": "#0284C7",
        "secondary": "#10B981",
        "accent": "#8B5CF6",
        "box": "#0284C7",
        "categorical": "deep",
        "heatmap_cmap": "coolwarm",
    },
    "dark": {
        "fig_bg": "#0F172A",
        "ax_bg": "#1E293B",
        "text": "#F1F5F9",
        "muted_text": "#94A3B8",
        "grid": "#334155",
        "spine": "#475569",
        "primary": "#38BDF8",
        "secondary": "#34D399",
        "accent": "#A78BFA",
        "box": "#38BDF8",
        "categorical": "bright",
        "heatmap_cmap": "coolwarm",
    },
}


def apply_plot_theme(
    fig: Figure,
    axes: list[Axes] | Axes | np.ndarray,
    dark_mode: bool = False,
) -> None:
    """Aplica configuraciones armónicas de color, ejes y grillas según el modo claro u oscuro."""
    theme = THEME_PALETTES["dark" if dark_mode else "light"]
    fig.patch.set_facecolor(theme["fig_bg"])

    if isinstance(axes, np.ndarray):
        axes_list = list(axes.flat)
    elif isinstance(axes, list):
        axes_list = axes
    else:
        axes_list = [axes]

    for ax in axes_list:
        ax.set_facecolor(theme["ax_bg"])
        ax.tick_params(colors=theme["muted_text"], which="both", labelsize=9)
        ax.xaxis.label.set_color(theme["text"])
        ax.yaxis.label.set_color(theme["text"])
        ax.title.set_color(theme["text"])
        for spine in ax.spines.values():
            spine.set_color(theme["spine"])
        ax.grid(True, color=theme["grid"], linestyle="--", linewidth=0.5, alpha=0.7)

        legend = ax.get_legend()
        if legend:
            legend.get_frame().set_facecolor(theme["ax_bg"])
            legend.get_frame().set_edgecolor(theme["spine"])
            for text in legend.get_texts():
                text.set_color(theme["text"])
            if legend.get_title():
                legend.get_title().set_color(theme["text"])


def plot_univariate_numeric(
    _df: pd.DataFrame, col: str, dark_mode: bool = False
) -> tuple[Figure, int]:
    """Histograma + KDE (eje X acotado al P99.9) y boxplot de una columna numérica.
    Devuelve (figura, cantidad_de_valores_fuera_de_rango)."""
    theme = THEME_PALETTES["dark" if dark_mode else "light"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

    values = cast(pd.Series, pd.to_numeric(_df[col], errors="coerce")).dropna()
    q0 = float(np.nanquantile(values, 0.0)) if len(values) else 0.0
    q99 = float(np.nanquantile(values, 0.999)) if len(values) else 0.0
    n_out_of_range = int((values > q99).sum()) if len(values) else 0

    plot_values = (
        values if len(values) <= 50_000 else cast(pd.Series, values.sample(50_000, random_state=42))
    )

    sns.histplot(
        x=plot_values, kde=True, ax=axes[0], color=theme["primary"], edgecolor=theme["spine"]
    )
    axes[0].set_xlim(q0, q99)
    axes[0].set_title(f"Distribución de {col} - Histograma + KDE")

    axes[1].boxplot(
        plot_values,
        orientation="vertical",
        patch_artist=True,
        boxprops=dict(facecolor=theme["box"], color=theme["spine"]),
        whiskerprops=dict(color=theme["spine"]),
        capprops=dict(color=theme["spine"]),
        medianprops=dict(color="#F59E0B" if not dark_mode else "#FBBF24", linewidth=2.0),
        flierprops=dict(markeredgecolor=theme["muted_text"], markersize=4),
    )
    axes[1].set_title(f"Boxplot {col}")

    apply_plot_theme(fig, axes, dark_mode=dark_mode)
    plt.tight_layout()
    return fig, n_out_of_range


def plot_univariate_temporal(_df: pd.DataFrame, col: str, dark_mode: bool = False) -> Figure:
    """Histograma de distribución temporal para una columna datetime."""
    theme = THEME_PALETTES["dark" if dark_mode else "light"]
    dates = pd.to_datetime(_df[col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(x=dates, ax=ax, color=theme["primary"], edgecolor=theme["spine"])
    ax.set_title(f"Distribución temporal de {col} ({len(dates)} fechas válidas)")
    apply_plot_theme(fig, ax, dark_mode=dark_mode)
    plt.tight_layout()
    return fig


def plot_univariate_categorical(_df: pd.DataFrame, col: str, dark_mode: bool = False) -> Figure:
    theme = THEME_PALETTES["dark" if dark_mode else "light"]
    fig, ax = plt.subplots(figsize=(10, 4))
    col_ser = cast(pd.Series, _df[col])
    vc = cast(pd.Series, col_ser.value_counts())
    order = list(cast(pd.Index, vc.index)[:15])
    sns.countplot(
        data=_df,
        y=col,
        order=order,
        hue=col,
        palette=theme["categorical"],
        legend=False,
        ax=ax,
        edgecolor=theme["spine"],
        linewidth=0.5,
    )
    ax.set_title(f"Frecuencia - {col} (Top 15)")
    apply_plot_theme(fig, ax, dark_mode=dark_mode)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    _df: pd.DataFrame,
    method: Literal["pearson", "spearman", "kendall"] | str = "pearson",
    dark_mode: bool = False,
) -> Figure | None:
    """Mapa de calor de correlación entre columnas numéricas.
    Devuelve None si hay menos de 2 columnas numéricas."""
    num_df = _df.select_dtypes(include=[np.number])
    n_cols = num_df.shape[1]
    if n_cols < 2:
        return None
    method_arg = cast(Literal["pearson", "spearman", "kendall"], method)
    corr = num_df.corr(method=method_arg, numeric_only=True)

    cell_width = 0.55
    fig_width = max(8, min(28, n_cols * cell_width))
    fig_height = max(6, min(22, n_cols * cell_width * 0.7))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    annot_font = max(5, 9 - (n_cols - 8) // 4)
    theme = THEME_PALETTES["dark" if dark_mode else "light"]

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=theme["heatmap_cmap"],
        center=0,
        ax=ax,
        square=False,
        linewidths=0.3,
        linecolor=theme["fig_bg"],
        annot_kws={"size": annot_font},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Correlación {method.title()} ({n_cols} columnas)")
    ax.tick_params(axis="x", rotation=45, labelsize=max(6, 9 - (n_cols - 8) // 5))
    ax.tick_params(axis="y", rotation=0, labelsize=max(6, 9 - (n_cols - 8) // 5))
    apply_plot_theme(fig, ax, dark_mode=dark_mode)

    if ax.collections and ax.collections[0].colorbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors=theme["muted_text"], labelsize=8)
        cbar.outline.set_edgecolor(theme["spine"])

    plt.tight_layout()
    return fig


def plot_scatter(
    _df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    dark_mode: bool = False,
) -> Figure:
    """Scatterplot entre dos columnas, con agrupación opcional por color (hue)."""
    if hue and hue != "Ninguno":
        hue_ser = cast(pd.Series, _df[hue])
        n_cat = int(hue_ser.nunique(dropna=True))
        if n_cat > HUE_MAX_CATEGORIES:
            raise ValueError(
                f"hue '{hue}' tiene {n_cat} categorías (> {HUE_MAX_CATEGORIES}): "
                f"leyenda no ilegible. Usa otra columna o 'Ninguno'."
            )
        if bool(hue_ser.isna().any()):
            n_null = int(hue_ser.isna().sum())
            raise ValueError(
                f"hue '{hue}' contiene {n_null} valores nulos: "
                "no se descartan filas en silencio. Imputa o usa 'Ninguno'."
            )

    plot_df = _df
    sample_notice = ""
    if len(_df) > 10_000:
        plot_df = _df.sample(n=10_000, random_state=42)
        sample_notice = " (muestra de 10,000 puntos)"

    fig, ax = plt.subplots(figsize=(8, 5))
    theme = THEME_PALETTES["dark" if dark_mode else "light"]

    if hue and hue != "Ninguno":
        sns.scatterplot(
            data=plot_df,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            palette=theme["categorical"],
            alpha=0.8,
            s=70,
            edgecolor=theme["fig_bg"],
            linewidth=0.5,
        )
    else:
        sns.scatterplot(
            data=plot_df,
            x=x,
            y=y,
            ax=ax,
            color=theme["primary"],
            alpha=0.8,
            s=70,
            edgecolor=theme["fig_bg"],
            linewidth=0.5,
        )
    ax.set_title(f"{y} vs {x}{sample_notice}")
    apply_plot_theme(fig, ax, dark_mode=dark_mode)
    plt.tight_layout()
    return fig


def plot_bivariate_bar(
    _df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    *,
    aggfunc: str = "mean",
    hue_col: str | None = None,
    top_n: int = 15,
    dark_mode: bool = False,
) -> Figure:
    """Gráfico de barras bivariada: una columna categórica (eje X) contra
    una numérica (eje Y) aplicando una función de agregación.
    Muestra solo las top_n categorías con mayor valor agregado."""
    if hue_col and hue_col != "Ninguno":
        hue_ser = cast(pd.Series, _df[hue_col])
        n_cat = int(hue_ser.nunique(dropna=True))
        if n_cat > HUE_MAX_CATEGORIES:
            raise ValueError(
                f"hue '{hue_col}' tiene {n_cat} categorías (> {HUE_MAX_CATEGORIES}): "
                f"leyenda no ilegible. Usa otra columna o 'Ninguno'."
            )

    if hue_col and hue_col != "Ninguno":
        agg = (
            _df.dropna(subset=[cat_col, num_col])
            .groupby([cat_col, hue_col], observed=True)[num_col]
            .agg(aggfunc)
            .reset_index()
        )
        cat_sums = cast(pd.Series, agg.groupby(cat_col, observed=True)[num_col].sum())
        top_cats = list(cat_sums.sort_values(ascending=False).head(top_n).index)
        agg = agg[agg[cat_col].isin(top_cats)]
        n_cats = len(top_cats)
    else:
        agg = (
            _df.dropna(subset=[cat_col, num_col])
            .groupby(cat_col, observed=True)[num_col]
            .agg(aggfunc)
            .reset_index()
            .nlargest(top_n, num_col)
        )
        n_cats = int(cast(pd.Series, agg[cat_col]).nunique())

    if agg.empty:
        raise ValueError(
            f"No hay datos válidos para '{cat_col}' + '{num_col}'. "
            "Verifica que la categórica tenga valores y la numérica no sea todo NaN."
        )

    fig_width = max(8, min(14, n_cats * 0.8 if n_cats > 5 else 8))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    theme = THEME_PALETTES["dark" if dark_mode else "light"]

    plot_data = cast(pd.DataFrame, agg)
    if hue_col and hue_col != "Ninguno":
        sns.barplot(
            data=plot_data,
            x=cat_col,
            y=num_col,
            hue=hue_col,
            palette=theme["categorical"],
            ax=ax,
            edgecolor=theme["spine"],
            linewidth=0.5,
        )
    else:
        sns.barplot(
            data=plot_data,
            x=cat_col,
            y=num_col,
            color=theme["primary"],
            ax=ax,
            edgecolor=theme["spine"],
            linewidth=0.5,
        )

    agg_cat_ser = cast(pd.Series, agg[cat_col])
    max_label_len = int(agg_cat_ser.astype(str).str.len().max()) if not agg.empty else 0
    if n_cats > 6 or max_label_len > 10:
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    else:
        ax.tick_params(axis="x", labelsize=9)

    ax.set_title(
        f"{aggfunc.upper()} de {num_col} por {cat_col}"
        + (f" (Top {top_n})" if n_cats >= top_n else "")
    )
    ax.set_xlabel(cat_col, fontsize=10)
    ax.set_ylabel(f"{aggfunc}({num_col})", fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

    apply_plot_theme(fig, ax, dark_mode=dark_mode)
    plt.tight_layout()
    return fig
