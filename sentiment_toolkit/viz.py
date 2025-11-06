# sentiment_toolkit/viz.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union

def _as_dataframe(obj: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=obj.name or "value")
    return obj

def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    return df.sort_index()

def _auto_baseline(values: pd.Series) -> Optional[float]:
    if values.empty or values.dropna().empty:
        return None
    v0 = values.dropna().iloc[0]
    if np.isfinite(v0) and np.allclose(v0, 1, atol=0.05):
        return 1.0
    m = values.mean(skipna=True)
    if np.isfinite(m) and np.allclose(m, 0, atol=0.05):
        return 0.0
    return None

def plot_each_series(
    df,
    baseline: "auto|float|None" = "auto",
    figsize=(14, 4),
    linewidth=2.0,
    grid=False,
    title_prefix="",
    ylabel=None,
    show=True,
    savepath=None,
):
    df = _prep_df(_as_dataframe(df)).dropna(axis=1, how="all")
    if df.shape[1] == 0:
        raise ValueError("No non-empty columns to plot.")

    figs_axes = []
    for col in df.columns:
        series = df[col]
        # resolve baseline per-series
        if baseline == "auto" or baseline is None:
            bline = _auto_baseline(series)
            if bline is None:
                bline = 0.0  # fallback so a dashed line always appears
        else:
            bline = baseline

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(series.index, series.values, linewidth=linewidth)
        ax.axhline(bline, linestyle="--", linewidth=1)   # always draw
        if grid:
            ax.grid(alpha=0.25)

        ttl = f"{title_prefix}{col}" if title_prefix else col
        ax.set_title(ttl)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.margins(x=0)
        fig.tight_layout()

        if savepath:
            safe_col = str(col).replace("/", "_")
            fig.savefig(f"{savepath}__{safe_col}.png", dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        figs_axes.append((fig, ax))
    return figs_axes

def _ensure_one_series(obj, role: str) -> pd.Series:
    """
    Accept a pandas Series or a 1-column DataFrame and return a named Series.
    """
    df = _as_dataframe(obj)
    if df.shape[1] != 1:
        raise ValueError(f"{role} must be a pandas Series or a 1-column DataFrame")
    s = df.iloc[:, 0]
    s.name = s.name or role
    # ensure datetime index & sorted
    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
    s = s.sort_index()
    return s

def rebase_series(s: pd.Series, base: float = 100.0) -> pd.Series:
    s = s.copy()
    non_na = s.dropna()
    if non_na.empty:
        return s
    first = non_na.iloc[0]
    if not np.isfinite(first) or first == 0:
        return s
    out = s / first * base
    out.name = s.name
    return out

def fixed_bands(levels: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
    # [(low, high, label)]
    return [(float(lo), float(hi), str(lbl)) for lo, hi, lbl in levels]

def bands_with_style(
    bands: List[Tuple[float, float, str]],
    label_styles: dict = None,
    default_alpha: float = 0.12,
) -> List[Tuple[float, float, str, float, str]]:
    """
    Map [(low, high, label)] -> [(low, high, color, alpha, label)]
    """
    label_styles = label_styles or {
        "Fear": ("tab:green", default_alpha),
        "Overheat": ("tab:red", default_alpha),
    }
    out = []
    for low, high, label in bands:
        color, alpha = label_styles.get(label, ("gray", default_alpha))
        out.append((low, high, color, alpha, label))
    return out

def plot_overlay(
    df: Union[pd.Series, pd.DataFrame],
    baseline: Optional[Union[str, float]] = None,
    figsize: Tuple[int, int] = (14, 6),
    alpha: float = 0.9,
    linewidth: float = 1.6,
    grid: bool = False,
    title: str = "Overlay Plot",
    ylabel: Optional[str] = None,
    legend: bool = True,
    show: bool = True,
    savepath: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    df = _prep_df(_as_dataframe(df)).dropna(axis=1, how="all")
    if df.shape[1] == 0:
        raise ValueError("No non-empty columns to plot.")

    if baseline == "auto":
        baseline_val = None
        for col in df.columns:
            b = _auto_baseline(df[col])
            if b is not None:
                baseline_val = b
                break
    else:
        baseline_val = baseline

    fig, ax = plt.subplots(figsize=figsize)
    for col in df.columns:
        ax.plot(df.index, df[col].values, linewidth=linewidth, alpha=alpha, label=str(col))
    if baseline_val is not None:
        ax.axhline(baseline_val, linestyle="--", color="gray", linewidth=1)
    if grid:
        ax.grid(alpha=0.25)

    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    ax.margins(x=0)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax

def plot_dual_vs_benchmark(
    index_series,
    benchmark_series,
    *,
    rebase: Union[bool, float] = False,   # False | True | 100.0
    bands: Optional[List[Tuple[float, float, str, float, str]]] = None,  # [(lo,hi,color,alpha,label)]
    figsize=(14, 6),
    linewidth_index=2.0,
    linewidth_bm=2.0,
    style_bm="--",
    grid=True,
    title="Index vs Benchmark (Dual Axis)",
    ylabel_left=None,
    ylabel_right=None,
    legend=True,
    color_left="tab:blue",
    color_right="tab:red",
    no_x_margin=True,
    show=True,
    savepath=None,
):
    idx = _ensure_one_series(index_series, "index")
    bm  = _ensure_one_series(benchmark_series, "benchmark")

    df = pd.concat([idx.rename("index"), bm.rename("benchmark")], axis=1, join="inner").dropna()
    if df.empty:
        raise ValueError("No overlapping dates between index and benchmark.")

    # optional plotting-time rebase (both)
    if rebase is not False:
        base = float(rebase) if isinstance(rebase, (int, float)) else 100.0
        df["index"]     = rebase_series(df["index"], base=base)
        df["benchmark"] = rebase_series(df["benchmark"], base=base)

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    ax_left.plot(df.index, df["index"].values, linewidth=linewidth_index, color=color_left,
                 label=df["index"].name or "index")
    ax_right.plot(df.index, df["benchmark"].values, linewidth=linewidth_bm, linestyle=style_bm, color=color_right,
                  label=df["benchmark"].name or "benchmark")

    # shaded bands on LEFT axis (sentiment scale)
    if bands:
        for lo, hi, color, alpha, label in bands:
            ax_left.axhspan(lo, hi, color=color, alpha=alpha, label=label)

    if grid:
        ax_left.grid(alpha=0.25)
    ax_left.set_title(title)
    if ylabel_left:
        ax_left.set_ylabel(ylabel_left)
    if ylabel_right:
        ax_right.set_ylabel(ylabel_right)

    if legend:
        lines, labels = [], []
        for a in (ax_left, ax_right):
            h, l = a.get_legend_handles_labels()
            lines.extend(h); labels.extend(l)
        # de-duplicate labels (bands may add dup labels)
        seen = set()
        uniq = [(h, l) for h, l in zip(lines, labels) if not (l in seen or seen.add(l))]
        if uniq:
            lines, labels = zip(*uniq)
            ax_left.legend(lines, labels, loc="best")

    if no_x_margin:
        ax_left.margins(x=0)
        ax_right.margins(x=0)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    return fig, (ax_left, ax_right)

def plot_components_vs_benchmark(
    aligned_df: pd.DataFrame,
    benchmark_series,
    indicators: list[str] | None = None,
    title_suffix: str = "vs 基准（双轴）",
    **kwargs,  # forwarded to plot_dual_vs_benchmark (colors, rebased, etc.)
):
    """
    Loop over chosen columns in aligned_df and plot each vs benchmark (dual axis).
    - aligned_df: DataFrame containing indicators and NOT the benchmark series (or if present, it will be ignored).
    - benchmark_series: Series or 1-col DataFrame (the benchmark).
    - indicators: list of column names to plot; if None, uses all columns except the benchmark's name.
    - kwargs: passed to plot_dual_vs_benchmark (e.g., color_left, color_right, no_x_margin, savepath, show, etc.)

    Returns: list of (fig, (ax_left, ax_right)) tuples.
    """
    bm = _ensure_one_series(benchmark_series, "benchmark")
    bm_name = bm.name or "benchmark"

    cols = [c for c in aligned_df.columns if c != bm_name]
    if indicators is not None and len(indicators) > 0:
        cols = [c for c in indicators if c in cols]
        if not cols:
            raise ValueError("No valid indicators found in aligned_df for the given names.")

    results = []
    for col in cols:
        s = aligned_df[col].dropna()
        s.name = col
        fig, axes = plot_dual_vs_benchmark(
            index_series=s,
            benchmark_series=bm,
            title=f"{col} {title_suffix}",
            **kwargs,
        )
        results.append((fig, axes))
    return results
