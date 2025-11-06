import matplotlib.pyplot as plt
import numpy as np

def enable_chinese_font():
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'STHeiti']
    plt.rcParams['axes.unicode_minus'] = False

def plot_each_series(
    df,
    baseline=None,
    figsize=(14,4),
):
    """
    Plot each series from a DataFrame in its own figure.

    df : pandas DataFrame (any timeseries)
    baseline : None (auto), number, or 'auto':
        None  → no baseline line
        'auto' → detect 1 (baseline normalized) or 0 (z-score centered)
        number → draw horizontal line at that value
    """
    # Auto baseline detection
    if baseline == 'auto':
        if np.allclose(df.iloc[0], 1, atol=0.05):
            baseline = 1
        elif np.allclose(df.mean(), 0, atol=0.05):
            baseline = 0
        else:
            baseline = None

    for col in df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(df.index, df[col], linewidth=2)
        ax.set_title(col)

        if baseline is not None:
            ax.axhline(baseline, linestyle="--", color="gray", linewidth=1)

        ax.margins(x=0)
        plt.show()


def plot_overlay(
    df,
    baseline=None,
    figsize=(14,6),
    alpha=0.9,
):
    """
    Plot all series on one overlay chart.

    df : pandas DataFrame (any timeseries)

    baseline behavior same as plot_each_series.
    """
    # Auto baseline detection
    if baseline == 'auto':
        if np.allclose(df.iloc[0], 1, atol=0.05):
            baseline = 1
        elif np.allclose(df.mean(), 0, atol=0.05):
            baseline = 0
        else:
            baseline = None

    fig, ax = plt.subplots(figsize=figsize)

    for col in df.columns:
        ax.plot(df.index, df[col], linewidth=1.6, alpha=alpha, label=col)

    if baseline is not None:
        ax.axhline(baseline, linestyle="--", color="gray", linewidth=1)

    ax.legend()
    ax.margins(x=0)
    plt.title("Overlay Plot")
    plt.show()
