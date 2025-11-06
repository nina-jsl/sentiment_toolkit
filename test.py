# test.py — auto-weighted (IC) sentiment: rolling z → IC weights → 0–100 → plot vs benchmark
from __future__ import annotations
import numpy as np
import pandas as pd

from sentiment_toolkit import (
    load_and_clean,
    resample_to_monthly,
    align_and_trim,
    rolling_zscore,
    weighted_index,
    to_scale_0_100,
    plot_dual_vs_benchmark,
    fixed_bands,
    bands_with_style,
    enable_chinese_font,
)

DATA_PATH   = "/Users/nina/Desktop/JPM/Market Insights/test_data.xlsx"
ROLL_WINDOW = 36
INVERSE_COLS = ["隐含风险溢价"]  # higher = worse → flip

def _as_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        s = x.iloc[:, 0]
        s.name = s.name or "benchmark"
        return s
    raise TypeError("Benchmark must be a Series or a 1-col DataFrame.")

def main():
    enable_chinese_font()

    # 1) Load → monthly → align
    cleaned, benchmark = load_and_clean(DATA_PATH)
    monthly = resample_to_monthly(cleaned)
    aligned  = align_and_trim(monthly, benchmark)

    # 2) Benchmark series
    bm = _as_series(benchmark)

    # 3) Components + flips
    comp_cols = [c for c in aligned.columns if c != "benchmark"]
    X = aligned[comp_cols].copy()
    for col in INVERSE_COLS:
        if col in X.columns:
            X[col] = -X[col]

    # 4) Rolling z
    Z = rolling_zscore(X, window=ROLL_WINDOW, min_periods=ROLL_WINDOW).dropna(how="all")
    if Z.empty:
        raise ValueError("Empty Z after rolling_zscore — increase data or reduce window.")

    # 5) Benchmark returns aligned
    y = bm.pct_change().reindex(Z.index)

    # 6) IC weights (use pandas corr; require >=2 data points)
    ic = {}
    yv = y.dropna()
    for col in Z.columns:
        zc = Z[col].dropna()
        idx = zc.index.intersection(yv.index)
        if len(idx) < 2:
            continue
        rho = zc.loc[idx].corr(yv.loc[idx])  # robust to dtype/shape issues
        if pd.notna(rho) and np.isfinite(rho):
            ic[col] = rho

    if not ic:
        raise ValueError("No valid ICs computed (too few overlaps or constant series).")

    w = pd.Series({k: abs(v) for k, v in ic.items()}, dtype=float)
    if not np.isfinite(w.sum()) or w.sum() == 0:
        # fallback equal weights across available Z columns
        w = pd.Series(1.0, index=Z.columns, dtype=float)
    w = w / w.sum()

    print("\nIC-based weights (|corr| normalized):")
    print(w.to_string())

    # 7) IC-weighted composite
    common = [c for c in w.index if c in Z.columns]
    z_comp = weighted_index(Z[common], weights=w, name="IC-Weighted Z")

    # 8) 0–100 scaling
    sent_0_100 = to_scale_0_100(z_comp, p_low=0.05, p_high=0.95, clip=True)
    sent_0_100.name = "情绪（0–100）"

    # 9) Align benchmark to index dates
    bm_aligned = bm.reindex(sent_0_100.index).dropna()
    idx_common = sent_0_100.index.intersection(bm_aligned.index)
    sent_0_100 = sent_0_100.reindex(idx_common)
    bm_aligned  = bm_aligned.reindex(idx_common)

    # 10) Bands
    bands = bands_with_style(fixed_bands([(0, 10, "Fear"), (90, 100, "Overheat")]))

    # 11) Plot
    plot_dual_vs_benchmark(
        index_series=sent_0_100,
        benchmark_series=bm_aligned,
        title="IC加权（滚动Z）情绪指数（0–100）vs Wind 全A",
        ylabel_left="情绪（0–100 百分位）",
        ylabel_right=bm_aligned.name or "Benchmark",
        bands=bands,
        grid=True,
        color_left="tab:blue",
        color_right="tab:orange",
        no_x_margin=True,
        show=True,
    )

if __name__ == "__main__":
    main()
