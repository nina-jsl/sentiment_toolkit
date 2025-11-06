# test.py — plot selected component indicators vs benchmark (dual-axis)
# Uses top-level imports thanks to __init__.py re-exports.

import os
import pandas as pd

from sentiment_toolkit import (
    load_and_clean,
    resample_to_monthly,
    align_and_trim,
    plot_components_vs_benchmark,
    enable_chinese_font,
)

# ============== CONFIG ==============
DATA_PATH = "/Users/nina/Desktop/JPM/Market Insights/test_data.xlsx"

# If empty or None → plot ALL indicators found in aligned_df (excluding the benchmark column)
INDICATORS = ["换手率", "新增开户"]

SAVE_PLOTS = False
OUT_DIR    = "./_plots"

# You can override defaults in viz.plot_components_vs_benchmark via **kwargs below
PLOT_KW = dict(
    rebased=False,           # dual-axis usually False (different units)
    ylabel_left="指标",
    ylabel_right="基准",
    grid=True,
    color_left="tab:blue",   # enforced in viz: index left
    color_right="tab:red",   # enforced in viz: benchmark right
    no_x_margin=True,
)
# ====================================

def ensure_series(bm) -> pd.Series:
    """Make sure benchmark is a Series even if a 1-col DataFrame is returned."""
    if isinstance(bm, pd.Series):
        s = bm.copy()
    elif isinstance(bm, pd.DataFrame):
        if bm.shape[1] == 0:
            raise ValueError("Benchmark DataFrame has no columns.")
        s = bm.iloc[:, 0].copy()
    else:
        raise TypeError("Benchmark must be a pandas Series or a 1-column DataFrame.")
    s.name = s.name or "benchmark"
    return s

def main():
    # 0) Ensure Chinese labels render
    enable_chinese_font()  # or enable_chinese_font("PingFang SC")

    # 1) Load raw Excel and get cleaned series + benchmark
    print("→ Loading …")
    cleaned, benchmark = load_and_clean(DATA_PATH)
    benchmark = ensure_series(benchmark)

    # 2) Resample per schema
    print("→ Resampling to monthly …")
    monthly = resample_to_monthly(cleaned)

    # 3) Align by time intersection & attach benchmark column
    print("→ Aligning …")
    aligned = align_and_trim(monthly, benchmark)
    print(f"  aligned columns: {list(aligned.columns)}")
    if not len(aligned):
        print("❌ Aligned DataFrame is empty.")
        return

    # 4) Decide which indicators to plot
    bm_name = benchmark.name or "benchmark"
    available = [c for c in aligned.columns if c != bm_name]
    if not INDICATORS:
        chosen = available
    else:
        chosen = [c for c in INDICATORS if c in available]
        missing = [c for c in INDICATORS if c not in available]
        if missing:
            print(f"⚠️ Missing (skipped): {missing}")
    if not chosen:
        print("❌ No valid indicators to plot.")
        return

    # 5) Plot components vs benchmark
    if SAVE_PLOTS:
        os.makedirs(OUT_DIR, exist_ok=True)

    print("→ Plotting component indicators vs benchmark (dual-axis) …")
    for col in chosen:
        savepath = os.path.join(OUT_DIR, f"dual_vs_benchmark__{col}.png") if SAVE_PLOTS else None
        plot_components_vs_benchmark(
            aligned_df=aligned,
            benchmark_series=benchmark,
            indicators=[col],                # one per call for separate files
            title_suffix="vs 基准（双轴）",
            savepath=savepath,               # passed only here
            show=not SAVE_PLOTS,             # passed only here
            **PLOT_KW,                       # no duplicate show/savepath now
        )


    print("✅ Done.")

if __name__ == "__main__":
    main()
