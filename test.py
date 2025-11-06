# test.py — minimal, readable demo

from __future__ import annotations
import pandas as pd
from sentiment_toolkit import (
    load_and_clean,
    resample_to_monthly,
    align_and_trim,
    build_weighted_sentiment_percentile,
    plot_dual_vs_benchmark,
    fixed_bands,
    bands_with_style,
    enable_chinese_font,
    ensure_series,            # from utils
    weights_from_aliases,     # from utils
)

DATA_PATH   = "/Users/nina/Desktop/JPM/Market Insights/test_data.xlsx"
ROLL_WINDOW = 36

# Canonical weights (user-facing)
WEIGHTS_CANON = {
    "Turnover Rate":            0.30,
    "New Investor Accounts":    0.25,
    "Limit-Up Count":           0.20,
    "New Equity Fund Issuance": 0.10,
    "Implied Risk Premium":     0.15,  # higher=worse → invert
    "Margin Balance (% Float)": 0.05,
}
INVERSE_CANON = {"Implied Risk Premium"}

# Canonical → actual column names in your Excel
ALIASES = {
    "Turnover Rate":            "换手率",
    "New Investor Accounts":    "新增开户",
    "Limit-Up Count":           "涨停家数",
    "New Equity Fund Issuance": "基金新发量",
    "Implied Risk Premium":     "隐含风险溢价",
    "Margin Balance (% Float)": "融资余额",
}

def main():
    enable_chinese_font()

    # 1) Load → monthly → align
    cleaned, benchmark = load_and_clean(DATA_PATH)
    monthly = resample_to_monthly(cleaned)
    aligned  = align_and_trim(monthly, benchmark)  # indicators (+ optional 'benchmark')
    bm_series = ensure_series(benchmark)

    # 2) Build weights using aliases (and figure out which columns to invert)
    comp_cols = [c for c in aligned.columns if c != "benchmark"]
    w, inv_cols, used = weights_from_aliases(comp_cols, WEIGHTS_CANON, ALIASES, INVERSE_CANON)

    print("\nWeights used (actual columns):")
    print(w.to_string())
    if inv_cols:
        print("\nInverted before z-score:", ", ".join(inv_cols))

    # 3) Build 0–100 percentile sentiment (rolling z + custom weights)
    sentiment_0_100, _Z = build_weighted_sentiment_percentile(
        sentiment_raw=aligned[w.index],
        weights=w,
        window=ROLL_WINDOW,
        inverse_cols=inv_cols,
        rank_method="average",
        clip_0_100=True,
    )

    # 4) Align benchmark to index dates
    bm_aligned = bm_series.reindex(sentiment_0_100.index).dropna()
    sentiment_0_100 = sentiment_0_100.reindex(bm_aligned.index)

    # 5) Fear/Overheat bands
    bands = bands_with_style(fixed_bands([(0, 10, "Fear"), (90, 100, "Overheat")]))

    # 6) Plot
    plot_dual_vs_benchmark(
        index_series=sentiment_0_100,
        benchmark_series=bm_aligned,
        title="市场情绪（滚动Z+加权，0–100）vs Wind 全A",
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
