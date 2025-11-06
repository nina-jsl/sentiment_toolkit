# test.py
from __future__ import annotations

import pandas as pd

try:
    from sentiment_toolkit import (
        load_and_clean,
        resample_to_monthly,
        align_and_trim,
        enable_chinese_font,
        plot_dual_vs_benchmark,
        fixed_bands,
        bands_with_style,
        build_weighted_sentiment_percentile,
    )
except ImportError:
    from sentiment_toolkit.io import (
        load_and_clean,
        resample_to_monthly,
        align_and_trim,
    )
    from sentiment_toolkit.viz import (
        enable_chinese_font,
        plot_dual_vs_benchmark,
        fixed_bands,
        bands_with_style,
    )
    from sentiment_toolkit.index import (
        build_weighted_sentiment_percentile,
    )

# === Update this to your local file ===
DATA_PATH = "/Users/nina/Desktop/JPM/Market Insights/test_data.xlsx"

# === Config: rolling window ===
ROLL_WINDOW = 36

# === User-facing (canonical) weight names (English) ===
WEIGHTS_CANON = {
    "Turnover Rate":            0.30,  # speculative trading heat
    "New Investor Accounts":    0.25,  # retail enthusiasm
    "Limit-Up Count":           0.20,  # speculative frenzy
    "New Equity Fund Issuance": 0.10,  # institutional optimism
    "Implied Risk Premium":     0.15,  # valuation fear (inverse sentiment)
    "Margin Balance (% Float)": 0.05,  # leveraged sentiment
}

# Columns where "higher = worse sentiment" → flip sign before z-scoring
INVERSE_CANON = {"Implied Risk Premium"}

# === Alias map: canonical name -> actual column name in your dataset ===
# Adjust here if your source column labels change.
ALIASES = {
    "Turnover Rate":            "换手率",
    "New Investor Accounts":    "新增开户",
    "Limit-Up Count":           "涨停家数",
    "New Equity Fund Issuance": "基金新发量",
    "Implied Risk Premium":     "隐含风险溢价",
    "Margin Balance (% Float)": "融资余额",      # if you later use % of float, change to that label
}

def _ensure_series(x) -> pd.Series:
    """Accept a Series or 1-col DataFrame for the benchmark; return a Series."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Benchmark DataFrame must have exactly 1 column.")
        s = x.iloc[:, 0]
        s.name = s.name or "benchmark"
        return s
    elif isinstance(x, pd.Series):
        return x
    else:
        raise TypeError("Benchmark must be a pandas Series or a 1-column DataFrame.")

def _map_weights_to_actual_columns(aligned_cols: list[str]) -> tuple[pd.Series, list[str], dict]:
    """
    Map canonical weight keys to actual dataset columns using ALIASES.
    Returns:
        w_actual  : pd.Series indexed by actual column names (renormalized to sum=1)
        inv_actual: list of actual column names that should be inverted
        mapping   : dict of canonical -> actual used (for logging)
    """
    mapping = {}
    actual_names = []
    weight_vals = []
    for canon_name, w in WEIGHTS_CANON.items():
        actual = ALIASES.get(canon_name, None)
        if actual is not None and actual in aligned_cols:
            mapping[canon_name] = actual
            actual_names.append(actual)
            weight_vals.append(float(w))
    if not actual_names:
        raise ValueError(
            "None of the weighted component columns were found.\n"
            f"Looked for: {list(ALIASES.values())}\n"
            f"Aligned has: {aligned_cols}"
        )
    w_actual = pd.Series(weight_vals, index=actual_names, dtype=float)
    w_actual = w_actual / w_actual.sum()

    # map inverse set to actual labels present
    inv_actual = []
    for canon in INVERSE_CANON:
        name = ALIASES.get(canon, None)
        if name in actual_names:
            inv_actual.append(name)

    return w_actual, inv_actual, mapping

def main():
    enable_chinese_font()

    # 1) Load → monthly → align
    cleaned, benchmark = load_and_clean(DATA_PATH)
    monthly = resample_to_monthly(cleaned)
    aligned = align_and_trim(monthly, benchmark)  # indicators df (may also include a 'benchmark' column)
    bm_series = _ensure_series(benchmark)

    # 2) Build weights mapped to actual dataset column labels
    aligned_cols = [c for c in aligned.columns if c != "benchmark"]
    w_actual, inv_actual, mapping = _map_weights_to_actual_columns(aligned_cols)

    # Log mapping and final weights
    print("\nColumn mapping (canonical → actual):")
    for k, v in mapping.items():
        print(f"  {k} → {v}")

    print("\nFinal Weights Used (aligned & normalized):")
    print(w_actual.to_string())

    if len(inv_actual):
        print("\nInverse columns (higher = worse, flipped before z):")
        for c in inv_actual:
            print(f"  {c}")

    # 3) Build 0–100 percentile sentiment using 36m rolling z + custom weights
    sentiment_0_100, Z = build_weighted_sentiment_percentile(
        sentiment_raw=aligned[w_actual.index],
        weights=w_actual,
        window=ROLL_WINDOW,
        inverse_cols=inv_actual,
        rank_method="average",
        clip_0_100=True,
    )

    # 4) Align benchmark to the sentiment index for clean plotting
    bm_aligned = bm_series.reindex(sentiment_0_100.index).dropna()
    idx_common = sentiment_0_100.index.intersection(bm_aligned.index)
    sentiment_0_100 = sentiment_0_100.reindex(idx_common)
    bm_aligned = bm_aligned.reindex(idx_common)

    # 5) Shaded bands on the left axis
    bands = fixed_bands([(0, 10, "Fear"), (90, 100, "Overheat")])
    bands = bands_with_style(bands)

    # 6) Plot (dual axis)
    plot_dual_vs_benchmark(
        index_series=sentiment_0_100,
        benchmark_series=bm_aligned,
        title="市场情绪（滚动Z+加权，0–100）vs Wind 全A",
        ylabel_left="情绪（0–100 百分位）",
        ylabel_right=bm_aligned.name or "Benchmark",
        grid=True,
        bands=bands,
        color_left="tab:blue",
        color_right="tab:orange",
        no_x_margin=True,
        show=True,
    )

if __name__ == "__main__":
    main()
