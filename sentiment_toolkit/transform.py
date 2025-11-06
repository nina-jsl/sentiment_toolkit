import pandas as pd

def normalize_indicators(
    aligned_df,
    benchmark_series,
    drop_cols=None,
    method="baseline"
):
    """
    Normalize individual sentiment indicators before composite calculation.

    aligned_df : full aligned dataframe (including benchmark)
    benchmark_series : Series returned separately from load_and_clean/align steps
    drop_cols : list[str] to remove unwanted indicators

    returns:
        benchmark_series (Series)
        indicators_raw (DataFrame)
        indicators_norm (DataFrame)
    """

    df = aligned_df.copy()

    if isinstance(benchmark_series, pd.DataFrame):
        if benchmark_series.shape[1] == 1:
            benchmark_series = benchmark_series.iloc[:, 0]
        else:
            raise ValueError("benchmark_series DataFrame has multiple columns; expected exactly one.")

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    df = df.drop(columns=[benchmark_series.name], errors="ignore")

    indicators_raw = df

    if method == "baseline":
        baseline = indicators_raw.iloc[0]
        indicators_norm = indicators_raw.div(baseline).iloc[1:]

    elif method == "zscore":
        indicators_norm = (indicators_raw - indicators_raw.mean()) / indicators_raw.std()

    else:
        raise ValueError(f"Unknown method: {method}")

    return benchmark_series, indicators_raw, indicators_norm