from __future__ import annotations

from typing import Iterable, Mapping, Optional, Tuple
import numpy as np
import pandas as pd

__all__ = [
    "equal_weight_index",
    "weighted_index",
    "rolling_zscore",
    "to_scale_0_100",
    "build_weighted_sentiment_percentile",
]

# ---------------------------
# helpers
# ---------------------------

def _as_dataframe(obj: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Accept Series or DataFrame; return DataFrame."""
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=obj.name or "value")
    return obj

def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex (if possible) and sort by index."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception:
            pass
    return out.sort_index()

def _validate_weights(
    cols: Iterable[str],
    weights: Optional[Mapping[str, float]]
) -> Optional[pd.Series]:
    if weights is None:
        return None
    w = pd.Series(weights, dtype=float)
    # keep only columns that exist
    w = w.reindex(list(cols)).dropna()
    if w.empty:
        return None
    # normalize weights to sum to 1 (for interpretability)
    s = w.sum()
    if s != 0 and np.isfinite(s):
        w = w / s
    return w

# ---------------------------
# core: index construction
# ---------------------------

def equal_weight_index(
    components: pd.DataFrame,
    *,
    min_non_na: int = 1,
    name: str = "Equal-Weighted Sentiment"
) -> pd.Series:
    """
    Row-wise *equal-weighted* composite.
    - Assumes inputs are already normalized the way you want (e.g., baseline-divide, z-scores, etc.).
    - Does NOT apply any rebasing or scaling. Pure averaging only.
    """
    df = _prep_df(_as_dataframe(components)).dropna(axis=1, how="all")
    if df.shape[1] == 0:
        raise ValueError("No non-empty component columns to aggregate.")

    cnt = df.notna().sum(axis=1)     # enforce min_non_na
    out = df.mean(axis=1)            # equal weights across available cols
    out[cnt < int(min_non_na)] = np.nan
    out.name = name
    return out


def weighted_index(
    components: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | None,
    *,
    min_non_na: int = 1,
    name: str = "Weighted Sentiment"
) -> pd.Series:
    """
    Row-wise *weighted* composite.
    - Assumes inputs are already normalized the way you want.
    - Does NOT apply any rebasing or scaling.
    - Weights are normalized to sum to 1 over the intersecting columns.
    """
    df = _prep_df(_as_dataframe(components)).dropna(axis=1, how="all")
    if df.shape[1] == 0:
        raise ValueError("No non-empty component columns to aggregate.")

    w = _validate_weights(df.columns, weights)
    if w is None:
        return equal_weight_index(df, min_non_na=min_non_na, name=name)

    # Align weights to columns; missing weights -> 0
    w = w.reindex(df.columns).fillna(0.0)

    vals = []
    idx = df.index
    for i in range(len(idx)):
        row = df.iloc[i]
        mask = row.notna() & (w != 0)
        if mask.sum() < int(min_non_na):
            vals.append(np.nan)
            continue
        w_sub = w[mask]
        s = w_sub.sum()
        if s == 0 or not np.isfinite(s):
            vals.append(np.nan)
            continue
        vals.append(float(np.dot(row[mask].values, (w_sub / s).values)))

    out = pd.Series(vals, index=idx, name=name)
    return out

# ---------------------------
# transforms
# ---------------------------

def rolling_zscore(
    df: pd.DataFrame,
    window: int = 36,
    *,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Column-wise rolling z-score: (x - mean_w) / std_w, with no look-ahead.
    """
    df = _prep_df(_as_dataframe(df))
    mp = window if min_periods is None else min_periods
    m = df.rolling(window=window, min_periods=mp).mean()
    s = df.rolling(window=window, min_periods=mp).std()
    z = (df - m) / s
    return z


def to_scale_0_100(
    s: pd.Series,
    p_low: float = 0.05,
    p_high: float = 0.95,
    *,
    clip: bool = True,
    anchor: float | None = None,
) -> pd.Series:
    """
    Map a series to a 0–100 gauge using quantiles.
    - If `anchor` is None: quantile min–max → [0,100].
    - If `anchor` is provided (e.g., 100.0 for a “baseline=100” input):
      anchor → 50; lower tail maps to 0; upper tail maps to 100.
    """
    x = s.astype(float)
    if x.dropna().empty:
        return s

    lo = float(np.nanquantile(x, p_low))
    hi = float(np.nanquantile(x, p_high))

    # Degenerate distribution
    if not np.isfinite(hi - lo) or abs(hi - lo) < 1e-12:
        return pd.Series(50.0, index=s.index, name=s.name)

    if anchor is None:
        out = (x - lo) / (hi - lo) * 100.0
    else:
        below = max(anchor - lo, 1e-12)
        above = max(hi - anchor, 1e-12)
        out = pd.Series(index=x.index, dtype=float)
        mask_lo = x < anchor
        mask_hi = ~mask_lo
        out.loc[mask_lo] = 50.0 - 50.0 * (anchor - x.loc[mask_lo]) / below
        out.loc[mask_hi] = 50.0 + 50.0 * (x.loc[mask_hi] - anchor) / above

    if clip:
        out = out.clip(0.0, 100.0)
    out.name = s.name
    return out

# ---------------------------
# end-to-end pipeline for testing/use in test.py
# ---------------------------

def build_weighted_sentiment_percentile(
    sentiment_raw: pd.DataFrame,
    *,
    weights: Mapping[str, float] | pd.Series,
    window: int = 36,
    inverse_cols: Iterable[str] = (),
    rank_method: str = "average",
    clip_0_100: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    1) Flip inverse_cols (higher=worse) by multiplying by -1.
    2) Rolling z-score per column (window).
    3) Weighted composite with on-the-fly NaN handling.
    4) Convert to 0–100 via historical percentile ranks.

    Returns:
        sentiment_0_100 : Series
        Z                : DataFrame of component z-scores (for inspection)
    """
    X = _prep_df(_as_dataframe(sentiment_raw))
    # flip inverse columns
    inv = set(inverse_cols).intersection(X.columns)
    if inv:
        X.loc[:, list(inv)] = -X.loc[:, list(inv)]

    Z = rolling_zscore(X, window=window, min_periods=window).dropna(how="all")
    if Z.empty:
        raise ValueError("Rolling z-scores are empty—check window size and data length.")

    comp = weighted_index(Z, weights=weights, min_non_na=1, name="Weighted Sentiment (Z)")

    # Percentile rank to 0–100
    ranks = comp.rank(pct=True, method=rank_method) * 100.0
    if clip_0_100:
        ranks = ranks.clip(0.0, 100.0)
    ranks.name = "Sentiment (0–100 Percentile)"
    return ranks, Z
