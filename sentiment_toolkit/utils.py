from __future__ import annotations
import pandas as pd

def ensure_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Accept a Series or a 1-column DataFrame and return a Series.
    """
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Benchmark DataFrame must have exactly one column.")
        s = x.iloc[:, 0]
        s.name = s.name or "benchmark"
        return s
    raise TypeError("Benchmark must be a pandas Series or a 1-column DataFrame.")

def weights_from_aliases(
    aligned_cols: list[str],
    weights_canon: dict[str, float],
    aliases: dict[str, str],
    inverse_canon: set[str] | frozenset[str] = frozenset(),
):
    """
    Map user-facing (canonical) weight keys to actual dataset columns via `aliases`.
    Returns:
        w_actual : pd.Series (index = actual column names, normalized to sum=1)
        inv_cols : list[str] of actual columns that should be inverted before z-scoring
        used_map : dict[canonical -> actual] for logging
    """
    present = {}
    used_map = {}
    for canon, w in weights_canon.items():
        actual = aliases.get(canon)
        if actual in aligned_cols:
            present[actual] = float(w)
            used_map[canon] = actual

    if not present:
        raise ValueError(
            "None of the weighted component columns were found in aligned data.\n"
            f"Aligned columns: {aligned_cols}\n"
            f"Expected any of: {list(aliases.values())}"
        )

    w_actual = pd.Series(present, dtype=float)
    w_actual /= w_actual.sum()

    inv_cols = []
    for canon in inverse_canon:
        actual = aliases.get(canon)
        if actual in w_actual.index:
            inv_cols.append(actual)

    return w_actual, inv_cols, used_map
