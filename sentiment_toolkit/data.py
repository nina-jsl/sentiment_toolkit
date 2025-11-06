import pandas as pd 
from functools import reduce
from .schema import DATA_SCHEMA

def load_and_clean(path):
    raw = pd.read_excel(path, sheet_name = None, header = None)
    cleaned = {}
    benchmark = None 

    for name, meta in DATA_SCHEMA.items():
        df = raw[meta["sheet"]].copy()

        # select required columsn 
        df = df.iloc[8:, list(meta["cols"].values())]
        df.columns = list(meta["cols"].keys())

        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")

        # handle zer-as-missing if sepcified
        if meta.get("zero_means_missing"):
            df = df.replace(0, pd.NA).dropna()

        if meta.get("benchmark", False):
            bench_col = [c for c in df.columns][0]   # first (and only) data column
            benchmark = df[[bench_col]].rename(columns={bench_col: "benchmark"})
            continue  

        cleaned[name] = df

        
    return cleaned, benchmark

        
def resample_to_monthly(cleaned):
    monthly = {}
    for name, df in cleaned.items():
        rule = DATA_SCHEMA[name]

        # daily → monthly conversion rules
        if rule.get("frequency") == "daily":
            if rule.get("resample") == "last":
                monthly[name] = df.resample("ME").last()
            else:
                monthly[name] = df.resample("ME").mean()

        # already monthly → copy as-is
        else:
            monthly[name] = df.copy()

    return monthly

def align_and_trim(monthly, benchmark):
    # 1. Combine all monthly series
    combined = pd.concat(monthly.values(), axis=1)
    combined.columns = list(monthly.keys())

    # 2. Add benchmark
    benchmark = benchmark.resample("ME").last()  # ensure same frequency
    full = pd.concat([combined, benchmark], axis=1)

    # 3. Drop rows where ANY indicator is missing (strict overlap)
    aligned = full.dropna(how="any")

    return aligned

