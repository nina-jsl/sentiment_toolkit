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

        cleaned[name] = df

        if meta.get("benchmark", False):
            # assume *exactly one* benchmark column is present (besides date)
            bench_col = [c for c in df.columns][0]
            benchmark = df[[bench_col]].rename(columns={bench_col: "benchmark"})
    
    return cleaned, benchmark

        
cleaned, benchmark = load_and_clean("/Users/nina/Desktop/JPM/Market Insights/test_data.xlsx")

print(cleaned.keys())
print(benchmark.head())
