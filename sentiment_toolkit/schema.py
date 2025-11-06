# zero_means_missing:
#   True  →  0 means "no market / not launched yet" → convert to NaN → drop
#   False →  0 is a real economic value → keep it
DATA_SCHEMA = {
    "新增开户": {
        "sheet": "新增开户",
        "cols": {"date": 0, "新增开户": 2},
        "frequency": "monthly",        
        "zero_means_missing": True 
    },
    "换手率": {
        "sheet": "换手率",
        "cols": {"date": 0, "换手率": 9},
        "frequency": "daily",
        "resample": "mean"
    },
    "融资余额": {
        "sheet": "融资余额",
        "cols": {"date": 0, "融资余额": 3},
        "frequency": "daily",
        "resample": "last"
    },
    "隐含风险溢价": {
        "sheet": "隐含风险溢价",
        "cols": {"date": 0, "隐含风险溢价": 5},
        "frequency": "monthly"
    },
    "基金新发量": {
        "sheet": "基金新发量",
        "cols": {"date": 0, "基金新发量": 2},
        "frequency": "monthly",
        "zero_means_missing": True
    },
    "涨停家数": {
        "sheet": "涨停家数",
        "cols": {"date": 0, "涨停家数": 1},
        "frequency": "daily",
        "resample": "mean"
    },
    "万得全A": {
        "sheet": "万得全A",
        "cols": {"date": 0, "万得全A": 1},
        "frequency": "monthly",        
        "zero_means_missing": True,
        "benchmark": True 
    },
}
