from sentiment_toolkit import (
    load_and_clean,
    resample_to_monthly,
    align_and_trim,
    normalize_indicators,
    plot_each_series,
    plot_overlay,
    enable_chinese_font
)
enable_chinese_font()
# Load test data
cleaned, benchmark = load_and_clean("/Users/nina/Desktop/JPM/Market Insights/test_data.xlsx")
monthly = resample_to_monthly(cleaned)
aligned = align_and_trim(monthly, benchmark)

benchmark_series, raw, normalized = normalize_indicators(
    aligned_df=aligned,
    benchmark_series=benchmark,   # ← This line
    method="baseline"
)


print("✅ Benchmark:")
print(benchmark_series.head(), "\n")
print(normalized.head())

plot_each_series(normalized, baseline='auto')
plot_overlay(normalized, baseline='auto')
