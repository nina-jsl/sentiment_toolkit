# sentiment_toolkit/style.py
import matplotlib.pyplot as plt

# A sensible default list across macOS/Win/Linux
_DEFAULT_FONT_CANDIDATES = [
    "PingFang SC",        # macOS
    "Hiragino Sans GB",   # macOS (older)
    "STHeiti",            # macOS (older)
    "Heiti SC",           # macOS (older)
    "SimHei",             # Windows
    "Microsoft YaHei",    # Windows
    "Noto Sans CJK SC",   # Linux, Google Noto
    "WenQuanYi Zen Hei"   # Linux (older distros)
]

def enable_chinese_font(preferred=None, minus_fix=True):
    """
    Enable a Chinese-capable sans-serif font so Matplotlib can render 中文.
    Call once at app startup or before plotting.

    preferred : str | list[str] | None
        - None -> use default candidate list (cross-platform)
        - str  -> a single exact font family name (e.g., "PingFang SC")
        - list -> your own prioritized candidates
    minus_fix : bool
        Replace the math minus with a normal hyphen so it displays correctly.

    Returns the final font family list set in rcParams.
    """
    if preferred is None:
        families = _DEFAULT_FONT_CANDIDATES
    elif isinstance(preferred, str):
        families = [preferred] + _DEFAULT_FONT_CANDIDATES
    else:
        # user provided a list
        families = list(preferred) + _DEFAULT_FONT_CANDIDATES

    plt.rcParams["font.sans-serif"] = families
    if minus_fix:
        plt.rcParams["axes.unicode_minus"] = False
    return plt.rcParams["font.sans-serif"]
