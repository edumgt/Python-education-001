import matplotlib as mpl
from matplotlib import font_manager

_CANDIDATES = [
    'NanumGothic', 'Malgun Gothic', 'AppleGothic',
    'Noto Sans CJK KR', 'NanumBarunGothic', 'NanumSquareRound',
]
_available = {f.name for f in font_manager.fontManager.ttflist}
_font = next((f for f in _CANDIDATES if f in _available), None)

if _font:
    mpl.rcParams['font.family'] = _font
mpl.rcParams['axes.unicode_minus'] = False
