import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import korean_font  # noqa: F401

os.makedirs("result", exist_ok=True)

print("=" * 65)
print("  yfinance 주가 데이터 정규화 비교 실습")
print("=" * 65)
print()
print("  정규화 방법 비유:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  Min-Max  : 키를 0~100 점수로 바꾸는 것 (0=최소)     │")
print("  │  Z-점수   : 평균에서 얼마나 떨어졌는지 표준화        │")
print("  │  로그수익률: 비율 변화 → 덧셈으로 표현 (안정적)      │")
print("  └──────────────────────────────────────────────────────┘")

# ── 1. 데이터 로드 ─────────────────────────────────────────
print("\n[1/7] 주가 데이터 로드 중...")
time.sleep(0.4)

try:
    import yfinance as yf
    from datetime import date
    today = date.today().isoformat()
    TICKERS = ['AAPL', '005930.KS', '^GSPC']   # 애플, 삼성, S&P500
    raw_data = {}
    for ticker in TICKERS:
        df = yf.download(ticker, start='2023-01-01', end=today,
                         auto_adjust=True, progress=False)
        if len(df) > 0:
            close = df['Close'].squeeze().dropna()   # DataFrame→Series 보정
            raw_data[ticker] = close
            print(f"   ✓ {ticker}: {len(raw_data[ticker])}일 데이터 로드")
        else:
            print(f"   ✗ {ticker}: 데이터 없음 (가상 데이터로 대체)")
    use_real = len(raw_data) > 0
except Exception as e:
    use_real = False
    print(f"   yfinance 연결 실패 ({e}) → 가상 데이터로 실습")

if not use_real:
    np.random.seed(42)
    days = 500
    t = np.arange(days)
    raw_data = {
        'AAPL':      pd.Series(150 + 0.15*t + 10*np.sin(t/30) + np.random.normal(0,3,days)),
        '005930.KS': pd.Series(70000 + 30*t + 4000*np.sin(t/40) + np.random.normal(0,1000,days)),
        '^GSPC':     pd.Series(4000 + 0.8*t + 200*np.sin(t/25) + np.random.normal(0,50,days)),
    }
    print("   → 가상 AAPL / 삼성 / S&P500 데이터 생성 완료")
time.sleep(0.3)

# ── 2. 세 가지 정규화 함수 정의 ───────────────────────────
print("\n[2/7] 세 가지 정규화 함수 정의 중...")
time.sleep(0.4)


def minmax_normalize(series):
    """Min-Max 정규화: 0~1 범위로 압축"""
    s_min, s_max = series.min(), series.max()
    return (series - s_min) / (s_max - s_min + 1e-8), s_min, s_max


def zscore_normalize(series):
    """Z-점수 정규화: 평균=0, 표준편차=1"""
    mean, std = series.mean(), series.std()
    return (series - mean) / (std + 1e-8), mean, std


def log_return(series):
    """로그 수익률: ln(P_t / P_{t-1}) — 가법적이고 정규분포에 가까움"""
    return np.log(series / series.shift(1)).dropna()


print("   ① Min-Max  정규화: x' = (x - min) / (max - min)")
print("   ② Z-점수   정규화: x' = (x - μ) / σ")
print("   ③ 로그 수익률    : r = ln(P_t / P_{t-1})")
time.sleep(0.3)

# ── 3. 정규화 적용 ────────────────────────────────────────
print("\n[3/7] 각 종목에 정규화 적용 중...")
time.sleep(0.5)

normalized_results = {}
for ticker, prices in raw_data.items():
    mm, mn, mx   = minmax_normalize(prices)
    zs, mu, sigma = zscore_normalize(prices)
    lr            = log_return(prices)
    normalized_results[ticker] = {
        'prices':  prices,
        'minmax':  mm,
        'zscore':  zs,
        'logret':  lr,
        'mm_range': (mn, mx),
        'zs_stat':  (mu, sigma),
    }
    print(f"   {ticker}: 원본 범위={prices.min():.1f}~{prices.max():.1f}  "
          f"MM=[0,1]  Z=[{zs.min():.1f}, {zs.max():.1f}]  "
          f"LogRet μ={lr.mean():.4f} σ={lr.std():.4f}")
time.sleep(0.3)

# ── 4. 역정규화 (Min-Max) ──────────────────────────────────
print("\n[4/7] Min-Max 역정규화 정확도 확인 중...")
time.sleep(0.4)
ticker0 = list(raw_data.keys())[0]
r       = normalized_results[ticker0]
mn, mx  = r['mm_range']
reconstructed = r['minmax'] * (mx - mn) + mn
max_err = (reconstructed - r['prices']).abs().max()
print(f"   {ticker0} 역정규화 최대 오차: {max_err:.6f} (≈0이면 완벽)")
time.sleep(0.3)

# ── 5. 로그수익률 통계 & 분포 ─────────────────────────────
print("\n[5/7] 로그수익률 통계 출력 중...")
time.sleep(0.4)
for ticker, res in normalized_results.items():
    lr = res['logret']
    print(f"   {ticker}: 평균={lr.mean():.5f}  σ={lr.std():.5f}  "
          f"최소={lr.min():.4f}  최대={lr.max():.4f}")
time.sleep(0.3)

# ── 6. 다종목 비교 (Z-점수 기준) ──────────────────────────
print("\n[6/7] 다종목 Z-점수 정규화 비교 중 (스케일 통일)...")
print("   Z-점수로 정규화하면 가격대가 달라도 같은 그래프에 비교 가능!")
time.sleep(0.4)

# ── 7. 시각화 ─────────────────────────────────────────────
print("\n[7/7] 시각화 저장 중...")
time.sleep(0.5)

tickers = list(normalized_results.keys())
colors  = ['steelblue', 'tomato', 'mediumseagreen']

fig = plt.figure(figsize=(14, 12))

# 원본 가격
ax1 = fig.add_subplot(3, 2, 1)
for ticker, color in zip(tickers, colors):
    prices = normalized_results[ticker]['prices']
    ax1.plot(prices.values, label=ticker, color=color, linewidth=1.2)
ax1.set_title("① 원본 가격 (스케일 불일치)")
ax1.set_xlabel("거래일")
ax1.set_ylabel("가격")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# Min-Max 정규화
ax2 = fig.add_subplot(3, 2, 2)
for ticker, color in zip(tickers, colors):
    mm = normalized_results[ticker]['minmax']
    ax2.plot(mm.values, label=ticker, color=color, linewidth=1.2)
ax2.set_title("② Min-Max 정규화 [0, 1]")
ax2.set_xlabel("거래일")
ax2.set_ylabel("정규화 값")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# Z-점수 정규화
ax3 = fig.add_subplot(3, 2, 3)
for ticker, color in zip(tickers, colors):
    zs = normalized_results[ticker]['zscore']
    ax3.plot(zs.values, label=ticker, color=color, linewidth=1.2)
ax3.axhline(0, linestyle='--', color='gray', linewidth=0.7)
ax3.set_title("③ Z-점수 정규화 (μ=0, σ=1)")
ax3.set_xlabel("거래일")
ax3.set_ylabel("표준편차 단위")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# 로그 수익률
ax4 = fig.add_subplot(3, 2, 4)
for ticker, color in zip(tickers, colors):
    lr = normalized_results[ticker]['logret']
    ax4.plot(lr.values, label=ticker, color=color, linewidth=0.8, alpha=0.8)
ax4.axhline(0, linestyle='--', color='gray', linewidth=0.7)
ax4.set_title("④ 로그 수익률 ln(P_t / P_{t-1})")
ax4.set_xlabel("거래일")
ax4.set_ylabel("로그 수익률")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# 로그수익률 분포
ax5 = fig.add_subplot(3, 2, 5)
for ticker, color in zip(tickers, colors):
    lr = normalized_results[ticker]['logret']
    ax5.hist(lr.values, bins=40, alpha=0.5, color=color, label=ticker, edgecolor='none')
ax5.set_title("⑤ 로그수익률 분포 (정규분포에 가까울수록 좋음)")
ax5.set_xlabel("로그 수익률")
ax5.set_ylabel("빈도")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

# 정규화 방법 비교 요약 텍스트
ax6 = fig.add_subplot(3, 2, 6)
ax6.axis('off')
summary = (
    "정규화 방법 비교 요약\n\n"
    "① Min-Max 정규화\n"
    "   • 범위: [0, 1]\n"
    "   • 장점: 직관적, 역변환 쉬움\n"
    "   • 단점: 이상치에 민감\n\n"
    "② Z-점수 정규화\n"
    "   • 범위: (−∞, +∞), μ=0 σ=1\n"
    "   • 장점: 이상치 강건, 다종목 비교 용이\n"
    "   • 단점: 값 범위 직접 해석 어려움\n\n"
    "③ 로그 수익률\n"
    "   • r = ln(P_t / P_{t-1})\n"
    "   • 장점: 정상성, 가법적, 분포 안정\n"
    "   • 단점: 절대 가격 정보 소실\n\n"
    "★ 딥러닝 입력에는 Min-Max or Z-점수\n"
    "  통계 분석·수익률 모델에는 로그 수익률"
)
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
         fontsize=8.5, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ── 한글 어노테이션 삽입 (plt.tight_layout 이전) ──────────

# 전체 요약 텍스트
fig.text(0.5, 0.98,
         "같은 기준으로 맞춰야(정규화) AI가 공평하게 학습할 수 있어요",
         ha='center', fontsize=9, color='#333', weight='bold')

# ── 원본 패널 (ax1) ──
ax1.text(0.5, -0.18,
         '단위가 달라서 비교가 어려워요 (삼성=7만원, 애플=150달러)',
         transform=ax1.transAxes, ha='center', fontsize=7, color='gray')

# ── MinMax 패널 (ax2) ──
ax2.text(0.5, -0.18,
         '모두 0~1 사이로 맞춰서 같은 눈금으로 비교해요',
         transform=ax2.transAxes, ha='center', fontsize=7, color='gray')

# ── Z점수 패널 (ax3) ──
ax3.text(0.5, -0.18,
         '평균에서 얼마나 떨어졌는지로 표현해요 (0=평균, +1=평균보다 1 표준편차 위)',
         transform=ax3.transAxes, ha='center', fontsize=7, color='gray')

# ── 로그수익률 패널 (ax4) ──
ax4.text(0.5, -0.18,
         '하루하루의 변화율이에요 — 위아래로 튀는 것이 정상이에요',
         transform=ax4.transAxes, ha='center', fontsize=7, color='gray')

# ── 분포 패널 (ax5) ──
ax5.text(0.5, -0.18,
         '종 모양에 가까울수록 통계 계산이 잘 맞아요',
         transform=ax5.transAxes, ha='center', fontsize=7, color='gray')

plt.subplots_adjust(top=0.93)
plt.tight_layout()
ticker_tag = "_".join(t.replace(".", "_").replace("^", "") for t in list(raw_data.keys()))
out_name = f"result/YfinanceNormalize_{ticker_tag}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: {out_name}")

print("\n✓ yfinance 주가 정규화 비교 실습 완료!\n")
