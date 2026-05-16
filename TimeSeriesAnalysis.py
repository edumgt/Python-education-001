import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

os.makedirs("result", exist_ok=True)

print("=" * 60)
print("  시계열 분석: 이동평균 / 볼린저밴드 / ACF·PACF / 정상성")
print("=" * 60)

# ── 1. 주가 데이터 로드 ────────────────────────────────────
print("\n[1/6] 주가 데이터 로드 중...")
time.sleep(0.5)
np.random.seed(42)
TICKER = '078935.KS'
prices = None
try:
    import yfinance as yf
    from datetime import date
    df = yf.download(TICKER, start='2020-01-01', end=date.today().isoformat(),
                     auto_adjust=True, progress=False)
    if len(df) > 50:
        prices = df['Close'].squeeze().dropna().values.flatten().astype(np.float32)
        print(f"   ✓ {TICKER}: {len(prices)}일 실제 데이터 로드")
except Exception as e:
    print(f"   yfinance 오류 ({e}) → 가상 데이터 사용")

if prices is None:
    days = 250
    t = np.arange(days)
    prices = (100 + 0.12 * t + 5 * np.sin(t / 20) + np.random.normal(0, 1.5, days)).astype(np.float32)
    prices = np.maximum(prices, 1)
    print(f"   → 가상 {days}일치 주가 생성")

days = len(prices)
print(f"   → {days}일치 주가  |  최소: {prices.min():.1f}  최대: {prices.max():.1f}  평균: {prices.mean():.1f}")
time.sleep(0.3)

# ── 2. 이동평균 ────────────────────────────────────────────
print("\n[2/6] 단순 이동평균(SMA) 계산 중...")
print("   수식: SMA_n = 최근 n일 주가의 평균")
print("   짧은 선(5일) = 민감, 긴 선(60일) = 완만 → 교차 시 매매 신호")
time.sleep(0.6)
sma5  = np.convolve(prices, np.ones(5)  / 5,  mode='valid')
sma20 = np.convolve(prices, np.ones(20) / 20, mode='valid')
sma60 = np.convolve(prices, np.ones(60) / 60, mode='valid')
print(f"   → SMA5 최신값: {sma5[-1]:.2f}  SMA20: {sma20[-1]:.2f}  SMA60: {sma60[-1]:.2f}")
time.sleep(0.3)

# ── 3. 볼린저밴드 ──────────────────────────────────────────
print("\n[3/6] 볼린저밴드 계산 중 (20일 이동평균 ± 2σ)...")
print("   상단선: 평균 + 2×표준편차  →  주가가 여기 닿으면 '과매수' 신호")
print("   하단선: 평균 - 2×표준편차  →  주가가 여기 닿으면 '과매도' 신호")
time.sleep(0.6)
window = 20
rolling_mean, rolling_std = [], []
for i in range(window - 1, days):
    w = prices[i - window + 1: i + 1]
    rolling_mean.append(w.mean())
    rolling_std.append(w.std())
rolling_mean = np.array(rolling_mean)
rolling_std  = np.array(rolling_std)
upper_band = rolling_mean + 2 * rolling_std
lower_band = rolling_mean - 2 * rolling_std
band_idx = np.arange(window - 1, days)
print(f"   → 최신 상단밴드: {upper_band[-1]:.2f}  하단밴드: {lower_band[-1]:.2f}")
time.sleep(0.3)

# ── 4. 일간 수익률 & 정상성 검정 ───────────────────────────
print("\n[4/6] 일간 수익률 계산 & ADF 정상성 검정 중...")
print("   정상성(Stationarity) = 평균·분산이 시간에 따라 일정한 성질")
print("   비정상 시계열(주가)을 차분(오늘-어제)하면 정상이 됩니다")
time.sleep(0.6)
returns = np.diff(prices) / prices[:-1]
adf_price   = adfuller(prices)
adf_returns = adfuller(returns)
print(f"   주가 ADF p-value:   {adf_price[1]:.4f}  "
      f"→  {'비정상 (차분 필요)' if adf_price[1] > 0.05 else '정상'}")
print(f"   수익률 ADF p-value: {adf_returns[1]:.4f}  "
      f"→  {'비정상' if adf_returns[1] > 0.05 else '정상 (ARIMA 등 모델에 바로 사용 가능)'}")
time.sleep(0.3)

# ── 5. ACF / PACF ──────────────────────────────────────────
print("\n[5/6] 자기상관(ACF) & 편자기상관(PACF) 분석 중...")
print("   ACF  = 현재 값이 n일 전 값과 얼마나 닮아 있는가")
print("   PACF = 중간 날들의 영향을 제거하고 n일 전과의 순수 상관")
print("   → ARIMA 모델의 p(AR차수), q(MA차수) 결정에 사용")
time.sleep(0.8)

# ── 6. 시각화 ─────────────────────────────────────────────
print("\n[6/6] 시각화 저장 중 (3개 그래프)...")
time.sleep(0.5)

fig = plt.figure(figsize=(14, 12))

# 전체 그래프 요약 제목
fig.text(0.5, 0.98, '주가의 흐름·규칙성·예측 가능성을 여러 방법으로 분석합니다',
         ha='center', fontsize=9, color='#333', weight='bold')

# 이동평균 & 볼린저밴드
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(prices, color='steelblue', linewidth=0.9, label='주가')
ax1.plot(np.arange(4, days), sma5,  color='orange',   linewidth=1.2, label='SMA5')
ax1.plot(np.arange(19, days), sma20, color='tomato',   linewidth=1.2, label='SMA20')
ax1.plot(np.arange(59, days), sma60, color='purple',   linewidth=1.2, label='SMA60')
ax1.fill_between(band_idx, upper_band, lower_band, alpha=0.15, color='gray', label='볼린저밴드')
ax1.plot(band_idx, upper_band, 'k--', linewidth=0.7)
ax1.plot(band_idx, lower_band, 'k--', linewidth=0.7)
ax1.set_title("이동평균(SMA) & 볼린저밴드")
ax1.set_xlabel("거래일")
ax1.set_ylabel("주가")
ax1.legend(fontsize=8)
# 볼린저밴드 상단 설명 어노테이션
ax1.annotate("이 선을 뚫고 올라가면 '과매수' — 너무 많이 올랐다는 신호",
             xy=(0.75, 0.92), xytext=(0.45, 0.98),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='darkred')
# 볼린저밴드 하단 설명 어노테이션
ax1.annotate("이 선 아래로 내려가면 '과매도' — 너무 많이 내렸다는 신호",
             xy=(0.75, 0.06), xytext=(0.45, 0.02),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
# 골든크로스 안내 텍스트
ax1.annotate("짧은 선이 긴 선을 위로 뚫으면 '골든크로스' 매수 신호",
             xy=(0.35, 0.55), xytext=(0.02, 0.75),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='orange'), fontsize=7, color='#333')
# SMA 설명 텍스트
ax1.text(0.5, -0.14, 'SMA = 최근 N일 평균 주가  |  N이 클수록 완만하고 느리게 반응합니다',
         transform=ax1.transAxes, ha='center', fontsize=7, color='gray')
print("   → 패널 1: 이동평균·볼린저밴드 완료")
time.sleep(0.3)

# 일간 수익률
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(returns, color='steelblue', linewidth=0.7, alpha=0.8)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.fill_between(range(len(returns)), returns, 0,
                 where=(np.array(returns) > 0), color='tomato', alpha=0.4, label='양수(수익)')
ax2.fill_between(range(len(returns)), returns, 0,
                 where=(np.array(returns) < 0), color='royalblue', alpha=0.4, label='음수(손실)')
ax2.set_title("일간 수익률 (차분된 정상 시계열)")
ax2.set_xlabel("거래일")
ax2.set_ylabel("수익률")
ax2.legend(fontsize=8)
# 양수 구간 설명
ax2.text(0.15, 0.80, '오른 날', transform=ax2.transAxes,
         fontsize=8, color='tomato', ha='center', weight='bold')
# 음수 구간 설명
ax2.text(0.15, 0.15, '내린 날', transform=ax2.transAxes,
         fontsize=8, color='royalblue', ha='center', weight='bold')
# 기준선 설명
ax2.annotate('기준선 (변동 없음)', xy=(0.05, 0.50), xytext=(0.25, 0.55),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
ax2.text(0.5, -0.14, '수익률 = (오늘 주가 - 어제 주가) ÷ 어제 주가  |  0 위면 상승, 0 아래면 하락',
         transform=ax2.transAxes, ha='center', fontsize=7, color='gray')
print("   → 패널 2: 일간 수익률 완료")
time.sleep(0.3)

# ACF & PACF (나란히)
ax3 = fig.add_subplot(3, 2, 5)
ax4 = fig.add_subplot(3, 2, 6)
plot_acf(returns,  ax=ax3, lags=30, title="자기상관 (ACF)  — 수익률",  zero=False)
plot_pacf(returns, ax=ax4, lags=30, title="편자기상관 (PACF) — 수익률", zero=False)
# ACF 파란 띠 설명
ax3.text(0.5, 0.85, '파란 띠 안에 있으면\n우연의 일치 — 관계없다는 뜻',
         transform=ax3.transAxes, ha='center', fontsize=7, color='#333',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
# ACF 첫 번째 막대 설명
ax3.annotate('가장 가까운 하루 전과의 관계',
             xy=(0.03, 0.5), xytext=(0.15, 0.65),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
ax3.text(0.5, -0.18, 'lag = 며칠 전과의 연관성인지를 나타냅니다',
         transform=ax3.transAxes, ha='center', fontsize=7, color='gray')
# PACF 파란 띠 설명
ax4.text(0.5, 0.85, '파란 띠 안에 있으면\n우연의 일치 — 관계없다는 뜻',
         transform=ax4.transAxes, ha='center', fontsize=7, color='#333',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
ax4.text(0.5, -0.18, '중간 날들의 영향을 제거한 순수한 상관관계입니다',
         transform=ax4.transAxes, ha='center', fontsize=7, color='gray')
print("   → 패널 3·4: ACF·PACF 완료")
time.sleep(0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
ticker_tag = TICKER.replace('.', '_')
plt.savefig(f"result/TimeSeriesAnalysis_{ticker_tag}.png", dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: result/TimeSeriesAnalysis_{ticker_tag}.png")

print("\n✓ 시계열 분석 실습 완료!\n")
