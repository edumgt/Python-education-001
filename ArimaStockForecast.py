import os
import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import korean_font  # noqa: F401
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")
os.makedirs("result", exist_ok=True)

print("=" * 60)
print("  ARIMA 모델: 주가 시계열 예측 실습")
print("=" * 60)
print("  ARIMA(p, d, q)")
print("  p = AR 차수: 몇 일 전까지 자기 자신의 영향을 받는가")
print("  d = 차분 횟수: 정상성을 만들기 위해 몇 번 뺄셈하는가")
print("  q = MA 차수: 몇 일 전까지 과거 오차의 영향을 받는가")

# ── 1. 데이터 로드 ─────────────────────────────────────────
print("\n[1/6] 주가 데이터 로드 중...")
time.sleep(0.5)
np.random.seed(42)
TICKER = '078935.KS'
prices = None
dates = None
try:
    import yfinance as yf
    from datetime import date
    df = yf.download(TICKER, start='2020-01-01', end=date.today().isoformat(),
                     auto_adjust=True, progress=False)
    if len(df) > 50:
        close_series = df['Close'].squeeze().dropna()
        prices = close_series.values.flatten().astype(np.float32)
        dates  = pd.DatetimeIndex(close_series.index)
        print(f"   ✓ {TICKER}: {len(prices)}일 실제 데이터 로드")
        print(f"   → 데이터 기간: {dates[0].date()} ~ {dates[-1].date()}")
except Exception as e:
    print(f"   yfinance 오류 ({e}) → 가상 데이터 사용")

if prices is None:
    from datetime import date, timedelta
    days_n = 300
    t = np.arange(days_n)
    prices = (50000 + 50 * t + 3000 * np.sin(t / 40) + np.random.normal(0, 500, days_n)).astype(np.float32)
    start = pd.Timestamp('2020-01-02')
    dates = pd.bdate_range(start=start, periods=days_n)
    print(f"   → 가상 {days_n}일치 주가 생성")

# ── 2. 정상성 검정 ─────────────────────────────────────────
print("\n[2/6] ADF 정상성 검정 중...")
print("   가설: 'H0 = 비정상 시계열이다'  →  p < 0.05 이면 H0 기각 = 정상")
time.sleep(0.5)
adf_raw  = adfuller(prices)
adf_diff = adfuller(np.diff(prices))
print(f"   원본 주가   p-value: {adf_raw[1]:.4f}  → {'비정상 → 차분 필요' if adf_raw[1] > 0.05 else '정상'}")
print(f"   1차 차분   p-value: {adf_diff[1]:.4f}  → {'비정상' if adf_diff[1] > 0.05 else '정상 → d=1 확정'}")
d = 1 if adf_raw[1] > 0.05 else 0
print(f"   → 차분 횟수(d) = {d}")
time.sleep(0.4)

# ── 3. p, q 결정 ──────────────────────────────────────────
print("\n[3/6] ARIMA 차수 결정 중 (p=2, d=1, q=2 사용)...")
print("   이 실습에서는 주가 예측 일반 조합 ARIMA(2,1,2)를 사용합니다")
p, q = 2, 2
print(f"   → 최종 설정: ARIMA({p}, {d}, {q})")
time.sleep(0.5)

# ── 4. 전체 데이터로 모델 학습 ────────────────────────────
print(f"\n[4/6] 전체 {len(prices)}일 데이터로 ARIMA 모델 학습 중...")
print("   AR(자기회귀): 과거 주가가 오늘 주가에 미치는 영향 학습")
print("   MA(이동평균): 과거 예측 오차가 오늘에 미치는 영향 학습")
time.sleep(0.8)
model  = ARIMA(prices, order=(p, d, q))
fitted = model.fit()
print(f"   → 학습 완료!")
print(f"      AIC: {fitted.aic:.2f}  (작을수록 더 좋은 모델)")
print(f"      BIC: {fitted.bic:.2f}")
time.sleep(0.5)

# ── 5. 오늘부터 1개월(22 거래일) 미래 예측 ───────────────
FORECAST_DAYS = 22
print(f"\n[5/6] 오늘({dates[-1].date()}) 이후 {FORECAST_DAYS} 거래일 예측 중...")
time.sleep(0.5)

forecast_result = fitted.get_forecast(steps=FORECAST_DAYS)
forecast_mean   = forecast_result.predicted_mean
conf_int        = forecast_result.conf_int(alpha=0.05)   # 95% 신뢰 구간
if hasattr(conf_int, 'iloc'):
    lower_ci = conf_int.iloc[:, 0].values
    upper_ci = conf_int.iloc[:, 1].values
else:
    lower_ci = conf_int[:, 0]
    upper_ci = conf_int[:, 1]
if hasattr(forecast_mean, 'values'):
    forecast_mean_vals = forecast_mean.values
else:
    forecast_mean_vals = np.asarray(forecast_mean)

# 미래 날짜: 마지막 거래일 다음 영업일부터 22일
future_dates = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS)
print(f"   → 예측 기간: {future_dates[0].date()} ~ {future_dates[-1].date()}")
print(f"   → 현재 주가:   {prices[-1]:,.0f}원")
print(f"   → 1개월 예측: {forecast_mean_vals[-1]:,.0f}원  "
      f"({'▲ 상승' if forecast_mean_vals[-1] > prices[-1] else '▼ 하락'} 예상)")
print(f"   → 95% 신뢰구간: {lower_ci[-1]:,.0f}원 ~ {upper_ci[-1]:,.0f}원")
time.sleep(0.4)

# ── 6. 시각화 ─────────────────────────────────────────────
print("\n[6/6] 시각화 저장 중...")
time.sleep(0.5)

# 하단 패널은 최근 6개월 + 예측 1개월
recent_cutoff = dates[-1] - pd.DateOffset(months=6)
mask_recent   = dates >= recent_cutoff
recent_dates  = dates[mask_recent]
recent_prices = prices[mask_recent]

fig, axes = plt.subplots(2, 1, figsize=(13, 9))

fig.text(0.5, 0.99,
         f"ARIMA({p},{d},{q}) — 전체 과거 데이터로 학습 후 오늘부터 1개월 미래 예측 ({TICKER})",
         ha='center', fontsize=9, color='#333', weight='bold', va='top')

# ── 상단: 전체 과거 + 미래 예측 ──────────────────────────
ax1 = axes[0]
ax1.plot(dates, prices, color='steelblue', linewidth=1.0, label='실제 주가 (과거)', alpha=0.9)
ax1.plot(future_dates, forecast_mean_vals, color='tomato', linewidth=2.0,
         label='ARIMA 예측 (미래)', zorder=5)
ax1.fill_between(future_dates, lower_ci, upper_ci,
                 alpha=0.25, color='tomato', label='95% 신뢰 구간')
ax1.axvline(dates[-1], color='gray', linestyle=':', linewidth=1.5)
ax1.set_title("전체 구간 — 과거 실제 주가 + 1개월 미래 예측")
ax1.set_xlabel("날짜")
ax1.set_ylabel("주가 (원)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
ax1.legend(fontsize=8)

ax1.annotate('오늘 (예측 시작)',
             xy=(dates[-1], prices[-1]),
             xytext=(0.76, 0.30), textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
ax1.text(0.98, 0.85,
         f"예측 마지막 날\n{future_dates[-1].date()}\n{forecast_mean_vals[-1]:,.0f}원",
         transform=ax1.transAxes, ha='right', fontsize=7, color='darkred',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.8))
ax1.text(0.5, -0.15, '분홍 띠 = 95% 신뢰 구간 (실제 주가가 이 범위 안에 있을 확률이 95%)',
         transform=ax1.transAxes, ha='center', fontsize=7, color='gray')

# ── 하단: 최근 6개월 + 미래 예측 확대 ───────────────────
ax2 = axes[1]
ax2.plot(recent_dates, recent_prices, color='steelblue', linewidth=1.5, label='실제 주가 (최근 6개월)')
ax2.plot(future_dates, forecast_mean_vals, color='tomato', linewidth=2.0,
         marker='o', markersize=3, label='ARIMA 예측 (1개월)', zorder=5)
ax2.fill_between(future_dates, lower_ci, upper_ci,
                 alpha=0.3, color='tomato', label='95% 신뢰 구간')
ax2.axvline(dates[-1], color='gray', linestyle=':', linewidth=1.5)
ax2.set_title(f"확대 — 최근 6개월 + 향후 1개월 예측  |  오늘: {prices[-1]:,.0f}원")
ax2.set_xlabel("날짜")
ax2.set_ylabel("주가 (원)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=7)
ax2.legend(fontsize=8)

# 첫 번째 예측 날 표시
ax2.annotate(f"{future_dates[0].date()}\n{forecast_mean_vals[0]:,.0f}원",
             xy=(future_dates[0], forecast_mean_vals[0]),
             xytext=(0.72, 0.20), textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='tomato'), fontsize=7, color='darkred')
# 마지막 예측 날 표시
ax2.annotate(f"{future_dates[-1].date()}\n{forecast_mean_vals[-1]:,.0f}원",
             xy=(future_dates[-1], forecast_mean_vals[-1]),
             xytext=(0.85, 0.65), textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='tomato'), fontsize=7, color='darkred')
ax2.text(0.5, -0.18,
         '예측값은 참고용입니다 — ARIMA는 추세를 학습하지만 급등락은 예측하기 어렵습니다',
         transform=ax2.transAxes, ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.subplots_adjust(top=0.95)
ticker_tag = TICKER.replace('.', '_')
out_name = f"result/ArimaStockForecast_{ticker_tag}.png"
plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: {out_name}")

print("\n✓ ARIMA 주가 예측 실습 완료!\n")
