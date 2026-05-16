import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
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
print("\n[1/7] 주가 데이터 로드 중...")
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
    days = 200
    t = np.arange(days)
    prices = (100 + 0.1 * t + 3 * np.sin(t / 15) + np.random.normal(0, 1.2, days)).astype(np.float32)
    print(f"   → 가상 {days}일치 주가 생성")

days = len(prices)
split_idx = int(days * 0.8)
train_prices = prices[:split_idx]
test_prices  = prices[split_idx:]
print(f"   → 학습 데이터: {len(train_prices)}일  |  테스트(예측 대상): {len(test_prices)}일")
time.sleep(0.3)

# ── 2. 정상성 검정 ─────────────────────────────────────────
print("\n[2/7] ADF 정상성 검정 중...")
print("   가설: 'H0 = 비정상 시계열이다'  →  p < 0.05 이면 H0 기각 = 정상")
time.sleep(0.5)
adf_raw  = adfuller(train_prices)
adf_diff = adfuller(np.diff(train_prices))
print(f"   원본 주가   p-value: {adf_raw[1]:.4f}  → {'비정상 → 차분 필요' if adf_raw[1] > 0.05 else '정상'}")
print(f"   1차 차분   p-value: {adf_diff[1]:.4f}  → {'비정상' if adf_diff[1] > 0.05 else '정상 → d=1 확정'}")
d = 1 if adf_raw[1] > 0.05 else 0
print(f"   → 차분 횟수(d) = {d}")
time.sleep(0.4)

# ── 3. p, q 결정 ──────────────────────────────────────────
print("\n[3/7] ARIMA 차수 결정 중 (p=2, d=1, q=2 사용)...")
print("   방법: ACF로 q, PACF로 p를 추정합니다")
print("   이 실습에서는 주가 예측 일반적 조합 ARIMA(2,1,2)를 사용합니다")
p, q = 2, 2
print(f"   → 최종 설정: ARIMA({p}, {d}, {q})")
time.sleep(0.5)

# ── 4. 모델 학습 ──────────────────────────────────────────
print("\n[4/7] ARIMA 모델 학습 중...")
print("   AR(자기회귀): 과거 주가가 오늘 주가에 미치는 영향 학습")
print("   MA(이동평균): 과거 예측 오차가 오늘에 미치는 영향 학습")
time.sleep(0.8)
model = ARIMA(train_prices, order=(p, d, q))
fitted = model.fit()
print(f"   → 학습 완료!")
print(f"      AIC: {fitted.aic:.2f}  (작을수록 더 좋은 모델)")
print(f"      BIC: {fitted.bic:.2f}")
time.sleep(0.5)

# ── 5. 학습 데이터 내 적합값 ──────────────────────────────
print("\n[5/7] 학습 기간 내 적합값(fitted values) 확인 중...")
time.sleep(0.4)
in_sample_pred = fitted.fittedvalues
residuals = train_prices[1:] - in_sample_pred[1:]
rmse_train = np.sqrt(np.mean(residuals ** 2))
print(f"   → 학습 기간 RMSE: {rmse_train:.4f}원")
time.sleep(0.3)

# ── 6. 미래 예측 ──────────────────────────────────────────
print("\n[6/7] 테스트 구간 예측 중 (40일)...")
print("   방법: 1일씩 예측 후 실제값을 반영해 다음 예측 (워킹 포워드)")
time.sleep(0.5)
history = list(train_prices)
one_step_preds = []
for i in range(len(test_prices)):
    m = ARIMA(history, order=(p, d, q))
    f = m.fit()
    yhat = f.forecast(steps=1)[0]
    one_step_preds.append(yhat)
    history.append(test_prices[i])
    if (i + 1) % 10 == 0:
        print(f"   {i + 1}/{len(test_prices)}일 예측 완료...")
        time.sleep(0.2)
one_step_preds = np.array(one_step_preds)
rmse_test = np.sqrt(np.mean((test_prices - one_step_preds) ** 2))
mae_test  = np.mean(np.abs(test_prices - one_step_preds))
print(f"   → 테스트 RMSE: {rmse_test:.4f}원  |  MAE: {mae_test:.4f}원")
time.sleep(0.3)

# ── 7. 시각화 ─────────────────────────────────────────────
print("\n[7/7] 시각화 저장 중...")
time.sleep(0.5)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 전체 그래프 요약 제목
fig.text(0.5, 0.98, '과거 주가의 패턴을 수식으로 정리해 미래를 예측합니다 (ARIMA)',
         ha='center', fontsize=9, color='#333', weight='bold')

# 전체 구간
ax = axes[0]
ax.plot(range(days), prices, color='steelblue', linewidth=1.2, label='실제 주가', alpha=0.8)
ax.plot(range(1, len(in_sample_pred)), in_sample_pred[1:],
        color='orange', linewidth=1.0, linestyle='--', label='ARIMA 적합값(학습)')
ax.plot(range(split_idx, days), one_step_preds,
        color='tomato', linewidth=1.5, label='ARIMA 예측(테스트)')
ax.axvline(split_idx, color='gray', linestyle=':', linewidth=1.5, label='학습/테스트 경계')
ax.set_title(f"ARIMA({p},{d},{q}) 주가 예측  |  테스트 RMSE={rmse_test:.2f}")
ax.set_xlabel("거래일")
ax.set_ylabel("주가")
ax.legend(fontsize=9)
# 학습 구간 설명
ax.text(0.35, 0.12, '컴퓨터가 공부한 구간', transform=ax.transAxes,
        ha='center', fontsize=8, color='steelblue',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.4))
# 테스트 구간 설명
ax.text(0.88, 0.12, '실제로 예측해본 구간\n(답을 보여주지 않음)', transform=ax.transAxes,
        ha='center', fontsize=8, color='tomato',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', alpha=0.4))
# 세로 점선 화살표 어노테이션
ax.annotate('여기서부터 진짜 예측 시작!', xy=(0.80, 0.60), xytext=(0.65, 0.78),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
# 예측선 설명
ax.annotate('ARIMA 예측값', xy=(0.88, 0.75), xytext=(0.70, 0.90),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='tomato'), fontsize=7, color='darkred')
ax.text(0.5, -0.12, 'ARIMA = 자기회귀(AR) + 차분(I) + 이동평균오차(MA)의 조합 모델',
        transform=ax.transAxes, ha='center', fontsize=7, color='gray')

# 테스트 구간 확대
ax2 = axes[1]
ax2.plot(range(len(test_prices)), test_prices,
         color='steelblue', linewidth=1.5, label='실제 주가')
ax2.plot(range(len(test_prices)), one_step_preds,
         color='tomato', linewidth=1.5, linestyle='--', label='예측 주가')
ax2.fill_between(range(len(test_prices)),
                 test_prices, one_step_preds, alpha=0.2, color='gray', label='오차 구간')
ax2.set_title("테스트 구간 확대 (실제 vs 예측)")
ax2.set_xlabel("테스트 인덱스")
ax2.set_ylabel("주가")
ax2.legend(fontsize=9)
# 회색 띠 설명
ax2.annotate('이 범위 안에 실제값이 있을 거라는 오차 구간',
             xy=(0.5, 0.5), xytext=(0.55, 0.25),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
# 오차 설명
ax2.annotate('이 차이가 오차예요',
             xy=(0.25, 0.65), xytext=(0.05, 0.85),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
ax2.text(0.5, -0.12, '실제선(파랑)과 예측선(빨강)이 가까울수록 예측이 잘 된 것입니다',
         transform=ax2.transAxes, ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
ticker_tag = TICKER.replace('.', '_')
plt.savefig(f"result/ArimaStockForecast_{ticker_tag}.png", dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: result/ArimaStockForecast_{ticker_tag}.png")

print("\n✓ ARIMA 주가 예측 실습 완료!\n")
