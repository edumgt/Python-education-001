import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import korean_font  # noqa: F401
from sklearn.linear_model import LinearRegression

os.makedirs("result", exist_ok=True)

print("=" * 55)
print("  선형 회귀: 거래량/변동성 → 다음 날 수익률 예측")
print("=" * 55)

# ── 실제 주가 데이터 다운로드 ────────────────────────────────────────────────
TICKER = '078935.KS'
prices_raw = None
try:
    import yfinance as yf
    from datetime import date
    df = yf.download(TICKER, start='2020-01-01', end=date.today().isoformat(),
                     auto_adjust=True, progress=False)
    if len(df) > 50:
        prices_raw = df['Close'].squeeze().dropna().values.flatten().astype(np.float32)
        print(f"   ✓ {TICKER}: {len(prices_raw)}일 실제 데이터 로드")
except Exception as e:
    print(f"   yfinance 오류 ({e}) → 가상 데이터 사용")


def compute_returns(prices):
    return np.diff(prices.astype(float)) / prices[:-1]


def compute_volume_change(n):
    """가상 거래량 변화율 (실제 거래량 없으면 noise로 대체)"""
    return np.random.normal(0, 0.5, n)

# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/4] 학습 데이터 준비 중...")
time.sleep(0.5)

if prices_raw is not None and len(prices_raw) > 50:
    np.random.seed(42)
    returns = compute_returns(prices_raw)           # shape: (N-1,)
    volume_change = compute_volume_change(len(returns))  # shape: (N-1,)

    # 다음 날 수익률을 레이블로 사용 (returns[i]가 i+1일의 수익률이므로
    # features는 returns[:-1], volume_change[:-1], 레이블은 returns[1:])
    feat_ret = returns[:-1]          # 전일 수익률 (%)
    feat_vol = volume_change[:-1]    # 전일 거래량 변화율
    label = returns[1:] * 100        # 다음 날 수익률 (%)

    # 극단값 제거 (|수익률| < 5%)
    mask = (np.abs(feat_ret) < 0.05) & (np.abs(label) < 5)
    feat_ret = feat_ret[mask] * 100
    feat_vol = feat_vol[mask]
    label = label[mask]

    X = np.column_stack([feat_vol, feat_ret])  # [거래량변화율, 전일수익률]
    y = label
    print(f"   → 실제 데이터 샘플: {len(X)}개  |  특성: [거래량변화율, 전일수익률(%)]")
    data_source = f"GS P&N({TICKER}) 실제 데이터"
    x_label_str = "거래량 변화율"
else:
    # ── fallback: 가상 데이터 ───────────────────────────────────────────────
    X = np.array([
        [12, 1.2], [18, 1.5], [25, 1.8],
        [30, 2.1], [38, 2.3], [45, 2.6],
    ])
    y = np.array([0.3, 0.45, 0.62, 0.76, 0.9, 1.05])
    print(f"   → 학습 샘플: {len(X)}개  |  특성: [거래량(만주), 변동성(%)]")
    for xi, yi in zip(X, y):
        print(f"      거래량={xi[0]:2.0f}만주, 변동성={xi[1]}%  →  다음날 수익률={yi}%")
        time.sleep(0.15)
    data_source = "가상 데이터"
    x_label_str = "거래량 (만주)"

print("\n[2/4] 선형 회귀 모델 학습 중 (최소제곱법)...")
print("   수식: 수익률 = w0 + w1×거래량변화율 + w2×전일수익률")
time.sleep(0.8)
model = LinearRegression()
model.fit(X, y)
print("   → 학습 완료!")
print(f"      절편(w0)          = {model.intercept_:.4f}")
print(f"      거래량변화율 계수(w1) = {model.coef_[0]:.4f}")
print(f"      전일수익률 계수(w2)  = {model.coef_[1]:.4f}")
time.sleep(0.5)

print("\n[3/4] 회귀 공식 확인...")
time.sleep(0.4)
print(f"   수익률 = {model.intercept_:.4f}"
      f" + {model.coef_[0]:.4f}×거래량변화율"
      f" + {model.coef_[1]:.4f}×전일수익률")
time.sleep(0.5)

print("\n[4/4] 새 데이터 예측: 거래량변화율=0.3, 전일수익률=0.5%")
time.sleep(0.5)
new_signal = np.array([[0.3, 0.5]])
predicted = model.predict(new_signal)
print(f"   → 예측 다음 날 수익률: {predicted[0]:.4f}%")

print("\n[5/4] 결과 시각화 중 (산점도 + 회귀선)...")
time.sleep(0.3)

# 거래량변화율(X의 첫 번째 특성)으로만 1D 예측선을 그린다
volume = X[:, 0]
volume_range = np.linspace(volume.min() - 0.5, volume.max() + 0.5, 200)

# 두 번째 특성을 평균값으로 고정한 뒤 첫 번째 특성만 변화시켜 예측
feat2_mean = X[:, 1].mean()
X_line = np.column_stack([volume_range, np.full_like(volume_range, feat2_mean)])
y_line = model.predict(X_line)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(volume, y, c='steelblue', edgecolors='k', linewidths=0.5,
           zorder=3, label='실제 데이터', alpha=0.5)
ax.plot(volume_range, y_line, color='tomato', linewidth=2, label='예측선 (회귀선)')
ax.set_xlabel(x_label_str)
ax.set_ylabel("다음 날 수익률 (%)")
ax.set_title(f"선형 회귀: {x_label_str} → 다음 날 수익률 예측\n({data_source})")
ax.legend()

# ── 초등학생도 이해할 수 있는 한글 설명 어노테이션 ──────────────────────────
# 각 점 설명 – 첫 번째 점에 화살표
ax.annotate('각 점 = 하루치 주가 데이터', xy=(volume[0], y[0]),
            xytext=(volume[0] + (volume.max() - volume.min()) * 0.2, y[0] - (y.max() - y.min()) * 0.15),
            fontsize=8, color='#333',
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.0),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# 회귀선 중간에 화살표
mid_idx = len(volume_range) // 2
ax.annotate('컴퓨터가 찾은 예측선\n(이 직선이 가까울수록 예측이 잘 맞아요)',
            xy=(volume_range[mid_idx], y_line[mid_idx]),
            xytext=(volume_range[mid_idx] - (volume.max() - volume.min()) * 0.4,
                    y_line[mid_idx] + (y.max() - y.min()) * 0.2),
            fontsize=7, color='darkred',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.0),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='tomato'))

# x축 보충 설명
ax.text(0.5, -0.13, f'{x_label_str}이 많을수록 →  (거래가 활발한 날일수록 오른쪽)',
        transform=ax.transAxes, ha='center', fontsize=7, color='gray')
# y축 보충 설명
ax.text(-0.18, 0.5, '위로 갈수록 더 많이 오른 날',
        transform=ax.transAxes, va='center', rotation=90, fontsize=7, color='gray')

# 전체 그래프 한 줄 요약
fig.text(0.5, 0.995,
         '거래량으로 다음날 수익률을 예측합니다 — 직선이 가까울수록 예측이 잘 맞아요',
         ha='center', fontsize=9, color='#333', weight='bold', va='top')
# ────────────────────────────────────────────────────────────────────────────

plt.tight_layout()
plt.savefig("result/LinearRegressionReturn_078935_KS.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/LinearRegressionReturn_078935_KS.png")

print("\n✓ 선형 회귀 실습 완료!\n")
