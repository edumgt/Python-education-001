import time

import numpy as np
from sklearn.linear_model import LinearRegression

print("=" * 55)
print("  선형 회귀: 거래량/변동성 → 다음 날 수익률 예측")
print("=" * 55)

print("\n[1/4] 학습 데이터 준비 중...")
time.sleep(0.5)
X = np.array([
    [12, 1.2], [18, 1.5], [25, 1.8],
    [30, 2.1], [38, 2.3], [45, 2.6],
])
y = np.array([0.3, 0.45, 0.62, 0.76, 0.9, 1.05])
print(f"   → 학습 샘플: {len(X)}개  |  특성: [거래량(만주), 변동성(%)]")
for xi, yi in zip(X, y):
    print(f"      거래량={xi[0]:2.0f}만주, 변동성={xi[1]}%  →  다음날 수익률={yi}%")
    time.sleep(0.15)

print("\n[2/4] 선형 회귀 모델 학습 중 (최소제곱법)...")
print("   수식: 수익률 = w0 + w1×거래량 + w2×변동성")
time.sleep(0.8)
model = LinearRegression()
model.fit(X, y)
print("   → 학습 완료!")
print(f"      절편(w0)       = {model.intercept_:.4f}")
print(f"      거래량 계수(w1) = {model.coef_[0]:.4f}")
print(f"      변동성 계수(w2) = {model.coef_[1]:.4f}")
time.sleep(0.5)

print("\n[3/4] 회귀 공식 확인...")
time.sleep(0.4)
print(f"   수익률 = {model.intercept_:.4f}"
      f" + {model.coef_[0]:.4f}×거래량"
      f" + {model.coef_[1]:.4f}×변동성")
time.sleep(0.5)

print("\n[4/4] 새 데이터 예측: 거래량=33만주, 변동성=2.0%")
time.sleep(0.5)
new_signal = np.array([[33, 2.0]])
predicted = model.predict(new_signal)
print(f"   → 예측 다음 날 수익률: {predicted[0]:.4f}%")

print("\n✓ 선형 회귀 실습 완료!\n")
