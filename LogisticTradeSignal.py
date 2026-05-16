import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs("result", exist_ok=True)

print("=" * 60)
print("  로지스틱 회귀: RSI/MACD 기반 매수/매도 신호 분류 실습")
print("=" * 60)

print("\n[1/6] 가상 RSI & MACD 데이터 생성 중 (300개 거래일)...")
time.sleep(0.5)
np.random.seed(42)
n = 300
rsi = np.random.uniform(20, 80, n)
macd = np.random.normal(0, 1, n)
y = ((rsi > 55) & (macd > 0.1)).astype(int)
noise_idx = np.random.choice(n, size=int(n * 0.08), replace=False)
y[noise_idx] = 1 - y[noise_idx]
print(f"   → {n}개 샘플  |  매수(1): {y.sum()}개  |  매도/관망(0): {(y == 0).sum()}개")
print(f"   (노이즈 {int(n * 0.08)}개 추가 → 현실적인 경계 모사)")
time.sleep(0.5)

print("\n[2/6] 학습/테스트 세트 분리 중 (75:25)...")
time.sleep(0.4)
X = np.column_stack([rsi, macd])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"   → 학습: {len(X_train)}개  |  테스트: {len(X_test)}개")
time.sleep(0.3)

print("\n[3/6] StandardScaler로 표준화 중...")
print("   이유: RSI(20~80)와 MACD(-3~3)의 스케일 차이 제거")
time.sleep(0.5)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("   → 표준화 완료 (평균=0, 표준편차=1)")
time.sleep(0.3)

print("\n[4/6] 로지스틱 회귀 모델 학습 중...")
print("   수식: P(매수) = sigmoid(w0 + w1×RSI + w2×MACD)")
print("   sigmoid: 어떤 숫자든 0~1 확률로 변환")
time.sleep(0.8)
model = LogisticRegression()
model.fit(X_train_s, y_train)
print("   → 학습 완료!")
print(f"      RSI 계수: {model.coef_[0][0]:.4f}  |  MACD 계수: {model.coef_[0][1]:.4f}")
print(f"      절편: {model.intercept_[0]:.4f}")
time.sleep(0.5)

print("\n[5/6] 테스트 세트 평가 중...")
time.sleep(0.5)
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(classification_report(y_test, y_pred, target_names=['매도/관망', '매수']))
print(f"   ROC-AUC: {auc:.4f}  (1.0에 가까울수록 완벽한 분류)")
time.sleep(0.5)

print("\n[6/6] 매수 확률 결정 경계 시각화 중...")
time.sleep(0.5)
step = 0.5
x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
grid = scaler.transform(np.c_[xx.ravel(), yy.ravel()])
zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, zz, levels=20, cmap='RdYlGn', alpha=0.4)
plt.colorbar(label='매수 확률')
buy_mask = y == 1
plt.scatter(X[buy_mask, 0], X[buy_mask, 1], c='tomato', label='매수(1)',
            edgecolors='k', linewidths=0.4, alpha=0.7)
plt.scatter(X[~buy_mask, 0], X[~buy_mask, 1], c='royalblue', label='매도/관망(0)',
            edgecolors='k', linewidths=0.4, alpha=0.7)
plt.xlabel("RSI")
plt.ylabel("MACD")
plt.title("로지스틱 회귀: 매수/매도 신호 분류")
plt.legend()
plt.tight_layout()
plt.savefig("result/LogisticTradeSignal.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/LogisticTradeSignal.png")

print("\n✓ 로지스틱 회귀 실습 완료!\n")
