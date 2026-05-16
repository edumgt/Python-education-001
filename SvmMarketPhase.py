import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

os.makedirs("result", exist_ok=True)

print("=" * 55)
print("  SVM: RSI/MACD 기반 시장 국면(상승/하락) 분류 실습")
print("=" * 55)

print("\n[1/5] 가상 기술지표 데이터 생성 중 (RSI, MACD, 거래량, 변동성)...")
time.sleep(0.5)
np.random.seed(42)
samples = 400
rsi = np.random.uniform(20, 80, samples)
macd = np.random.normal(0, 1.2, samples)
volume_change = np.random.normal(0, 6, samples)
volatility = np.random.uniform(0.5, 3.5, samples)
X = np.column_stack([rsi, macd, volume_change, volatility])
y = ((rsi > 50) & (macd > 0) & (volume_change > -1) & (volatility < 2.8)).astype(int)
print(f"   → {samples}개 샘플 생성")
print(f"      상승장(1): {y.sum()}개  |  하락장(0): {(y == 0).sum()}개")
time.sleep(0.5)

print("\n[2/5] 학습/테스트 세트 분리 중 (8:2)...")
time.sleep(0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   → 학습: {len(X_train)}개  |  테스트: {len(X_test)}개")
time.sleep(0.3)

print("\n[3/5] SVM RBF 커널 모델 학습 중...")
print("   RBF = 방사형 기저 함수 (곡선 경계로 복잡한 패턴 학습)")
print("   gamma=0.2: 결정 경계의 곡률 조절 (작을수록 완만)")
time.sleep(0.8)
clf = SVC(gamma=0.2)
clf.fit(X_train, y_train)
print(f"   → 학습 완료!  서포트 벡터 수: {sum(clf.n_support_)}개")
time.sleep(0.5)

print("\n[4/5] 테스트 세트 예측 & 성능 평가 중...")
time.sleep(0.5)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"   → 정확도: {acc:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=['하락장', '상승장']))
time.sleep(0.5)

print("\n[5/5] 예측 결과 시각화 중 (RSI vs MACD)...")
time.sleep(0.5)
points = X_test[:40]
preds = y_pred[:40]
colors = np.where(preds == 1, 'tomato', 'royalblue')
plt.figure(figsize=(7, 4))
plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.7)
plt.title("RSI-MACD 기반 시장 국면 예측")
plt.xlabel("RSI")
plt.ylabel("MACD")
plt.tight_layout()
plt.savefig("result/SvmMarketPhase.png", dpi=150, bbox_inches="tight")
print("   → 그래프 저장: result/SvmMarketPhase.png")

print("\n✓ SVM 시장 국면 분류 실습 완료!\n")
