import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

os.makedirs("result", exist_ok=True)

print("=" * 55)
print("  SVM: 모멘텀/거래량 기반 매매 신호 분류 실습")
print("=" * 55)

print("\n[1/5] 가상 매매 신호 데이터 생성 중...")
time.sleep(0.5)
np.random.seed(42)
X = np.random.randn(240, 2)
X[:, 0] = X[:, 0] * 2.0 + 0.5
X[:, 1] = X[:, 1] * 1.5 - 0.2
y = ((X[:, 0] > 0.3) & (X[:, 1] > -0.1)).astype(int)
buy_count = y.sum()
print(f"   → 총 {len(X)}개 샘플 생성")
print(f"      매수 신호(1): {buy_count}개  |  매도/보류(0): {len(y) - buy_count}개")
time.sleep(0.5)

print("\n[2/5] 학습/테스트 세트 분리 중 (7:3)...")
time.sleep(0.4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"   → 학습: {len(X_train)}개  |  테스트: {len(X_test)}개")
time.sleep(0.3)

print("\n[3/5] SVM 선형 커널 모델 학습 중...")
print("   원리: 두 클래스 사이 경계선(초평면)을 찾아 마진을 최대화")
time.sleep(0.8)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
print(f"   → 학습 완료!  서포트 벡터 수: {sum(model.n_support_)}개")
time.sleep(0.5)

print("\n[4/5] 테스트 세트 예측 & 정확도 평가 중...")
time.sleep(0.5)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
correct = (y_pred == y_test).sum()
print(f"   → {len(X_test)}개 중 {correct}개 정확히 분류")
print(f"   → 신호 분류 정확도: {acc:.4f} ({acc * 100:.1f}%)")
time.sleep(0.5)

print("\n[5/5] 결정 경계 시각화 중...")
time.sleep(0.5)


def plot_signal_boundary(features, labels, clf):
    step = 0.02
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    grid_pred = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, grid_pred, cmap=plt.cm.coolwarm, alpha=0.25)
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(f"SVM 매매 신호 경계 (accuracy: {acc:.2f})")
    plt.xlabel("모멘텀")
    plt.ylabel("거래량 변화율")
    plt.tight_layout()
    plt.savefig("result/SvmTradeSignal.png", dpi=150, bbox_inches="tight")
    print("   → 그래프 저장: result/SvmTradeSignal.png")


plot_signal_boundary(X, y, model)
print("\n✓ SVM 매매 신호 분류 실습 완료!\n")
