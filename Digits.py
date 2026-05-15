import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 가상 기술지표 데이터 생성: RSI, MACD, 거래량증가율, 변동성
np.random.seed(42)
samples = 400
rsi = np.random.uniform(20, 80, samples)
macd = np.random.normal(0, 1.2, samples)
volume_change = np.random.normal(0, 6, samples)
volatility = np.random.uniform(0.5, 3.5, samples)

X = np.column_stack([rsi, macd, volume_change, volatility])

# 라벨: 상승장(1) / 하락장(0)
y = ((rsi > 50) & (macd > 0) & (volume_change > -1) & (volatility < 2.8)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = SVC(gamma=0.2)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))
print("분류 리포트:")
print(classification_report(y_test, y_pred))

# 예측 결과 일부 시각화
points = X_test[:40]
preds = y_pred[:40]
colors = np.where(preds == 1, 'tomato', 'royalblue')

plt.figure(figsize=(7, 4))
plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.7)
plt.title("RSI-MACD 기반 시장 국면 예측")
plt.xlabel("RSI")
plt.ylabel("MACD")
plt.tight_layout()
plt.show()
