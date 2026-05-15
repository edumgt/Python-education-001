import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# 2개 특성(모멘텀, 거래량 변화율) 기반 매수/매도 신호 데이터
np.random.seed(42)
X = np.random.randn(240, 2)
X[:, 0] = X[:, 0] * 2.0 + 0.5    # 모멘텀
X[:, 1] = X[:, 1] * 1.5 - 0.2    # 거래량 변화율

y = ((X[:, 0] > 0.3) & (X[:, 1] > -0.1)).astype(int)  # 1: 매수, 0: 보류/매도

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"신호 분류 정확도: {acc:.2f}")


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
    plt.show()


plot_signal_boundary(X, y, model)
