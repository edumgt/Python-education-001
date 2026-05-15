from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
X, y = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_clusters_per_class=1, n_classes=2, random_state=42
)

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# SVM 학습
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 예측 및 정확도
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)


# 시각화
def plot_classification(X, y, clf):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title(f"SVM Classification (accuracy: {acc:.2f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


plot_classification(X, y, model)
