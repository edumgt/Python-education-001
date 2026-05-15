import numpy as np
from sklearn.linear_model import LinearRegression

# 입력: 거래량(만주), 변동성(%) / 출력: 다음 날 수익률(%)
X = np.array([
    [12, 1.2],
    [18, 1.5],
    [25, 1.8],
    [30, 2.1],
    [38, 2.3],
    [45, 2.6],
])
y = np.array([0.3, 0.45, 0.62, 0.76, 0.9, 1.05])

model = LinearRegression()
model.fit(X, y)

print("회귀 계수:", model.coef_)
print("절편:", model.intercept_)

new_signal = np.array([[33, 2.0]])
predicted_return = model.predict(new_signal)
print("예측 다음 날 수익률(%):", predicted_return[0])
