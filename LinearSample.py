from sklearn.linear_model import LinearRegression
import numpy as np

# 입력: 공부 시간 (X), 출력: 시험 점수 (y)
X = np.array([[1], [2], [3], [4], [5]])  # 공부 시간
y = np.array([55, 60, 65, 70, 75])       # 시험 점수

model = LinearRegression()
model.fit(X, y)

print("기울기:", model.coef_)     # 시간당 점수 증가량
print("절편:", model.intercept_)  # 공부 안 해도 나오는 점수
print("예측값:", model.predict([[6]]))  # 6시간 공부하면?

# 결과:
# 기울기: [5.]
# 절편: 50.0
# 예측값: [80.]
