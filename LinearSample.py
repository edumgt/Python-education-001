import numpy as np                              # 배열 생성 및 수치 연산 라이브러리
from sklearn.linear_model import LinearRegression  # 선형 회귀 모델 클래스

# 입력: 거래량(만주), 변동성(%) / 출력: 다음 날 수익률(%)
X = np.array([        # 학습 입력 행렬: 각 행이 하나의 거래일 데이터
    [12, 1.2],        # 거래량 12만주, 변동성 1.2%
    [18, 1.5],        # 거래량 18만주, 변동성 1.5%
    [25, 1.8],        # 거래량 25만주, 변동성 1.8%
    [30, 2.1],        # 거래량 30만주, 변동성 2.1%
    [38, 2.3],        # 거래량 38만주, 변동성 2.3%
    [45, 2.6],        # 거래량 45만주, 변동성 2.6%
])
y = np.array([0.3, 0.45, 0.62, 0.76, 0.9, 1.05])  # 각 거래일의 다음 날 수익률(%) 정답 레이블

model = LinearRegression()  # 선형 회귀 모델 객체 생성 (수익률 = w0 + w1*거래량 + w2*변동성)
model.fit(X, y)             # 최소제곱법으로 최적 가중치(회귀 계수) 산출

print("회귀 계수:", model.coef_)         # 각 특성(거래량, 변동성)에 대한 기울기 출력
print("절편:", model.intercept_)         # 회귀식의 상수항(y절편) 출력

new_signal = np.array([[33, 2.0]])       # 예측할 새 데이터: 거래량 33만주, 변동성 2.0%
predicted_return = model.predict(new_signal)  # 학습된 모델로 다음 날 수익률 예측
print("예측 다음 날 수익률(%):", predicted_return[0])  # 예측 수익률 출력
