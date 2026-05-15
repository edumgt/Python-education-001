import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# 예제 데이터 (주택 면적, 방 개수, 가격)
data = pd.DataFrame({
    '면적': [50, 70, 80, 100, 120, 150, 180, 200, 250, 300],
    '방 개수': [5, 5, 1, 1, 1, 5, 5, 6, 6, 7],
    '가격': [3000, 5000, 5500, 7000, 8500, 11000,
           13000, 15000, 18000, 22000]  # 단위: 만 원
})

# 입력(X)과 출력(y) 정의
X = data[['면적', '방 개수']]
y = data['가격']

# 데이터 분할 (훈련 80%, 테스트 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 모델 성능 평가 (MAE 사용)
mae = mean_absolute_error(y_test, y_pred)
print(f"평균 절대 오차(MAE): {mae:.2f}")

# 새 데이터 예측
new_data = pd.DataFrame({'면적': [220], '방 개수': [5]})
predicted_price = model.predict(new_data)
print(f"예측 가격: {predicted_price[0]:.2f} 만 원")
