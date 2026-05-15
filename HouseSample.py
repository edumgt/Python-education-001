import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 종목 펀더멘털 예시 데이터 (PER, PBR, ROE -> 기대 수익률)
data = pd.DataFrame({
    'PER': [8.1, 10.5, 7.8, 15.2, 11.3, 9.6, 12.4, 6.9, 14.7, 8.8],
    'PBR': [0.8, 1.2, 0.7, 1.8, 1.1, 0.9, 1.4, 0.6, 1.7, 0.85],
    'ROE': [14.2, 12.5, 16.1, 9.2, 11.8, 13.7, 10.1, 17.4, 8.6, 15.3],
    '기대수익률': [9.5, 7.2, 10.4, 4.8, 6.9, 8.1, 5.9, 11.0, 4.5, 9.0],
})

X = data[['PER', 'PBR', 'ROE']]
y = data['기대수익률']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"평균 절대 오차(MAE): {mae:.2f}")

new_stock = pd.DataFrame({'PER': [9.2], 'PBR': [0.95], 'ROE': [14.8]})
predicted_return = model.predict(new_stock)
print(f"예상 수익률: {predicted_return[0]:.2f}%")
