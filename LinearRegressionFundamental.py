import time

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

print("=" * 55)
print("  선형 회귀: PER/PBR/ROE → 종목 기대 수익률 예측")
print("=" * 55)

print("\n[1/5] 종목 펀더멘털 데이터 로딩 중...")
time.sleep(0.5)
data = pd.DataFrame({
    'PER': [8.1, 10.5, 7.8, 15.2, 11.3, 9.6, 12.4, 6.9, 14.7, 8.8],
    'PBR': [0.8, 1.2, 0.7, 1.8, 1.1, 0.9, 1.4, 0.6, 1.7, 0.85],
    'ROE': [14.2, 12.5, 16.1, 9.2, 11.8, 13.7, 10.1, 17.4, 8.6, 15.3],
    '기대수익률': [9.5, 7.2, 10.4, 4.8, 6.9, 8.1, 5.9, 11.0, 4.5, 9.0],
})
print(data.to_string(index=False))
time.sleep(0.5)

print("\n[2/5] 학습/테스트 세트 분리 중 (8:2)...")
time.sleep(0.5)
X = data[['PER', 'PBR', 'ROE']]
y = data['기대수익률']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   → 학습: {len(X_train)}개, 테스트: {len(X_test)}개")
time.sleep(0.3)

print("\n[3/5] 선형 회귀 모델 학습 중...")
print("   수식: 기대수익률 = w0 + w1×PER + w2×PBR + w3×ROE")
time.sleep(0.8)
model = LinearRegression()
model.fit(X_train, y_train)
print("   → 학습 완료!")
for feat, coef in zip(['PER', 'PBR', 'ROE'], model.coef_):
    print(f"      {feat} 계수: {coef:.4f}")
    time.sleep(0.2)

print("\n[4/5] 테스트 세트 성능 평가 중 (MAE)...")
time.sleep(0.5)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
for real, pred in zip(y_test, y_pred):
    print(f"   실제={real:.1f}%  예측={pred:.2f}%  오차={abs(real - pred):.2f}%")
    time.sleep(0.2)
print(f"   → 평균 절대 오차(MAE): {mae:.4f}%")
time.sleep(0.3)

print("\n[5/5] 새 종목 기대 수익률 예측 중 (PER=9.2, PBR=0.95, ROE=14.8)...")
time.sleep(0.5)
new_stock = pd.DataFrame({'PER': [9.2], 'PBR': [0.95], 'ROE': [14.8]})
predicted = model.predict(new_stock)
print(f"   → 예상 기대 수익률: {predicted[0]:.2f}%")

print("\n✓ 펀더멘털 선형 회귀 실습 완료!\n")
