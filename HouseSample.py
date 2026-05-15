import pandas as pd                                    # 데이터프레임 기반 데이터 처리 라이브러리
from sklearn.linear_model import LinearRegression     # 선형 회귀 모델
from sklearn.metrics import mean_absolute_error       # MAE(평균 절대 오차) 계산 함수
from sklearn.model_selection import train_test_split  # 학습/테스트 세트 분리 함수

# 종목 펀더멘털 예시 데이터 (PER, PBR, ROE -> 기대 수익률)
data = pd.DataFrame({
    'PER': [8.1, 10.5, 7.8, 15.2, 11.3, 9.6, 12.4, 6.9, 14.7, 8.8],      # PER(주가수익비율): 낮을수록 저평가 경향
    'PBR': [0.8, 1.2, 0.7, 1.8, 1.1, 0.9, 1.4, 0.6, 1.7, 0.85],          # PBR(주가순자산비율): 낮을수록 자산 대비 저평가
    'ROE': [14.2, 12.5, 16.1, 9.2, 11.8, 13.7, 10.1, 17.4, 8.6, 15.3],   # ROE(자기자본이익률, %): 높을수록 수익성 우수
    '기대수익률': [9.5, 7.2, 10.4, 4.8, 6.9, 8.1, 5.9, 11.0, 4.5, 9.0],  # 목표 변수: 해당 종목의 기대 연간 수익률(%)
})

X = data[['PER', 'PBR', 'ROE']]  # 입력 특성: PER·PBR·ROE 3개 컬럼 선택
y = data['기대수익률']            # 출력 타깃: 기대수익률 컬럼 선택

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 전체의 20%(2개 샘플)를 테스트용으로 분리, 시드 42로 고정
)

model = LinearRegression()       # 선형 회귀 모델 객체 생성 (y = w0 + w1*PER + w2*PBR + w3*ROE 형태)
model.fit(X_train, y_train)      # 학습 데이터로 최적 회귀 계수(가중치) 산출

y_pred = model.predict(X_test)                    # 테스트 입력에 대해 기대수익률 예측
mae = mean_absolute_error(y_test, y_pred)         # 예측값과 실제값의 절대 오차 평균 계산
print(f"평균 절대 오차(MAE): {mae:.2f}")           # MAE 출력 (단위: %)

new_stock = pd.DataFrame({'PER': [9.2], 'PBR': [0.95], 'ROE': [14.8]})  # 새로운 미지 종목의 펀더멘털 데이터
predicted_return = model.predict(new_stock)        # 학습된 모델로 해당 종목의 기대수익률 예측
print(f"예상 수익률: {predicted_return[0]:.2f}%")  # 예측 수익률 출력
