import matplotlib.pyplot as plt                                    # 산점도 등 그래프 시각화 라이브러리
import numpy as np                                                # 수치 배열 연산 라이브러리
import korean_font  # noqa: F401                                  # 한글 폰트 설정 모듈 (사이드이펙트로 폰트 적용)
from sklearn.metrics import accuracy_score, classification_report  # 정확도 및 상세 분류 성능 지표
from sklearn.model_selection import train_test_split              # 데이터를 학습/테스트 세트로 분리
from sklearn.svm import SVC                                       # Support Vector Classification 모델

# 가상 기술지표 데이터 생성: RSI, MACD, 거래량증가율, 변동성
np.random.seed(42)   # 난수 고정 → 실행마다 동일한 데이터 생성
samples = 400        # 생성할 샘플(가상 거래일) 수
rsi = np.random.uniform(20, 80, samples)        # RSI 지표: 20~80 사이 균등분포로 생성 (과매도~과매수 범위)
macd = np.random.normal(0, 1.2, samples)        # MACD 지표: 평균 0, 표준편차 1.2인 정규분포로 생성
volume_change = np.random.normal(0, 6, samples) # 거래량 변화율(%): 평균 0, 표준편차 6인 정규분포로 생성
volatility = np.random.uniform(0.5, 3.5, samples)  # 변동성(%): 0.5~3.5 사이 균등분포로 생성

X = np.column_stack([rsi, macd, volume_change, volatility])  # 4개 지표를 열 방향으로 합쳐 (400×4) 특성 행렬 생성

# 라벨: 상승장(1) / 하락장(0)
# RSI > 50(강세), MACD > 0(상승 모멘텀), 거래량 감소 없음, 변동성 낮음 → 상승장으로 정의
y = ((rsi > 50) & (macd > 0) & (volume_change > -1) & (volatility < 2.8)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 80%를 학습, 20%를 테스트용으로 분리, 시드 고정
)

clf = SVC(gamma=0.2)      # RBF 커널 SVM 생성, gamma는 결정 경계의 곡률 조절 (작을수록 완만)
clf.fit(X_train, y_train) # 학습 데이터로 SVM 모델 훈련

y_pred = clf.predict(X_test)                             # 테스트 입력으로 상승/하락 클래스 예측
print("정확도:", accuracy_score(y_test, y_pred))          # 예측이 정답과 일치한 비율 출력
print("분류 리포트:")                                     # 헤더 출력
print(classification_report(y_test, y_pred))             # 클래스별 정밀도·재현율·F1·지지도 출력

# 예측 결과 일부 시각화
points = X_test[:40]   # 테스트 세트에서 앞 40개 샘플만 시각화용으로 추출
preds = y_pred[:40]    # 위 40개 샘플에 대한 예측 레이블 추출
colors = np.where(preds == 1, 'tomato', 'royalblue')  # 상승장(1)은 빨강, 하락장(0)은 파랑으로 색 지정

plt.figure(figsize=(7, 4))                                   # 가로 7인치, 세로 4인치 그래프 창 생성
plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.7) # RSI(x축) vs MACD(y축) 산점도, 투명도 0.7
plt.title("RSI-MACD 기반 시장 국면 예측")  # 그래프 제목
plt.xlabel("RSI")                           # x축: RSI 값
plt.ylabel("MACD")                          # y축: MACD 값
plt.tight_layout()                          # 여백 자동 조정으로 레이블 잘림 방지
plt.savefig("Digits.png", dpi=150, bbox_inches="tight")  # PNG 파일로 저장
print("그래프 저장: Digits.png")            # 저장 완료 메시지 출력
