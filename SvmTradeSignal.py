import matplotlib.pyplot as plt                   # 결정 경계 및 산점도 시각화 라이브러리
import numpy as np                               # 배열 생성 및 메시그리드 연산 라이브러리
import korean_font  # noqa: F401                 # 한글 폰트 설정 모듈 (임포트 시 폰트 자동 적용)
from sklearn.metrics import accuracy_score       # 분류 정확도 계산 함수
from sklearn.model_selection import train_test_split  # 학습/테스트 세트 분리 함수
from sklearn.svm import SVC                     # Support Vector Classification 모델

# 2개 특성(모멘텀, 거래량 변화율) 기반 매수/매도 신호 데이터
np.random.seed(42)            # 난수 시드 고정 → 실행마다 동일한 데이터 생성
X = np.random.randn(240, 2)  # 표준정규분포에서 240개 샘플 × 2개 특성 난수 생성
X[:, 0] = X[:, 0] * 2.0 + 0.5   # 첫 번째 특성: 평균 0.5, 표준편차 2.0으로 스케일 조정 (모멘텀)
X[:, 1] = X[:, 1] * 1.5 - 0.2   # 두 번째 특성: 평균 -0.2, 표준편차 1.5로 스케일 조정 (거래량 변화율)

# 모멘텀 > 0.3 이고 거래량 변화율 > -0.1 이면 매수 신호(1), 아니면 매도/보류(0)
y = ((X[:, 0] > 0.3) & (X[:, 1] > -0.1)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # 70% 학습, 30% 테스트로 분리, 시드 고정
)

model = SVC(kernel='linear')   # 선형 커널 SVM 생성 (직선 결정 경계로 두 클래스를 분리)
model.fit(X_train, y_train)    # 학습 데이터로 최적 결정 경계(초평면) 학습

y_pred = model.predict(X_test)           # 테스트 샘플에 대한 매수/매도 신호 예측
acc = accuracy_score(y_test, y_pred)     # 예측 정확도 계산
print(f"신호 분류 정확도: {acc:.2f}")    # 정확도 출력


def plot_signal_boundary(features, labels, clf):
    step = 0.02  # 메시그리드 격자 간격 (작을수록 결정 경계가 세밀하게 그려짐)
    x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1  # x축 범위: 데이터 최솟값-1 ~ 최댓값+1
    y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1  # y축 범위: 데이터 최솟값-1 ~ 최댓값+1
    # 지정 범위를 step 간격으로 격자화하여 2D 메시그리드 생성 (결정 경계 시각화용)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    # 메시그리드의 모든 격자점을 1D로 펼쳐 SVM으로 예측 후 원래 격자 형태로 복원
    grid_pred = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, grid_pred, cmap=plt.cm.coolwarm, alpha=0.25)  # 결정 영역을 색으로 채움 (투명도 0.25)
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap=plt.cm.coolwarm, edgecolors='k')  # 실제 데이터 산점도 (경계선 포함)
    plt.title(f"SVM 매매 신호 경계 (accuracy: {acc:.2f})")  # 정확도를 제목에 표시
    plt.xlabel("모멘텀")          # x축: 모멘텀 값
    plt.ylabel("거래량 변화율")   # y축: 거래량 변화율 값
    plt.tight_layout()            # 레이블이 잘리지 않도록 여백 자동 조정
    plt.savefig("SvmSample.png", dpi=150, bbox_inches="tight")  # 고해상도 PNG로 저장
    print("그래프 저장: SvmSample.png")  # 저장 완료 메시지 출력


plot_signal_boundary(X, y, model)  # 전체 데이터와 학습된 모델로 결정 경계 시각화 함수 호출
