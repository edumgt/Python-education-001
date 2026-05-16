import matplotlib.pyplot as plt  # 그래프 그리기 라이브러리
import numpy as np               # 수치 계산 및 배열 연산 라이브러리
import korean_font  # noqa: F401 # 한글 폰트 설정 모듈 (경고 억제용 주석 포함)

np.random.seed(42)  # 난수 시드 고정 → 실행마다 동일한 결과 보장

# 1) 가상 주가 데이터 생성 (추세 + 주기 + 노이즈)
days = 220  # 전체 거래일 수 (약 10개월치 데이터)
t = np.arange(days)  # 0, 1, 2, ... 219 로 구성된 시간 인덱스 배열
# 선형 추세(0.08*t) + 사인 파동(주기적 등락) + 정규분포 노이즈로 가상 주가 생성
prices = 100 + 0.08 * t + 2.5 * np.sin(t / 8) + np.random.normal(0, 0.5, days)

# 2) 최근 5일 -> 다음 1일 예측 형태로 변환
window_size = 5  # 입력으로 사용할 과거 일수 (슬라이딩 윈도우 크기)
X, y = [], []    # 입력(특성) 리스트, 출력(정답) 리스트 초기화
for i in range(days - window_size):  # 윈도우가 끝까지 슬라이드할 수 있는 횟수만큼 반복
    X.append(prices[i:i + window_size])  # i번째부터 5일치 주가를 입력으로 추가
    y.append(prices[i + window_size])    # 5일 이후 다음 날 주가를 정답으로 추가

X = np.array(X)           # 리스트를 2D NumPy 배열로 변환 (샘플 수 × 5)
y = np.array(y).reshape(-1, 1)  # 1D 배열을 열벡터(샘플 수 × 1)로 변환

# 3) 정규화
x_min, x_max = X.min(), X.max()  # X 전체의 최솟값과 최댓값 추출
y_min, y_max = y.min(), y.max()  # y 전체의 최솟값과 최댓값 추출
norm_eps = 1e-8  # 분모가 0이 되는 것을 방지하는 아주 작은 값
X_norm = (X - x_min) / (x_max - x_min + norm_eps)  # X를 0~1 범위로 Min-Max 정규화
y_norm = (y - y_min) / (y_max - y_min + norm_eps)  # y를 0~1 범위로 Min-Max 정규화

# 4) 학습/테스트 분리
split = int(len(X_norm) * 0.8)              # 전체의 80%를 학습용으로 사용할 인덱스 계산
X_train, X_test = X_norm[:split], X_norm[split:]  # X를 학습/테스트 세트로 분리
y_train, y_test = y_norm[:split], y_norm[split:]  # y를 학습/테스트 세트로 분리

# 5) 간단 1-은닉층 신경망
input_size = window_size  # 입력층 뉴런 수 = 윈도우 크기(5)
hidden_size = 8           # 은닉층 뉴런 수 (하이퍼파라미터)
output_size = 1           # 출력층 뉴런 수 = 다음 날 주가 1개
learning_rate = 0.05      # 경사하강법의 학습률 (가중치 업데이트 보폭)
epochs = 2000             # 전체 학습 데이터를 반복 학습할 횟수
sigmoid_clip_range = 500  # sigmoid 계산 시 exp 오버플로 방지용 클리핑 범위
weight_init_scale = 0.1   # 가중치 초기화 시 표준편차 스케일 (너무 크면 기울기 폭발)

W1 = np.random.randn(input_size, hidden_size) * weight_init_scale  # 입력→은닉 가중치 행렬 (5×8), 작은 난수로 초기화
b1 = np.zeros((1, hidden_size))                                     # 은닉층 편향 벡터 (1×8), 0으로 초기화
W2 = np.random.randn(hidden_size, output_size) * weight_init_scale  # 은닉→출력 가중치 행렬 (8×1), 작은 난수로 초기화
b2 = np.zeros((1, output_size))                                     # 출력층 편향 스칼라 (1×1), 0으로 초기화


def sigmoid(x):
    clipped = np.clip(x, -sigmoid_clip_range, sigmoid_clip_range)  # exp(-x) 계산 전 값 클리핑하여 오버플로 방지
    return 1 / (1 + np.exp(-clipped))  # 시그모이드 함수: 입력을 0~1 사이 확률값으로 변환


def sigmoid_derivative(activated):
    return activated * (1 - activated)  # 시그모이드 미분: f(x)*(1-f(x)), 역전파에서 기울기 계산에 사용


for epoch in range(epochs):  # 지정한 에폭 수만큼 학습 반복
    # --- 순전파(Forward Pass) ---
    z1 = X_train @ W1 + b1      # 입력층 → 은닉층 선형 변환 (행렬곱 + 편향)
    a1 = sigmoid(z1)            # 은닉층에 시그모이드 활성화 함수 적용
    y_pred = a1 @ W2 + b2       # 은닉층 → 출력층 선형 변환 (회귀이므로 활성화 없음)

    loss = np.mean((y_pred - y_train) ** 2)  # MSE 손실: 예측값과 정답의 평균 제곱 오차

    # --- 역전파(Backward Pass) ---
    d_y_pred = 2 * (y_pred - y_train) / len(y_train)  # 출력층에서의 MSE 기울기 (손실의 y_pred에 대한 미분)
    d_W2 = a1.T @ d_y_pred                             # W2의 기울기: 은닉층 출력의 전치 × 출력 기울기
    d_b2 = np.sum(d_y_pred, axis=0, keepdims=True)     # b2의 기울기: 배치 방향으로 합산

    d_a1 = d_y_pred @ W2.T                      # 은닉층 출력(a1)에 대한 기울기: 출력 기울기 × W2 전치
    d_z1 = d_a1 * sigmoid_derivative(a1)        # 시그모이드 미분을 곱해 은닉층 입력(z1)의 기울기 계산
    d_W1 = X_train.T @ d_z1                     # W1의 기울기: 입력의 전치 × z1 기울기
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)  # b1의 기울기: 배치 방향으로 합산

    # --- 가중치 업데이트(경사하강법) ---
    W2 -= learning_rate * d_W2  # 학습률 × 기울기만큼 W2를 손실이 줄어드는 방향으로 이동
    b2 -= learning_rate * d_b2  # 학습률 × 기울기만큼 b2 업데이트
    W1 -= learning_rate * d_W1  # 학습률 × 기울기만큼 W1 업데이트
    b1 -= learning_rate * d_b1  # 학습률 × 기울기만큼 b1 업데이트

    if epoch % 400 == 0:  # 400 에폭마다 한 번씩 학습 현황 출력
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")  # 현재 에폭 번호와 손실값 출력

# 6) 테스트 MAE
z1_test = X_test @ W1 + b1      # 테스트 입력을 학습된 W1, b1로 선형 변환
a1_test = sigmoid(z1_test)      # 은닉층 활성화 적용
y_test_pred_norm = a1_test @ W2 + b2  # 출력층 계산 → 정규화된 예측값

y_test_real = y_test * (y_max - y_min) + y_min            # 정규화된 정답을 원래 주가 스케일로 역변환
y_test_pred = y_test_pred_norm * (y_max - y_min) + y_min  # 정규화된 예측값을 원래 주가 스케일로 역변환
mae = np.mean(np.abs(y_test_real - y_test_pred))           # MAE: 예측과 실제의 절대 오차 평균
print(f"\n테스트 MAE: {mae:.4f}")  # 테스트 세트 MAE 출력

# 7) 마지막 5일로 다음 날 주가 예측
last_window = prices[-window_size:]  # 전체 주가에서 마지막 5일 슬라이스
last_window_norm = (last_window - x_min) / (x_max - x_min + norm_eps)  # 학습 때와 동일한 기준으로 정규화
last_window_norm = np.clip(last_window_norm, 0.0, 1.0)                  # 이상치 방지용 0~1 클리핑
z1_next = last_window_norm.reshape(1, -1) @ W1 + b1  # 1×5 형태로 변환 후 은닉층 선형 변환
a1_next = sigmoid(z1_next)                            # 은닉층 활성화 적용
next_day_norm = a1_next @ W2 + b2                     # 출력층 계산 → 정규화된 예측값
next_day_price = next_day_norm[0, 0] * (y_max - y_min) + y_min  # 역정규화로 실제 주가 단위로 변환
print(f"다음 날 예측 주가: {next_day_price:.2f}")  # 내일 예측 주가 출력

# 8) 시각화
plt.figure(figsize=(10, 4))                           # 가로 10인치, 세로 4인치 그래프 창 생성
plt.plot(y_test_real, label="실제 주가", linewidth=2)  # 실제 주가 실선으로 표시
plt.plot(y_test_pred, label="예측 주가", linestyle="--")  # 예측 주가 점선으로 표시
plt.title("역전파 기반 주가 예측 결과")  # 그래프 제목 설정
plt.xlabel("테스트 인덱스")              # x축 레이블: 테스트 샘플 순서
plt.ylabel("주가")                       # y축 레이블: 주가 단위
plt.legend()                             # 범례(실제/예측 구분) 표시
plt.tight_layout()                       # 레이블이 잘리지 않도록 여백 자동 조정
plt.savefig("BackpropStockSample.png", dpi=150, bbox_inches="tight")  # 고해상도 PNG로 저장
print("그래프 저장: BackpropStockSample.png")  # 저장 완료 메시지 출력
