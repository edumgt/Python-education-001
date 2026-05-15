import numpy as np
import matplotlib.pyplot as plt


# 재현 가능한 난수 시드
np.random.seed(42)

# 1) 쉬운 가상 주가 데이터 생성 (추세 + 주기 + 작은 노이즈)
days = 220
t = np.arange(days)
prices = 100 + 0.08 * t + 2.5 * np.sin(t / 8) + np.random.normal(0, 0.5, days)

# 2) 시계열을 지도학습 형태로 변환 (최근 5일 -> 다음 1일)
window_size = 5
X, y = [], []
for i in range(days - window_size):
    X.append(prices[i:i + window_size])
    y.append(prices[i + window_size])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# 3) 정규화 (학습 안정화를 위해 0~1 스케일)
x_min, x_max = X.min(), X.max()
y_min, y_max = y.min(), y.max()
X_norm = (X - x_min) / (x_max - x_min)
y_norm = (y - y_min) / (y_max - y_min)

# 4) 훈련/테스트 분할
split = int(len(X_norm) * 0.8)
X_train, X_test = X_norm[:split], X_norm[split:]
y_train, y_test = y_norm[:split], y_norm[split:]

# 5) 신경망 구조 (입력 5 -> 은닉층 8 -> 출력 1), 역전파 직접 구현
input_size = window_size
hidden_size = 8
output_size = 1
learning_rate = 0.05
epochs = 2000

W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


for epoch in range(epochs):
    # 순전파
    z1 = X_train @ W1 + b1
    a1 = sigmoid(z1)
    y_pred = a1 @ W2 + b2

    # 손실 (MSE)
    loss = np.mean((y_pred - y_train) ** 2)

    # 역전파
    d_y_pred = 2 * (y_pred - y_train) / len(y_train)
    d_W2 = a1.T @ d_y_pred
    d_b2 = np.sum(d_y_pred, axis=0, keepdims=True)

    d_a1 = d_y_pred @ W2.T
    d_z1 = d_a1 * sigmoid_derivative(a1)
    d_W1 = X_train.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # 가중치 업데이트
    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1

    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

# 6) 테스트 평가
z1_test = X_test @ W1 + b1
a1_test = sigmoid(z1_test)
y_test_pred_norm = a1_test @ W2 + b2

y_test_real = y_test * (y_max - y_min) + y_min
y_test_pred = y_test_pred_norm * (y_max - y_min) + y_min
mae = np.mean(np.abs(y_test_real - y_test_pred))
print(f"\n테스트 MAE: {mae:.4f}")

# 7) 마지막 5일로 다음 날 주가 예측
last_window = prices[-window_size:]
last_window_norm = (last_window - x_min) / (x_max - x_min)
z1_next = last_window_norm.reshape(1, -1) @ W1 + b1
a1_next = sigmoid(z1_next)
next_day_norm = a1_next @ W2 + b2
next_day_price = next_day_norm[0, 0] * (y_max - y_min) + y_min
print(f"다음 날 예측 주가: {next_day_price:.2f}")

# 8) 시각화
plt.figure(figsize=(10, 4))
plt.plot(y_test_real, label="실제 주가", linewidth=2)
plt.plot(y_test_pred, label="예측 주가", linestyle="--")
plt.title("주가 예측 결과")
plt.xlabel("테스트 구간 인덱스")
plt.ylabel("주가")
plt.legend()
plt.tight_layout()
plt.show()
