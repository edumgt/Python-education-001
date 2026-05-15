import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

# 1) 가상 주가 데이터 생성 (추세 + 주기 + 노이즈)
days = 220
t = np.arange(days)
prices = 100 + 0.08 * t + 2.5 * np.sin(t / 8) + np.random.normal(0, 0.5, days)

# 2) 최근 5일 -> 다음 1일 예측 형태로 변환
window_size = 5
X, y = [], []
for i in range(days - window_size):
    X.append(prices[i:i + window_size])
    y.append(prices[i + window_size])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# 3) 정규화
x_min, x_max = X.min(), X.max()
y_min, y_max = y.min(), y.max()
norm_eps = 1e-8
X_norm = (X - x_min) / (x_max - x_min + norm_eps)
y_norm = (y - y_min) / (y_max - y_min + norm_eps)

# 4) 학습/테스트 분리
split = int(len(X_norm) * 0.8)
X_train, X_test = X_norm[:split], X_norm[split:]
y_train, y_test = y_norm[:split], y_norm[split:]

# 5) 간단 1-은닉층 신경망
input_size = window_size
hidden_size = 8
output_size = 1
learning_rate = 0.05
epochs = 2000
sigmoid_clip_range = 500
weight_init_scale = 0.1

W1 = np.random.randn(input_size, hidden_size) * weight_init_scale
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * weight_init_scale
b2 = np.zeros((1, output_size))


def sigmoid(x):
    clipped = np.clip(x, -sigmoid_clip_range, sigmoid_clip_range)
    return 1 / (1 + np.exp(-clipped))


def sigmoid_derivative(activated):
    return activated * (1 - activated)


for epoch in range(epochs):
    z1 = X_train @ W1 + b1
    a1 = sigmoid(z1)
    y_pred = a1 @ W2 + b2

    loss = np.mean((y_pred - y_train) ** 2)

    d_y_pred = 2 * (y_pred - y_train) / len(y_train)
    d_W2 = a1.T @ d_y_pred
    d_b2 = np.sum(d_y_pred, axis=0, keepdims=True)

    d_a1 = d_y_pred @ W2.T
    d_z1 = d_a1 * sigmoid_derivative(a1)
    d_W1 = X_train.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    W2 -= learning_rate * d_W2
    b2 -= learning_rate * d_b2
    W1 -= learning_rate * d_W1
    b1 -= learning_rate * d_b1

    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

# 6) 테스트 MAE
z1_test = X_test @ W1 + b1
a1_test = sigmoid(z1_test)
y_test_pred_norm = a1_test @ W2 + b2

y_test_real = y_test * (y_max - y_min) + y_min
y_test_pred = y_test_pred_norm * (y_max - y_min) + y_min
mae = np.mean(np.abs(y_test_real - y_test_pred))
print(f"\n테스트 MAE: {mae:.4f}")

# 7) 마지막 5일로 다음 날 주가 예측
last_window = prices[-window_size:]
last_window_norm = (last_window - x_min) / (x_max - x_min + norm_eps)
last_window_norm = np.clip(last_window_norm, 0.0, 1.0)
z1_next = last_window_norm.reshape(1, -1) @ W1 + b1
a1_next = sigmoid(z1_next)
next_day_norm = a1_next @ W2 + b2
next_day_price = next_day_norm[0, 0] * (y_max - y_min) + y_min
print(f"다음 날 예측 주가: {next_day_price:.2f}")

# 8) 시각화
plt.figure(figsize=(10, 4))
plt.plot(y_test_real, label="실제 주가", linewidth=2)
plt.plot(y_test_pred, label="예측 주가", linestyle="--")
plt.title("역전파 기반 주가 예측 결과")
plt.xlabel("테스트 인덱스")
plt.ylabel("주가")
plt.legend()
plt.tight_layout()
plt.show()
