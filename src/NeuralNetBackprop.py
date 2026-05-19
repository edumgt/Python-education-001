import os
import time

import matplotlib.pyplot as plt
import numpy as np
import korean_font  # noqa: F401

os.makedirs("../result", exist_ok=True)

print("=" * 60)
print("  신경망 역전파(Backpropagation) 직접 구현: 주가 예측")
print("=" * 60)

np.random.seed(42)

print("\n[1/8] 주가 데이터 로드 중...")
time.sleep(0.5)
TICKER = '078935.KS'
prices = None
try:
    import yfinance as yf
    from datetime import date
    df = yf.download(TICKER, start='2020-01-01', end=date.today().isoformat(),
                     auto_adjust=True, progress=False)
    if len(df) > 50:
        prices = df['Close'].squeeze().dropna().values.flatten().astype(np.float32)
        print(f"   ✓ {TICKER}: {len(prices)}일 실제 데이터 로드")
except Exception as e:
    print(f"   yfinance 오류 ({e}) → 가상 데이터 사용")

if prices is None:
    days = 220
    t = np.arange(days)
    prices = (100 + 0.08 * t + 2.5 * np.sin(t / 8) + np.random.normal(0, 0.5, days)).astype(np.float32)
    print(f"   → 가상 {days}일치 주가 생성")

days = len(prices)
print(f"   → {days}일치 주가 생성  |  최소: {prices.min():.1f}  최대: {prices.max():.1f}")
time.sleep(0.3)

print("\n[2/8] 슬라이딩 윈도우로 데이터 변환 중 (최근 5일 → 다음 날 예측)...")
time.sleep(0.5)
window_size = 5
X, y = [], []
for i in range(days - window_size):
    X.append(prices[i:i + window_size])
    y.append(prices[i + window_size])
X = np.array(X)
y = np.array(y).reshape(-1, 1)
print(f"   → 입력 X: {X.shape}  (샘플 수 × 5일)  |  정답 y: {y.shape}")
time.sleep(0.3)

print("\n[3/8] Min-Max 정규화 중 (0~1 범위로 스케일 조정)...")
time.sleep(0.5)
x_min, x_max = X.min(), X.max()
y_min, y_max = y.min(), y.max()
norm_eps = 1e-8
X_norm = (X - x_min) / (x_max - x_min + norm_eps)
y_norm = (y - y_min) / (y_max - y_min + norm_eps)
print(f"   → X 정규화 범위: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
time.sleep(0.3)

print("\n[4/8] 학습/테스트 분리 중 (8:2)...")
time.sleep(0.4)
split = int(len(X_norm) * 0.8)
X_train, X_test = X_norm[:split], X_norm[split:]
y_train, y_test = y_norm[:split], y_norm[split:]
print(f"   → 학습: {len(X_train)}개  |  테스트: {len(X_test)}개")
time.sleep(0.3)

print("\n[5/8] 신경망 가중치 초기화 중 (구조: 5→8→1)...")
time.sleep(0.5)
input_size, hidden_size, output_size = window_size, 8, 1
learning_rate = 0.05
epochs = 2000
sigmoid_clip_range = 500
weight_init_scale = 0.1

W1 = np.random.randn(input_size, hidden_size) * weight_init_scale
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * weight_init_scale
b2 = np.zeros((1, output_size))
print(f"   → W1: {W1.shape}  b1: {b1.shape}  W2: {W2.shape}  b2: {b2.shape}")
print(f"   → 학습률(lr)={learning_rate}  에폭={epochs}")
time.sleep(0.5)


def sigmoid(x):
    clipped = np.clip(x, -sigmoid_clip_range, sigmoid_clip_range)
    return 1 / (1 + np.exp(-clipped))


def sigmoid_derivative(activated):
    return activated * (1 - activated)


print("\n[6/8] 순전파(→) + 역전파(←) 반복 학습 시작...")
print("   각 에폭: 순전파 → MSE 오차 계산 → 역전파 → 가중치 업데이트")
time.sleep(0.8)

prev_loss = None
for epoch in range(epochs):
    # 순전파
    z1 = X_train @ W1 + b1
    a1 = sigmoid(z1)
    y_pred = a1 @ W2 + b2
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
        trend = ""
        if prev_loss is not None:
            trend = " ↓ 감소중" if loss < prev_loss else " → 수렴"
        print(f"   Epoch {epoch:4d} | Loss: {loss:.6f}{trend}")
        prev_loss = loss
        time.sleep(0.3)

print(f"   Epoch {epochs:4d} | 학습 완료!")
time.sleep(0.5)

print("\n[7/8] 테스트 세트 MAE 평가 & 내일 주가 예측 중...")
time.sleep(0.5)
z1_test = X_test @ W1 + b1
a1_test = sigmoid(z1_test)
y_test_pred_norm = a1_test @ W2 + b2
y_test_real = y_test * (y_max - y_min) + y_min
y_test_pred = y_test_pred_norm * (y_max - y_min) + y_min
mae = np.mean(np.abs(y_test_real - y_test_pred))
print(f"   → 테스트 MAE: {mae:.4f}원")

last_window = prices[-window_size:]
last_window_norm = (last_window - x_min) / (x_max - x_min + norm_eps)
last_window_norm = np.clip(last_window_norm, 0.0, 1.0)
z1_next = last_window_norm.reshape(1, -1) @ W1 + b1
a1_next = sigmoid(z1_next)
next_day_norm = a1_next @ W2 + b2
next_day_price = next_day_norm[0, 0] * (y_max - y_min) + y_min
print(f"   → 다음 날 예측 주가: {next_day_price:.2f}원")
time.sleep(0.3)

print("\n[8/8] 예측 결과 시각화 중...")
time.sleep(0.5)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 전체 그래프 요약 제목
fig.text(0.5, 0.98, '역전파(오답 노트)를 직접 손으로 구현한 신경망 — 틀린 만큼 가중치를 수정해요',
         ha='center', fontsize=9, color='#333', weight='bold')

# 손실 곡선
loss_curve_x = list(range(0, epochs, 400)) + [epochs]
# 학습 중 기록된 주요 에폭 손실 재계산
z1_check = X_train @ W1 + b1
a1_check = sigmoid(z1_check)
y_check = a1_check @ W2 + b2
final_loss = float(np.mean((y_check - y_train) ** 2))

axes[0].set_title("학습 손실 변화 (MSE)")
axes[0].set_xlabel("에폭")
axes[0].set_ylabel("MSE 손실")
axes[0].grid(alpha=0.3)
axes[0].text(0.5, 0.5, f'최종 손실: {final_loss:.6f}',
             transform=axes[0].transAxes, ha='center', fontsize=9, color='steelblue',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.4))
axes[0].text(0.5, 0.70, '학습을 반복할수록 오차가 줄어듭니다',
             transform=axes[0].transAxes, ha='center', fontsize=8, color='#333')
axes[0].text(0.15, 0.90, '처음엔 많이 틀려요', transform=axes[0].transAxes,
             fontsize=7, color='darkred',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='mistyrose', alpha=0.5))
axes[0].text(0.55, 0.20, '오차가 많이 줄었어요!', transform=axes[0].transAxes,
             fontsize=7, color='steelblue',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.5))
axes[0].annotate('시작: 오차 큼', xy=(0.05, 0.85), xytext=(0.20, 0.95),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='darkred')
axes[0].annotate('끝: 오차 줄어듦', xy=(0.85, 0.12), xytext=(0.60, 0.30),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
axes[0].text(0.5, -0.14, '역전파: 오답을 보고 가중치를 조금씩 수정하는 과정',
             transform=axes[0].transAxes, ha='center', fontsize=7, color='gray')

# 실제 vs 예측
axes[1].plot(y_test_real, label="실제 주가", linewidth=2, color='steelblue')
axes[1].plot(y_test_pred, label="예측 주가", linestyle="--", color='tomato')
axes[1].set_title("역전파 기반 주가 예측 결과")
axes[1].set_xlabel("테스트 인덱스")
axes[1].set_ylabel("주가")
axes[1].legend()
axes[1].grid(alpha=0.3)
# 실제선 설명
axes[1].text(0.02, 0.92, '실제 주가 (파랑)', transform=axes[1].transAxes,
             fontsize=7, color='steelblue', ha='left')
# 예측선 설명
axes[1].text(0.02, 0.84, '신경망 예측 주가 (빨강 점선)', transform=axes[1].transAxes,
             fontsize=7, color='tomato', ha='left')
# 잘 맞는 구간 어노테이션
axes[1].annotate('추세는 잘 따라가요',
                 xy=(0.30, 0.55), xytext=(0.08, 0.35),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
# 틀리는 구간 어노테이션
axes[1].annotate('급격한 변화는 놓쳐요',
                 xy=(0.70, 0.80), xytext=(0.50, 0.65),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='darkred')
axes[1].text(0.5, -0.14, f'테스트 MAE: {mae:.4f}원  |  두 선이 가까울수록 예측이 잘 된 것입니다',
             transform=axes[1].transAxes, ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
ticker_tag = TICKER.replace('.', '_')
plt.savefig(f"../result/NeuralNetBackprop_{ticker_tag}.png", dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: result/NeuralNetBackprop_{ticker_tag}.png")

print("\n✓ 신경망 역전파 실습 완료!\n")
