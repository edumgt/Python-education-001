import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import korean_font  # noqa: F401

os.makedirs("result", exist_ok=True)

print("=" * 65)
print("  PyTorch LSTM: 주가 시계열 예측 실습")
print("=" * 65)
print()
print("  LSTM 게이트 구조:")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  망각 게이트(f): 과거 기억을 얼마나 지울지 결정       │")
print("  │  입력 게이트(i): 새 정보를 얼마나 기억할지 결정       │")
print("  │  출력 게이트(o): 은닉 상태로 무엇을 내보낼지 결정     │")
print("  │  셀 상태(C_t) : 장기 기억 — RNN의 가장 큰 개선점     │")
print("  └──────────────────────────────────────────────────────┘")

torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cpu")

# ── 1. 데이터 생성 ─────────────────────────────────────────
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
    days = 300
    t = np.arange(days, dtype=float)
    prices = (100 + 0.12 * t + 6 * np.sin(t / 18) + np.random.normal(0, 1.5, days)).astype(np.float32)
    print(f"   → 가상 {days}일치 주가 생성")

days = len(prices)
print(f"   → {days}일치 주가  |  최소: {prices.min():.1f}  최대: {prices.max():.1f}")
time.sleep(0.3)

# ── 2. 슬라이딩 윈도우 & 정규화 ───────────────────────────
print("\n[2/8] 슬라이딩 윈도우 변환 & Min-Max 정규화 중...")
print("   입력: 최근 20일 주가  →  출력: 다음 날 주가 1개")
time.sleep(0.5)
SEQ_LEN = 20
p_min, p_max = prices.min(), prices.max()
norm = (prices - p_min) / (p_max - p_min + 1e-8)

X_list, y_list = [], []
for i in range(len(norm) - SEQ_LEN):
    X_list.append(norm[i:i + SEQ_LEN])
    y_list.append(norm[i + SEQ_LEN])

X_all = np.array(X_list, dtype=np.float32)          # (샘플, 20)
y_all = np.array(y_list, dtype=np.float32)

split = int(len(X_all) * 0.8)
X_train = torch.tensor(X_all[:split]).unsqueeze(-1).to(DEVICE)   # (N, 20, 1)
X_test  = torch.tensor(X_all[split:]).unsqueeze(-1).to(DEVICE)
y_train = torch.tensor(y_all[:split]).unsqueeze(-1).to(DEVICE)   # (N, 1)
y_test  = torch.tensor(y_all[split:]).unsqueeze(-1).to(DEVICE)
print(f"   → 학습: {X_train.shape}  테스트: {X_test.shape}")
time.sleep(0.3)

# ── 3. LSTM 모델 정의 ─────────────────────────────────────
print("\n[3/8] LSTM 모델 정의 중...")
print("   구조: LSTM(1 → 32) → Dropout(0.2) → Linear(32 → 1)")
time.sleep(0.5)


class LSTMStock(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, seq, hidden)
        out = self.dropout(out[:, -1, :])  # 마지막 타임스텝만 사용
        return self.fc(out)


model = LSTMStock().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f"   → 총 파라미터 수: {total_params:,}개")
print(model)
time.sleep(0.5)

# ── 4. 손실함수 & 옵티마이저 ──────────────────────────────
print("\n[4/8] 학습 설정 중 (MSE Loss + Adam)...")
print("   MSE Loss: 예측값과 실제값의 평균 제곱 오차")
print("   Adam: 학습률을 파라미터마다 자동으로 조정하는 최적화 알고리즘")
time.sleep(0.5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
EPOCHS     = 300
BATCH_SIZE = 32
print(f"   → 에폭={EPOCHS}  배치={BATCH_SIZE}  초기 lr=0.001")
time.sleep(0.3)

# ── 5. 학습 루프 ──────────────────────────────────────────
print("\n[5/8] LSTM 학습 시작...")
print("   순전파 → MSE 손실 → 역전파(autograd) → Adam 업데이트")
time.sleep(0.8)

loss_history = []
prev_loss = None
model.train()
for epoch in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    epoch_loss = 0.0
    for start in range(0, len(X_train), BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        xb, yb = X_train[idx], y_train[idx]

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(xb)

    scheduler.step()
    avg_loss = epoch_loss / len(X_train)
    loss_history.append(avg_loss)

    if epoch % 50 == 0:
        trend = ""
        if prev_loss is not None:
            trend = " ↓" if avg_loss < prev_loss else " →"
        lr_now = optimizer.param_groups[0]['lr']
        print(f"   Epoch {epoch:4d} | Loss: {avg_loss:.6f}{trend}  lr={lr_now:.5f}")
        prev_loss = avg_loss
        time.sleep(0.25)

print(f"   Epoch {EPOCHS:4d} | 학습 완료!")
time.sleep(0.4)

# ── 6. 테스트 평가 ─────────────────────────────────────────
print("\n[6/8] 테스트 세트 평가 중...")
time.sleep(0.4)
model.eval()
with torch.no_grad():
    pred_norm = model(X_test).cpu().numpy().flatten()
    true_norm = y_test.cpu().numpy().flatten()

pred_real = pred_norm * (p_max - p_min) + p_min
true_real = true_norm * (p_max - p_min) + p_min
mae  = np.mean(np.abs(pred_real - true_real))
rmse = np.sqrt(np.mean((pred_real - true_real) ** 2))
print(f"   → 테스트 MAE:  {mae:.4f}원")
print(f"   → 테스트 RMSE: {rmse:.4f}원")
time.sleep(0.3)

# ── 7. 다음 날 예측 ────────────────────────────────────────
print("\n[7/8] 내일 주가 예측 중...")
time.sleep(0.4)
last_seq = torch.tensor(norm[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
with torch.no_grad():
    next_norm = model(last_seq).item()
next_price = next_norm * (p_max - p_min) + p_min
print(f"   → 오늘 실제 주가: {prices[-1]:.2f}원")
print(f"   → 내일 예측 주가: {next_price:.2f}원  "
      f"({'▲ 상승' if next_price > prices[-1] else '▼ 하락'} 예상)")
time.sleep(0.3)

# ── 8. 시각화 ─────────────────────────────────────────────
print("\n[8/8] 시각화 저장 중...")
time.sleep(0.5)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 전체 그래프 요약 제목
fig.text(0.5, 0.98, '장기 기억(셀 상태)과 단기 기억(은닉 상태)을 분리해 더 오래된 주가 흐름도 기억합니다',
         ha='center', fontsize=9, color='#333', weight='bold')

# 손실 곡선
axes[0].plot(loss_history, color='steelblue', linewidth=1.5)
axes[0].set_title("LSTM 학습 손실 곡선 (MSE)")
axes[0].set_xlabel("에폭")
axes[0].set_ylabel("MSE 손실")
axes[0].grid(alpha=0.3)
# 초반 급격한 하락 어노테이션
axes[0].annotate('빠르게 배우는 중!',
                 xy=(0.05, 0.80), xytext=(0.20, 0.92),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='darkred')
# 후반 평탄 어노테이션
axes[0].annotate('더 배워도 크게 좋아지지 않아요 (수렴)',
                 xy=(0.85, 0.15), xytext=(0.45, 0.35),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
axes[0].text(0.5, -0.18, '손실(MSE)이 낮을수록 예측값이 실제값에 가깝습니다',
             transform=axes[0].transAxes, ha='center', fontsize=7, color='gray')

# 실제 vs 예측
axes[1].plot(true_real, color='steelblue', linewidth=1.5, label='실제 주가')
axes[1].plot(pred_real, color='tomato',    linewidth=1.5, linestyle='--', label='LSTM 예측')
axes[1].set_title(f"LSTM 주가 예측  |  MAE={mae:.2f}원  RMSE={rmse:.2f}원")
axes[1].set_xlabel("테스트 인덱스")
axes[1].set_ylabel("주가")
axes[1].legend()
axes[1].grid(alpha=0.3)
# 실제선 설명
axes[1].text(0.02, 0.92, '실제 주가', transform=axes[1].transAxes,
             fontsize=7, color='steelblue', ha='left')
# 예측선 설명
axes[1].text(0.02, 0.84, 'LSTM이 예측한 주가', transform=axes[1].transAxes,
             fontsize=7, color='tomato', ha='left')
# 잘 맞춘 구간 어노테이션
axes[1].annotate('잘 맞춘 구간',
                 xy=(0.25, 0.55), xytext=(0.10, 0.35),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='#333')
# 틀린 구간 어노테이션
axes[1].annotate('틀린 구간 (급등락은 어려워요)',
                 xy=(0.75, 0.85), xytext=(0.50, 0.70),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=7, color='darkred')
axes[1].text(0.5, -0.18, '두 선이 가까울수록 예측이 잘 된 것입니다',
             transform=axes[1].transAxes, ha='center', fontsize=7, color='gray')

plt.tight_layout()
plt.subplots_adjust(top=0.90)
ticker_tag = TICKER.replace('.', '_')
plt.savefig(f"result/LstmStockPyTorch_{ticker_tag}.png", dpi=150, bbox_inches="tight")
print(f"   → 그래프 저장: result/LstmStockPyTorch_{ticker_tag}.png")

print("\n✓ PyTorch LSTM 주가 예측 실습 완료!\n")
